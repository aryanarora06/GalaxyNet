import argparse
import logging
import math
import os
import random
import psutil  # NEW: Added for memory monitoring
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, datasets
from torchvision.transforms import v2  # NEW: Using v2 exclusively
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
import torch._inductor.config as inductor_config
inductor_config.max_autotune_gemm = False

def default_data_dir():
    candidates = [
        os.environ.get("GALAXY_DATA_DIR"),
        "/kaggle/input/galaxy-morphology/data",
        "/kaggle/input/galaxy-morphology",
        "data",
    ]
    return next((path for path in candidates if path and os.path.isdir(path)), candidates[1])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test EfficientNet-B2 for galaxy morphology classification."
    )
    parser.add_argument("--data-dir", default=default_data_dir(), help="ImageFolder dataset directory.")
    parser.add_argument("--output-model", default="best_galaxy_model.pth", help="Where to save the best checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=260)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument(
        "--parallel",
        choices=["auto", "none", "dataparallel"],
        default="auto",
        help="Use DataParallel automatically on multi-GPU CUDA systems, disable it, or force it.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet weights, useful for offline smoke tests.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile even when CUDA is available.")
    return parser.parse_args()

args = parse_args()

def resolve_parallel_mode(requested_mode):
    if requested_mode == "none" or not torch.cuda.is_available():
        return "none"

    gpu_count = torch.cuda.device_count()
    if requested_mode == "dataparallel" and gpu_count < 2:
        print("[!] DataParallel requested, but fewer than 2 CUDA GPUs are available. Using a single device.")
        return "none"
    if requested_mode == "auto" and gpu_count < 2:
        return "none"
    return "dataparallel"

def unwrap_model(model):
    unwrapped = model
    if isinstance(unwrapped, nn.DataParallel):
        unwrapped = unwrapped.module
    return getattr(unwrapped, "_orig_mod", unwrapped)

def clean_state_dict(model):
    cleaned = {}
    for key, value in model.state_dict().items():
        for prefix in ("module.", "_orig_mod."):
            while key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DATA_DIR = args.data_dir

BATCH_SIZE    = args.batch_size
IMAGE_SIZE    = args.image_size
EPOCHS        = args.epochs
PATIENCE      = args.patience
LEARNING_RATE = args.learning_rate
WARMUP_EPOCHS = args.warmup_epochs
OUTPUT_MODEL  = args.output_model

NUM_WORKERS = args.num_workers
PIN_MEMORY  = True
PERSISTENT_WORKERS = NUM_WORKERS > 0

CUTMIX_ALPHA = 1.0
MIXUP_ALPHA  = 0.2
CUTMIX_PROB  = 0.0
MIXUP_PROB   = 0.3

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP    = torch.cuda.is_available()
AMP_DEVICE = 'cuda' if USE_AMP else 'cpu'
GPU_COUNT  = torch.cuda.device_count() if torch.cuda.is_available() else 0
PARALLEL_MODE = resolve_parallel_mode(args.parallel)

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 2. In-Memory Data Pre-loading (With OOM Guard)
# ==========================================
print("Scanning directory structure...")
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_DIR}\n"
        "Pass --data-dir /path/to/data or set GALAXY_DATA_DIR."
    )
base_dataset = datasets.ImageFolder(root=DATA_DIR)
NUM_CLASSES  = len(base_dataset.classes)
print(f"Found {NUM_CLASSES} classes: {base_dataset.classes}")

total_images = len(base_dataset)
print(f"Pre-loading {total_images} images into system RAM as uint8...")

all_images, all_labels = [], []
for path, label in tqdm(base_dataset.samples, desc="Loading to RAM"):
    # IMPROVEMENT: Monitor RAM to prevent system crashes
    if psutil.virtual_memory().percent > 90:
        print("\n[!] WARNING: RAM usage exceeded 90%. Stopping caching early to prevent OOM!")
        break
        
    img = base_dataset.loader(path)
    all_images.append(np.array(img, dtype=np.uint8))
    all_labels.append(label)

all_labels = np.array(all_labels)
print("Data successfully cached in RAM.")

class_counts = np.bincount(all_labels)
print("\nClass distribution:")
for name, count in zip(base_dataset.classes, class_counts):
    print(f"  {name}: {count} ({count / len(all_labels) * 100:.1f}%)")

# ==========================================
# 3. Stratified Splitting
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp 
)
print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==========================================
# 4. Dataset & Native v2 Transforms
# ==========================================
class InMemoryGalaxyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# IMPROVEMENT: Upgraded entirely to v2 transforms pipeline
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomChoice([
        v2.RandomRotation((0, 0)),
        v2.RandomRotation((90, 90)),
        v2.RandomRotation((180, 180)),
        v2.RandomRotation((270, 270))
    ]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

eval_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = InMemoryGalaxyDataset(X_train, y_train, transform=train_transforms)
val_dataset   = InMemoryGalaxyDataset(X_val,   y_val,   transform=eval_transforms)
test_dataset  = InMemoryGalaxyDataset(X_test,  y_test,  transform=eval_transforms)

# ==========================================
# 5. Weighted Sampler (Class Imbalance)
# ==========================================
train_class_counts = np.bincount(y_train)
class_weights  = 1.0 / train_class_counts
sample_weights = class_weights[y_train]

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).float(),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=PERSISTENT_WORKERS, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=PERSISTENT_WORKERS, drop_last=True)

# ==========================================
# 6. Native v2 CutMix & MixUp Implementation
# ==========================================
# IMPROVEMENT: Removed 30+ lines of custom logic. Using highly optimized native v2 implementations.
cutmix = v2.CutMix(num_classes=NUM_CLASSES, alpha=CUTMIX_ALPHA)
mixup  = v2.MixUp(num_classes=NUM_CLASSES, alpha=MIXUP_ALPHA)

# ==========================================
# 7. Model Setup (EfficientNet-B2)
# ==========================================
print("\nInitializing EfficientNet-B2...")
weights = None if args.no_pretrained else models.EfficientNet_B2_Weights.DEFAULT
model = models.efficientnet_b2(weights=weights)

num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(device)

_v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
_torch_ge_2 = _v >= (2, 0)

if PARALLEL_MODE == "dataparallel":
    device_names = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
    print(f"Using DataParallel across {GPU_COUNT} GPUs: {device_names}")
    if _torch_ge_2 and torch.cuda.is_available() and not args.no_compile:
        print("Skipping torch.compile because DataParallel is enabled.")
    model = nn.DataParallel(model)
elif _torch_ge_2 and torch.cuda.is_available() and not args.no_compile:
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")

# ==========================================
# 8. Scheduler Helper
# ==========================================
def build_scheduler(optimizer, warmup_steps, total_steps):
    cosine_steps = max(total_steps - warmup_steps, 1)
    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

# ==========================================
# 9. Optimizer & Dual Loss Setup
# ==========================================
_base_model = unwrap_model(model)

optimizer = optim.AdamW([
    {'params': _base_model.features.parameters(),   'lr': LEARNING_RATE * 0.05},
    {'params': _base_model.classifier.parameters(), 'lr': LEARNING_RATE}
], weight_decay=1e-2)

# IMPROVEMENT: PyTorch CE doesn't allow label_smoothing=0.05 on soft targets (probabilities). 
# We split the criterions so clean batches get smoothing, and mixed batches get standard CE.
criterion_clean = nn.CrossEntropyLoss(label_smoothing=0.05)
criterion_mixed = nn.CrossEntropyLoss()

scheduler = build_scheduler(optimizer, warmup_steps=WARMUP_EPOCHS, total_steps=EPOCHS)
scaler    = torch.amp.GradScaler(AMP_DEVICE, enabled=USE_AMP)

# ==========================================
# 10. Training Loop (with Early Stopping)
# ==========================================
best_val_acc = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    running_loss  = 0.0
    correct_train = 0
    total_train   = 0
    total_train_samples = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_n = inputs.size(0)

        # Apply Native CutMix/MixUp at the batch level
        r = random.random()
        use_mixed = False
        if r < CUTMIX_PROB:
            inputs, labels_target = cutmix(inputs, labels)
            use_mixed = True
        elif r < CUTMIX_PROB + MIXUP_PROB:
            inputs, labels_target = mixup(inputs, labels)
            use_mixed = True
        else:
            labels_target = labels

        optimizer.zero_grad()

        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            outputs = model(inputs)
            # Route to the correct loss function based on target type
            if use_mixed:
                loss = criterion_mixed(outputs, labels_target)
            else:
                loss = criterion_clean(outputs, labels_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss        += loss.item() * batch_n
        total_train_samples += batch_n

        # We only calculate accuracy on clean batches, as mixed batch argmax is noisy
        if not use_mixed:
            _, predicted   = torch.max(outputs, 1)
            total_train   += batch_n
            correct_train += (predicted == labels).sum().item()

        train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_loss = running_loss / total_train_samples
    train_acc  = correct_train / total_train if total_train > 0 else float('nan')

    # --- Validation ---
    model.eval()
    val_loss      = 0.0
    correct_val   = 0
    total_val     = 0
    total_val_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_n = inputs.size(0)
            with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                outputs = model(inputs)
                loss    = criterion_clean(outputs, labels)
            
            val_loss          += loss.item() * batch_n
            total_val_samples += batch_n
            _, predicted       = torch.max(outputs, 1)
            total_val         += batch_n
            correct_val       += (predicted == labels).sum().item()

    val_loss = val_loss / total_val_samples
    val_acc  = correct_val / total_val
    scheduler.step()

    lrs    = [pg['lr'] for pg in optimizer.param_groups]
    lr_str = (f"LR: {lrs[0]:.2e}" if len(lrs) == 1
              else f"LR backbone: {lrs[0]:.2e} | LR head: {lrs[1]:.2e}")
    acc_str = f"{train_acc:.4f}" if not math.isnan(train_acc) else "n/a (all mixed)"
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc (clean): {acc_str} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | {lr_str}")

    # --- Early Stopping & Model Saving ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        checkpoint_state = clean_state_dict(model)
        output_dir = os.path.dirname(OUTPUT_MODEL)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(checkpoint_state, OUTPUT_MODEL)
        print(f"  --> Saved best model checkpoint (Val Acc: {best_val_acc:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"  --> No improvement. Early stopping counter: {epochs_without_improvement}/{PATIENCE}")
        if epochs_without_improvement >= PATIENCE:
            print(f"\n[!] Early stopping triggered at Epoch {epoch+1}. Moving to TTA inference.")
            break

# ==========================================
# 11. Fast GPU Test-Time Augmentation (TTA)
# ==========================================
TTA_N = args.tta_n

print("\nLoading best model for test evaluation...")
best_model = models.efficientnet_b2(weights=None)

best_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(best_model.classifier[1].in_features, NUM_CLASSES)
)
best_model.load_state_dict(
    torch.load(OUTPUT_MODEL, map_location=device, weights_only=True)
)
best_model = best_model.to(device)
if PARALLEL_MODE == "dataparallel":
    best_model = nn.DataParallel(best_model)
best_model.eval()

tta_base_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

tta_gpu_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomChoice([
        v2.RandomRotation((0, 0)),
        v2.RandomRotation((90, 90)),
        v2.RandomRotation((180, 180)),
        v2.RandomRotation((270, 270))
    ]),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

fast_tta_dataset = InMemoryGalaxyDataset(X_test, y_test, transform=tta_base_transforms)
fast_tta_loader  = DataLoader(
    fast_tta_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    drop_last=False
)

def predict_tta_gpu(model, base_tensors, n_augments=TTA_N):
    accumulated = None
    for _ in range(n_augments):
        augmented_batch = tta_gpu_transforms(base_tensors)
        with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            probs = torch.softmax(model(augmented_batch), dim=1)
        accumulated = probs if accumulated is None else accumulated + probs
    return (accumulated / n_augments).argmax(dim=1)

correct_test, total_test = 0, 0
for base_tensors, labels in tqdm(fast_tta_loader, desc=f"Testing with Fast GPU TTA (n={TTA_N})"):
    base_tensors = base_tensors.to(device)
    labels       = labels.to(device)
    predicted    = predict_tta_gpu(best_model, base_tensors, n_augments=TTA_N)
    total_test  += labels.size(0)
    correct_test += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy (GPU TTA n={TTA_N}): {correct_test / total_test * 100:.2f}%")

baseline_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS, drop_last=False
)
correct_base, total_base = 0, 0
with torch.no_grad():
    for inputs, labels in baseline_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            outputs = best_model(inputs)
        _, predicted  = torch.max(outputs, 1)
        total_base   += labels.size(0)
        correct_base += (predicted == labels).sum().item()

print(f"Final Test Accuracy (no TTA):         {correct_base / total_base * 100:.2f}%")
