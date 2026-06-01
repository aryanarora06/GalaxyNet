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

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DATA_DIR = '/kaggle/input/datasets/aryansanjeevarora/galaxy-morphology/data'

BATCH_SIZE    = 32
IMAGE_SIZE    = 260   
EPOCHS        = 30    
PATIENCE      = 5     
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 5     

NUM_WORKERS = 4
PIN_MEMORY  = True

CUTMIX_ALPHA = 1.0
MIXUP_ALPHA  = 0.2
CUTMIX_PROB  = 0.0
MIXUP_PROB   = 0.3

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP    = torch.cuda.is_available()
AMP_DEVICE = 'cuda' if USE_AMP else 'cpu'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 2. In-Memory Data Pre-loading (With OOM Guard)
# ==========================================
print("Scanning directory structure...")
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
                          persistent_workers=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=True, drop_last=True)

# ==========================================
# 6. Native v2 CutMix & MixUp Implementation
# ==========================================
# IMPROVEMENT: Removed 30+ lines of custom logic. Using highly optimized native v2 implementations.
cutmix = v2.CutMix(num_classes=NUM_CLASSES, alpha=CUTMIX_ALPHA)
mixup  = v2.MixUp(num_classes=NUM_CLASSES, alpha=MIXUP_ALPHA)

# ==========================================
# 7. Model Setup (EfficientNet-B2)
# ==========================================
print("\nInitializing EfficientNet-B2 with pre-trained ImageNet weights...")
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(device)

_v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
_torch_ge_2 = _v >= (2, 0)

if _torch_ge_2 and torch.cuda.is_available():
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
_base_model = getattr(model, '_orig_mod', model)

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
        clean_state_dict = {k.replace('_orig_mod.', ''): v
                            for k, v in model.state_dict().items()}
        torch.save(clean_state_dict, 'best_galaxy_model.pth')
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
TTA_N = 8

print("\nLoading best model for test evaluation...")
best_model = models.efficientnet_b2(weights=None)

best_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(best_model.classifier[1].in_features, NUM_CLASSES)
)
best_model.load_state_dict(
    torch.load('best_galaxy_model.pth', map_location=device, weights_only=True)
)
best_model = best_model.to(device)
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
    persistent_workers=True,
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
    num_workers=NUM_WORKERS, persistent_workers=True, drop_last=False
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