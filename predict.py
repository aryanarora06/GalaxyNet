import argparse
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from PIL import Image

CLASS_NAMES = [
    'Barred_Spiral', 'Cigar_Smooth', 'Disturbed', 'Edge_On_Bulge',
    'Edge_On_No_Bulge', 'Inbetween_Smooth', 'Merging', 'Round_Smooth',
    'Unbarred_Loose_Spiral', 'Unbarred_Tight_Spiral'
]
NUM_CLASSES = len(CLASS_NAMES)

def parse_args():
    parser = argparse.ArgumentParser(description="Classify one galaxy image.")
    parser.add_argument("image_path", nargs="?", help="Path to a galaxy image.")
    parser.add_argument("--model-path", default="best_galaxy_model.pth", help="Path to trained checkpoint.")
    parser.add_argument("--image-size", type=int, default=260)
    parser.add_argument("--tta-n", type=int, default=8)
    return parser.parse_args()

args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP    = torch.cuda.is_available()
AMP_DEVICE = 'cuda' if USE_AMP else 'cpu'

# ==========================================
# Load Model
# ==========================================
model = models.efficientnet_b2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ==========================================
# Transforms
# ==========================================
# Normalize is applied once here, not inside the TTA loop
base_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((args.image_size, args.image_size), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Spatial augmentations only — no normalization
tta_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomChoice([
        v2.RandomRotation((0, 0)),
        v2.RandomRotation((90, 90)),
        v2.RandomRotation((180, 180)),
        v2.RandomRotation((270, 270))
    ]),
])

# ==========================================
# Predict
# ==========================================
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = base_transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    accumulated = None
    with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
        for _ in range(args.tta_n):
            probs = torch.softmax(model(tta_transform(tensor)), dim=1)
            accumulated = probs if accumulated is None else accumulated + probs

    avg_probs   = accumulated / args.tta_n
    confidence, pred_idx = avg_probs.squeeze().max(dim=0)

    print(f"Image     : {image_path}")
    print(f"Prediction: {CLASS_NAMES[pred_idx.item()]}")
    print(f"Confidence: {confidence.item() * 100:.1f}%")
    print()
    print("All class probabilities:")
    for name, prob in zip(CLASS_NAMES, avg_probs.squeeze().tolist()):
        bar = '█' * int(prob * 40)
        print(f"  {name:<25} {prob * 100:5.1f}%  {bar}")

if __name__ == '__main__':
    image_path = args.image_path or input("Enter path to image: ").strip()
    predict(image_path)
