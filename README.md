# Galaxy Morphology Classifier

EfficientNet-B2 fine-tuned to classify galaxy images into 10 morphological categories, trained on the [Galaxy10 DECals Dataset](https://astronn.readthedocs.io/en/latest/galaxy10.html).

## Classes

Classes are assigned alphabetically by folder name — this is the order used by both `train.py` and `predict.py`.

## Technical Overview

**Model:** EfficientNet-B2 pretrained on ImageNet, with the classifier head replaced by `Dropout(0.3) → Linear(1408 → 10)`. On PyTorch ≥ 2.0 with CUDA, the model is compiled with `torch.compile` for extra speed.

**Data:** All 17,736 images are loaded into RAM as `uint8` arrays before training, eliminating disk I/O across epochs. A guard halts caching if RAM exceeds 90%. The dataset is split 80/10/10 (train/val/test) with stratification to preserve class proportions.

**Training:** AdamW optimizer with differential learning rates — the backbone is fine-tuned at `1.5e-5` and the head at `3e-4`. The schedule is a linear warmup over 5 epochs followed by cosine annealing. Mixed precision (`torch.amp`) is used throughout. Class imbalance is handled by `WeightedRandomSampler`. Early stopping triggers after 5 epochs without val accuracy improvement.

**Augmentation:** Training applies random horizontal/vertical flips, random 90° rotations, and MixUp (`α=0.2`, 30% of batches). Because PyTorch's `CrossEntropyLoss` does not accept soft targets with `label_smoothing`, mixed batches are routed to a separate criterion without smoothing.

**Inference:** Test-Time Augmentation (TTA) runs 8 stochastic forward passes per image with random flips and rotations, averaging the softmax probabilities for the final prediction.

## Requirements

```
torch
torchvision
numpy
scikit-learn
tqdm
psutil
h5py
Pillow
```

All of the above are pre-installed on Kaggle.

## Files

| File | Description |
|------|-------------|
| `convert.py` | Converts `Galaxy10_DECals.h5` to an `ImageFolder`-compatible directory |
| `train.py` | Full training and evaluation script |
| `predict.py` | Classifies a single image using the trained model |
| `best_galaxy_model.pth` | Best model weights — saved automatically during training |

## Step 1 — Download the Dataset

Download `Galaxy10_DECals.h5` (link above).

## Step 2 — Convert to Image Folders

Open `convert.py` and set the paths (change paths):

```python
H5_PATH = r'C:\path\to\Galaxy10_DECals.h5'
OUT_DIR = r'C:\path\to\data'
```

Then run:

```bash
python convert.py
```

This produces a `data/` folder organised by class:

```
data/
  Barred_Spiral/
  Cigar_Smooth/
  Disturbed/
  Edge_On_Bulge/
  Edge_On_No_Bulge/
  Inbetween_Smooth/
  Merging/
  Round_Smooth/
  Unbarred_Loose_Spiral/
  Unbarred_Tight_Spiral/
```

## Step 3 — Zip the Data Folder

Use 7-Zip or other compression utilities.

## Step 4 — Upload to Kaggle and Train

1. Go to kaggle.com
2. Create a new notebook
3. Upload `data.zip` and add dataset. **Name it exactly `galaxy-morphology`** and UPDATE DATA PATH IN train.py (path of data folder)
4. Set the accelerator to **GPU T4 x2**
5. Upload `train.py` and run:

```bash
python train.py
```

Training runs for up to 30 epochs with early stopping. The best checkpoint is saved to `best_galaxy_model.pth` in the notebook's working directory. Final test accuracy (with and without TTA) is printed at the end.

## Step 5 — Download the Model

In the Kaggle notebook sidebar, go to **Output** and download `best_galaxy_model.pth` to your local machine.

## Step 6 — Run Inference

Place `best_galaxy_model.pth` in the same directory as `predict.py`, then run (update path:

```bash
# Pass image path directly
python predict.py path/to/galaxy.jpg

# Or run and enter path when prompted
python predict.py
```

Example output:

```
Image     : galaxy.jpg
Prediction: Barred_Spiral
Confidence: 87.3%

All class probabilities:
  Barred_Spiral              87.3%  ██████████████████████████████████
  Unbarred_Tight_Spiral       6.1%  ██
  Unbarred_Loose_Spiral       3.4%  █
  ...
```
