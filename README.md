# Galaxy Morphology Classifier

EfficientNet-B2 fine-tuned to classify galaxy images into 10 morphological categories, trained on the [Galaxy10 DECals Dataset](https://astronn.readthedocs.io/en/latest/galaxy10.html).

## Classes

`train.py` uses `torchvision.datasets.ImageFolder`, so class ids are assigned alphabetically by folder name. `predict.py` uses the same alphabetical class order:

```text
Barred_Spiral
Cigar_Smooth
Disturbed
Edge_On_Bulge
Edge_On_No_Bulge
Inbetween_Smooth
Merging
Round_Smooth
Unbarred_Loose_Spiral
Unbarred_Tight_Spiral
```

## Requirements

```bash
pip install -r requirements.txt
```

On Kaggle, these packages are usually already installed.

## Files

| File | Description |
|------|-------------|
| `convert.py` | Converts `Galaxy10_DECals.h5` to an ImageFolder-compatible directory |
| `train.py` | Full training and evaluation script |
| `predict.py` | Classifies a single image using the trained model |
| `requirements.txt` | Python dependencies |
| `best_galaxy_model.pth` | Best model weights, saved automatically during training |

## Use the Kaggle Dataset

Add this dataset to a Kaggle notebook:

```text
aryansanjeevarora/galaxy-morphology
```

Kaggle normally mounts it at:

```text
/kaggle/input/galaxy-morphology
```

If the dataset contains the class folders directly, run:

```bash
python train.py --data-dir /kaggle/input/galaxy-morphology
```

If the dataset contains a nested `data/` folder, run:

```bash
python train.py --data-dir /kaggle/input/galaxy-morphology/data
```

Training runs for up to 30 epochs with early stopping. The best checkpoint is saved as `best_galaxy_model.pth` in the notebook working directory.

## Convert From H5

If you have the original `Galaxy10_DECals.h5` file instead of class folders:

```bash
python convert.py --h5-path path/to/Galaxy10_DECals.h5 --out-dir data
python train.py --data-dir data
```

This creates:

```text
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

## Local Kaggle CLI Download

If you want to download the Kaggle dataset locally, install/configure Kaggle credentials first, then run:

```bash
pip install kaggle
kaggle datasets download -d aryansanjeevarora/galaxy-morphology -p dataset --unzip
python train.py --data-dir dataset/data
```

If the unzipped folder contains class folders directly instead of `dataset/data`, use:

```bash
python train.py --data-dir dataset
```

## Quick Smoke Test

For a fast local sanity check on a small machine, use fewer epochs/workers. `--no-pretrained` avoids downloading ImageNet weights:

```bash
python train.py --data-dir data --epochs 1 --batch-size 4 --num-workers 0 --no-pretrained --no-compile
```

## Inference

After training, download or keep `best_galaxy_model.pth`, then classify one image:

```bash
python predict.py path/to/galaxy.jpg --model-path best_galaxy_model.pth
```

Or run interactively:

```bash
python predict.py --model-path best_galaxy_model.pth
```
