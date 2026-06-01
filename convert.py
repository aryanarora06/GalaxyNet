import argparse
import os

import h5py
from PIL import Image

CLASS_NAMES = [
    'Disturbed', 'Merging', 'Round_Smooth', 'Inbetween_Smooth',
    'Cigar_Smooth', 'Barred_Spiral', 'Unbarred_Tight_Spiral',
    'Unbarred_Loose_Spiral', 'Edge_On_No_Bulge', 'Edge_On_Bulge'
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Galaxy10_DECals.h5 into ImageFolder class directories."
    )
    parser.add_argument(
        "--h5-path",
        default=os.environ.get("GALAXY_H5_PATH", "Galaxy10_DECals.h5"),
        help="Path to Galaxy10_DECals.h5.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("GALAXY_DATA_DIR", "data"),
        help="Directory to write ImageFolder-style class folders.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    with h5py.File(args.h5_path, 'r') as f:
        images = f['images'][:]   # (17736, 256, 256, 3), uint8
        labels = f['ans'][:]      # (17736,), int

    for cls_name in CLASS_NAMES:
        os.makedirs(os.path.join(args.out_dir, cls_name), exist_ok=True)

    for i, (img, lbl) in enumerate(zip(images, labels)):
        cls_name = CLASS_NAMES[int(lbl)]
        path = os.path.join(args.out_dir, cls_name, f'{i:05d}.png')
        Image.fromarray(img).save(path)

    print(f"Done. Wrote {len(labels)} images to {args.out_dir}")

if __name__ == "__main__":
    main()
