# Script to split images by asset into train, validation, and test sets
# Importing required libraries
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def extract_base_name(filename: str) -> str:
    """Strip _hires or _ml suffix from a filename stem."""
    return filename.rsplit("_", 1)[0]

def split_images_by_base(asset_name, source_dir, output_dir, train_pct=0.7, val_pct=0.2, test_pct=0.1):
    # Group all image files by base name
    image_files = list(source_dir.glob("*.jpg"))
    base_to_files = defaultdict(list)

    for img_path in image_files:
        base = extract_base_name(img_path.stem)
        base_to_files[base].append(img_path)

    base_names = list(base_to_files.keys())
    random.shuffle(base_names)

    total = len(base_names)
    train_end = int(total * train_pct)
    val_end = train_end + int(total * val_pct)

    splits = {
        "train": base_names[:train_end],
        "val": base_names[train_end:val_end],
        "test": base_names[val_end:]
    }

    for split_name, split_bases in splits.items():
        split_dir = output_dir / split_name / asset_name
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for base in split_bases:
            for file_path in base_to_files[base]:
                if file_path.stem.endswith("_ml"):
                    dest = split_dir / file_path.name
                    shutil.copy(file_path, dest)
                    count += 1

        print(f"✅ {asset_name}: {count} _ml images copied to {split_name}/")

def split_all_assets(base_data_path, asset_names):
    output_dir = base_data_path / "splits"
    for asset in asset_names:
        source_dir = base_data_path / asset / "converted_images"
        if not source_dir.exists():
            print(f"⚠️ Skipping {asset} — no converted_images found.")
            continue
        split_images_by_base(asset, source_dir, output_dir)

# --- Run ---
base_data_path = Path("C:/Users/Stuart/Python/PID_MLOPS/digitised-pid-mlops/data")
asset_names = ["GE", "Scott"]

split_all_assets(base_data_path, asset_names)
