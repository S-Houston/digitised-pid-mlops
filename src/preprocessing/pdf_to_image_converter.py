# Script to convert PDF files to images
# Importing required libraries
import os
import subprocess
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import csv

# Allow very large images
Image.MAX_IMAGE_PIXELS = None

# Conversion settings
DPI = 400
OUTPUT_FORMAT = 'jpg'
MAX_SAFE_PIXELS = 250_000_000
MAX_WIDTH = 8000
MAX_HEIGHT = 8000
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

def convert_pdf_to_images(pdf_path, output_folder, asset_name, suspicious_folder, log_entries):
    try:
        images = convert_from_path(
            pdf_path,
            dpi=DPI,
            use_cropbox=False,
            thread_count=4,
            grayscale=False
        )
    except Exception as e:
        log_entries.append((pdf_path.name, asset_name, "conversion_failure", str(e)))
        return

    base_filename = Path(pdf_path).stem
    for i, image in enumerate(images):
        image = image.convert("RGB")
        width, height = image.size
        print(f"Processing {pdf_path.name} page {i+1} | size: {width}x{height}")

        output_stem = f"{asset_name}_{base_filename}_{i + 1}"
        full_output = output_folder / f"{output_stem}_hires.{OUTPUT_FORMAT}"
        ml_output = output_folder / f"{output_stem}_ml.{OUTPUT_FORMAT}"

        # Handle suspiciously small images
        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            small_img_path = suspicious_folder / f"{output_stem}_suspicious.{OUTPUT_FORMAT}"
            image.save(small_img_path)
            log_entries.append((pdf_path.name, asset_name, "tiny_image_saved", f"{width}x{height}"))
            print(f"Saved suspicious image: {small_img_path.name}")
            continue

        # Save high-resolution version
        image.save(full_output)
        print(f"Saved full-res: {full_output.name} | Size: {image.size}")

        # Save ML version (downscaled if too large)
        if width * height > MAX_SAFE_PIXELS:
            image_ml = image.copy()
            image_ml.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)
            image_ml.save(ml_output)
            print(f"Saved downscaled ML version: {ml_output.name} | Size: {image_ml.size}")
        else:
            image.save(ml_output)

def convert_pdfs_for_assets(base_data_path, asset_names):
    log_entries = []

    for asset in asset_names:
        input_folder = base_data_path / asset / 'original_pids'
        output_folder = base_data_path / asset / 'converted_images'
        suspicious_folder = base_data_path / asset / 'suspicious_images'

        output_folder.mkdir(parents=True, exist_ok=True)
        suspicious_folder.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found for asset {asset} in {input_folder}")
            continue

        for pdf_file in pdf_files:
            try:
                convert_pdf_to_images(pdf_file, output_folder, asset, suspicious_folder, log_entries)
            except Exception as e:
                msg = f"Unexpected failure for {pdf_file.name}: {e}"
                print(msg)
                log_entries.append((pdf_file.name, asset, "unknown_error", str(e)))

    # Write log to CSV
    log_path = base_data_path / "conversion_log.csv"
    with open(log_path, mode='w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Filename", "Asset", "Error Type", "Message"])
        writer.writerows(log_entries)
    print(f"\nConversion log saved to: {log_path}")

# Run the conversion
base_data_path = Path('C:/Users/Stuart/Python/PID_MLOPS/digitised-pid-mlops/data')
asset_names = ['GE', 'Scott']
convert_pdfs_for_assets(base_data_path, asset_names)

