# Script to convert PDF files to images
# Importing required libraries
import os
import subprocess
from pdf2image import convert_from_path
from pathlib import Path

# Function to check if Poppler is installed
def check_poppler():
    try:
        subprocess.run(["pdfinfo", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print("Poppler is not installed or not found in PATH.")
        raise e

# Function to convert a single PDF to images
def convert_pdf_to_images(pdf_path, output_folder, asset_name, output_format='jpg'):
    images = convert_from_path(pdf_path)
    base_filename = Path(pdf_path).stem
    for i, image in enumerate(images):
        image = image.convert("L")  # Grayscale
        output_filename = output_folder / f"{asset_name}_{base_filename}_{i + 1}.{output_format.lower()}"
        image.save(output_filename)
    print(f"Converted: {pdf_path.name} -> {len(images)} images")

# Function to convert all PDFs in asset folders
def convert_pdfs_for_assets(base_data_path, asset_names, output_format='jpg'):
    check_poppler()
    for asset in asset_names:
        input_folder = base_data_path / asset / 'original_pids'
        output_folder = base_data_path / asset / 'converted_images'
        output_folder.mkdir(parents=True, exist_ok=True)

        pdf_files = [pdf for pdf in input_folder.glob("*.pdf")]
        for pdf_file in pdf_files:
            convert_pdf_to_images(pdf_file, output_folder, asset, output_format=output_format)

# Base directory for assets
base_data_path = Path('C:/Users/Stuart/Python/PID_MLOPS/digitised-pid-mlops/data')
asset_names = ['GE', 'Scott']

# Run the conversion for all assets
convert_pdfs_for_assets(base_data_path, asset_names)