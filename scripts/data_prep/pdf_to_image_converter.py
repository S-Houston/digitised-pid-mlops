# Script to convert PDF files to images
# Importing required libraries
import os
import os
import subprocess
from pdf2image import convert_from_path
from pathlib import Path

# Function to check if Poppler is installed
def check_poppler():
    try:
        subprocess.run(["pdfinfo", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Check if Poppler is installed
    except Exception as e:
        print("Poppler is not installed or not found in PATH.")
        raise e

# Function to convert PDF to images
def convert_pdf_to_images(pdf_path, output_format='jpg', output_folder='output_images'):
    check_poppler() # Check if Poppler is installed
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)
    base_filename = Path(pdf_path).stem
    for i, image in enumerate(images):
        image = image.convert("L")
        output_filename = f"{output_folder}/{base_filename}_{i + 1}.{output_format.lower()}"
        image.save(output_filename)

# Function to convert all PDFs in a folder to images
def convert_all_pdfs_in_folder(input_folder, output_folder='output_images', output_format='jpg'):
    pdf_files = [filename for filename in os.listdir(input_folder) if filename.lower().endswith(".pdf")]
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_filename)
        convert_pdf_to_images(pdf_path, output_format=output_format, output_folder=output_folder)

# Convert all PDFs in a folder to images
input_folder = Path('C:/Users/Stuart/Documents/Honours_Project/Dataset/Demo/Original Images')
output_folder = Path('C:/Users/Stuart/Documents/Honours_Project/Dataset/Demo/Images')

# Call the function to convert all PDFs in a folder to images
convert_all_pdfs_in_folder(input_folder, output_folder=output_folder)