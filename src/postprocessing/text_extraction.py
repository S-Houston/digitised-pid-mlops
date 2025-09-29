# Description: This script is used to process text files created by Tesseract OCR and store the extracted text in a DataFrame.
# Import necessary libraries
import os
import pandas as pd
import re

# Function to load text files created by Tesseract OCR
def load_text_files(directory):
    text_files = []
    for filename in os.listdir(directory):
        if filename.startswith("text_extraction_") and filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                patch_id, original_filename = extract_patch_and_original_filename(filename)  # Extract patch ID and original filename
                text_files.append({'patch_id': patch_id, 'filename': original_filename, 'text': text})
    return pd.DataFrame(text_files)

# Function to extract patch ID and original filename from file name
def extract_patch_and_original_filename(filename):
    match = re.search(r'_patch_(\d+_\d+)\.jpg', filename)
    if match:
        patch_id = match.group(1)
        original_filename = filename.split('_patch_')[0].replace('text_extraction_', '')
        return patch_id, original_filename
    else:
        return None, None

# Define a function to clean text data
def clean_text(text):
    # Cleaning process
    text = text.replace('\n', ' ')  # Replace newline characters with spaces
    text = ' '.join(text.split())   # Remove extra whitespace
    return text

# Main function to process text files and store in DataFrame
def process_text_files(text_detection_dir):
    df = load_text_files(text_detection_dir)
    
    if df.empty or 'text' not in df.columns:
        print(f"No text files found or 'text' column missing in DataFrame from {text_detection_dir}.")
        return pd.DataFrame()  # Return empty DataFrame or handle the error as needed
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Consolidate extracted text based on original filename
    consolidated_df = df.groupby('filename')['cleaned_text'].apply(lambda x: ' '.join(x)).reset_index()
    consolidated_df.columns = ['filename', 'consolidated_text']
    
    # Save the consolidated extracted text to a CSV file
    consolidated_csv_filename = os.path.join(text_detection_dir, 'consolidated_extracted_text.csv')
    consolidated_df.to_csv(consolidated_csv_filename, index=False)
    
    print(f"Consolidated extracted text saved to: {consolidated_csv_filename}")
    
    return consolidated_df