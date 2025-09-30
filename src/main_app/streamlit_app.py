# Description: Streamlit application for object and text detection on images.
# Import necessary libraries
import streamlit as st
import os
import re
import pandas as pd
from PIL import Image, ImageDraw
import torch
import glob
import sys

# Import custom scripts
from src.detection.yolo_object_detection import detect_objects_and_draw_boxes, detect_objects, load_model
from src.detection.east_text_detector import detect_text
from src.postprocessing.text_extraction import process_text_files
from src.postprocessing.image_deconstruction import slice_images
from src.postprocessing.image_reconstruction import reconstruct_images

# Switch between full Dataset or Demo mode
demo_mode = True

# Base source directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset'))

if demo_mode:
    source_dir = os.path.join(base_dir, 'Demo')
else:
    source_dir = base_dir

# Define directories
image_dir = os.path.join(source_dir, 'Images')
patches_dir = os.path.join(source_dir, 'Patches')
object_detection_dir = os.path.join(source_dir, 'ObjectDetection')
text_detection_dir = os.path.join(source_dir, 'TextDetection')
output_dir = os.path.join(source_dir, 'Output')

# Get the most recent training run directory for YOLOv5 model
runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov5', 'runs', 'train'))
run_dirs = sorted(glob.glob(os.path.join(runs_dir, 'exp*')), key=os.path.getmtime, reverse=True)
if run_dirs:
    latest_run_dir = run_dirs[0]
    model_path = os.path.join(latest_run_dir, 'weights', 'best.pt')
else:
    raise FileNotFoundError("No YOLOv5 training runs found. Please train the model first.")

model = load_model(model_path)

# Load the pre-trained EAST text detection model
text_model_path = r"C:\Users\Stuart\Python\PID_MLOPS\digitised-pid-mlops\src\detection\models\frozen_east_text_detection.pb"
print("File exists:", os.path.exists(text_model_path))

# Streamlit app
st.title('P&ID Image Processing Application')
st.subheader('Carries out object detection, text detection & extraction')
st.subheader('Reconstruction of Images with Bounding Boxes')
st.subheader('Displays count of objects and extracted text for each image, and provides download option for consolidated extracted text CSV.')

# Step 1: Slice the original image into patches
if not os.path.exists(patches_dir) or not os.listdir(patches_dir):
    st.subheader("Step 1: Slicing images into patches")
    slice_images(image_dir, patches_dir)

# Step 2: Perform object detection on patches and draw bounding boxes
if 'object_boxes' not in st.session_state:
    st.subheader("Step 2: Performing object detection on patches")
    st.session_state.object_boxes = detect_objects_and_draw_boxes(patches_dir, object_detection_dir, model)
object_boxes = st.session_state.object_boxes

# Function to get counts of symbols for a specific image
def get_symbol_counts_for_image(image_filename):
    counts = {}
    for patch_filename, boxes in object_boxes.items():
        if patch_filename.startswith(image_filename):
            for box in boxes:
                class_name = box['class_name']
                if class_name in counts:
                    counts[class_name] += 1
                else:
                    counts[class_name] = 1
    return counts

# Step 3: Text detection on patches, draw bounding boxes on the patches with objects and extract text
if 'extracted_texts' not in st.session_state:
    st.subheader("Step 3: Performing text detection on patches")
    extracted_texts = []
    for filename in os.listdir(object_detection_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(object_detection_dir, filename)
            patch_id = os.path.splitext(filename)[0]  # Assuming patch_id is based on filename without extension
            texts, csv_filepath = detect_text(image_path, text_model_path, text_detection_dir, patch_id)
            extracted_texts.extend(texts)
    st.session_state.extracted_texts = extracted_texts
else:
    extracted_texts = st.session_state.extracted_texts

# Step 4: Reconstruct the original image from the patches that have the bounding boxes already overlayed
if 'reconstructed_image_paths' not in st.session_state:
    st.subheader("Step 4: Reconstructing the original image from patches")
    reconstructed_image_dir = reconstruct_images(text_detection_dir, output_dir)
    if reconstructed_image_dir:
        reconstructed_image_paths = sorted(glob.glob(os.path.join(reconstructed_image_dir, '*.jpg')))
        st.session_state.reconstructed_image_paths = reconstructed_image_paths
    else:
        st.error("Failed to reconstruct the images.")
        reconstructed_image_paths = None
else:
    reconstructed_image_paths = st.session_state.reconstructed_image_paths

# Select an image to view
selected_image_path = st.selectbox('Select Image', reconstructed_image_paths, index=0) if reconstructed_image_paths else None

# Function to extract original filename from reconstructed image filename
def extract_original_filename(filename):
    match = re.search(r'reconstructed_(.+)\.jpg', filename)
    if match:
        return match.group(1)
    else:
        return None

# Step 5: Display extracted text and identified symbols for the selected image
if selected_image_path:
    st.image(Image.open(selected_image_path), caption='Selected Image', use_container_width=True)
    
    # Process text files and get the DataFrame for selected image
    df = process_text_files(text_detection_dir)
    
    original_filename = extract_original_filename(os.path.basename(selected_image_path))
    
    # Extracted text
    col1, col2 = st.columns(2)
    
    with col1:
        if not df.empty:
            selected_df = df[df['filename'] == original_filename]
            if not selected_df.empty:
                st.subheader('Extracted Text')
                st.table(selected_df[['filename', 'consolidated_text']])
            else:
                st.error(f"No extracted text found for {original_filename}.")
        else:
            st.error("Failed to process text files. Check if text files exist in the directory.")
    
    # Identified symbols
    with col2:
        st.subheader("Identified Symbols")
        symbol_counts = get_symbol_counts_for_image(original_filename)
        if symbol_counts:
            symbol_df = pd.DataFrame(list(symbol_counts.items()), columns=['Symbol', 'Count'])
            st.table(symbol_df)
        else:
            st.write("No symbols detected for the selected image.")
else:
    st.warning("No reconstructed images available.")

# Provide download option for consolidated extracted text
text_extraction_csv = os.path.join(text_detection_dir, 'consolidated_extracted_text.csv')
if os.path.exists(text_extraction_csv):
    with open(text_extraction_csv, 'rb') as file:
        st.download_button(
            label='Download Consolidated Extracted Text CSV',
            data=file,
            file_name='consolidated_extracted_text.csv',
            mime='text/csv'
        )