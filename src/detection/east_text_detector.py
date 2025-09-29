# Description: Script to extract text from images using the EAST text detector and Tesseract OCR.
# Import necessary libraries
import os
import cv2
import numpy as np
import pytesseract
import csv
import re

# Specify Tesseract executable path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

# Function to extract patch ID from file name
def extract_patch_id(filename):
    match = re.search(r'patch_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

# Function to process an image using the EAST text detector and Tesseract OCR
def detect_text(image_path, model_path, output_dir, patch_id, newW=448, newH=448, min_confidence=0.3):
    print(f"Processing image: {os.path.basename(image_path)}")

    # Load the pre-trained EAST model
    net = cv2.dnn.readNet(model_path)

    # Use CUDA if available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Specify output layer names for EAST text detector
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Initialize lists to store extracted texts and bounding box coordinates
    extracted_texts = []
    bounding_boxes = []

    # Read the original image
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"Failed to load image: {image_path}")
        return [], ""

    (H, W) = orig.shape[:2]

    # Resize the original image
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(orig, (newW, newH))
    (H, W) = image.shape[:2]

    # Prepare the image for processing
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Extract bounding boxes and confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Apply non-maxima suppression to suppress weak overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(rects, confidences, 0.3, 0.4)

    # Extract text using Tesseract OCR within each bounding box
    if len(indices) > 0:
        for i in indices.flatten():
            (startX, startY, endX, endY) = rects[i]
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # Draw bounding box on the original image (before resizing)
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the region of interest (ROI) using the bounding box coordinates
            roi = orig[startY:endY, startX:endX]

            # Check if ROI is valid before resizing and processing
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                # Resize ROI if needed (adjust dimensions as necessary)
                roi = cv2.resize(roi, (roi.shape[1]*4, roi.shape[0]*4))  # Resize by as needed for better OCR results

                # Use Tesseract to OCR the cropped region
                text = pytesseract.image_to_string(roi, config='--psm 12')  # Adjust PSM (Page Segmentation Mode) as needed
                extracted_texts.append(text.strip())  # Append extracted text to list

                # Save bounding box coordinates adjusted for original image size
                bounding_boxes.append((startX, startY, endX, endY))

    # Save extracted texts to a text file with patch ID in the name
    # Remove patch_id from the text_filename to prevent appending to the start
    text_filename = f"text_extraction_{os.path.basename(image_path)}.txt"
    text_filepath = os.path.join(output_dir, text_filename)
    with open(text_filepath, 'w', encoding='utf-8') as text_file:
        for text in extracted_texts:
            text_file.write(text + '\n')

    # Save bounding box coordinates to a CSV file with patch ID in the name
    # Remove patch_id from the csv_filename to prevent appending to the start
    csv_filename = f"bounding_boxes_{os.path.basename(image_path)}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    with open(csv_filepath, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['startX', 'startY', 'endX', 'endY'])  # Write header
        for bbox in bounding_boxes:
            csv_writer.writerow(bbox)

    # Save image with bounding boxes drawn for verification
    output_image_path = os.path.join(output_dir, f"{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, orig)

    return extracted_texts, csv_filepath

# Function to load image paths from a directory
def load_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(directory, filename))
    return image_paths