# Import the necessary packages
import os
import cv2
import numpy as np

# Function to load image paths from a directory
def load_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(directory, filename))
    return image_paths

# Function to process an image using the EAST text detector
def process_image(image_path, net, newW=448, newH=448):
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Resize image
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Prepare the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Get the rectangles
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
            if scoresData[x] < 0.5:
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

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

    # Draw the bounding boxes if there are any
    if len(indices) > 0:
        for i in indices.flatten():
            (startX, startY, endX, endY) = rects[i]
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Return the processed image
    return orig

# Directories for training and validation datasets
training_dir = 'C:/Users/Stuart/Documents/Honours_Project/Dataset/Training/Patches'
validation_dir = 'C:/Users/Stuart/Documents/Honours_Project/Dataset/Validation/Patches'

# Load image paths
training_image_paths = load_image_paths(training_dir)
validation_image_paths = load_image_paths(validation_dir)

# Load the pre-trained EAST model
net = cv2.dnn.readNet('C:/Users/Stuart/Documents/Honours_Project/East/frozen_east_text_detection.pb')

# Use CUDA if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define the output layer names (these are the final layers of the EAST model)
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# Process and save training images
for image_path in training_image_paths:
    processed_image = process_image(image_path, net)
    output_path = os.path.join('C:/Users/Stuart/Documents/Honours_Project/Dataset/Training/Text_Processed', os.path.basename(image_path))
    cv2.imwrite(output_path, processed_image)

# Process and save validation images
for image_path in validation_image_paths:
    processed_image = process_image(image_path, net)
    output_path = os.path.join('C:/Users/Stuart/Documents/Honours_Project/Dataset/Validation/Text_Processed', os.path.basename(image_path))
    cv2.imwrite(output_path, processed_image)
