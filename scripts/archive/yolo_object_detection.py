# Description: This script is used to detect objects in image patches using the YOLO object detector and draw bounding boxes around them.
# Import the necessary packages
import torch
import os
import csv
from PIL import Image, ImageDraw

# Function to load the YOLOv5 model.
def load_model(model_path):
    # Load the YOLOv5 model from the given path.
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

# Function to detect objects in an image.
def detect_objects(image_path, model):
    # Load the image.
    img = Image.open(image_path)
    results = model(img)

    class_names = model.names  # Assuming YOLO model has names attribute for class names

    boxes = []
    for *xyxy, conf, cls in results.xyxy[0]:
        box = {
            'image_filename': os.path.basename(image_path),
            'bbox_type': 'YOLOv5',
            'bbox_coordinates': xyxy,
            'confidence_score': conf.item(),
            'class_label': int(cls.item()),
            'class_name': class_names[int(cls.item())]  # Add class name
        }
        boxes.append(box)
    return boxes

# Function to draw bounding boxes on the image.
def draw_boxes(image_path, boxes, output_path):
    # Draw bounding boxes on the image and save it.
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        xyxy = box['bbox_coordinates']
        draw.rectangle(xyxy, outline="red", width=4)
    img.save(output_path)

# Function to detect objects in image patches and draw bounding boxes.
def detect_objects_and_draw_boxes(patches_dir, output_dir, model):
    # Detect objects in image patches and draw bounding boxes.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(patches_dir) if os.path.isfile(os.path.join(patches_dir, f))]
    boxes_dict = {}

    for image_file in image_files:
        image_path = os.path.join(patches_dir, image_file)
        boxes = detect_objects(image_path, model)
        boxes_dict[image_file] = boxes

        output_path = os.path.join(output_dir, image_file)
        draw_boxes(image_path, boxes, output_path)

    return boxes_dict

# Function to detect objects in image patches and save bounding box info to CSV.
def detect_objects_and_save_to_csv(patches_dir, output_dir, model):
    # Create output directory if it doesn't exist (optional, if not already done)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_csv = os.path.join(output_dir, 'bbox_info.csv')  # Full path including filename

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_filename', 'bbox_type', 'bbox_coordinates', 'confidence_score', 'class_label', 'class_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        image_files = [f for f in os.listdir(patches_dir) if os.path.isfile(os.path.join(patches_dir, f))]

        for image_file in image_files:
            image_path = os.path.join(patches_dir, image_file)
            boxes = detect_objects(image_path, model)

            for box in boxes:
                writer.writerow(box)