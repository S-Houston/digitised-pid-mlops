# Description: Runs YOLOv5 training on Windows, ensures GPU is used, and auto-launches TensorBoard.

# Import necessary libraries
import os
import subprocess
import sys
import shutil

# Fix OpenMP DLL conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths

PYTHON_EXE = sys.executable  # current Python executable
YOLO_TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov5', 'train.py'))
DATA_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataset.yaml'))
HYP_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov5', 'data', 'hyps', 'hyp.scratch-low.yaml'))
TRAIN_CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'Training', 'labels.cache.npy')
VAL_CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'Validation', 'labels.cache.npy')


# Clean up YOLO cache files
for cache_file in [TRAIN_CACHE, VAL_CACHE]:
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"Deleted cache: {cache_file}")
        except PermissionError:
            print(f"Could not delete cache (permission issue): {cache_file}")


# Build training command
train_cmd = [
    PYTHON_EXE,
    YOLO_TRAIN,
    "--weights", "yolov5m.pt",
    "--data", DATA_YAML,
    "--hyp", HYP_YAML,
    "--epochs", "100",
    "--batch-size", "16",
    "--imgsz", "448",
    "--device", "0"
]


# Run YOLO training
try:
    print(f"Running YOLOv5 training...\nCommand: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"\nERROR: YOLOv5 training failed with exit code {e.returncode}")
    print(f"Command: {e.cmd}")
    sys.exit(e.returncode)
except FileNotFoundError:
    print("\nERROR: train.py not found. Check YOLOv5 path!")
    sys.exit(1)

print("YOLOv5 training completed successfully.")

