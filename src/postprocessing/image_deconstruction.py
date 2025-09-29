# Description: This script is used to slice images into patches of a specified size. The patches are saved in a specified directory.
# Import necessary libraries
import os
import numpy as np
from PIL import Image
from patchify import patchify

# Define the size of patches to split each image into
patch_size = (448, 448, 3)
step_size = 416  # Step size for patchifying

# Function to slice images in a directory
def slice_images(image_dir, patches_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"): 
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = np.array(img)

            # Create patches
            patches = patchify(img, patch_size, step=step_size)

            # Save patches
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    single_patch = patches[i, j, :, :, :]
                    single_patch = np.squeeze(single_patch)  # Remove single-dimensional entries from the shape of an array.
                    patch_filename = f"{filename.split('.')[0]}_patch_{i}_{j}.jpg"
                    patch_filepath = os.path.join(patches_dir, patch_filename)
                    Image.fromarray(np.uint8(single_patch)).save(patch_filepath)

