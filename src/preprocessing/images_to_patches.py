# Split images into patches so they can be annotated

# Import libraries
import os
import numpy as np
from PIL import Image
from patchify import patchify

# Define the path to the data
source_dir = 'Dataset/'

training_dir = source_dir + 'Training/'
validation_dir = source_dir + 'Validation/'
test_dir = source_dir + 'Test/'

# Define the size of patches to split each image into
patch_size = (448, 448, 3)

# Function to slice images in a directory
def slice_images(directory):
    # Create a subdirectory for patches
    patches_dir = directory + 'Patches/'
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"): 
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img = np.array(img)

            # Create patches
            patches = patchify(img, patch_size, step=416)

            # Save patches
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    single_patch = patches[i, j, :, :, :]
                    single_patch = np.squeeze(single_patch)  # Remove single-dimensional entries from the shape of an array.
                    Image.fromarray(np.uint8(single_patch)).save(patches_dir + filename.split('.')[0] + '_patch_' + str(i) + '_' + str(j) + '.jpg')

# Slice images in each directory
slice_images(training_dir)
slice_images(validation_dir)
slice_images(test_dir)