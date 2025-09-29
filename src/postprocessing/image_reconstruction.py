# Description: Script to reconstruct image from patches
# Import necessary libraries
import os
import numpy as np
from PIL import Image

# Define the size of patches to split each image into
patch_size = (448, 448, 3)
step_size = 416  # Step size for patchifying

# Function to reconstruct images from patches
def reconstruct_images(patches_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all patch filenames
    patch_files = [f for f in os.listdir(patches_dir) if f.endswith(".jpg")]

    if not patch_files:
        print(f"No patch files found in {patches_dir}. Exiting reconstruction process.")
        return None

    # Group patches by base name
    patches_by_base_name = {}
    for filename in patch_files:
        if '_patch_' in filename:
            parts = filename.split('_patch_')
            base_name = parts[0]
            if base_name not in patches_by_base_name:
                patches_by_base_name[base_name] = []
            patches_by_base_name[base_name].append(filename)

    # Reconstruct images for each base name
    for base_name, filenames in patches_by_base_name.items():
        print(f"Reconstructing image for base name: {base_name}")

        # Extract patch IDs from filenames
        patch_ids = []
        for filename in filenames:
            parts = filename.split('_patch_')
            ij_part = parts[1].split('.')[0]
            i, j = map(int, ij_part.split('_'))
            patch_id = (i, j)
            patch_ids.append(patch_id)

        if not patch_ids:
            print(f"No valid patch IDs extracted from filenames for {base_name}. Skipping.")
            continue

        # Determine image dimensions based on patch IDs
        max_i = max(patch_ids, key=lambda x: x[0])[0] + 1
        max_j = max(patch_ids, key=lambda x: x[1])[1] + 1
        img_height = max_i * step_size + patch_size[0] - step_size
        img_width = max_j * step_size + patch_size[1] - step_size
        #print(f"Determined image dimensions: height={img_height}, width={img_width}")
        reconstructed_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Load patches and reconstruct image
        for patch_id in patch_ids:
            i, j = patch_id
            patch_filename = f"{base_name}_patch_{i}_{j}.jpg"
            patch_filepath = os.path.join(patches_dir, patch_filename)
            
            if not os.path.exists(patch_filepath):
                #print(f"Patch file {patch_filename} not found. Skipping this patch.")
                continue
            
            patch_img = Image.open(patch_filepath)
            patch_img = np.array(patch_img)
            start_i = i * step_size
            start_j = j * step_size
            end_i = start_i + patch_img.shape[0]
            end_j = start_j + patch_img.shape[1]
            #print(f"Placing patch {patch_filename} at: start_i={start_i}, end_i={end_i}, start_j={start_j}, end_j={end_j}")
            reconstructed_img[start_i:end_i, start_j:end_j, :] = patch_img

        if np.max(reconstructed_img) == 0:
            #print(f"Reconstructed image for {base_name} is empty. Check patch loading and dimensions.")
            continue

        # Save reconstructed image
        reconstructed_img = Image.fromarray(reconstructed_img)
        output_path = os.path.join(output_dir, f'reconstructed_{base_name}.jpg')
        reconstructed_img.save(output_path)
        #print(f"Reconstructed image saved to: {output_path}")

    return output_dir

