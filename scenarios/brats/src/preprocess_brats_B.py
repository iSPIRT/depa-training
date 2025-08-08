import os
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Note: This script is designed to preprocess the BraTS dataset by extracting the middle axial slice from each NIfTI file and saving as PNG.

# Commented out since NiFTI files are bulky and time consuming for this demo. The preprocessed datasets are already present in the repository.
# To run the preprocessing step if you have the NIfTI files available in the specified input directories (scenario/mri-tumor-segmentation/data), uncomment the code below.

'''
def get_middle_axial_slice(nifti_path):
    """Load NIfTI file and return center axial slice"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Get center axial slice
    axial_slices = data.shape[2]
    center_slice = data[:, :, axial_slices // 2]
    
    return center_slice


def normalize_slice(slice_data):
    """Normalize slice to 0-255 range"""
    slice_data = slice_data.astype(np.float32)
    if np.max(slice_data) > 0:  # avoid division by zero
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    return slice_data.astype(np.uint8)


## Process all NIfTI files in directory structure and save as PNGs (middle axial slice)

input_root = "/mnt/input/data"
output_root = "/mnt/output/preprocessed/"

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith('.nii.gz'):
            # Create output path maintaining structure
            rel_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process NIfTI file
            input_path = os.path.join(root, file)
            try:
                center_slice = get_middle_axial_slice(input_path)
                normalized_slice = normalize_slice(center_slice)
                
                # Create PNG filename (replace .nii.gz with .png)
                png_filename = file.replace('.nii.gz', '.png')
                output_path = os.path.join(output_dir, png_filename)
                
                # Save as PNG
                Image.fromarray(normalized_slice).save(output_path)
                print(f"Processed: {input_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
'''

print("Preprocessed BraTS_B dataset saved to data/brats_B/preprocessed/")