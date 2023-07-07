# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:21:05 2023

@author: mittal
"""

#%%
# imports
import zipfile
import pandas as pd
import numpy as np
import os
import nibabel
import matplotlib.pyplot as plt

#%%
# Check if the target folder for storing the data already exists. If not, create it.
if not os.path.exists('./fMRI_data'):
    os.mkdir('./fMRI_data')

# Specify the path to the ZIP file
zip_file_path = 'C:\\Users\\mittal\\Downloads\\siddharth\\coding\\fMRI\DATA\\sid-fMRI-data\\sid_fMRI.zip'

# Extract the ZIP file to the target folder
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('./fMRI_data')
    
#%%Direct Images
import nibabel as nib

# Define the file paths
file_paths = [
    "sub-sidtest_ses-001_task-bar_run-01_hemi-L_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-01_hemi-R_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-02_hemi-L_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-02_hemi-R_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz",
    "sub-sidtest_ses-001_task-bar_run-0102avg_hemi-R_bold.nii.gz",
    "task-bar_apertures.nii.gz"
]

# Loop through each file and load the data
for file_path in file_paths:
    img = nib.load(file_path)
    data = img.get_fdata()

    # Create the MRI image
    mri_image = nib.Nifti1Image(data, img.affine, img.header)

    # Save the MRI image
    output_path = file_path.replace(".nii.gz", ".nii")
    nib.save(mri_image, output_path)
    print(f"Saved MRI image as: {output_path}")


#%%Plot images
# Create a subplot with 6 columns and 1 row
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 4))

# Loop through each file path and display the MRI image
for i, file_path in enumerate(file_paths):
    img = nib.load(file_path)
    data = img.get_fdata()

    # Reshape the data to remove singleton dimensions
    data = data.squeeze()

    # Display every 10th image (interval of 10 images)
    image_to_display = data[..., ::10]

    # Choose the corresponding subplot and display the image
    ax = axes[i]
    ax.imshow(image_to_display, cmap='gray')
    ax.set_title(file_path)

# Adjust the layout and display the subplots
plt.tight_layout()
plt.show()