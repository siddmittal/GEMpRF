# -*- coding: utf-8 -*-

from extract_data import create_local_folder_and_extract_data
import os
import nibabel as nib
import matplotlib.pyplot as plt

# Replace 'zip_file_path' with the actual path to your zip file
zip_file_path = "./../../DATASETS/sid-prf-fmri-data.zip"
create_local_folder_and_extract_data(zip_file_path)

# def build_structural_images(nifti_file_path):
#     # Load the NIfTI file
#     img = nib.load(nifti_file_path)

#     # Get the 3D image data as a NumPy array
#     image_data = img.get_fdata()

#      # Determine the number of slices in the z-direction
#     num_slices = image_data.shape[2]

#      # Plot each slice as a separate figure
#     for i in range(num_slices):
#         #plt.imshow(image_data[:, :, i], cmap='gray')
#         plt.imshow(image_data[i, :, :], cmap='gray')
#         plt.title(f'Slice {i}')
#         plt.axis('off')
#         plt.show()
        
def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def generate_and_save_slice_images(image_data, orientation, output_folder):
    if orientation == 'traversal':
        for i in range(image_data.shape[2]):
            plt.imshow(image_data[:, :, i], cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(output_folder, f'slice_{i}.png'), bbox_inches='tight')
            plt.close()
    elif orientation == 'sagittal':
        for i in range(image_data.shape[0]):
            plt.imshow(image_data[i, :, :].T, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(output_folder, f'slice_{i}.png'), bbox_inches='tight')
            plt.close()
    elif orientation == 'coronal':
        for i in range(image_data.shape[1]):
            plt.imshow(image_data[:, i, :].T, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(output_folder, f'slice_{i}.png'), bbox_inches='tight')
            plt.close()

def build_structural_images(nifti_file_path):
    # Load the NIfTI file
    img = nib.load(nifti_file_path)

    # Get the 3D image data as a NumPy array
    image_data = img.get_fdata()

    # Determine the output folder path
    output_folder = os.path.join(os.getcwd(), 'sid-mri-result-slice-images')
    traversal_folder = os.path.join(output_folder, 'traversal')
    sagittal_folder = os.path.join(output_folder, 'sagittal')
    coronal_folder = os.path.join(output_folder, 'coronal')

    # Create the required folders
    create_folder_if_not_exist(output_folder)
    create_folder_if_not_exist(traversal_folder)
    create_folder_if_not_exist(sagittal_folder)
    create_folder_if_not_exist(coronal_folder)

    # Save the slices in traversal, sagittal, and coronal orientations
    generate_and_save_slice_images(image_data, 'traversal', traversal_folder)
    generate_and_save_slice_images(image_data, 'sagittal', sagittal_folder)
    generate_and_save_slice_images(image_data, 'coronal', coronal_folder)        
        
        
# Path to the NIfTI file
file_path = './../../local-extracted-datasets/sid-prf-fmri-data/sub-sidtest_ses-001_desc-preproc_T1w.nii.gz'
#file_path = r'D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data\sub-sidtest_ses-001_desc-preproc_T1w.nii.gz'


#Main
if __name__ == "__main__":
    build_structural_images(file_path)






















