from extract_data import create_local_folder_and_extract_data
import os
import nibabel as nib
import matplotlib.pyplot as plt

# Replace 'zip_file_path' with the actual path to your zip file
zip_file_path = "./../../DATASETS/sid-prf-fmri-data.zip"

create_local_folder_and_extract_data(zip_file_path)

# Path to the NIfTI file
file_path = r'D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data\sub-sidtest_ses-001_desc-preproc_T1w.nii.gz'

def build_structural_images(nifti_file_path):
    # Load the NIfTI file
    img = nib.load(nifti_file_path)

    # Get the 3D image data as a NumPy array
    image_data = img.get_fdata()

    # You can perform various image processing or analysis tasks here, if needed
    # For example, you can apply filters, thresholding, or other transformations

    # Visualize the structural image using matplotlib
    plt.imshow(image_data[:, :, image_data.shape[2] // 2], cmap='gray')
    plt.title('Structural Image')
    plt.axis('off')
    plt.show()

    # Save the structural image as an image file (optional)
    output_file_path = 'structural_image.png'
    plt.imsave(output_file_path, image_data[:, :, image_data.shape[2] // 2], cmap='gray')
    print(f"Structural image saved as: {os.path.abspath(output_file_path)}")

if __name__ == "__main__":
    build_structural_images(file_path)