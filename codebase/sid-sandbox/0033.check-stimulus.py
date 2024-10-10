import nibabel as nib
import matplotlib.pyplot as plt

def plot_nifti_images(nifti_file):
    # Load the NIfTI file
    img = nib.load(nifti_file)
    
    # Get the image data as a NumPy array
    data = img.get_fdata()

    # Loop through the image slices and plot them
    num_slices = data.shape[-1]
    for i in range(num_slices):
        plt.figure()
        plt.imshow(data[..., i], cmap='gray')
        plt.title(f"Slice {i+1}")
        plt.axis('off')
        plt.show()

# Usage example:
nifti_file = "D:/results/docker-analysis/01-test/BIDS/stimuli/sub-001_ses-20200320_task-prf_apertures.nii.gz"
plot_nifti_images(nifti_file)

print()
