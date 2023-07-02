# Time Course of Voxels

import nibabel as nib
import matplotlib.pyplot as plt

# Define the file paths of the MRI images and stimulus video
fmri_file_path = "sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz"
stimulus_file_path = "task-bar_apertures.nii.gz"

# Load the fMRI data from the V1 region
fmri_img = nib.load(fmri_file_path)
fmri_data = fmri_img.get_fdata()

# Reshape the fMRI data to 2D
fmri_data = fmri_data.reshape(-1, fmri_data.shape[-1])

# Get the shape of the fMRI data
num_voxels, num_timepoints = fmri_data.shape

# Display the fMRI data as transversal slices
for i in range(num_voxels):
    voxel_data = fmri_data[i]
    plt.plot(voxel_data)
    plt.title(f"Voxel {i + 1}")
    plt.xlabel("Time")
    plt.ylabel("Signal Intensity")
    plt.show()

# Load the stimulus video
stimulus_img = nib.load(stimulus_file_path)
stimulus_data = stimulus_img.get_fdata()

# Display the first frame of the stimulus video
plt.imshow(stimulus_data[:, :, 0], cmap='gray')
plt.title("Stimulus Video - Frame 0")
plt.axis('off')
plt.show()
