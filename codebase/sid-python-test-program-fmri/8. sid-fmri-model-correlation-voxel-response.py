import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os


# IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths of the BOLD responses and stimulus video
bold_response_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz")
stimulus_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz")


# Set the number of voxels to analyze
num_voxels_to_analyze = 5

# Load the BOLD response data
bold_response_img = nib.load(bold_response_file_path)
bold_response_data = bold_response_img.get_fdata()
bold_response_data = bold_response_data.reshape(-1, bold_response_data.shape[-1])

# Load the stimulus video
stimulus_img = nib.load(stimulus_file_path)
stimulus_data = stimulus_img.get_fdata()

# Define the model based on the stimulus
model = stimulus_data.squeeze()  # Adjust the model creation based on your specific stimulus

# Calculate correlations voxel by voxel
correlations = np.empty(bold_response_data.shape[0])
for voxel_index in range(bold_response_data.shape[0]):
    voxel_time_course = bold_response_data[voxel_index, :]
    correlations[voxel_index] = np.corrcoef(voxel_time_course, model.flatten())[0, 1]

# Find voxels with highest correlation
top_voxel_indices = np.argsort(correlations)[::-1][:num_voxels_to_analyze]

# Plot the model
plt.figure()
plt.imshow(model, cmap='gray')
plt.title('Stimulus Model')
plt.axis('off')
plt.show()

# Plot voxel time courses with overlay
plt.figure()
for idx, voxel_index in enumerate(top_voxel_indices):
    voxel_time_course = bold_response_data[voxel_index, :]
    plt.plot(voxel_time_course, label=f'Voxel {voxel_index + 1}')
plt.plot(model.flatten(), 'k--', label='Model')
plt.xlabel('Time')
plt.ylabel('BOLD Response')
plt.title('Voxel Time Courses with Model Overlay')
plt.legend()
plt.show()
