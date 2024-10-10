import nibabel as nib
import matplotlib.pyplot as plt

# Define the file paths of the BOLD responses and stimulus video
bold_response_file_path = r"D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz"
stimulus_file_path = r"D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data/task-bar_apertures.nii.gz"

# Load the BOLD response data
bold_response_img = nib.load(bold_response_file_path)
bold_response_data = bold_response_img.get_fdata()

# Reshape the BOLD response data to 2D
bold_response_data = bold_response_data.reshape(-1, bold_response_data.shape[-1])

# Get the shape of the BOLD response data (num_voxels=4413, num_timepoints=300)
num_voxels, num_timepoints = bold_response_data.shape

# Load the stimulus video
stimulus_img = nib.load(stimulus_file_path)
stimulus_data = stimulus_img.get_fdata()

# Display the BOLD response data as time courses of voxels
for i in range(num_voxels):
    voxel_data = bold_response_data[i]
    frame_idx = min(i, stimulus_data.shape[-1] - 1)  # Get the corresponding frame index from the stimulus video
    frame = stimulus_data[..., frame_idx]  # Extract a single frame
    
    plt.subplot(121)  # Plot the BOLD response data on the left side
    plt.plot(voxel_data)
    plt.title(f"Voxel {i + 1}")
    plt.xlabel("Time")
    plt.ylabel("Signal Intensity")

    plt.subplot(122)  # Plot the stimulus frame on the right side
    plt.imshow(frame, cmap='gray')
    plt.title(f"Stimulus Video - Frame {frame_idx}")
    plt.axis('off')
    
    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()
