import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths of the BOLD responses and stimulus video
bold_response_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz")
stimulus_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz")

# Load the BOLD response data
bold_response_img = nib.load(bold_response_file_path)
bold_response_data = bold_response_img.get_fdata()

# Reshape the BOLD response data to 2D
bold_response_data = bold_response_data.reshape(-1, bold_response_data.shape[-1])

# Get the shape of the BOLD response data (num_voxels=4413, num_timepoints=300)
num_voxels, num_timepoints = bold_response_data.shape

# Compute the average bold response for each time point
average_bold_response = np.mean(bold_response_data, axis=0)

# Load the stimulus video
stimulus_img = nib.load(stimulus_file_path)
stimulus_data = stimulus_img.get_fdata()

# Iterate over the time points
for i in range(num_timepoints):
    # Select the current time point
    current_timepoint = i
    
    frame_idx = min(i, stimulus_data.shape[-1] - 1)  # Get the corresponding frame index from the stimulus video
    frame = stimulus_data[..., frame_idx]  # Extract a single frame
    
    # Display the average BOLD response as a time course
    plt.subplot(121)  # Plot the average BOLD response
    plt.plot(average_bold_response)
    plt.title("Average BOLD Response")
    plt.xlabel("Time")
    plt.ylabel("Signal Intensity")
    
    # Add a vertical red line to indicate the current time point
    plt.axvline(x=current_timepoint, color='red')
    
    # Display the stimulus frame on the right side
    plt.subplot(122)  # Plot the stimulus frame on the right side
    plt.imshow(frame, cmap='gray')
    plt.title(f"Stimulus Video - Frame {frame_idx}")
    plt.axis('off')
    
    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()
