
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import nibabel as nib
from scipy.ndimage import zoom
from scipy import integrate
import math

#Generate Gaussian
def generate_2d_gaussian(total_size, sigma, mean_x, mean_y):                    
    # mean = [mean_x, mean_y]
    mean = [mean_x, mean_y]
    start_stop_borders = math.floor(total_size/2)
    stimuls_size_in_degrees = 9
    # x = np.linspace( - start_stop_borders, + start_stop_borders, total_size) # +1 at the end to make it odd number to have (0,0) origin in the middle
    # y = np.linspace( - start_stop_borders, + start_stop_borders, total_size)
    x = np.linspace( - stimuls_size_in_degrees, + stimuls_size_in_degrees, total_size) # +1 at the end to make it odd number to have (0,0) origin in the middle
    y = np.linspace( - stimuls_size_in_degrees, + stimuls_size_in_degrees, total_size)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X - mean[0])**2 / (2 * sigma**2)) * np.exp(-(Y - mean[1])**2 / (2 * sigma**2))
    return X, Y, Z


# Generate the timecourse for a Gaussian Curve (for a particular pixel location)
# by taking the Stimulus frames into account
def generate_pixel_location_time_course(X, Y, Z, stimulus_frame_data):
    time_points = stimulus_frame_data.shape[2]
    area_under_gaussian = np.zeros(time_points)

    for t in range(time_points):
        # Use stimulus_frame_data as a binary mask
        mask = (stimulus_frame_data[:, :, t] > 0).astype(int)
        
        # Apply mask to the Gaussian curve
        masked_gaussian = Z * mask

        # Compute area under the Gaussian curve after applying the mask
        area_under_gaussian[t] = np.sum(masked_gaussian)
        
    return area_under_gaussian

#Plot timecourse
def plot_timecourse(timecourse_data, title):
    plt.figure(figsize=(10, 6))
    #plt.plot(range(len(timecourse_data)), timecourse_data, marker='o', linestyle='-', color='b')
    plt.plot(range(len(timecourse_data)), timecourse_data, linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(title)
    plt.grid()
    plt.show()

# #Grid fit: Try to find the best fit for the measured responses within the array of expected responses (grid) for each pixel location of the visual stimulus
# def match_voxel_responses_to_expected_pixel_postion_timecourses(measured_voxel_timecourses, pixel_locations_expected_responses_grid):
#     print("Grid fitting started...")
#     for voxel_idx in range(len(measured_voxel_timecourses)):
#         best_match_pixel_index = -1
#         for expected_responses_idx in range (len(pixel_locations_expected_responses_grid)):
#             inProgress = True


# Grid fit: Find the best fit for the measured responses within the array of expected responses (grid)
def match_voxel_responses_to_expected_pixel_postion_timecourses(measured_voxel_timecourses, pixel_locations_expected_responses_grid):
    matched_indices = []  # List to store the indices of best-matched expected responses for each measured response

    for idx, measured_timecourse in enumerate(measured_voxel_timecourses):
        best_match_index = -1  # Initialize with -1 (no best match found)
        min_mse = float('inf')  # Initialize minimum MSE with a high value

        for e_idx, expected_timecourse in enumerate(pixel_locations_expected_responses_grid):
            mse = np.mean((measured_timecourse - expected_timecourse) ** 2)  # Calculate Mean Squared Error
            if mse < min_mse:
                min_mse = mse
                best_match_index = e_idx

        matched_indices.append([idx, best_match_index])

    return np.array(matched_indices)


#################################################################
########################----Program-------#######################
#################################################################
##---Constant definitions
DESIRED_STIMULUS_SIZE = 101
SIGMA = 2
NOISE_STD_DEV = 0.1 

##---Load voxels data (BOLD)
bold_response_file_path = r"D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz"
bold_response_img = nib.load(bold_response_file_path)
bold_response_data = bold_response_img.get_fdata()
bold_response_data = bold_response_data.reshape(-1, bold_response_data.shape[-1])

##---Load Stimulus data
stimulus_file_path = r"D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data/task-bar_apertures.nii.gz"
stimulus_img = nib.load(stimulus_file_path)
stimulus_data = stimulus_img.get_fdata()

##---Resample Stimulus
original_stimulus_shape = stimulus_data.shape # (1024, 1024, 1, 300)
resampled_stimulus_shape = (DESIRED_STIMULUS_SIZE, DESIRED_STIMULUS_SIZE, original_stimulus_shape[2])
resampling_factors = (
    resampled_stimulus_shape[0] / (original_stimulus_shape[0] -1),
    resampled_stimulus_shape[1] / (original_stimulus_shape[1] - 1),
    1  # Keep the number of time points unchanged
)
resampled_stimulus_data = zoom(stimulus_data.squeeze(), resampling_factors, order=1)

# Expected response Grid: Initialize an empty list to store the timecourse arrays for each pixel location
expected_timecourses_grid = []

#Simulated measured BOLD responses
simulated_measured_timecourses = []

# Traverse through all pixels of the stimulus frame
# for y in range(0, resampled_stimulus_data.shape[0], 20): #rows
#     for x in range(0, resampled_stimulus_data.shape[1], 10): #columns

X,Y,_ = generate_2d_gaussian(DESIRED_STIMULUS_SIZE, SIGMA, 0, 0)
xy_mask = np.sqrt(X**2 + Y**2) <= 9
X_stim = X[xy_mask]
Y_stim = Y[xy_mask]

for x, y in zip(X_stim, Y_stim):
    X, Y, Z = generate_2d_gaussian(DESIRED_STIMULUS_SIZE, SIGMA, x, y)
    pixel_location_timecourse = generate_pixel_location_time_course(X,Y,Z, resampled_stimulus_data)

    pixel_location_timecourse /= pixel_location_timecourse.max()

    expected_timecourses_grid.append(pixel_location_timecourse)

    # Add random Gaussian noise to the pixel location timecourse
    noisy_timecourse = pixel_location_timecourse + np.random.normal(0, NOISE_STD_DEV, len(pixel_location_timecourse))
    simulated_measured_timecourses.append(noisy_timecourse)

# for y in range(resampled_stimulus_data.shape[0]): #rows
#     for x in range(resampled_stimulus_data.shape[1]): #columns
#         X, Y, Z = generate_2d_gaussian(DESIRED_STIMULUS_SIZE, SIGMA, x, y)
#         pixel_location_timecourse = generate_pixel_location_time_course(X,Y,Z, resampled_stimulus_data)        
#         expected_timecourses_grid.append(pixel_location_timecourse)

#         # Add random Gaussian noise to the pixel location timecourse
#         noisy_timecourse = pixel_location_timecourse + np.random.normal(0, NOISE_STD_DEV, len(pixel_location_timecourse))
#         simulated_measured_timecourses.append(noisy_timecourse)

#     plot_timecourse(pixel_location_timecourse, f"Exptected Time Course - ({y}, {math.floor(resampled_stimulus_data.shape[0]/2)})")
#     plot_timecourse(noisy_timecourse, f"Noisy Time Course - ({y}, {math.floor(resampled_stimulus_data.shape[0]/2)})")
#     print(f"Row finished - {y}")

# matched_indices = match_voxel_responses_to_expected_pixel_postion_timecourses(simulated_measured_timecourses, expected_timecourses_grid)
# print("Program finished")      

# #%%
# average_timecourse = np.mean(expected_timecourses_grid[0]) # just for testing
# print(f"Average timecourse: {average_timecourse}")


