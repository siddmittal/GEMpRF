
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import nibabel as nib
from scipy.ndimage import zoom
from scipy import integrate
import math, sys
from pathlib import Path

# HRF Generator
hrf_module_path = (Path(__file__).resolve().parent / '../external-packages/nipy-hrf-generator').resolve()
sys.path.append(str(hrf_module_path))
from hrf_generator_script import spm_hrf_compat

# Meshgrid generator for Gaussian
def get_meshgrid_for_gaussian(stimulus_size_in_degrees, stimulus_frame_size):
    x = np.linspace( - stimulus_size_in_degrees, + stimulus_size_in_degrees, stimulus_frame_size)
    y = np.linspace( - stimulus_size_in_degrees, + stimulus_size_in_degrees, stimulus_frame_size)
    X, Y = np.meshgrid(x, y)
    return X,Y

# Generate Gaussian
def generate_2d_gaussian(meshgrid_X, meshgrid_Y, sigma, mean_x, mean_y):                    
    mean_corrdinates = [mean_x, mean_y]        
    Z = np.exp(-(meshgrid_X - mean_corrdinates[0])**2 / (2 * sigma**2)) * np.exp(-(meshgrid_Y - mean_corrdinates[1])**2 / (2 * sigma**2))
    return Z

# Generate the timecourse for a Gaussian Curve (for a particular pixel location)
# by taking the Stimulus frames into account
def generate_pixel_location_time_course(Z, stimulus_frame_data):
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
    plt.plot(range(len(timecourse_data)), timecourse_data, linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(title)
    plt.grid()
    plt.show()

# Grid fit: Find the best fit for the measured responses within the array of expected responses (grid)
def match_voxel_responses_to_expected_pixel_postion_timecourses(measured_voxel_timecourses, expected_responses_grid):
    matched_indices = []  # List to store the indices of best-matched expected responses for each measured response

    for idx, measured_timecourse in enumerate(measured_voxel_timecourses):
        best_match_index = -1  # Initialize with -1 (no best match found)
        min_mse = float('inf')  # Initialize minimum MSE with a high value

        for e_idx, expected_timecourse in enumerate(expected_responses_grid):
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
GRID_SIZE_X = 4
GRID_SIZE_Y = 4
DESIRED_STIMULUS_SIZE = 101
SIGMA = 2
NOISE_STD_DEV = 0.1 
STIMULUS_SIZE_IN_DEGREES = 9

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

#Initialize meshgrid and stimulus
meshgrid_X, meshgrid_Y = get_meshgrid_for_gaussian(STIMULUS_SIZE_IN_DEGREES, DESIRED_STIMULUS_SIZE)
xy_mask = np.sqrt(meshgrid_X**2 + meshgrid_Y**2) <= STIMULUS_SIZE_IN_DEGREES
X_stim = meshgrid_X[xy_mask]
Y_stim = meshgrid_Y[xy_mask]

#dummy - to get the timecourse lenth
dummy_2d_gaussian = generate_2d_gaussian(meshgrid_X, meshgrid_Y, SIGMA, 0, 0)
timecourse_length = len(generate_pixel_location_time_course(dummy_2d_gaussian, resampled_stimulus_data))

# Get HRF Curve
hrf_t = np.linspace(0,30,31)
# hrf_t = np.linspace(0,timecourse_length,timecourse_length+1)
hrf_curve = spm_hrf_compat(hrf_t) 
plot_timecourse(hrf_curve, "HRF Curve")        

# Create Expected Responses Grid: Traverse through all locations of the defined grid
for x in range(GRID_SIZE_X):
    for y in range(GRID_SIZE_Y):
        Z = generate_2d_gaussian(meshgrid_X, meshgrid_Y, SIGMA, x, y)
        pixel_location_timecourse = generate_pixel_location_time_course(Z, resampled_stimulus_data)
        pixel_location_timecourse /= pixel_location_timecourse.max()
        r_t = np.convolve(pixel_location_timecourse, hrf_curve, mode='full')[:len(pixel_location_timecourse)]         
        expected_timecourses_grid.append(pixel_location_timecourse)
        plt.figure()
        plt.plot(pixel_location_timecourse, '-') 
        plt.plot(r_t)
        plt.show()
        print('done')


# Generate simulated Voxels responses (for tests: noisy version of expected responses)        
for idx in range(50):
    expected_response_to_choose_from_idx = math.floor((idx/GRID_SIZE_X))    
    noisy_timecourse = expected_timecourses_grid[expected_response_to_choose_from_idx] + abs(np.random.normal(0, NOISE_STD_DEV, len(pixel_location_timecourse)))
    simulated_measured_timecourses.append(noisy_timecourse)
    # plot_timecourse(noisy_timecourse, f"Simulated timecourse - ({idx}), chose from Expected - {expected_response_to_choose_from_idx}")

# Matching - find best matching for the voxel responses within the expected response grid
# matched_indices = match_voxel_responses_to_expected_pixel_postion_timecourses(simulated_measured_timecourses, expected_timecourses_grid)

print("Done!")
