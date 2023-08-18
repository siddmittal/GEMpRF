
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
from hrf_generator_script import spm_hrf_compat
import os

# IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths of the BOLD responses and stimulus video
bold_response_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz")
stimulus_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz")

##---Load Stimulus data
stimulus_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz")
stimulus_img = nib.load(stimulus_file_path)
stimulus_data = stimulus_img.get_fdata()

# Plot top five best matched voxel responses
def plot_best_matched_responses(matched_indices_array, measured_voxel_responses, modelled_responses):
     # Get the indices of the top five lowest min_mse values
    top_min_mse_indices = np.argsort(matched_indices_array[:, 0])[:5]
    top_min_mse_values = matched_indices_array[top_min_mse_indices]
    
    for mse, voxel_idx, match_idx in top_min_mse_values:   
        voxel_signal = measured_voxel_responses[int(voxel_idx)]
        model_signal = modelled_responses[int(match_idx)]
        y_hat = get_y_hat(voxel_signal, model_signal)                   
        plt.plot(voxel_signal, label='Observed Voxel Timecourse')
        plt.plot(y_hat, label='Modelled Response (Scaled)')
        plt.xlabel('time')
        plt.ylabel('response')
        plt.title(f'Best GLM Fitting - MSE = {mse}')
        plt.legend()
        plt.show()

# Plot top five best matched voxel responses
def plot_worst_matched_responses(matched_indices_array, measured_voxel_responses, modelled_responses):
     # Get the indices of the top five lowest min_mse values
    worst_min_mse_indices = np.argsort(matched_indices_array[:, 0])[::-1][:5]
    worst_min_mse_values = matched_indices_array[worst_min_mse_indices]
    
    for mse, voxel_idx, match_idx in worst_min_mse_values:   
        voxel_signal = measured_voxel_responses[int(voxel_idx)]
        model_signal = modelled_responses[int(match_idx)]
        y_hat = get_y_hat(voxel_signal, model_signal)                   
        plt.plot(voxel_signal, label='Observed Voxel Timecourse')
        plt.plot(y_hat, label='Modelled Response (Scaled)')
        plt.xlabel('time')
        plt.ylabel('response')
        plt.title(f'Worst GLM Fitting - MSE = {mse}')
        plt.legend()
        plt.show() 

# Grid fit with GLM fitting
def match_voxel_responses_to_expected_pixel_position_timecourses(measured_voxel_timecourses, expected_responses_grid):
    num_voxels, num_timepoints = measured_voxel_timecourses.shape
    num_expected_responses = len(expected_responses_grid)
    
    matched_indices = []  # List to store the indices of best-matched expected responses for each measured response
    min_mse_values = [] # store the values of best 'Mean Squared Error' value computed for each measured voxel reponse
    
    # Iterate through each voxel's timecourse
    for voxel_idx in range(num_voxels):        
        observed_voxel_timecourse = measured_voxel_timecourses[voxel_idx]        
        
        # Find the best-matching expected response model using least squares
        best_match_index = -1  # Initialize with -1 (no best match found)
        min_mse = float('inf')  # Initialize minimum MSE with a high value
        for expected_idx in range(num_expected_responses):
            expected_response = expected_responses_grid[expected_idx] 
            if expected_response.ndim == 1:
                expected_response = expected_response[:, np.newaxis]
            y_hat = get_y_hat(observed_voxel_timecourse, expected_response)           
            mse = np.mean(((observed_voxel_timecourse - y_hat)) ** 2)  # Calculate Mean Squared Error
            if mse < min_mse:
                min_mse = mse
                best_match_index = expected_idx

        matched_indices.append([min_mse, voxel_idx, best_match_index])
        min_mse_values.append(min_mse)
        
    return np.array(matched_indices)

# Get measured BOLD response data for all voxels
def get_measured_voxels_data():
    bold_response_img = nib.load(bold_response_file_path)
    bold_response_data = bold_response_img.get_fdata()
    # Reshape the BOLD response data to 2D
    bold_response_data = bold_response_data.reshape(-1, bold_response_data.shape[-1])

    # Get the shape of the BOLD response data (num_voxels=4413, num_timepoints=300)
    num_voxels, num_timepoints = bold_response_data.shape

    return num_voxels, num_timepoints, bold_response_data

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

# GLM fittiing: compute y_hat
def get_y_hat(voxel_signal, model_signal):
    if model_signal.ndim == 1:
        # Add back a singleton axis
        # otherwise stacking will give an error
        model_signal = model_signal[:, np.newaxis]

    intercept = np.ones((model_signal.size, 1))
    X = np.hstack((intercept, model_signal))
    betas = np.linalg.inv(X.T @ X) @ X.T @ voxel_signal

    y_hat = X@betas

    return y_hat

#################################################################
########################----Program-------#######################
#################################################################
##---Constant definitions
GRID_SIZE_X = 10
GRID_SIZE_Y = 10
DESIRED_STIMULUS_SIZE = 101
SIGMA = 2
NOISE_STD_DEV = 0.1 
STIMULUS_SIZE_IN_DEGREES = 9

##---Load voxels data (BOLD)
num_voxels, num_timepoints, bold_response_data = get_measured_voxels_data()

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

#Initialize meshgrid and stimulus
meshgrid_X, meshgrid_Y = get_meshgrid_for_gaussian(STIMULUS_SIZE_IN_DEGREES, DESIRED_STIMULUS_SIZE)
xy_mask = np.sqrt(meshgrid_X**2 + meshgrid_Y**2) <= STIMULUS_SIZE_IN_DEGREES
X_stim = meshgrid_X[xy_mask]
Y_stim = meshgrid_Y[xy_mask]

# Get HRF Curve
hrf_t = np.linspace(0,30,31)
hrf_curve = spm_hrf_compat(hrf_t) 

# Create Expected Responses Grid: Traverse through all locations of the defined grid
for x in range(GRID_SIZE_X):
    for y in range(GRID_SIZE_Y):
        Z = generate_2d_gaussian(meshgrid_X, meshgrid_Y, SIGMA, x, y)
        pixel_location_timecourse = generate_pixel_location_time_course(Z, resampled_stimulus_data)
        pixel_location_timecourse /= pixel_location_timecourse.max()
        r_t = np.convolve(pixel_location_timecourse, hrf_curve, mode='full')[:len(pixel_location_timecourse)]         
        expected_timecourses_grid.append(r_t)

# Matching - find best matching for the voxel responses within the expected response grid
matched_indices = match_voxel_responses_to_expected_pixel_position_timecourses(bold_response_data[0:500], expected_timecourses_grid)
plot_best_matched_responses(matched_indices, bold_response_data[0:500], expected_timecourses_grid)
plot_worst_matched_responses(matched_indices, bold_response_data[0:500], expected_timecourses_grid)

print("Done!")
