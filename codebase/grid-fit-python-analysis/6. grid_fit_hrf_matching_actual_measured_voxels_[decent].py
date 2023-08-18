
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import nibabel as nib
from scipy.ndimage import zoom
from scipy import integrate
from hrf_generator_script import spm_hrf_compat
import math
import os
from sklearn.linear_model import LinearRegression
from scipy.signal import butter, filtfilt

# IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths of the BOLD responses and stimulus video
bold_response_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz")
stimulus_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz")


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

#Plot timecourse
def plot_timecourse(timecourse_data, title):
    plt.figure(figsize=(10, 6))    
    plt.plot(range(len(timecourse_data)), timecourse_data, linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(title)
    plt.grid()
    plt.show()

# Apply smoothing to observed timecourse
def smooth_voxel_timecourse(voxel_timecourse, sampling_rate=1.0, cutoff_frequency=0.1):
    # Calculate Nyquist frequency
    nyquist_frequency = 0.5 * sampling_rate
    
    # Normalize the cutoff frequency
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    
    # Design a low-pass Butterworth filter
    b, a = butter(N=4, Wn=normalized_cutoff, btype='low')
    
    # Apply the filter to the voxel timecourse
    smoothed_timecourse = filtfilt(b, a, voxel_timecourse)
    
    return smoothed_timecourse

# Resample measured voxel timecourse and apply linear regressiong
def get_linear_regression_curve(voxel_timecourse):
    num_timepoints = len(voxel_timecourse)
    
    # Create a time vector for the original timecourse
    original_time = np.arange(num_timepoints)
    
    # Resample the timecourse every second
    new_time = np.arange(0, num_timepoints, step=1)
    
    # Initialize the linear regression model
    model = LinearRegression()
    
    # Fit the model using the original time and voxel timecourse
    model.fit(original_time.reshape(-1, 1), voxel_timecourse)
    
    # Predict the voxel timecourse using the linear model and new time
    linear_regression_curve = model.predict(new_time.reshape(-1, 1))

    # plt.plot(voxel_timecourse)
    # plt.plot(linear_regression_curve)
    # plt.show()
    
    return linear_regression_curve

#normalize a timecourse
def normalize_timecourse(timecourse):
    return (timecourse - np.mean(timecourse)) / np.std(timecourse)

# Grid fit with GLM fitting
def match_voxel_responses_to_expected_pixel_position_timecourses(measured_voxel_timecourses, expected_responses_grid):
    num_voxels, num_timepoints = measured_voxel_timecourses.shape
    num_expected_responses = len(expected_responses_grid)
    
    matched_indices = []  # List to store the indices of best-matched expected responses for each measured response
    min_mse_values = [] # store the values of best 'Mean Squared Error' value computed for each measured voxel reponse

    
    # Iterate through each voxel's timecourse
    for voxel_idx in range(num_voxels):        
        observed_voxel_timecourse = measured_voxel_timecourses[voxel_idx]
        smoothened_voxel_timecourse = smooth_voxel_timecourse(observed_voxel_timecourse)
        # plt.plot(observed_voxel_timecourse)
        # plt.plot(smoothened_voxel_timecourse)
        # plt.show()

        # normalize measured+smoothened voxel curve
        normalized_smoothened_voxel_timecourse = normalize_timecourse(smoothened_voxel_timecourse)

        # get vertical shift (beta value)
        linear_regression_curve = get_linear_regression_curve(normalized_smoothened_voxel_timecourse)
        
        # Find the best-matching expected response model using least squares
        best_match_index = -1  # Initialize with -1 (no best match found)
        min_mse = float('inf')  # Initialize minimum MSE with a high value
        for expected_idx in range(num_expected_responses):
            expected_response = expected_responses_grid[expected_idx]
            vertically_shifted_expected_response = expected_response + linear_regression_curve
            mse = np.mean((normalized_smoothened_voxel_timecourse - vertically_shifted_expected_response) ** 2)  # Calculate Mean Squared Error
            if mse < min_mse:
                min_mse = mse
                best_match_index = expected_idx

        matched_indices.append([min_mse, voxel_idx, best_match_index])
        min_mse_values.append(min_mse)
        
        #debug plot
        # plot_glm_fitting_result(normalized_smoothened_voxel_timecourse
        #                         , expected_responses_grid[expected_idx] + linear_regression_curve)

    
    # # Get the indices of the top five highest min_mse_values
    # best_min_mse_values = np.array(min_mse_values)
    # best_min_mse_indices = np.argsort(best_min_mse_values)[:5]

    # top_min_mse_values = []
    # for i  in best_min_mse_indices:
    #     min_mse_value = min_mse_values[i]
    #     top_min_mse_values.append(min_mse_value)
    #     # plot_glm_fitting_result( normalize_timecourse(smooth_voxel_timecourse(measured_voxel_timecourses[i]))
    #     #                         ,1
    #     # )
    
    # print("Top 5 values:", top_min_mse_values)
    # print("Corresponding indices:", best_min_mse_indices)

    return np.array(matched_indices)

            #######-----PLOTTING---------------##########
            # #Vertically Shift the expected response curve
            # linear_regression_curve = apply_linear_regression_to_timecourse(normalized_smoothened_voxel_timecourse)
            # shifted_expected_response_1 = expected_response + linear_regression_curve
            # # shifted_expected_response_2 = expected_response + ((linear_regression_curve - np.mean(linear_regression_curve))/ np.std(linear_regression_curve))
            # plt.plot(normalized_smoothened_voxel_timecourse, label='Normalized Observed Voxel Timecourse')
            # plt.plot(shifted_expected_response_1, label='Modelled Response (Vertically Shifted: Direct)')
            # # plt.plot(shifted_expected_response_2, label='Shifted Response 2: Normalized')
            # plt.xlabel('time')
            # plt.ylabel('response')
            # plt.title('GLM Fitting')
            # plt.legend()
            # plt.show()
            
            # print("debug")


# Plot top five best matched voxel responses
def plot_best_matched_responses(matched_indices_array, measured_voxel_responses, modelled_responses):
     # Get the indices of the top five lowest min_mse values
    top_min_mse_indices = np.argsort(matched_indices_array[:, 0])[:5]
    top_min_mse_values = matched_indices_array[top_min_mse_indices]
    
    for mse, voxel_idx, match_idx in top_min_mse_values:
        normalized_smoothened_measured_voxel_timecourse = normalize_timecourse(smooth_voxel_timecourse((measured_voxel_responses[int(voxel_idx)])))        
        linear_regression_curve = get_linear_regression_curve(normalized_smoothened_measured_voxel_timecourse)
        shifted_expected_response = modelled_responses[int(match_idx)] + linear_regression_curve        
        plt.plot(normalized_smoothened_measured_voxel_timecourse, label='Normalized Observed Voxel Timecourse')
        plt.plot(shifted_expected_response, label='Modelled Response (Vertically Shifted: Direct)')
        plt.xlabel('time')
        plt.ylabel('response')
        plt.title(f'GLM Fitting - MSE = {mse}')
        plt.legend()
        plt.show()

# Plot top five best matched voxel responses
def plot_worst_matched_responses(matched_indices_array, measured_voxel_responses, modelled_responses):
     # Get the indices of the top five lowest min_mse values
    worst_min_mse_indices = np.argsort(matched_indices_array[:, 0])[::-1][:5]
    worst_min_mse_values = matched_indices_array[worst_min_mse_indices]
    
    for mse, voxel_idx, match_idx in worst_min_mse_values:
        normalized_smoothened_measured_voxel_timecourse = normalize_timecourse(smooth_voxel_timecourse((measured_voxel_responses[int(voxel_idx)])))        
        linear_regression_curve = get_linear_regression_curve(normalized_smoothened_measured_voxel_timecourse)
        shifted_expected_response = modelled_responses[int(match_idx)] + linear_regression_curve        
        plt.plot(normalized_smoothened_measured_voxel_timecourse, label='Normalized Observed Voxel Timecourse')
        plt.plot(shifted_expected_response, label='Modelled Response (Vertically Shifted: Direct)')
        plt.xlabel('time')
        plt.ylabel('response')
        plt.title(f'GLM Fitting - MSE = {mse}')
        plt.legend()
        plt.show()      

# GLM fitting plots
def plot_glm_fitting_result(normalized_smoothened_voxel_timecourse, shifted_expected_response):
    plt.plot(normalized_smoothened_voxel_timecourse, label='Normalized Observed Voxel Timecourse')
    plt.plot(shifted_expected_response, label='Modelled Response (Vertically Shifted: Direct)')
    plt.xlabel('time')
    plt.ylabel('response')
    plt.title('GLM Fitting')
    plt.legend()
    plt.show()




# Grid fit: Find the best fit for the measured responses within the array of expected responses (grid)
def OLD_match_voxel_responses_to_expected_pixel_postion_timecourses(measured_voxel_timecourses, expected_responses_grid):
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
GRID_SIZE_X = 10
GRID_SIZE_Y = 10
DESIRED_STIMULUS_SIZE = 101
SIGMA = 2
NOISE_STD_DEV = 0.1 
STIMULUS_SIZE_IN_DEGREES = 9

##---Load voxels data (BOLD)
# bold_response_file_path = r"D:\code\sid-git\fmri\local-extracted-datasets\sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz"
# bold_response_img = nib.load(bold_response_file_path)
# bold_response_data = bold_response_img.get_fdata()
# bold_response_data = bold_response_data.reshape(-1, bold_response_data.shape[-1])
num_voxels, num_timepoints, bold_response_data = get_measured_voxels_data()

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

#test plots - Just plot the matching of the the "expected timecourse" overlayed by the "simulated_measured_timecourses"

# matchIdx = matched_indices[2500]
# plt.plot(expected_timecourses_grid[matchIdx[1]])
# plt.plot(bold_response_data[matchIdx[0]])
# plt.plot(simulated_measured_timecourses[1])
# plt.show()

print("Done!")
