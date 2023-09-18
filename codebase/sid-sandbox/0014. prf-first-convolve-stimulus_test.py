import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.linear_model import LinearRegression
import os, sys

from pathlib import Path

# HRF Generator
hrf_module_path = (Path(__file__).resolve().parent / '../external-packages/nipy-hrf-generator').resolve()
sys.path.append(str(hrf_module_path))
from hrf_generator_script import spm_hrf_compat

# IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths of the BOLD responses and stimulus video
bold_response_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-0102avg_hemi-L_bold.nii.gz")
stimulus_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz")

##---Load Stimulus data
stimulus_file_path = os.path.join(script_directory, "../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz")
stimulus_img = nib.load(stimulus_file_path)
stimulus_data = stimulus_img.get_fdata()

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
        # Apply mask to the Gaussian curve
        masked_gaussian = Z * stimulus_frame_data[:, :, t]

        # Compute area under the Gaussian curve after applying the mask
        area_under_gaussian[t] = np.sum(masked_gaussian)
        
    return area_under_gaussian

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

# HRF convolved stimulus version
stimulus_location_convolved_timecourses = np.empty_like(resampled_stimulus_data, dtype=float)
for row in range (resampled_stimulus_shape[0]):
    for col in range (resampled_stimulus_shape[1]):
        location_tc = resampled_stimulus_data[row, col, :]
        stimulus_location_convolved_timecourses[row, col, :] = np.convolve(location_tc, hrf_curve, mode='full')[:len(location_tc)]     
        print("")  

# Create Expected Responses Grid: Traverse through all locations of the defined grid
mu_x = -1
mu_y = -1
for x in range(GRID_SIZE_X):
    for y in range(GRID_SIZE_Y):
        mu_x = meshgrid_X[x][y]
        mu_y = meshgrid_Y[x][y]
        Z = generate_2d_gaussian(meshgrid_X, meshgrid_Y, SIGMA, mu_x, mu_y)
        pixel_location_timecourse = generate_pixel_location_time_course(Z, stimulus_location_convolved_timecourses)
        pixel_location_timecourse /= pixel_location_timecourse.max()        
        expected_timecourses_grid.append(pixel_location_timecourse)  

print("Done!")
