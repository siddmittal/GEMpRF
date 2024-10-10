import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import cProfile

def gpu_gridfit():
    # Initialize CuPy device
    cp.cuda.Device(0).use()  # You can specify the GPU index

    ###############################################--------Stimulus-------------------------#############
    ###---3d Stimulus
    stim_dimension = 101
    num_stimulus_frames = 300
    stimulus = cp.random.randint(2, size=(num_stimulus_frames, stim_dimension, stim_dimension))

    ##################################--------Dummy Measured/Simulated Voxel Responses----------#############
    # Define the number of time series and their length
    num_time_series = 100

    # Generate all simulated time series at once using cuPy
    simulated_voxel_responses = cp.random.normal(0, 1, (num_time_series, num_stimulus_frames))

    # 'simulated_voxel_responses' now contains the 100 simulated time series

    # To access a specific simulated time series:
    # Example: Access the first simulated time series
    first_simulated_time_series = simulated_voxel_responses[0]


    ###############################################------------X and Y Ranges----------------------------##############
    range_points_xx = cp.linspace(-9, +9, stim_dimension)
    range_points_yy = cp.linspace(-9, +9, stim_dimension)
    range_points_X, range_points_Y = cp.meshgrid(range_points_xx, range_points_yy)

    ###############################################--------Gaussian Curves-------------------------#############
    testSpaceDim = 11
    xx = cp.linspace(-9, +9, testSpaceDim)
    yy = cp.linspace(-9, +9, testSpaceDim)
    mean_points_X, mean_points_Y = cp.meshgrid(xx, yy)

    # Define the standard deviation (you can adjust this value)
    std_dev = 1.0

    # Initialize an empty list to store the Gaussian curves
    gaussian_curves = []

    # Loop through each pair of mean values
    for mean_x, mean_y in zip(mean_points_X.flatten(), mean_points_Y.flatten()):
        # Compute the Gaussian curve for this mean using cuPy
        gaussian_curve = cp.exp(-((range_points_X - mean_x) ** 2 + (range_points_Y - mean_y) ** 2) / (2 * std_dev**2))
        gaussian_curves.append(gaussian_curve)
    # 'gaussian_curves' now contains a list of 2D Gaussian curves for each mean pair


    ###############################################--------Time Series----------------------#############
    # Initialize an empty list to store the time series
    model_signals = []

    # Loop through each Gaussian curve
    for gaussian_curve in gaussian_curves:
        frame_series = []
        # Loop through each time frame
        for frame in stimulus:
            # Multiply the Gaussian curve with the frame using cuPy
            weighted_frame = gaussian_curve * frame
            # Sum along rows and columns using cuPy
            weighted_sum = cp.sum(weighted_frame)
            frame_series.append(weighted_sum)
        # Add the frame's time series to the list
        model_signals.append(frame_series)

    # 'time_series' now contains the time series data

    # To access the time series for a specific frame:
    # Example: Time series for the first frame
    first_frame_series = model_signals[0]

    # Perform the rest of the code using CuPy
    # NOTE: I have skipped the necessary Orthogonalization parts
    grid_s_prime = cp.vstack([timecourse for timecourse in model_signals]).T
    grid_y = cp.vstack([timecourse for timecourse in simulated_voxel_responses]).T

    # (y^T . s_prime)^2
    proj_squared = (grid_s_prime.T @ grid_y) ** 2

    # find best matches along the rows (vertical axis) of the array
    best_fit_proj = cp.argmax(proj_squared, axis=0)

print("done")

if __name__ == '__main__':
    cProfile.run('gpu_gridfit()', sort='cumulative')
