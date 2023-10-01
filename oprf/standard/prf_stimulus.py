import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os

class Stimulus:
    def __init__(self, relative_file_path, size_in_degrees):
        # IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        stimulus_file_path = os.path.join(script_directory, relative_file_path)
        stimulus_img = nib.load(stimulus_file_path)
        self.size_in_degrees = size_in_degrees
        self.org_data = (stimulus_img.get_fdata()).squeeze()
        self.resampled_data = None
        self.resampled_hrf_convolved_data = None

    def compute_resample_stimulus_data(self
                               , resampled_stimulus_shape # e.g. resampled_stimulus_shape = (DESIRED_STIMULUS_SIZE_X, DESIRED_STIMULUS_SIZE_Y, original_stimulus_shape[2])
                               ):  
        original_stimulus_shape = self.org_data.shape # e.g. (1024, 1024, 1, 300)        
        resampling_factors = (
        resampled_stimulus_shape[0] / (original_stimulus_shape[0] -1),
        resampled_stimulus_shape[1] / (original_stimulus_shape[1] - 1),
        resampled_stimulus_shape[2] / (original_stimulus_shape[2] - 1), # 1  # Keep the number of time points unchanged        
        )   
        self.resampled_data = zoom(self.org_data.squeeze(), resampling_factors, order=1)
    
    def compute_hrf_convolved_stimulus_data(self, hrf_curve):  
        stimulus_location_convolved_timecourses = np.empty_like(self.resampled_data, dtype=float)
        for row in range (self.resampled_data.shape[0]):
            for col in range (self.resampled_data.shape[1]):
                location_tc = self.resampled_data[row, col, :]
                stimulus_location_convolved_timecourses[row, col, :] = np.convolve(location_tc, hrf_curve, mode='full')[:len(location_tc)]  
        
        self.resampled_hrf_convolved_data = stimulus_location_convolved_timecourses
