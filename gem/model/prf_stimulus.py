import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

class Stimulus:
    def __init__(self, relative_file_path, size_in_degrees, stim_config):
        # IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        stimulus_file_path = os.path.join(script_directory, relative_file_path)
        stimulus_img = nib.load(stimulus_file_path)
        self.size_in_degrees = size_in_degrees
        self.org_data = (stimulus_img.get_fdata()).squeeze()
        self.resampled_data = None
        self.__resampled_hrf_convolved_data = None
        self.__flattened_columnmajor_stimulus_data_gpu = None
        self.__flattened_columnmajor_stimulus_data_cpu = None
        self.width = int(stim_config["width"])
        self.height = int(stim_config["height"])
        self.x_range_cpu = np.linspace(-float(stim_config["visual_field"]), +float(stim_config["visual_field"]), int(stim_config["width"]))
        self.y_range_cpu = np.linspace(-float(stim_config["visual_field"]), +float(stim_config["visual_field"]), int(stim_config["height"]))
        self.x_range_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=(self.x_range_cpu,))
        self.y_range_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=(self.y_range_cpu,))

    """Column major stimulus data on GPU"""
    @property
    def stimulus_data_gpu(self):
        if self.__flattened_columnmajor_stimulus_data_gpu is None:
            self.__compute_flattened_columnmajor_stimulus_data()
        return self.__flattened_columnmajor_stimulus_data_gpu    

    """Column major stimulus data on CPU"""
    @property
    def stimulus_data_cpu(self):
        if self.__flattened_columnmajor_stimulus_data_cpu is None:
            self.__compute_flattened_columnmajor_stimulus_data()
        return self.__flattened_columnmajor_stimulus_data_cpu
    
    @property
    def Height(self):
        return self.height
    
    @property
    def Width(self):
        return self.width
    
    @property
    def NumFrames(self):
        return self.org_data.shape[2]

    def compute_resample_stimulus_data(self
                               , resampled_stimulus_shape # e.g. resampled_stimulus_shape = (DESIRED_STIMULUS_SIZE_X, DESIRED_STIMULUS_SIZE_Y, original_stimulus_shape[2])
                               ):  
        original_stimulus_shape = self.org_data.shape # e.g. (1024, 1024, 1, 300)        
        resampling_factors = (
        resampled_stimulus_shape[0] / (original_stimulus_shape[0]),  # TODO: MAybe "-1" needs to be removed !!!!
        resampled_stimulus_shape[1] / (original_stimulus_shape[1]), # TODO: MAybe "-1" needs to be removed !!!!
        resampled_stimulus_shape[2] / (original_stimulus_shape[2]), # 1  # Keep the number of time points unchanged        
        )   
        self.resampled_data = zoom(self.org_data.squeeze(), resampling_factors, order=1)
    
    def compute_hrf_convolved_stimulus_data(self, hrf_curve):  
        stimulus_location_convolved_timecourses = np.empty_like(self.resampled_data, dtype=float)
        for row in range (self.resampled_data.shape[0]):
            for col in range (self.resampled_data.shape[1]):
                location_tc = self.resampled_data[row, col, :]
                stimulus_location_convolved_timecourses[row, col, :] = np.convolve(location_tc, hrf_curve, mode='full')[:len(location_tc)]  
        
        self.__resampled_hrf_convolved_data = stimulus_location_convolved_timecourses

    def __compute_flattened_columnmajor_stimulus_data(self):  ####<----This one is more correct
        stimulus_flat_data_cpu = self.__resampled_hrf_convolved_data.flatten('F')

        # GPU
        if self.__flattened_columnmajor_stimulus_data_gpu is None:
            stimulus_flat_data_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=(stimulus_flat_data_cpu,))
            stim_height, stim_width, stim_frames = self.resampled_data.shape
            self.__flattened_columnmajor_stimulus_data_gpu = cp.reshape(stimulus_flat_data_gpu, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame

        # CPU
        if self.__flattened_columnmajor_stimulus_data_cpu is None: 
            self.__flattened_columnmajor_stimulus_data_cpu = np.reshape(stimulus_flat_data_cpu, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame            
            
    def data_shape(self):
        return self.__resampled_hrf_convolved_data.shape