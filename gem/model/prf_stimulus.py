# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Siddharth Mittal",
"@Desc    :   None",     
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
import os
from gem.utils.gem_gpu_manager import GemGpuManager as ggm
from gem.utils.logger import Logger

class Stimulus:
    def __init__(self, relative_file_path, size_in_degrees, stim_config, binarize, binarize_threshold, high_temporal_resolution_info=None, stimulus_task_name=None):
        # IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        stimulus_file_path = os.path.join(script_directory, relative_file_path)
        stimulus_img = nib.load(stimulus_file_path)
        self.size_in_degrees = size_in_degrees
        self.org_data = (stimulus_img.get_fdata()).squeeze()
        if binarize:
            if not np.all(np.isin(self.org_data, [0, 1])):  
                Logger.print_yellow_message("Warning: Stimulus data contains values other than 0 and 1. Binarizing...", print_file_name=False)
                self.org_data = (self.org_data > binarize_threshold).astype(np.uint8)  # Convert to 0s and 1s
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
        self.__header = stimulus_img.header
        self.__stimulus_task_name = stimulus_task_name

        # high temporal resolution stimulus params
        if high_temporal_resolution_info:
            self.__high_temporal_resolution = True
            self.__num_frames_downsampled = high_temporal_resolution_info['num_frames_downsampled']
            self.__slice_time_ref = high_temporal_resolution_info['slice_time_ref']
        else:
            self.__high_temporal_resolution = False
            self.__num_frames_downsampled = None
            self.__slice_time_ref = None

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
    
    @property
    def HighTemporalResolutionEnabled(self):
        return self.__high_temporal_resolution
    
    @property
    def NumFramesDownsampled(self):
        return self.__num_frames_downsampled
    
    @property
    def SliceTimeRef(self):
        return self.__slice_time_ref

    @property
    def Header(self):
        return self.__header

    @property
    def StimulusTaskName(self):
        return self.__stimulus_task_name

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
        
        # # To use High Resolution Stimulus
        # self.resampled_data = self.org_data

    def compute_hrf_convolved_stimulus_data(self, hrf_curve):
        """
        Computes HRF-convolved stimulus timecourses for all pixels
        using a vectorized FFT-based approach.
        """
        rows, cols, T = self.resampled_data.shape
        N = rows * cols

        # Flatten spatial dimensions so each row is a pixel timecourse (i.e. each row would contain the values a pixel at all time points)
        flat = self.resampled_data.reshape(N, T)  # shape: (N, T)

        # FFT-based convolution along time axis
        conv_full = fftconvolve(flat, hrf_curve[None, :], mode='full', axes=1)

        # Truncate to original length
        conv_same = conv_full[:, :T]

        # Reshape back to original 3D shape
        self.__resampled_hrf_convolved_data = conv_same.reshape(rows, cols, T)
        return self.__resampled_hrf_convolved_data        

    def __compute_flattened_columnmajor_stimulus_data(self):  ####<----This one is more correct
        stimulus_flat_data_cpu = self.__resampled_hrf_convolved_data.flatten('C')

        # GPU
        if self.__flattened_columnmajor_stimulus_data_gpu is None:
            stimulus_flat_data_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=(stimulus_flat_data_cpu,))
            stim_height, stim_width, stim_frames = self.resampled_data.shape
            self.__flattened_columnmajor_stimulus_data_gpu = cp.reshape(stimulus_flat_data_gpu, (stim_height * stim_width, stim_frames), order='C') # each row contains a flat stimulus frame

        # CPU
        if self.__flattened_columnmajor_stimulus_data_cpu is None: 
            self.__flattened_columnmajor_stimulus_data_cpu = np.reshape(stimulus_flat_data_cpu, (stim_height * stim_width, stim_frames), order='C') # each row contains a flat stimulus frame            
            
    def data_shape(self):
        return self.__resampled_hrf_convolved_data.shape