# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:14:51 2024

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Medical University of Vienna",
"@Desc    :   None",
        
"""


import numpy as np
import cupy as cp
from typing import List

from gem.model.prf_model import PRFModel
from gem.model.selected_prf_model import SelectedPRFModel
from gem.model.prf_stimulus import Stimulus
from gem.space.PRFSpace import PRFSpace
from gem.utils.hpc_cupy_utils import HpcUtils as gpu_utils

  
class PRFGaussianModel(PRFModel):
    def __init__(self, visual_field_radius) -> None:
        super().__init__()
        self.__model_type = SelectedPRFModel.GAUSSIAN
        # self.__stimulus = stimulus
        # self.__prf_points = prf_points
        self.__visual_field_radius = visual_field_radius
        self.__num_dimensions = 3
        self.model_signals_batches = None
        self.derivative_signals_batches = None
        self.__cuda_module = gpu_utils.get_raw_module('gaussian_kernel')
        self.__gc_kernel = self.__cuda_module.get_function("gc_using_args_arrays_cuda_Kernel")
        self._derivative_kernels_list = [self.__cuda_module.get_function("dgc_dx_using_args_arrays_cuda_Kernel"), 
                                         self.__cuda_module.get_function("dgc_dy_using_args_arrays_cuda_Kernel"), 
                                         self.__cuda_module.get_function("dgc_dsigma_using_args_arrays_cuda_Kernel")]
        self.__model_parameters_list = ['x0', 'y0', 'sigma']

    ####################------------Abstract methods implementations------------####################
    def create_model(self, model_name: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        print('entered in PRF Gaussian Model...')
        pass

    def extract_text(self, full_file_path: str) -> dict:
        """Overrides FormalParserInterface.extract_text()"""
        pass

    def get_validated_sampling_points_indices(self, multidim_points : np.ndarray) -> np.ndarray:
        """
        Overrides FormalParserInterface.get_validated_sampling_points_indices().
        
        Extracts the validated pRF sampling points from the multidimensional points.

        Parameters
        ----------
        multidim_points : np.ndarray
            The multidimensional points to be validated.

        Returns
        -------
        np.ndarray
            The indices of validated pRF sampling points.
        """

        # NOTE NOTE: Initial validation
        print("Extracting validated pRF sampling points...")        
        valid_indices = np.where((multidim_points[:, 0]**2 + multidim_points[:, 1]**2) < (self.__visual_field_radius**2)) # Calculate the condition (x^2 + y^2) < (radius^2)        
        return valid_indices[0] # "valid_indices" is a tuple so, extracting the array   
        
        # # # NOTE: New validation which includes the check for pRF size within the lower and upper bounds
        # # eccentricity = np.sqrt(multidim_points[:, 0]**2 + multidim_points[:, 1]**2)

        # # # pRF size bounds
        # # lower_boundary = 0.3333 * eccentricity + 0.5
        # # upper_boundary = 0.1667 * eccentricity + 2.75

        # # within_radius_indices = np.where((eccentricity) < self.__visual_field_radius)
        # # valid_pRF_indices = np.where((multidim_points[:, 2] >= lower_boundary) & (multidim_points[:, 2] <= upper_boundary))
        # # valid_indices = np.intersect1d(within_radius_indices[0], valid_pRF_indices[0]) # combined

        # # return valid_indices
        

    ####################------------Properties------------####################
    @property
    def model_parameters_list(self) -> list:
        """Overrides FormalParserInterface.model_parameters_list()"""
        return self.__model_parameters_list

    # @property
    # def prf_points(self) -> PRFPoints:
    #     """Overrides FormalParserInterface.prf_points()"""
    #     return self.__prf_points      
    
    @property
    def num_dimensions(self) -> int:
        """Overrides FormalParserInterface.num_dimensions()"""
        return self.__num_dimensions

    @property
    def model_curves_kernel(self) -> cp.RawKernel:
        """Overrides FormalParserInterface.get_model_curves_kernel()"""
        return self.__gc_kernel

    @property
    def derivatives_kernels_list(self) -> list:
        """Overrides FormalParserInterface.get_derivatives_kernels_list()"""
        return self._derivative_kernels_list
    
    # @property
    # def stimulus(self) -> Stimulus:
    #     return self.__stimulus        

    @property
    def model_type(self) -> SelectedPRFModel:
        """Overrides FormalParserInterface.model_type()"""
        return self.__model_type                