# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:14:23 2024

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Medical University of Vienna",
"@Desc    :   None",
        
"""
import abc
import cupy as cp
import numpy as np
from typing import List
from enum import Enum


import gem.model.selected_prf_model as SelectedPRFModel
import gem.space.PRFSpace as PRFSpace
import gem.model.prf_stimulus as Stimulus

class GaussianModelParams(Enum):
    NONE = -1
    X0 = 0
    Y0 = 1
    SIGMA = 2

class DoGModelParams(Enum):
    X0 = 1
    Y0 = 2
    SIGMA_MAJOR = 3  
    SIGMA_MINOR = 4  


class PRFModel(metaclass=abc.ABCMeta):
    """A Parser metaclass that will be used for parser class creation. This interface is used for concrete classes to inherit from.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                # create_model
                hasattr(subclass, 'create_model') and 
                callable(subclass.create_model) and 

                # extract_text
                hasattr(subclass, 'extract_text') and 
                callable(subclass.extract_text) and 
                
                # model_type
                hasattr(subclass, 'model_type') and 
                callable(subclass.model_type)  and

                # keep valditaed sample points
                hasattr(subclass, 'get_validated_sampling_points_indices') and 
                callable(subclass.get_validated_sampling_points_indices)  and
                
                # get_model_curves_kernel
                hasattr(subclass, 'get_model_curves_kernel') and
                callable(subclass.get_model_curves_kernel) and

                # get_derivatives_kernels_list
                hasattr(subclass, 'get_derivatives_kernels_list') and
                callable(subclass.get_derivatives_kernels_list) 
                
                # and

                # # prf_points
                # hasattr(subclass, 'prf_points') and
                # callable(subclass.prf_points)

                or NotImplemented)

    #-------Abtract Methods-------------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def create_model(self, model_name: str, file_name: str):
        """Load in the data set"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_validated_sampling_points_indices(self, model_name: str, file_name: str):
        """Get the indices of the validated sampling points (i.e. check that the pRF point fullfills the requirement of the selected model)"""
        raise NotImplementedError    

    @abc.abstractmethod
    def extract_text(self, full_file_path: str):
        """Extract text from the data set"""
        raise NotImplementedError
        
    #-------Abtract Properties------------------------------------------------------------------------------------------------------------
    @abc.abstractproperty
    def model_type(self) -> SelectedPRFModel:
            """Get the model type"""
            raise NotImplementedError     

    @abc.abstractproperty
    def num_dimensions(self) -> int:
            """Get the model type"""
            raise NotImplementedError  

#     @abc.abstractproperty
#     def prf_points(self) -> PRFPoints:
#             """Get the model type"""
#             raise NotImplementedError   
    
    # Properties
    @abc.abstractproperty
    def model_curves_kernel(self) -> cp.RawKernel:
            """Get the CUDA kernel for the model curves"""
            raise NotImplementedError

    @abc.abstractproperty
    def derivatives_kernels_list(self) -> List[cp.RawKernel]:
            """Get the CUDA kernels for the derivatives"""
            raise NotImplementedError
    
#     @abc.abstractproperty
#     def stimulus(self) -> Stimulus:
#             """Get the stimulus used for the model"""
#             raise NotImplementedError
    
    @abc.abstractproperty
    def model_parameters_list(self) -> list:
            """Get the model parameters"""
            raise NotImplementedError

#     @abc.abstractproperty
#     def model_parameters_enum(self) -> Enum:
#             """Get the model parameters"""
#             raise NotImplementedError