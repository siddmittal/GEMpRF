# -*- coding: utf-8 -*-
"""

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

  
class PRFAnalysis():
    def __init__(self, prf_space : PRFSpace, stimulus : Stimulus) -> None:
        self.__prf_points = prf_space
        self.__stimulus = stimulus
        self.__model_signals_batches = None
        self.__dervatives_signals_batches_list = None
        self.__error_e = None
        self.__error_de_dtheta_list = None # contains a list of all derivative errors w.r.t each parameter

    @property
    def prf_points(self)-> PRFSpace:
        return self.__prf_points

    @property
    def stimulus(self)-> Stimulus:
        return self.__stimulus

    @property
    def orthonormalized_S_batches(self)-> List[cp.ndarray]:
        return self.__model_signals_batches
    
    @orthonormalized_S_batches.setter
    def orthonormalized_S_batches(self, new_value)-> None:
        self.__model_signals_batches = new_value

    @property
    def orthonormalized_dS_dtheta_batches_list(self)-> List[List[cp.ndarray]]:
        return self.__dervatives_signals_batches_list
    
    @orthonormalized_dS_dtheta_batches_list.setter
    def orthonormalized_dS_dtheta_batches_list(self, new_value)-> None:
        self.__dervatives_signals_batches_list = new_value

    @property
    def error_e(self)-> cp.ndarray:
        return self.__error_e
    
    @error_e.setter
    def error_e(self, new_value)-> None:
        self.__error_e = new_value

    @property
    def error_de_dtheta_list(self)-> List[cp.ndarray]:
        return self.__error_de_dtheta_list
    
    @error_de_dtheta_list.setter
    def error_de_dtheta_list(self, new_value)-> None:
        self.__error_de_dtheta_list = new_value

    