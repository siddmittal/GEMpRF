import numpy as np
import cupy as cp
import os
import sys
import nibabel as nib
from enum import Enum

# Define an enumeration class
class DataType(Enum):
    measured_data = 1
    simulated_data = 2

class ObservedData:
    def __init__(self, data_type):
        self.data_type = data_type # measured_data or simulated_data

    def get_y_signals(self, filepath):
        if self.data_type is DataType.measured_data:
          Y_signals_cpu = self._load_data_from_file(filepath)

        return Y_signals_cpu

    def _load_data_from_file(self, filepath):
        # load the BOLD response data
        bold_response_img = nib.load(filepath)
        Y_signals_cpu = bold_response_img.get_fdata()

        # reshape the BOLD response data to 2D
        Y_signals_cpu = Y_signals_cpu.reshape(-1, Y_signals_cpu.shape[-1])

        # just to make them column vectors
        Y_signals_cpu = Y_signals_cpu.T

        return Y_signals_cpu

