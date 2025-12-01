# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024 - 2025, Medical University of Vienna",
"@Desc    :   None",
        
"""

import nibabel as nib
from enum import Enum
import numpy as np
from gem.utils.logger import Logger

# Define an enumeration class
class DataSource(Enum):
    measured_data = 1
    simulated_data = 2

class ObservedData:
    def __init__(self, data_source : DataSource):
        self.data_source = data_source

    def get_y_signals(self, filepath):
        if self.data_source is DataSource.measured_data:
          Y_signals_cpu = self._load_data_from_file(filepath)

        return Y_signals_cpu

    def _load_data_from_file(self, filepath):
        # load the BOLD response data
        bold_response_img = nib.load(filepath)

        # GIFTI files (from fmriprep) contain one or more "data arrays"
        if isinstance(bold_response_img, nib.gifti.GiftiImage):
            # Stack all data arrays (each one corresponds to a surface vertex timeseries)
            Y_signals_cpu = np.column_stack([darray.data for darray in bold_response_img.darrays])
        else:
            Y_signals_cpu = bold_response_img.get_fdata()

        # Shape example: (x, y, z, t) or (n_vertices, t)
        shape = Y_signals_cpu.shape
        spatial_dims = shape[:-1]

        # Count how many spatial dims are > 1
        active_dims = sum(d > 1 for d in spatial_dims)

        # reshape the BOLD response data to 2D
        Y_signals_cpu = Y_signals_cpu.reshape(-1, Y_signals_cpu.shape[-1])

        # Logger.print_yellow_message(f"Data shape {shape} {['', f'→ {Y_signals_cpu.shape}'][(active_dims > 1)]}", print_file_name=False)
        Logger.print_yellow_message(f"Data shape {shape} → {Y_signals_cpu.shape}", print_file_name=False)

        # just to make them column vectors
        Y_signals_cpu = Y_signals_cpu.T

        return Y_signals_cpu

