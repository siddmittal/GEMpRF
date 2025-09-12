# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Siddharth Mittal",
"@Desc    :   None",     
"""

import os
import h5py
import numpy as np
import cupy as cp


class GemWriteToFile:
    _instance = None  # Singleton instance

    def __new__(cls, result_dir, debugging_enabled=False):
        if cls._instance is None:
            cls._instance = super(GemWriteToFile, cls).__new__(cls)
            cls._instance.__initialize(result_dir, debugging_enabled)
        return cls._instance

    def __initialize(self, result_dir, debugging_enabled):
        self.__result_dir = result_dir
        self.__debugging_enabled = debugging_enabled

    @classmethod
    def get_instance(cls):
        return cls._instance

    def write_array_to_h5(self, data, variable_path, append_to_existing_variable=False):
        """
        Write a NumPy/CuPy array or list of arrays into an HDF5 file with hierarchical groups.
        If variable_path corresponds to 'model_signals' or 'derivative_model_signals_*', 
        concatenates list of arrays along axis=0 before writing.
        """
        if not self.__debugging_enabled:
            return

        filepath = os.path.join(self.__result_dir, "debug_model_data.h5")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert variable_path to a slash-separated string
        if isinstance(variable_path, (list, tuple)):
            variable_path_str = "/".join(variable_path)
        else:
            variable_path_str = variable_path

        # Concatenate list of arrays if needed
        if isinstance(data, list):
            # Keywords for special handling
            special_keys = [
                "model_signals",
                "model_signals_derivative",
                "orthonormalized_model_signals",
                "orthonormalized_model_signals_derivative",
            ]

            if any(key in variable_path_str for key in special_keys):
                # Flatten list of lists
                flat_list = []
                for item in data:
                    flat_list.extend(item if isinstance(item, list) else [item])

                concat_arrays = []
                for arr in flat_list:
                    if isinstance(arr, cp.ndarray):
                        arr = cp.asnumpy(arr)

                    # Transpose first if it's an orthonormalized variant
                    if any(key in variable_path_str for key in [
                        "orthonormalized_model_signals",
                        "orthonormalized_model_signals_derivative"
                    ]):
                        arr = arr.T

                    concat_arrays.append(arr)

                # Now concatenate safely
                data_to_write = np.concatenate(concat_arrays, axis=0)
            else:
                # Not a special variable, just convert list → numpy
                data_to_write = np.array(data)                
        else:
            # Already array-like, move to CPU if needed
            data_to_write = cp.asnumpy(data) if isinstance(data, cp.ndarray) else data


        # Handle string arrays
        if np.issubdtype(data_to_write.dtype, np.str_) or np.issubdtype(data_to_write.dtype, np.object_):
            data_to_write = np.array(data_to_write, dtype=h5py.string_dtype(encoding='utf-8'))

        # Open HDF5 file
        with h5py.File(filepath, "a") as f:
            if variable_path_str in f:
                dset = f[variable_path_str]
                if append_to_existing_variable:
                    # Handle 1D arrays
                    if data_to_write.ndim == 1:
                        data_to_write = data_to_write.reshape(-1, 1)
                        dset_shape = (dset.shape[0], 1)
                    else:
                        dset_shape = dset.shape

                    if data_to_write.shape[1:] != dset_shape[1:]:
                        raise ValueError(
                            f"Shape mismatch: cannot append array {data_to_write.shape} "
                            f"to existing dataset {dset.shape}"
                        )
                    dset.resize((dset.shape[0] + data_to_write.shape[0]), axis=0)
                    dset[-data_to_write.shape[0]:] = data_to_write
                else:
                    del f[variable_path_str]
                    maxshape = (None,) + data_to_write.shape[1:] if data_to_write.ndim > 1 else (None,)
                    f.create_dataset(variable_path_str, data=data_to_write, maxshape=maxshape)
            else:
                maxshape = (None,) + data_to_write.shape[1:] if data_to_write.ndim > 1 else (None,)
                f.create_dataset(variable_path_str, data=data_to_write, maxshape=maxshape)