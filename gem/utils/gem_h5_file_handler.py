# -*- coding: utf-8 -*-

"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Siddharth Mittal",
"@Desc    :   None",
        
"""
import h5py


class H5FileManager:

    @classmethod
    def get_key_value(cls, filepath, key = None, type = None):        
        with h5py.File(filepath, 'r') as f:
            if key:
                if type:
                    return f[key][()].astype(type)
                return f[key][()]
        return None
