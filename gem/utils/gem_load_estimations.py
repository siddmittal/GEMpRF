# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Siddharth Mittal",
"@Desc    :   None",     
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

class EstimationsUtils:
    dummy_variable = None  # Class variable

    @classmethod
    def get_estimation_data(cls, filepath):
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)
            return data
        
    @classmethod
    def compare_estimation_results(cls, new_results , benchmark_results , tolerance=0.01):        
        # helper NaN function
        def is_nan(value):
            try:
                return math.isnan(value)
            except TypeError:
                return False
        
        max_differece = 0
        for bench_res, new_res in zip(benchmark_results, new_results):
            for key in bench_res:
                if key in new_res:
                    bench_value = bench_res[key]
                    new_value = new_res[key]
                    
                     # Check if both values are NaN
                    if is_nan(bench_value) and is_nan(new_value):
                        continue

                     # Check if one value is NaN and the other is not
                    if is_nan(bench_value) or is_nan(new_value):
                        print(f"Difference in {key}: {bench_value} (benchmark) vs {new_value} (new) [One value is NaN]")
                        max_differece = sys.maxint                                                
                        break

                    # Check if the values are lists (e.g., modelpred)
                    if isinstance(bench_value, list) and isinstance(new_value, list):
                        for i, (bv, nv) in enumerate(zip(bench_value, new_value)):
                            if is_nan(bv) and is_nan(nv):
                                continue
                            elif nv is None: # e.g. in case of new values, the modelpred is None because we don't write the values to keep JSON short
                                continue
                            elif is_nan(bv) or is_nan(nv):
                                print(f"Difference in {key} at index {i}: {bv} (benchmark) vs {nv} (new) [One value is NaN]")
                                max_differece = sys.maxint                                                
                                break
                            elif abs(bv - nv) > tolerance:
                                    max_differece = abs(bv - nv)
                                    print(f"Difference in {key} at index {i}: {bv} (benchmark) vs {nv} (new)")
                    else:
                        if abs(bench_value - new_value) > tolerance:
                            max_differece = abs(bench_value - new_value)
                            print(f"Difference in {key}: {bench_value} (benchmark) vs {new_value} (new)")  

        return max_differece
    
    @classmethod
    def get_avg_2d_gaussian_estimated_values(cls, json_data):
        # Filter out entries with NaN values
        # filtered_data = [entry for entry in data if not any(np.isnan(value) for value in [entry['Centerx0'], entry['Centery0'], entry['sigmaMajor']])]
        filtered_data = [entry for entry in json_data 
                        if not (any(np.isnan(value) for value in [entry['Centerx0'], entry['Centery0'], entry['sigmaMajor']]) 
                                                        or entry['R2'] < 0.1)]
        
        # Compute mean values of Centerx0, Centery0, and sigmaMajor
        mean_Centerx0 = np.mean([entry['Centerx0'] for entry in filtered_data])
        mean_Centery0 = np.mean([entry['Centery0'] for entry in filtered_data])
        mean_sigmaMajor = np.mean([entry['sigmaMajor'] for entry in filtered_data])

        return mean_Centerx0, mean_Centery0, mean_sigmaMajor

