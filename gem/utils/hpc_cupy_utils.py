# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Siddharth Mittal",
"@Desc    :   None",     
"""

import os
import cupy as cp
import numpy as np
from typing import List
import subprocess


class HpcUtils:
    usage_label = None  # Class attribute to hold the label widget
    gpu_memory_pool = None  # Class attribute to hold the GPU memory pool
    root = None  # Class attribute to hold the Tkinter root window
    num_gpus = -1

    @classmethod
    def get_number_of_gpus(cls):
        if cls.num_gpus != -1:
            return cls.num_gpus
        
        try:            
            cls.num_gpus = cp.cuda.runtime.getDeviceCount()
            return cls.num_gpus
        except Exception as e:            
            return f"Error: {str(e)}"
        
    # Get RAW Kernel
    @classmethod
    def get_raw_kernel(cls, kernel_filename, kernel_name):
        # Get the path of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct the path to the CUDA kernel file
        kernel_file_path = os.path.join(script_dir, ".." , 'kernels', kernel_filename) # os.path.join(script_dir, kernel_filename)

        # Load the CUDA kernel file
        with open(kernel_file_path, 'r') as kernel_file:
            kernel_code = kernel_file.read()

        # Compile the kernel code using CuPy
        kernel = cp.RawKernel(kernel_code, kernel_name)

        return kernel

    # Compile CUDA .CU to PTX
    @classmethod
    def compile_cuda_to_ptx(cls, cu_file, output_ptx):
        try:
            subprocess.run(
                ['nvcc', '-ptx', cu_file, '-o', output_ptx],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print("Compilation of .cu to .ptx failed:")
            print(e.stderr)

    # Get RAW Module
    @classmethod
    def get_raw_module(cls, module_filename_without_ext):
        # Get the path of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))
        gpu_kernels_dir = os.path.join(script_dir, "..", 'kernels')

        # Construct the path to the CUDA kernel file
        kernel_file_path = os.path.join(gpu_kernels_dir, f"{module_filename_without_ext}.cu")

        # check if the .ptx file already exists, generate one if not
        ptx_file_path = os.path.join(gpu_kernels_dir, f"{module_filename_without_ext}.ptx")
        if not os.path.exists(ptx_file_path):
            cls.compile_cuda_to_ptx(kernel_file_path, ptx_file_path)

        # # # Load the CUDA kernel file
        # # with open(kernel_file_path, 'r') as kernel_file:
        # #     module_code = kernel_file.read()

        # # Compile the kernel code using CuPy
        # # raw_module = cp.RawModule(code=module_code)
        raw_module = cp.RawModule(path=ptx_file_path)

        return raw_module

    # Release the GPU memory kept by the variables which are either gone out of scope or deleted
    @classmethod
    def free_device_memory(cls):
        if cls.gpu_memory_pool is None:
            cls.gpu_memory_pool = cp.get_default_memory_pool()

        cls.gpu_memory_pool.free_all_blocks()
    
    @classmethod
    def delete_gpu_variables(cls, variable : List[cp.ndarray]):
        for v in variable:
            del v
        cls.free_device_memory()

    @classmethod
    def print_gpu_memory_stats(cls): 
        if cls.gpu_memory_pool is None:
            cls.gpu_memory_pool = cp.get_default_memory_pool()

        total_mempool = cls.gpu_memory_pool.total_bytes() / (1024 * 1024)
        used_mempool = cls.gpu_memory_pool.used_bytes() / (1024 * 1024)
        print(f'Total Mempool Size: {total_mempool} MB')
        print(f'Used Mempool Size: {used_mempool} MB')   
    
    @classmethod
    def device_available_mem_bytes(cls, device_id): 
        total_free_mem = 0 # bytes
        with cp.cuda.Device(device_id):    

            # get available memory in the already allocated mempool
            free_mem_in_allocated_pool = 0
            if cls.gpu_memory_pool is None:
                cls.gpu_memory_pool = cp.get_default_memory_pool()
                total_mempool = cls.gpu_memory_pool.total_bytes()
                used_mempool = cls.gpu_memory_pool.used_bytes()
                free_mem_in_allocated_pool = total_mempool - used_mempool

            # get still available free memory on GPU
            available, total = cp.cuda.runtime.memGetInfo()

            # total free
            total_free_mem = free_mem_in_allocated_pool + available

        return total_free_mem
   

    @classmethod
    def gpu_mem_required_in_gb(cls, num_elements): 
        # get number of bytes in a single element of float64 type
        num_bytes_in_single_float_element =  np.dtype(np.float64).itemsize

        # total bytes needed by all the elements
        num_bytes_required = num_elements * num_bytes_in_single_float_element
        size_kb = num_bytes_required / 1024
        size_mb = size_kb / 1024
        size_gb = size_mb / 1024
        return size_gb 