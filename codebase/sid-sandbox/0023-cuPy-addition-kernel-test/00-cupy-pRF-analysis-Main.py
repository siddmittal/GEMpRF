# Add my "oprf" package path
import sys
sys.path.append("D:/code/sid-git/fmri/")

# config
from config.config import Configuration

# imports
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy import fft as fft_gpu
from scipy import fft
import cProfile
import os

# Local Imports
from oprf.standard.prf_stimulus import Stimulus
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator

array_cpu = np.random.randint(0, 255, size=(2000, 2000) )
array_gpu = cp.asarray(array_cpu)

def cpu_computation():    
    fft.fftn(array_cpu)

def gpu_computation():
    fft_gpu.fftn(array_gpu)

def test_raw_cuda_kernel():
    # Get the path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the CUDA kernel file
    kernel_file_path = os.path.join(script_dir, 'kernels/add_matrices.cu')

    # Load the CUDA kernel file
    with open(kernel_file_path, 'r') as kernel_file:
        kernel_code = kernel_file.read()

    # Compile the kernel code using CuPy
    kernel = cp.RawKernel(kernel_code, 'addMatrices')

    # Define matrix dimensions
    rows, cols = 3, 3

    # Generate random matrices A and B
    A = cp.random.rand(rows, cols, dtype=cp.float32)
    B = cp.random.rand(rows, cols, dtype=cp.float32)

    # Allocate memory for the result matrix C
    C = cp.empty((rows, cols), dtype=cp.float32)

    # Specify grid and block dimensions
    grid_size = (1, 1)
    block_size = (cols, rows)

    # Call the CUDA kernel to add matrices A and B
    kernel(grid_size, block_size, (A, B, C, rows, cols))

    # Convert the CuPy array to a NumPy array for printing
    result = C.get()    

    # Print the result
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix C (A + B):")
    print(result)

########################----------------main---------------------################
def main():
    config = Configuration()
    print(config.search_space_rows)
    print("done")


if __name__ == "__main__":
    #main()
    # cProfile.run('main()', sort='cumulative')
    # cProfile.run('cpu_computation()', sort='cumulative')
    # cProfile.run('gpu_computation()', sort='cumulative')
    # cProfile.run('test_raw_cuda_kernel()', sort='cumulative')
    test_raw_cuda_kernel()
    print("done yes")