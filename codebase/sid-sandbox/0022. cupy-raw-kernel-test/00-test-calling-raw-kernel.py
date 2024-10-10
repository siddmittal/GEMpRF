## Add my "oprf" package path
#import sys
#sys.path.append("D:/code/sid-git/fmri/")

## config
#from config.config import Configuration

## imports
#import numpy as np
#import matplotlib.pyplot as plt
#import cupy as cp
#from cupyx.scipy import fft as fft_gpu
#from scipy import fft
#import cProfile
#import os

## Local Imports
#from oprf.standard.prf_stimulus import Stimulus
#from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator

#array_cpu = np.random.randint(0, 255, size=(2000, 2000) )
#array_gpu = cp.asarray(array_cpu)

#def cpu_computation():    
#    fft.fftn(array_cpu)

#def gpu_computation():
#    fft_gpu.fftn(array_gpu)

#def test_raw_cuda_kernel():
#    # Get the path of the current script
#    script_dir = os.path.dirname(os.path.realpath(__file__))

#    # Construct the path to the CUDA kernel file
#    kernel_file_path = os.path.join(script_dir, 'kernels/multiple_test_kernels.cu')

#    # Load the CUDA kernel file
#    with open(kernel_file_path, 'r') as kernel_file:
#        kernel_code = kernel_file.read()

#    # Compile the kernel code using CuPy
#    kernel = cp.RawKernel(kernel_code, 'applyMatricesOperation')

#    # Define matrix dimensions
#    rows, cols = 3, 3

#    # Generate random matrices A and B
#    A = cp.random.rand(rows, cols, dtype=cp.float32)
#    B = cp.random.rand(rows, cols, dtype=cp.float32)

#    # Allocate memory for the result matrix C
#    C = cp.empty((rows, cols), dtype=cp.float32)

#    # Specify grid and block dimensions
#    grid_size = (1, 1)
#    block_size = (cols, rows)

#    # Call the CUDA kernel to add matrices A and B
#    func_ptr = cp.RawKernel(kernel_code, 'addValues') #kernel.get_function("addValues")
#    kernel(grid_size, block_size, (func_ptr, A, B, C, rows, cols))

#    # Convert the CuPy array to a NumPy array for printing
#    result = C.get()    

#    # Print the result
#    print("Matrix A:")
#    print(A)
#    print("\nMatrix B:")
#    print(B)
#    print("\nMatrix C (A + B):")
#    print(result)



#    ####### Gaussian Test
#    #kernel_2 = cp.RawKernel(kernel_code, 'addOne')
#    #kernel_2(grid_size, block_size, (A, C, rows, cols))
#    #result2 = C.get()
#    #print(result2)


#########################----------------main---------------------################
#def main():
#    config = Configuration()
#    print(config.search_space_rows)
#    print("done")


#if __name__ == "__main__":
#    #main()
#    # cProfile.run('main()', sort='cumulative')
#    # cProfile.run('cpu_computation()', sort='cumulative')
#    # cProfile.run('gpu_computation()', sort='cumulative')
#    # cProfile.run('test_raw_cuda_kernel()', sort='cumulative')
#    test_raw_cuda_kernel()
#    print("done yes")


import cupy as cp

# Define the CUDA kernel code
kernel_code = """
extern "C" {
    __device__ float addValues(float a, float b)
    {
        float sum = a + b;
        printf("computed...");
        return sum;
    }
    typedef float(*FuncPtrAddition)(float, float);
	__device__ FuncPtrAddition d_ptrAddition = addValues;

    __global__ void addMatrices(float (*funcPtrAddValues)(float, float),
        const float* A,
        const float* B,
        float* C,
        int rows,
        int cols)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        

        if (col < cols && row < rows) 
        {
            int index = row * cols + col;
            printf("here...");
            C[index] = funcPtrAddValues(A[index], B[index]);
        }
    }
}
"""

# Create a RawModule from the kernel code
raw_module = cp.RawModule(code=kernel_code)

# Load the addMatrices kernel from the module
addMatrices_kernel = raw_module.get_function("addMatrices")

# Define the dimensions for your matrix
rows, cols = 3, 3

# Allocate device memory for input and output arrays
A = cp.random.rand(rows, cols).astype(cp.float32)
B = cp.random.rand(rows, cols).astype(cp.float32)
C = cp.empty((rows, cols), dtype=cp.float32)

# Define grid and block dimensions
block_dim = (3, 3)
grid_dim = (rows // block_dim[0], cols // block_dim[1])

# Launch the kernel, passing the function pointer as an argument
funcPtr_gpu = raw_module.get_global("d_ptrAddition")
funcPtr_cpu = funcPtr_gpu.get() #<-----------trying to match my C++ implementation
addMatrices_kernel(grid_dim, block_dim, (funcPtr_cpu, A, B, C, rows, cols))

# Synchronize and release memory
cp.cuda.Device().synchronize()

# Copy the result back to the host
result = C.get()
print(result)