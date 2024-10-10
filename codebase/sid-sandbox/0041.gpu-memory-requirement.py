import cupy as cp
import numpy as np

def get_mem_required_gb(num_elements):
    # get number of bytes in a single element of float64 type
    num_bytes_in_single_float_element =  np.dtype(np.float64).itemsize

    # total bytes needed by all the elements
    num_bytes_required = num_elements * num_bytes_in_single_float_element
    size_kb = num_bytes_required / 1024
    size_mb = size_kb / 1024
    size_gb = size_mb / 1024
    return size_gb





################--------------------------main()---------################
h = 151
w = 151
sigma = 16
stim_h = 101
stim_w  =101

# number of elements in your array
num_elements = h * w * 16 * stim_h * stim_w

# required memory
required_mem_gb = get_mem_required_gb(num_elements)

# The code below wold throw an EXCEPTION because of the huge memory requirement
arr = cp.zeros((151*151*16*101*101), dtype=cp.float64)


