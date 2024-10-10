import cupy as cp

# Create a raw module
ptx_code_path = "D:/code/sid-git/fmri/codebase/sid-sandbox/0012_use_PTX/kernel.ptx"
raw_module = cp.RawModule(path=ptx_code_path)

# Get the kernel functions from the raw module
vector_add_kernel = raw_module.get_function('vector_add')
vector_subtract_kernel = raw_module.get_function('vector_subtract')

def vector_add(a, b):
    size = a.size
    result = cp.empty(size, dtype=cp.int32)
    vector_add_kernel(grid=(size,), block=(256,), args=(a, b, result, size))
    return result

def vector_subtract(a, b):
    size = a.size
    result = cp.empty(size, dtype=cp.int32)
    vector_subtract_kernel(grid=(size,), block=(256,), args=(a, b, result, size))
    return result

# Example usage
a = cp.array([1, 2, 3, 4])
b = cp.array([5, 6, 7, 8])

# Perform vector addition and subtraction
result_add = vector_add(a, b)
result_subtract = vector_subtract(a, b)

print("Vector Addition Result:", result_add)
print("Vector Subtraction Result:", result_subtract)
