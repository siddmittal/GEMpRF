# NOTE: If there an exception like "cupy_backends.cuda.libs.nvrtc.NVRTCError: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE (7)" or about "nvrtc-builtins64_118.dll"...
# ...then install the required version of CUDA Toolkit e.g. in the above case "118" means that we need to install CUDA Toolkit 11.8

import cupy as cp

def main():
    # Define two 1D arrays
    array1 = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
    array2 = cp.array([10, 20, 30, 40, 50], dtype=cp.float32)

    # Perform element-wise addition on the GPU
    result = array1 + array2

    # Transfer the result back to the CPU (if needed)
    result_cpu = cp.asnumpy(result)

    # Print the result
    print("Array 1:", array1)
    print("Array 2:", array2)
    print("Result (GPU):", result)
    print("Result (CPU):", result_cpu)

if __name__ == "__main__":
    main()
