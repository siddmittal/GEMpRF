extern "C"
{
    __global__ void vector_add(const int* a, const int* b, int* result, int size) 
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) 
        {
            result[tid] = a[tid] + b[tid];
        }
    }

    __global__ void vector_subtract(const int* a, const int* b, int* result, int size) 
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            result[tid] = a[tid] - b[tid];
        }
    }

}
