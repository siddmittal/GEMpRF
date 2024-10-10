extern "C" 
{
    // substraction
    __device__ float substractValues(float a, float b)
    {
        float sum = a + b;
        return sum;
    }
    typedef float(*FuncPtrSubstraction)(float, float);
    __device__ FuncPtrSubstraction d_ptrSubstraction = substractValues;

    // addtion
    __device__ float addValues(float a, float b)
    {
        float sum = a + b;
        return sum;
    }
	typedef float(*FuncPtrAddition)(float, float);
	__device__ FuncPtrAddition d_ptrAddition = addValues;

    // main Kernel
    __global__ void applyMatricesOperation(float (*funcPtrOperation)(float, float),
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
            C[index] = funcPtrOperation(A[index] , B[index]); // A[index] + B[index]; OR A[index] - B[index]
            printf("computed...\n");
        }
    }

    // simple addition Kernel
    __global__ void addMatrices(
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
            C[index] = A[index] + B[index]; // OR A[index] - B[index]
        }
    }
}