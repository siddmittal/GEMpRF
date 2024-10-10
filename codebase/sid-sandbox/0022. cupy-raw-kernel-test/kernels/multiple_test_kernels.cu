extern "C"
{
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

    // simple substract Kernel
    __global__ void addOne(
        const float* A,
        float* C,
        int rows,
        int cols)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col < cols && row < rows)
        {
            int index = row * cols + col;
            C[index] = A[index] + 1; // OR A[index] - B[index]
        }
    }
}