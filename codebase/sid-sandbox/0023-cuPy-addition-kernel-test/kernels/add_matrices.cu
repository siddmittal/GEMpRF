extern "C" 
{
    __global__ void addMatrices(const float* A, const float* B, float* C, int rows, int cols) 
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col < cols && row < rows) 
        {
            int index = row * cols + col;
            C[index] = A[index] + B[index];
        }
    }
}