extern "C"
{
	__global__ void test_kernel(double* large_array)
	{
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int frame = blockIdx.z * blockDim.z + threadIdx.z;
	}
}

