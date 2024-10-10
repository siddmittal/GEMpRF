#include "../header/ModelSignals.h"
#include <cuda_runtime.h>



void ModelSignals::freeDeviceMemoryWrapper(double* floatDevicePtrToFree)
{
	cudaFree(floatDevicePtrToFree);
}