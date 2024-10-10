#include "../header/ModelSignals.h"
#include "../../utils/arrays/Linspace.h"

#include <math.h>
#include <string>

// gpu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void computeGaussianAndStimulusMultiplicationKernel(double* result_stimulus_multiplied_gaussian_curves
	, double* flattened_3d_stimulusData
	, double* flattened_gaussian_curves_data
	, int singleGaussianCurveLength
	, int nTotalGaussianCurvesPerStimulusFrame
	, int nStimulusFrames
	, int nStimulusRows
	, int nStimulusCols)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int frame = blockIdx.z * blockDim.z + threadIdx.z;

	if (y < nTotalGaussianCurvesPerStimulusFrame && x < singleGaussianCurveLength && frame < nStimulusFrames)
	{
		int stimIdx = frame * (nStimulusRows * nStimulusCols) + x; // MIND that sitmIdx is not using the term "(y * nStimulusCols + x)" because each frame of the stimulus is considered to be a flattened 1d array
		int gaussIdx = (y * singleGaussianCurveLength + x);
		double stimulusCellValue = flattened_3d_stimulusData[stimIdx];
		double gaussianCurveValue = flattened_gaussian_curves_data[gaussIdx];
		double mulitplicationResult = stimulusCellValue * gaussianCurveValue;
		int resultIdx = (frame * singleGaussianCurveLength * nTotalGaussianCurvesPerStimulusFrame) + (y * singleGaussianCurveLength + x);

		// Print array values for debugging in a single statement
		//printf("Thread: (%d, %d, %d), x: %d, y: %d, frame: %d, stimulusCellValue[%d]: %.6f, gaussianCurveValue: %.6e, result: %.6e at [%d]\n", threadIdx.x, threadIdx.y, threadIdx.z, x, y, frame, stimIdx ,stimulusCellValue, gaussianCurveValue, mulitplicationResult, resultIdx);

		result_stimulus_multiplied_gaussian_curves[resultIdx] = mulitplicationResult;

	}
}

double* ModelSignals::applyStimulusMask(double* dev_computed_gaussian_curves)
{
	cudaError_t cudaStatus;

	// device
	double* dev_stimulus_data = 0;
	double* dev_intermediate_result_stimulus_multiplied_gaussian_curves = 0;

	// Memory allocations	
	cudaStatus = cudaMalloc((void**)&dev_stimulus_data, _nStimulusFrames * _nStimulusRows * _nStimulusCols * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_intermediate_result_stimulus_multiplied_gaussian_curves, _nTotalGaussianCurvesPerStimulusFrame * (_nStimulusRows * _nStimulusCols) * _nStimulusFrames * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Memory tranfers - host to device
	cudaStatus = cudaMemcpy(dev_stimulus_data, _flattenedHRFConvolvedStimulusData, _nStimulusFrames * _nStimulusRows * _nStimulusCols * sizeof(double), cudaMemcpyHostToDevice);

	// Launch kernel
	dim3 blockSizeModelSignalsKernel(32, 32, 1); // Equivalent to dim3 blockSizeModelSignalsKernel(Tx, Ty, Tz);
	int bx2 = (_singleFlattenedGaussianCurvelength + blockSizeModelSignalsKernel.x - 1) / blockSizeModelSignalsKernel.x;
	int by2 = (_nTotalGaussianCurvesPerStimulusFrame + blockSizeModelSignalsKernel.y - 1) / blockSizeModelSignalsKernel.y;
	int bz2 = (_nStimulusFrames + blockSizeModelSignalsKernel.z - 1) / blockSizeModelSignalsKernel.z;
	dim3 gridSizeModelSignalsKernel = dim3(bx2, by2, bz2);
	computeGaussianAndStimulusMultiplicationKernel << <gridSizeModelSignalsKernel, blockSizeModelSignalsKernel >> > (dev_intermediate_result_stimulus_multiplied_gaussian_curves
		, dev_stimulus_data
		, dev_computed_gaussian_curves
		, _singleFlattenedGaussianCurvelength
		, _nTotalGaussianCurvesPerStimulusFrame
		, _nStimulusFrames
		, _nStimulusRows
		, _nStimulusCols
		);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeGaussianAndStimulusMultiplicationKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//-----##########################----------------------Verification Part---------------######################--------//
	if (false)
	{
		// Copy the intermediate (Gaussian Curves * stimulus) results back to the host
		double* intermediate_result_stimulus_multiplied_gaussian_curves = new double[_nTotalGaussianCurvesPerStimulusFrame * (_nStimulusRows * _nStimulusCols) * (_nStimulusFrames)];
		cudaStatus = cudaMemcpy(intermediate_result_stimulus_multiplied_gaussian_curves, dev_intermediate_result_stimulus_multiplied_gaussian_curves, _nTotalGaussianCurvesPerStimulusFrame * (_nStimulusRows * _nStimulusCols) * (_nStimulusFrames) * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		delete[] intermediate_result_stimulus_multiplied_gaussian_curves;
	}



Error:

	return dev_intermediate_result_stimulus_multiplied_gaussian_curves;

}