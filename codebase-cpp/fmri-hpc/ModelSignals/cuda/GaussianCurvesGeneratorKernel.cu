#include "../header/ModelSignals.h"
#include "../../utils/arrays/Linspace.h"

#include <math.h>
#include <string>

// gpu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__constant__  double PI = 3.14159265358979323846;


// Gaussian Model
__device__ double gaussianModel(double x, double y, double x_mean, double y_mean, double sigma)
{
	double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);
	return exp(exponent);
}
typedef double(*FuncPtrGaussianModel)(double, double, double, double, double);
__device__ FuncPtrGaussianModel d_ptrGaussianModel = gaussianModel;


// Kernel to compute model curves based on the selected/parsed model (as function pointer)
__global__ void generateGaussianKernel(double (*model)(double, double, double, double, double)
	, double* result_gaussian_curves
	, double* modelspace_vf_points_x
	, double* modelspace_vf_points_y
	, double* stimulus_vf_points_x
	, double* stimulus_vf_points_y
	, int nGaussianRows
	, int nGaussianCols
	, int nStimulusRows
	, int nStimulusCols
	, double sigma)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < nGaussianRows && col < nGaussianCols)
	{
		double y_mean = modelspace_vf_points_y[row];
		double x_mean = modelspace_vf_points_x[col];
		int meanPairIdx = row * nGaussianCols + col;
		int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

		int gaussIdx = currentGaussianCurveStartIdx;
		//printf("Thread: (%d, %d), meanPairIdx: %d,  currentGaussianCurveStartIdx: %d\n", row, col, meanPairIdx, currentGaussianCurveStartIdx);
		for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
		{
			for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
			{
				double y = stimulus_vf_points_y[stim_vf_row];
				double x = stimulus_vf_points_x[stim_vf_col];
				result_gaussian_curves[gaussIdx] = model(x, y, x_mean, y_mean, sigma); //exp(exponent); // / (2 * PI * sigma * sigma);				
				//printf("Thread: (%d, %d), Ux: %f, Uy: %f, result_gaussian_curves[%d]: %.6e\n", col, row, x_mean, y_mean, gaussIdx, result_gaussian_curves[gaussIdx]);

				gaussIdx++;
			}
		}
	}
}

// Function to write Gaussian curves to text files
#include <fstream>
#include <string>
void writeGaussianCurvesToFile(const std::string& dataFolder
	, double* result_gaussian_curves
	, double* model_vf_points_x
	, double* model_vf_points_y
	, int nModelSpaceGridRows
	, int nModelSpaceGridCols
	, int nStimulusRows
	, int nStimulusCols)
{
	for (int row = 0; row < nModelSpaceGridRows; row++)
	{
		for (int col = 0; col < nModelSpaceGridCols; col++)
		{
			double mean_x = model_vf_points_x[col];
			double mean_y = model_vf_points_y[row];
			std::string filename = dataFolder + "gc-(" + std::to_string(mean_x) + ", " + std::to_string(mean_y) + ").txt";
			std::ofstream outfile(filename);

			// indexing
			int meanPairIdx = row * nModelSpaceGridCols + col;
			int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size
			int gaussIdx = currentGaussianCurveStartIdx;

			for (int i = 0; i < nStimulusRows; i++)
			{
				for (int j = 0; j < nStimulusCols; j++)
				{
					double value = result_gaussian_curves[gaussIdx];
					outfile << value << "\n";
					gaussIdx++;
				}
			}
			outfile.close();
		}
	}
}

double* ModelSignals::computeModelCurves()
{
	// For GAUSSIAN
	Linspace range_xx(-9.0, 9.0, _nStimulusCols);
	Linspace range_yy(-9.0, 9.0, _nStimulusRows);
	//Linspace stimulusVisualFieldPointsLinspace(-9.0, 9.0, _nStimulusRows);
	double* stimulus_vf_points_x = range_xx.GenerateLinspaceArr();
	double* stimulus_vf_points_y = range_yy.GenerateLinspaceArr();

	// For MEANS (i.e. muX and muY)
	//Linspace testspaceVisualFieldPointsLinspace(-9.0, 9.0, _nQuadrilateralSpaceGridRows);
	Linspace quad_space_xx(-9.0, 9.0, _nQuadrilateralSpaceGridCols);
	Linspace quad_space_yy(-9.0, 9.0, _nQuadrilateralSpaceGridRows);
	double* testspace_vf_points_x = quad_space_xx.GenerateLinspaceArr(); // values for muX
	double* testspace_vf_points_y = quad_space_yy.GenerateLinspaceArr(); // values for muY

	// device
	double* dev_stimulus_vf_points_x = 0;
	double* dev_stimulus_vf_points_y = 0;
	double* dev_testspace_vf_points_x = 0;
	double* dev_testspace_vf_points_y = 0;
	double* dev_result_gaussian_curves = 0;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Declare a function pointer and initialize it with the device function
	FuncPtrGaussianModel h_ptrGassuianModel;
	cudaStatus = cudaMemcpyFromSymbol(&h_ptrGassuianModel, d_ptrGaussianModel, sizeof(FuncPtrGaussianModel)); // We can specify different models (like Gaussian, Gamma, derivative of gaussian etc.

	// Memory allocations
	cudaStatus = cudaMalloc((void**)&dev_stimulus_vf_points_x, _nStimulusCols * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_stimulus_vf_points_y, _nStimulusRows * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_testspace_vf_points_x, _nQuadrilateralSpaceGridCols * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_testspace_vf_points_y, _nQuadrilateralSpaceGridRows * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_result_gaussian_curves, (_nStimulusRows * _nStimulusCols) * (_nQuadrilateralSpaceGridRows * _nQuadrilateralSpaceGridCols) * sizeof(double));

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Memory tranfers - host to device
	cudaStatus = cudaMemcpy(dev_stimulus_vf_points_x, stimulus_vf_points_x, _nStimulusCols * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_stimulus_vf_points_y, stimulus_vf_points_y, _nStimulusRows * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_testspace_vf_points_x, testspace_vf_points_x, _nQuadrilateralSpaceGridCols * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_testspace_vf_points_y, testspace_vf_points_y, _nQuadrilateralSpaceGridRows * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Kernel - Model Curves (such as gaussian curves) Computation
	int TX = 32;
	int TY = 32;
	dim3 blockSizeGaussianKernel(TX, TY); // Equivalent to dim3 blockSizeGaussianKernel(TX, TY, 1);
	int bx1 = (_nQuadrilateralSpaceGridCols + blockSizeGaussianKernel.x - 1) / blockSizeGaussianKernel.x;
	int by1 = (_nQuadrilateralSpaceGridRows + blockSizeGaussianKernel.y - 1) / blockSizeGaussianKernel.y;
	dim3 gridSizeGaussianKernel = dim3(bx1, by1);
	generateGaussianKernel << <gridSizeGaussianKernel, blockSizeGaussianKernel >> > (h_ptrGassuianModel
		, dev_result_gaussian_curves
		, dev_testspace_vf_points_x
		, dev_testspace_vf_points_y
		, dev_stimulus_vf_points_x
		, dev_stimulus_vf_points_y
		, _nQuadrilateralSpaceGridRows
		, _nQuadrilateralSpaceGridCols
		, _nStimulusRows
		, _nStimulusCols
		, _sigma);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "generateGaussianKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

	 //Copy the Gaussian Curves results back to the host
	if (false)
	{
		double* result_gaussian_curves_host = new double[_nQuadrilateralSpaceGridRows * _nQuadrilateralSpaceGridCols * _nStimulusRows * _nStimulusCols];
		cudaStatus = cudaMemcpy(result_gaussian_curves_host, dev_result_gaussian_curves, (_nQuadrilateralSpaceGridRows * _nQuadrilateralSpaceGridCols) * (_nStimulusRows * _nStimulusCols) * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Call the function to write Gaussian curves to text files
		writeGaussianCurvesToFile("D:/code/sid-git/fmri/local-extracted-datasets/gpu-tests/resulting-gaussian-curves/"
			, result_gaussian_curves_host
			, testspace_vf_points_x
			, testspace_vf_points_y
			, _nQuadrilateralSpaceGridRows
			, _nQuadrilateralSpaceGridCols
			, _nStimulusRows
			, _nStimulusCols
		);

		delete[] result_gaussian_curves_host;
	}


Error:
	cudaFree(dev_stimulus_vf_points_x);
	cudaFree(dev_stimulus_vf_points_y);
	cudaFree(dev_testspace_vf_points_x);
	cudaFree(dev_testspace_vf_points_y);

	return dev_result_gaussian_curves;
}


