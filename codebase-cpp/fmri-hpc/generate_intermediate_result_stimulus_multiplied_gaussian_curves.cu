//#include <iostream>
//#include <fstream>
//#include <string>
//#include <math.h>
//
//#include <stdio.h>
//#include <stdlib.h>
//
//// gpu
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//
//#include "utils/arrays/Linspace.h"
//#include "utils/arrays/meshgrid.h"
//
//__constant__  double PI = 3.14159265358979323846;
//
//
//__global__ void generateGaussian(double* result_gaussian_curves
//	, double* modelspace_vf_points_x
//	, double* modelspace_vf_points_y
//	, double* stimulus_vf_points_x
//	, double* stimulus_vf_points_y
//	, int nGaussianRows
//	, int nGaussianCols
//	, int nStimulusRows
//	, int nStimulusCols)
//{
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (row < nGaussianRows && col < nGaussianCols)
//	{
//		double y_mean = modelspace_vf_points_y[row];
//		double x_mean = modelspace_vf_points_x[col];
//		double sigma = 2.0;  // Standard deviation
//		int meanPairIdx = row * nGaussianCols + col;
//		int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size
//
//		int gaussIdx = currentGaussianCurveStartIdx;
//		//printf("Thread: (%d, %d), meanPairIdx: %d,  currentGaussianCurveStartIdx: %d\n", row, col, meanPairIdx, currentGaussianCurveStartIdx);
//		for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
//		{
//			for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
//			{				
//				double y = stimulus_vf_points_y[stim_vf_row];
//				double x = stimulus_vf_points_x[stim_vf_col];
//				double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);
//				result_gaussian_curves[gaussIdx] = exp(exponent); // / (2 * PI * sigma * sigma);				
//				//printf("Thread: (%d, %d), Ux: %f, Uy: %f, result_gaussian_curves[%d]: %.6e\n", col, row, x_mean, y_mean, gaussIdx, result_gaussian_curves[gaussIdx]);
//
//				gaussIdx++;
//			}
//		}
//	}
//}
//
//
//__global__ void computeGaussianAndStimulusMultiplicationKernel(double* result_stimulus_multiplied_gaussian_curves
//	, double* flattened_3d_stimulusData
//	, double* flattened_gaussian_curves_data
//	, int singleGaussianCurveLength
//	, int nTotalGaussianCurvesPerStimulusFrame
//	, int nStimulusFrames
//	, int nStimulusRows
//	, int nStimulusCols)
//{
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int frame = blockIdx.z * blockDim.z + threadIdx.z; 
//
//	if (y < nTotalGaussianCurvesPerStimulusFrame && x < singleGaussianCurveLength && frame < nStimulusFrames)
//	{
//		int stimIdx = frame * (nStimulusRows * nStimulusCols) +  x; // MIND that sitmIdx is not using the term "(y * nStimulusCols + x)" because each frame of the stimulus is considered to be a flattened 1d array
//		int gaussIdx = (y * singleGaussianCurveLength + x);
//		double stimulusCellValue = flattened_3d_stimulusData[stimIdx];
//		double gaussianCurveValue = flattened_gaussian_curves_data[gaussIdx];
//		double mulitplicationResult = stimulusCellValue * gaussianCurveValue;
//		int resultIdx = (frame * singleGaussianCurveLength * nTotalGaussianCurvesPerStimulusFrame) + (y * singleGaussianCurveLength + x);
//
//		// Print array values for debugging in a single statement
//		//printf("Thread: (%d, %d, %d), x: %d, y: %d, frame: %d, stimulusCellValue[%d]: %.6f, gaussianCurveValue: %.6e, result: %.6e at [%d]\n", threadIdx.x, threadIdx.y, threadIdx.z, x, y, frame, stimIdx ,stimulusCellValue, gaussianCurveValue, mulitplicationResult, resultIdx);
//
//		result_stimulus_multiplied_gaussian_curves[resultIdx] = mulitplicationResult;
//
//	}
//}
//
//// Function to write Gaussian curves to text files
//void writeGaussianCurvesToFile(const std::string& dataFolder
//	, double* result_gaussian_curves
//	, double* model_vf_points_x
//	, double* model_vf_points_y
//	, int nModelSpaceGridRows
//	, int nModelSpaceGridCols
//	, int nStimulusRows
//	, int nStimulusCols)
//{
//	for (int row = 0; row < nModelSpaceGridRows; row++)
//	{
//		for (int col = 0; col < nModelSpaceGridCols; col++)
//		{
//			double mean_x = model_vf_points_x[col];
//			double mean_y = model_vf_points_y[row];
//			std::string filename = dataFolder + "gc-(" + std::to_string(mean_x) + ", " + std::to_string(mean_y) + ").txt";
//			std::ofstream outfile(filename);
//
//			// indexing
//			int meanPairIdx = row * nModelSpaceGridCols + col;
//			int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size
//			int gaussIdx = currentGaussianCurveStartIdx;
//
//			for (int i = 0; i < nStimulusRows; i++) 
//			{
//				for (int j = 0; j < nStimulusCols; j++)
//				{
//					double value = result_gaussian_curves[gaussIdx]; 
//					outfile << value << "\n";
//					gaussIdx++;
//				}
//			}
//			outfile.close();
//		}
//	}
//}
//
//
//int main_working()
//{
//	const int nStimulusRows = 5;
//	const int nStimulusCols = 5;
//	const int nStimulsFrames = 2;
//	const int nQuadrilateralSpaceGridRows = 3; // could be test space or search space (or model signals space)
//	const int nQuadrilateralSpaceGridCols = 3;
//	const int singleFlattenedGaussianCurvelength = nStimulusRows * nStimulusCols;
//	const int nTotalGaussianCurves = nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols;
//
//	// Dummy Stimulus data
//	/*double dummyHRFConvolvedStimulusData[nStimulsFrames * nStimulusRows * nStimulusCols] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };*/
//	//double dummyHRFConvolvedStimulusData[nStimulsFrames * nStimulusRows * nStimulusCols] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//
//	// stimulus of size 3 x 3 = 9
//	//double dummyHRFConvolvedStimulusData[nStimulsFrames * nStimulusRows * nStimulusCols] = {
//	//	 1, 2, 3, 4, 5, 6, 7, 8, 9, // 9 values for frame-1
//	//	 1, 2, 3, 4, 5, 6, 7, 8, 9, // 9 values for frame-2
//	//};
//
//	// stimulus of size 5 x 5 = 25
//	double dummyHRFConvolvedStimulusData[nStimulsFrames * nStimulusRows * nStimulusCols] = {
//		 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,1, 2, 3, 4, 5,1, 2, 3, 4, 5,1, 2, 3, 4, 5, // 25 values for frame-1
//		 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,1, 2, 3, 4, 5,1, 2, 3, 4, 5,1, 2, 3, 4, 5  // 25 values for frame-2
//	};
//
//	// For GAUSSIAN
//	Linspace stimulusVisualFieldPointsLinspace(-9.0, 9.0, nStimulusRows);
//	double* stimulus_vf_points_x = stimulusVisualFieldPointsLinspace.GenerateLinspaceArr();
//	double* stimulus_vf_points_y = stimulusVisualFieldPointsLinspace.GenerateLinspaceArr();
//
//	// For MEANS (i.e. muX and muY)
//	Linspace testspaceVisualFieldPointsLinspace(-9.0, 9.0, nQuadrilateralSpaceGridRows);
//	double* testspace_vf_points_x = testspaceVisualFieldPointsLinspace.GenerateLinspaceArr(); // values for muX
//	double* testspace_vf_points_y = testspaceVisualFieldPointsLinspace.GenerateLinspaceArr(); // values for muY
//
//	// device
//	double* dev_stimulus_data = 0;
//	double* dev_stimulus_vf_points_x = 0;
//	double* dev_stimulus_vf_points_y = 0;
//	double* dev_testspace_vf_points_x = 0;
//	double* dev_testspace_vf_points_y = 0;
//	double* dev_result_gaussian_curves = 0;
//	double* dev_intermediate_result_stimulus_multiplied_gaussian_curves = 0;
//
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	// Memory allocations
//	cudaStatus = cudaMalloc((void**)&dev_stimulus_data, nStimulsFrames * nStimulusRows * nStimulusCols * sizeof(double));
//	cudaStatus = cudaMalloc((void**)&dev_stimulus_vf_points_x, nStimulusCols * sizeof(double));
//	cudaStatus = cudaMalloc((void**)&dev_stimulus_vf_points_y, nStimulusRows * sizeof(double));
//	cudaStatus = cudaMalloc((void**)&dev_testspace_vf_points_x, nQuadrilateralSpaceGridCols * sizeof(double));
//	cudaStatus = cudaMalloc((void**)&dev_testspace_vf_points_y, nQuadrilateralSpaceGridRows * sizeof(double));
//	cudaStatus = cudaMalloc((void**)&dev_result_gaussian_curves, (nStimulusRows * nStimulusCols) * (nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols) * sizeof(double));
//	cudaStatus = cudaMalloc((void**)&dev_intermediate_result_stimulus_multiplied_gaussian_curves, nTotalGaussianCurves * (nStimulusRows * nStimulusCols) * nStimulsFrames * sizeof(double));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Memory tranfers - host to device
//	cudaStatus = cudaMemcpy(dev_stimulus_data, dummyHRFConvolvedStimulusData, nStimulsFrames * nStimulusRows * nStimulusCols * sizeof(double), cudaMemcpyHostToDevice);
//	cudaStatus = cudaMemcpy(dev_stimulus_vf_points_x, stimulus_vf_points_x, nStimulusCols * sizeof(double), cudaMemcpyHostToDevice);
//	cudaStatus = cudaMemcpy(dev_stimulus_vf_points_y, stimulus_vf_points_y, nStimulusRows * sizeof(double), cudaMemcpyHostToDevice);
//	cudaStatus = cudaMemcpy(dev_testspace_vf_points_x, testspace_vf_points_x, nQuadrilateralSpaceGridCols * sizeof(double), cudaMemcpyHostToDevice);
//	cudaStatus = cudaMemcpy(dev_testspace_vf_points_y, testspace_vf_points_y, nQuadrilateralSpaceGridRows * sizeof(double), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Kernel - Gaussian Curves Computation
//	int TX = 3;
//	int TY = 3;
//	dim3 blockSizeGaussianKernel(TX, TY); // Equivalent to dim3 blockSizeGaussianKernel(TX, TY, 1);
//	int bx1 = (nQuadrilateralSpaceGridCols + blockSizeGaussianKernel.x - 1) / blockSizeGaussianKernel.x;
//	int by1 = (nQuadrilateralSpaceGridRows + blockSizeGaussianKernel.y - 1) / blockSizeGaussianKernel.y;
//	dim3 gridSizeGaussianKernel = dim3(bx1, by1);
//	generateGaussian << <gridSizeGaussianKernel, blockSizeGaussianKernel >> > (dev_result_gaussian_curves
//		, dev_testspace_vf_points_x
//		, dev_testspace_vf_points_y
//		, dev_stimulus_vf_points_x
//		, dev_stimulus_vf_points_y
//		, nQuadrilateralSpaceGridRows
//		, nQuadrilateralSpaceGridCols
//		, nStimulusRows
//		, nStimulusCols);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Kernel - Model Signals Computation
//	dim3 blockSizeModelSignalsKernel(16, 16, nStimulsFrames); // Equivalent to dim3 blockSizeModelSignalsKernel(Tx, Ty, Tz);
//	int bx2 = (singleFlattenedGaussianCurvelength + blockSizeModelSignalsKernel.x - 1) / blockSizeModelSignalsKernel.x;
//	int by2 = (nTotalGaussianCurves + blockSizeModelSignalsKernel.y - 1) / blockSizeModelSignalsKernel.y;
//	int bz2 = (nStimulsFrames + blockSizeModelSignalsKernel.z - 1) / blockSizeModelSignalsKernel.z;
//	dim3 gridSizeModelSignalsKernel = dim3(bx2, by2, bz2);
//	computeGaussianAndStimulusMultiplicationKernel << <gridSizeModelSignalsKernel, blockSizeModelSignalsKernel >> > (dev_intermediate_result_stimulus_multiplied_gaussian_curves
//		, dev_stimulus_data
//		, dev_result_gaussian_curves
//		, singleFlattenedGaussianCurvelength
//		, nTotalGaussianCurves
//		, nStimulsFrames
//		, nStimulusRows
//		, nStimulusCols
//		);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	//-----##########################----------------------Verification Part---------------######################--------//
//
//	// Copy the Gaussian Curves results back to the host
//	double* result_gaussian_curves_host = new double[nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols * nStimulusRows * nStimulusCols];
//	cudaStatus = cudaMemcpy(result_gaussian_curves_host, dev_result_gaussian_curves, (nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols) * (nStimulusRows * nStimulusCols) * sizeof(double), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Copy the intermediate (Gaussian Curves * stimulus) results back to the host
//	double* intermediate_result_stimulus_multiplied_gaussian_curves = new double[nTotalGaussianCurves * (nStimulusRows * nStimulusCols) * (nStimulsFrames)];
//	cudaStatus = cudaMemcpy(intermediate_result_stimulus_multiplied_gaussian_curves, dev_intermediate_result_stimulus_multiplied_gaussian_curves, nTotalGaussianCurves *(nStimulusRows * nStimulusCols) * (nStimulsFrames) * sizeof(double), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Print the copied back results (intermediate_result_stimulus_multiplied_gaussian_curves) for debugging
//	//for (int idx = 0; idx < nTotalGaussianCurves * nStimulsFrames * nStimulusRows * nStimulusCols; idx++)
//	//{
//	//	double value = intermediate_result_stimulus_multiplied_gaussian_curves[idx];
//	//	printf("Result[%d] = %.6e\n", idx, value);
//	//}
//
//	// Call the function to write Gaussian curves to text files
//	//writeGaussianCurvesToFile("D:/code/sid-git/fmri/local-extracted-datasets/gpu-tests/resulting-gaussian-curves/"
//	//	, result_gaussian_curves_host
//	//	, testspace_vf_points_x
//	//	, testspace_vf_points_y
//	//	, nQuadrilateralSpaceGridRows
//	//	, nQuadrilateralSpaceGridCols
//	//	, nStimulusRows
//	//	, nStimulusCols
//	//);
//
//
//Error:
//	cudaFree(dev_stimulus_data);
//	cudaFree(dev_stimulus_vf_points_x);
//	cudaFree(dev_stimulus_vf_points_y);
//	cudaFree(dev_testspace_vf_points_x);
//	cudaFree(dev_testspace_vf_points_y);
//	cudaFree(dev_result_gaussian_curves);
//	cudaFree(dev_intermediate_result_stimulus_multiplied_gaussian_curves);
//
//	//Cleanup
//	delete[] result_gaussian_curves_host;
//	delete[] intermediate_result_stimulus_multiplied_gaussian_curves;
//
//	return cudaStatus;
//}
