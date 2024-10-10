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
//__global__ void dummyTimecourse(double* result_sum
//	, double* gaussian_curves_data
//	, int nSingleFlattenedGaussianCurveLength
//	, int nTotalGaussianCurvesPerStimulusFrame
//	, int numStimulusFrames
//)
//{
//	extern __shared__ double s_singleRowGaussData[];
//
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	int frame = blockIdx.z * blockDim.z + threadIdx.z;
//
//	if (x >= nSingleFlattenedGaussianCurveLength || y >= nTotalGaussianCurvesPerStimulusFrame || frame >= numStimulusFrames)
//		return;
//
//	int gaussDataIdx = (frame * nSingleFlattenedGaussianCurveLength * nTotalGaussianCurvesPerStimulusFrame) + (y * nSingleFlattenedGaussianCurveLength + x);
//
//	// shared memory related ---- NOTE: I guess we can make the SHARED MEMORY ARRAY INDEX Computation single dimensonal (i.e. only x-direction)
//	int s_x = threadIdx.x;
//	int s_y = threadIdx.y;
//	int s_frame = threadIdx.z;
//	int s_w = blockDim.x;
//	int s_h = blockDim.y;
//	int s_d = blockDim.z; // depth
//	int s_gaussDataIdx = (s_frame * s_w * s_h) + (s_y * s_w + s_x);
//
//	s_singleRowGaussData[s_gaussDataIdx] = gaussian_curves_data[gaussDataIdx];
//	//printf("s_singleRowGaussData[%d]: gaussian_curves_data[%d] = %.6f\n", s_gaussDataIdx, gaussDataIdx, gaussian_curves_data[gaussDataIdx]);
//
//	__syncthreads();
//
//	// sumup the row values gathered in the shared memory
//	if (s_x == 0 && s_y == 0 && s_frame == 0)
//	{
//		for (int d = 0; d < s_d; d++)
//		{
//			for (int row = 0; row < s_h; row++)
//			{
//				// add the shared memory sum for this chunk of row Atomically to the actual row-wise sum for the result
//				int res_x_idx = 0; // 0 because result is a colum vector // blockIdx.x * blockDim.x + col;
//				int res_y_idx = blockIdx.y * blockDim.y + row;
//				int res_z_idx = blockIdx.z * blockDim.z + d;
//
//				if (res_y_idx >= nTotalGaussianCurvesPerStimulusFrame || res_z_idx >= numStimulusFrames)
//					continue;
//
//				double chunkRowSum = 0;
//				for (int col = 0; col < s_w; col++)
//				{
//					if(col < nSingleFlattenedGaussianCurveLength)
//						chunkRowSum = chunkRowSum + s_singleRowGaussData[(d * s_w * s_h) + (row * s_w) + col];
//					//if (d == 0)
//					//	printf("%.2f, ", s_singleRowGaussData[row * s_w + col]);
//				}
//				//if(d == 0)
//				//	printf("chunkRowSum = %.2f \n", chunkRowSum);
//
//				//int idx = res_z_idx * nTotalGaussianCurvesPerStimulusFrame + res_y_idx; 
//				 int idx = res_y_idx * numStimulusFrames + res_z_idx;
//				
//				//printf("[%d]: (res_z_idx * nTotalGaussianCurvesPerStimulusFrame + res_y_idx) = (%d * %d + %d) \n", idx, res_z_idx, nTotalGaussianCurvesPerStimulusFrame, res_y_idx);
//
//				//if (d == 0)
//				//	printf("result_sum[%d] = %.2f + %.2f\n",idx, result_sum[idx], chunkRowSum);
//
//				atomicAdd(&result_sum[idx], chunkRowSum);
//				//printf(" = %.6f\n", result_sum[idx]);
//
//				//printf("Thread(%d, %d, %d): _block_x:%d, _block_y: %d, _block_z: %d, res_y_idx: %d, res_z_idx: %d, result_idx: %d\n", threadIdx.x, threadIdx.y, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z, res_y_idx, res_z_idx, idx);
//			}
//		}
//	}
//}
//
//int main()
//{
//	cudaError_t cudaStatus;
//
//	const int nStimulusRows = 3;
//	const int nStimulusCols = 3;
//	const int nStimulsFrames = 2;
//	const int nQuadrilateralSpaceGridRows = 2; // could be test space or search space (or model signals space)
//	const int nQuadrilateralSpaceGridCols = 1;
//	const int singleFlattenedGaussianCurvelength = nStimulusRows * nStimulusCols;
//	const int nTotalGaussianCurvesPerStimulusFrame = nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols;
//
//	// stimulus of size 3 x 3 = 9
//	int nGaussianCurvesPerStimulusFrame = 2;
//	int nSingleFlattenedGaussianCurveLength = 9;
//	int nStimulusFrames = 2;
//	double h_dummyData[36] = {
//		 1, 2, 3, 4, 5, 6, 7, 8, 10, // 9 values for frame-1 - curve-1
//		 1, 2, 3, 4, 5, 6, 7, 8, 12, // 9 values for frame-1 - curve-2
//		 1, 2, 3, 4, 5, 6, 7, 8, 14, // 9 values for frame-2 - curve-1
//		 1, 2, 3, 4, 5, 6, 7, 8, 16 // 9 values for frame-2 - curve-2
//	};
//
//	//double h_dummyData[18] = {
//	//		 1, 2, 3, 4, 5, 6, 7, 8, 10, // 9 values for frame-1 - curve-1
//	//		1, 2, 3, 4, 5, 6, 7, 8, 12,		// 9 values for frame-1 - curve-2	 
//	//};
//
//	double h_dummyResult[4] = {
//		0,
//		0,
//		0,
//		0
//	};
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	// device
//	double* d_stimulusMultipliedGaussianCurvesData = 0;
//	double* d_result_rows_summation = 0;
//
//	cudaStatus = cudaMalloc((void**)&d_stimulusMultipliedGaussianCurvesData, 36 * sizeof(double));
//	cudaStatus = cudaMemcpy(d_stimulusMultipliedGaussianCurvesData, h_dummyData, 36 * sizeof(double), cudaMemcpyHostToDevice);
//	cudaStatus = cudaMalloc((void**)&d_result_rows_summation, 4 * sizeof(double));
//	cudaStatus = cudaMemcpy(d_result_rows_summation, h_dummyResult, 4 * sizeof(double), cudaMemcpyHostToDevice);
//
//	int Tx = 3;
//	int Ty = 3;
//	int Tz = 1;
//	dim3 blockSizeColumnSummationKernel(Tx, Ty, Tz); // Equivalent to dim3 blockSizeColumnSummationKernel(TX, TY, 1);
//	int bx1 = (nSingleFlattenedGaussianCurveLength + blockSizeColumnSummationKernel.x - 1) / blockSizeColumnSummationKernel.x;
//	int by1 = (nGaussianCurvesPerStimulusFrame + blockSizeColumnSummationKernel.y - 1) / blockSizeColumnSummationKernel.y;
//	int bz1 = (nStimulusFrames + blockSizeColumnSummationKernel.z - 1) / blockSizeColumnSummationKernel.z;
//	dim3 gridSizeColumnSummationKernel = dim3(bx1, by1, bz1);
//	const size_t smemsize = (Tx * Ty * Tz) *  sizeof(double); // THROW ERROR if the size of the shared memory is greater than 48 kb
//	dummyTimecourse << <gridSizeColumnSummationKernel, blockSizeColumnSummationKernel, smemsize >> > (d_result_rows_summation
//		, d_stimulusMultipliedGaussianCurvesData // stimulus multipled gaussian curves data (all curves multiplied with stimuls at all time frames)
//		, nSingleFlattenedGaussianCurveLength
//		, nGaussianCurvesPerStimulusFrame
//		, nStimulusFrames
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
//	double* h_result_columns_summation = new double[nGaussianCurvesPerStimulusFrame * nStimulusFrames];
//	cudaStatus = cudaMemcpy(h_result_columns_summation, d_result_rows_summation, nGaussianCurvesPerStimulusFrame * nStimulusFrames * sizeof(double), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	for (int i = 0; i < 4; i++)
//		printf("result[%d]: %.6f\n", i, h_result_columns_summation[i]);
//
//
//Error:
//
//	delete[] h_result_columns_summation;
//
//	cudaFree(d_stimulusMultipliedGaussianCurvesData);
//	cudaFree(d_result_rows_summation);
//	return cudaStatus;
//}
