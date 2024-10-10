#include "../header/ModelSignals.h"

#include <math.h>
#include <string>

// gpu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


__global__ void computeRowwiseSumKernel(double* result_sum
	, double* gaussian_curves_data
	, int nSingleFlattenedGaussianCurveLength
	, int nTotalGaussianCurvesPerStimulusFrame
	, int numStimulusFrames
)
{
	extern __shared__ double s_singleRowGaussData[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int frame = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= nSingleFlattenedGaussianCurveLength || y >= nTotalGaussianCurvesPerStimulusFrame || frame >= numStimulusFrames)
		return;

	//if (x == 0 && y == 0 && frame == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
	//	for (int i = 0; i < 18; i++)
	//	{
	//		printf("gaussian_curves_data[%d]: %.2f\n", i, gaussian_curves_data[i]);
	//	}

	int gaussDataIdx = (frame * nSingleFlattenedGaussianCurveLength * nTotalGaussianCurvesPerStimulusFrame) + (y * nSingleFlattenedGaussianCurveLength + x);

	// shared memory related ---- NOTE: I guess we can make the SHARED MEMORY ARRAY INDEX Computation single dimensonal (i.e. only x-direction)
	int s_x = threadIdx.x;
	int s_y = threadIdx.y;
	int s_frame = threadIdx.z;
	int s_w = blockDim.x;
	int s_h = blockDim.y;
	int s_d = blockDim.z; // depth
	int s_gaussDataIdx = (s_frame * s_w * s_h) + (s_y * s_w + s_x);

	s_singleRowGaussData[s_gaussDataIdx] = gaussian_curves_data[gaussDataIdx];
	//printf("s_singleRowGaussData[%d]: gaussian_curves_data[%d] = %.6f\n", s_gaussDataIdx, gaussDataIdx, gaussian_curves_data[gaussDataIdx]);

	__syncthreads();

	// sumup the row values gathered in the shared memory
	if (s_x == 0 && s_y == 0 && s_frame == 0)
	{
		for (int d = 0; d < s_d; d++)
		{
			for (int row = 0; row < s_h; row++)
			{
				// add the shared memory sum for this chunk of row Atomically to the actual row-wise sum for the result
				int res_x_idx = 0; // 0 because result is a colum vector // blockIdx.x * blockDim.x + col;
				int res_y_idx = blockIdx.y * blockDim.y + row;
				int res_z_idx = blockIdx.z * blockDim.z + d;

				if (res_y_idx >= nTotalGaussianCurvesPerStimulusFrame || res_z_idx >= numStimulusFrames)
					continue;

				double chunkRowSum = 0;
				for (int col = 0; col < s_w; col++)
				{
					if (col < nSingleFlattenedGaussianCurveLength)
						chunkRowSum = chunkRowSum + s_singleRowGaussData[(d * s_w * s_h) + (row * s_w) + col];
					//if (d == 0)
					//printf("%.2f, ", s_singleRowGaussData[row * s_w + col]);
				}
				//if(d == 0)
				//printf("chunkRowSum = %.2f \n", chunkRowSum);

				//int idx = res_z_idx * nTotalGaussianCurvesPerStimulusFrame + res_y_idx;
				 int idx = res_y_idx * numStimulusFrames + res_z_idx;
				//printf("[%d]: (res_z_idx * nTotalGaussianCurvesPerStimulusFrame + res_y_idx) = (%d * %d + %d) \n", idx, res_z_idx, nTotalGaussianCurvesPerStimulusFrame, res_y_idx);

				//if (d == 0)
					//printf("result_sum[%d] = %.2f + %.2f\n",idx, result_sum[idx], chunkRowSum);

				atomicAdd(&result_sum[idx], chunkRowSum);
				//printf("chunkRowSum = %0.2f , result_sum[%d] = %.6f\n", chunkRowSum, idx, result_sum[idx]);

				//printf("Thread(%d, %d, %d): _block_x:%d, _block_y: %d, _block_z: %d, res_y_idx: %d, res_z_idx: %d, result_idx: %d\n", threadIdx.x, threadIdx.y, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z, res_y_idx, res_z_idx, idx);
			}
		}
	}
}

double* ModelSignals::computeRowwiseSum(double* dev_stimulusMaskAppliedGaussianCurves)
{
	cudaError_t cudaStatus;
	// device	
	double* d_result_rows_summation = 0;

	// Memory allocations		
	cudaStatus = cudaMalloc((void**)&d_result_rows_summation, _nStimulusFrames * _nTotalGaussianCurvesPerStimulusFrame * sizeof(double));

	// Memory tranfers - host to device
	//---none required
	cudaMemset(d_result_rows_summation, 0, _nStimulusFrames * _nTotalGaussianCurvesPerStimulusFrame * sizeof(int)); // must initialize the result with zeros

	// Launch kernel
	int Tx = 32;
	int Ty = 32;
	int Tz = 1;
	dim3 blockSizeColumnSummationKernel(Tx, Ty, Tz); // Equivalent to dim3 blockSizeColumnSummationKernel(TX, TY, 1);
	int bx1 = (_singleFlattenedGaussianCurvelength + blockSizeColumnSummationKernel.x - 1) / blockSizeColumnSummationKernel.x;
	int by1 = (_nTotalGaussianCurvesPerStimulusFrame + blockSizeColumnSummationKernel.y - 1) / blockSizeColumnSummationKernel.y;
	int bz1 = (_nStimulusFrames + blockSizeColumnSummationKernel.z - 1) / blockSizeColumnSummationKernel.z;
	dim3 gridSizeColumnSummationKernel = dim3(bx1, by1, bz1);
	const size_t smemsize = (Tx * Ty * Tz) * sizeof(double); // THROW ERROR if the size of the shared memory is greater than 48 kb	
	computeRowwiseSumKernel << <gridSizeColumnSummationKernel, blockSizeColumnSummationKernel, smemsize >> >
		(d_result_rows_summation
			, dev_stimulusMaskAppliedGaussianCurves
			, _singleFlattenedGaussianCurvelength
			, _nTotalGaussianCurvesPerStimulusFrame
			, _nStimulusFrames
			);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeRowwiseSumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
		double* h_result_columns_summation = new double[_nTotalGaussianCurvesPerStimulusFrame * _nStimulusFrames];
		cudaStatus = cudaMemcpy(h_result_columns_summation, d_result_rows_summation, _nTotalGaussianCurvesPerStimulusFrame * _nStimulusFrames * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		for (int i = 0; i < 4; i++)
			printf("result[%d]: %.6f\n", i, h_result_columns_summation[i]);
	}

Error:

	return d_result_rows_summation;
}