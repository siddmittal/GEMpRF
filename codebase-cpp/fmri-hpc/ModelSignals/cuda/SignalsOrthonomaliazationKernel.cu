#include "../header/ModelSignals.h"
#include "../../utils/matrix/Matrix.h"

// gpu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>


// CUDA kernel for orthogonalization of S  ( e.g. 6x4) result
__global__ void computeOrthogonalizationMatrixKernel(
    double* R,
    double* result_O,
    int nRows_R,
    int nCols_R
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nRows_R && col < nCols_R)
    {
        double diff = (row == col) ? 1.0f : 0.0f; // IMPORTANT: Don'T forget to uncomment it
        for (int k = 0; k < nCols_R; k++)
        {
            diff -= R[row * nCols_R + k] * R[col * nCols_R + k];

            //if (row == 0 && col == 2)
            //    printf("(%.2f * %.2f) +  ", R[row * nCols_R + k], R[col * nCols_R + k]);
        }
        result_O[row * nRows_R + col] = diff;
        //if (row == 0 && col == 2)
        //    printf(" = %.20f \n\n", diff);
    }
}

// CUDA kernel for orthogonalization of S  ( e.g. 6x4) result
__global__ void orthogonalizeSignalsKernel(
    double* S, 
    double* O, 
    int nRows_R,
    int nCols_R,
    int nRows_S,
    double* result_s_prime
    )
{
    int s_row = blockIdx.y * blockDim.y + threadIdx.y;
    int s_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (s_row < nRows_S && s_col < nCols_R)
    {
        // Logic:
        // ...multiply a row of O with a col of S
        // ...find summation of the multiplcation
        // ...make summation the result of s_prime at (s_row, s_col)
        // Logic: each element in the res_row, res_col of result_s_prime is equal to the summation of multiplication of the elements of S in s_row with the elements of O in the row at index s_col
        int r_row = s_col;
        double summation = 0.0f;
        for (int r_col = 0; r_col < nCols_R; r_col++)
        {
            double o = O[s_col * nCols_R + r_col];
            double s = S[s_row * nCols_R + r_col];
            summation = summation + O[s_col * nCols_R + r_col] * S[s_row * nCols_R + r_col];

            //if (s_row == 0 && s_col == 1)
            //    printf("%.1f * %.1f + ", o, s);
        }
        result_s_prime[s_row * nCols_R + s_col] = summation;
        //if (s_row == 0 && s_col == 1)
        //    printf(" = %.1f\n", summation);
    }
}


double* ModelSignals::orthonormalizeSignals(double* d_S, double* R)
{
    cudaError_t cudaStatus;

    int nRows_R = _nStimulusFrames;
    int nCols_R = _nStimulusFrames;
    int nRows_S = _nStimulusFrames * _nTotalGaussianCurvesPerStimulusFrame;

    // device
    double* d_R, * d_O, * d_result;

    // Allocate memory on the GPU
    cudaStatus = cudaMalloc((void**)&d_R, nRows_R * nCols_R * sizeof(double));
    cudaStatus = cudaMalloc((void**)&d_O, nRows_R * nCols_R * sizeof(double));
    cudaStatus = cudaMalloc((void**)&d_result, nRows_S * nRows_R * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    // Copy data from CPU to GPU
    cudaStatus = cudaMemcpy(d_R, R, nRows_R * nCols_R * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }
    
    // kernels config
    int TX = 32;
    int TY = 32;
    dim3 blockSize(TX, TY);
    int bx = (nCols_R + blockSize.x - 1) / blockSize.x;

    // Compute orthogonalization Matrix "O"
    int by1 = (nRows_R + blockSize.y - 1) / blockSize.y;
    dim3 gridOrthogonalizationMatrixSize = dim3(bx, by1);
    computeOrthogonalizationMatrixKernel << <gridOrthogonalizationMatrixSize, blockSize >> > (d_R, d_O, nRows_R, nCols_R);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "computeOrthogonalizationMatrixKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error;
    }

    // Orthogonalize signals
    int by2 = (nRows_S + blockSize.y - 1) / blockSize.y;
    dim3 gridOrthogonalizationSize = dim3(bx, by2);
    orthogonalizeSignalsKernel<< <gridOrthogonalizationSize, blockSize >> > (d_S, d_O, nRows_R, nCols_R, nRows_S, d_result);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "orthogonalizeSignalsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:
    // Free GPU memory
    cudaFree(d_R);
    cudaFree(d_O);   

    return d_result;
}










