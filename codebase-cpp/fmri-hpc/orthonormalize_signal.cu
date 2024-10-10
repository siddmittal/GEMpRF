#include <stdio.h>
#include <Eigen/Dense>
#include "utils/matrix/Matrix.h"

// gpu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//
//#define nRows_R 4  // Number of rows in R
//#define nCols_R 4  // Number of columns in R
//#define nCols_S 6  // Number of columns in S
//
//// CUDA kernel for matrix multiplication
//// Each Row in "s" represents a signal signal
//__global__ void matrixMul(double* R, double* S, double* result) 
//{
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    double sum = 0.0;
//
//    if (row < nRows_R && col < nCols_S) 
//    {
//        for (int k = 0; k < nCols_R; k++) 
//        {
//            sum += R[row * nCols_R + k] * S[k * nCols_S + col];
//        }
//        result[row * nCols_S + col] = sum;
//    }
//}
//
//int main() {
//    double R[nRows_R * nCols_R] = { // 4x4 matrix
//        1, 1, 1, 1,
//        2, 2, 3, 4, 
//        1 , 1, 2, 2, 
//        0, 0, 1, 1
//    };
//    double S[nCols_R * nCols_S] = { // 4x6 matrix
//        1, 1, 1, 1, 1, 1, 
//        2, 2, 3, 4, 1, 1, 
//        1, 1, 1, 1, 1, 1,
//        1, 1, 1, 1, 1, 1
//    };
//    double result[nRows_R * nCols_S]; // Result matrix
//
//
//
//
//    // Initialize matrices R and S with your data
//
//    double* d_R, * d_S, * d_result;
//
//    // Allocate memory on the GPU
//    cudaMalloc((void**)&d_R, nRows_R * nCols_R * sizeof(double));
//    cudaMalloc((void**)&d_S, nCols_R * nCols_S * sizeof(double));
//    cudaMalloc((void**)&d_result, nRows_R * nCols_S * sizeof(double));
//
//    // Copy data from CPU to GPU
//    cudaMemcpy(d_R, R, nRows_R * nCols_R * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_S, S, nCols_R * nCols_S * sizeof(double), cudaMemcpyHostToDevice);
//
//    // Define thread and block dimensions
//    int TX = 3;
//	int TY = 3;
//	dim3 blockSizeGaussianKernel(TX, TY); // Equivalent to dim3 blockSizeGaussianKernel(TX, TY, 1);
//	int bx1 = (nCols_S + blockSizeGaussianKernel.x - 1) / blockSizeGaussianKernel.x;
//	int by1 = (nRows_R + blockSizeGaussianKernel.y - 1) / blockSizeGaussianKernel.y;
//	dim3 gridSizeGaussianKernel = dim3(bx1, by1);
//    dim3 threadsPerBlock(16, 16);
//    dim3 numBlocks((nCols_S + threadsPerBlock.x - 1) / threadsPerBlock.x, (nRows_R + threadsPerBlock.y - 1) / threadsPerBlock.y);
//
//    // Launch the kernel
//    matrixMul << <numBlocks, threadsPerBlock >> > (d_R, d_S, d_result);
//
//    // Copy the result back to the CPU
//    cudaMemcpy(result, d_result, nRows_R * nCols_S * sizeof(double), cudaMemcpyDeviceToHost);
//
//    // Verify the result
//    // You can compare 'result' with the CPU matrix multiplication here
//    Matrix resultMat(nRows_R, nCols_S, result);
//    resultMat.printMatrix("s*");
//
//    // Free GPU memory
//    cudaFree(d_R);
//    cudaFree(d_S);
//    cudaFree(d_result);
//
//    return 0;
//}


#include <stdio.h>

#define nRows_R 4  // Number of rows in R
#define nCols_R 4  // Number of columns in R
#define nRows_S 6  // Number of rows in transposed S (which is equivalent to the columns in original S)

// CUDA kernel for matrix multiplication with transposed S and a 6x4 result
__global__ void matrixMul(double* O, double* S, double* result) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if (row < nRows_S && col < nRows_R) 
    { // Swap row and col for nRows_S and nRows_R
        for (int k = 0; k < nCols_R; k++) 
        {
            sum += S[row * nCols_R + k] * O[col * nCols_R + k]; // Adjusted indexing for transposed S
        }
        result[row * nRows_R + col] = sum; // Swap row and col for result
    }
}

//// CUDA kernel for orthogonalization of S  ( e.g. 6x4) result
//__global__ void orthogonalizationKernel_MERGED(double* R, double* S, double* result)
//{
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    double sum = 0.0;
//
//    if (row < nRows_S && col < nRows_R) 
//    { // Swap row and col for P and N
//        for (int k = 0; k < nRows_S; k++) 
//        {
//            // Calculate (np.eye(N) - R @ R.T) on the fly within the kernel
//            double diff = (row == k && col == k) ? 1.0f : 0.0f;
//            for (int i = 0; i < nCols_R; i++) 
//            {
//                diff -= R[row * nCols_R + i] * R[k * nCols_R + i];
//            }
//            sum += diff; //* S[k * nCols_R + col]; // Adjusted indexing for (np.eye(N) - R @ R.T) and S
//        }
//        result[row * nRows_S + col] = sum; // Swap row and col for result
//    }
//}

// CUDA kernel for orthogonalization of S  ( e.g. 6x4) result
__global__ void computeOrthogonalizationMatrix(double* R, double* result_O)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nRows_R && col < nCols_R) 
    {
        double diff = (row == col) ? 1.0f : 0.0f;
        for (int k = 0; k < nCols_R; k++)
        {
            diff -= R[row * nCols_R + k] * R[col * nCols_R + k];
        }
        result_O[row * nRows_R + col] = diff;
    }
}

// CUDA kernel for orthogonalization of S  ( e.g. 6x4) result
__global__ void orthogonalizeSignals(double* S, double* O, double* result_s_prime)
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

            if(s_row == 0 && s_col == 1)
                printf("%.1f * %.1f + ", o, s);
        }
        result_s_prime[s_row * nCols_R + s_col] = summation;
        if (s_row == 0 && s_col == 1)
            printf(" = %.1f\n", summation);
    }
}

int main_working_2() 
{    
    double result[nRows_S * nCols_R]; // Result matrix, 6x4

    // Initialize matrices R and S with your data
	double R[nRows_R * nCols_R] = { // 4x4 matrix
	1, 1, 1, 1,
	2, 2, 3, 4,
	1 , 1, 2, 2,
	0, 0, 1, 1
	};
	double S[nCols_R * nRows_S] = { // 6 x 4  matrix
       1, 2, 1, 1,
       1, 2, 1, 1,
       1, 3, 1, 1,
       1, 4, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1
	};

    double* d_R, *d_O, * d_S, * d_result;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_R, nRows_R * nCols_R * sizeof(double));
    cudaMalloc((void**)&d_O, nRows_R * nCols_R * sizeof(double));
    cudaMalloc((void**)&d_S, nRows_S * nCols_R * sizeof(double)); // Adjusted size for transposed S
    cudaMalloc((void**)&d_result, nRows_S * nRows_R * sizeof(double)); // Adjusted size for 6x4 result

    // Copy data from CPU to GPU
    cudaMemcpy(d_R, R, nRows_R * nCols_R * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, S, nRows_S * nCols_R * sizeof(double), cudaMemcpyHostToDevice); // Adjusted size for transposed S

    // Define thread and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nRows_R + threadsPerBlock.x - 1) / threadsPerBlock.x, (nRows_S + threadsPerBlock.y - 1) / threadsPerBlock.y); // Swap N and P

    // Launch the kernel
    computeOrthogonalizationMatrix << <numBlocks, threadsPerBlock >> > (d_R, d_O);
    cudaDeviceSynchronize();
    cudaMemcpy(R, d_O, nRows_R * nCols_R * sizeof(double), cudaMemcpyDeviceToHost); // Adjusted size for 6x4 result
    cudaDeviceSynchronize();
    Matrix oMat(nRows_R, nCols_R, R);
    oMat.printMatrix("O");

    // Verify S
    cudaMemcpy(S, d_S, nRows_S * nCols_R * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();    
    Matrix sMat(nRows_S, nCols_R, S);
    sMat.printMatrix("S");


    // orthogonalize Signals
	int TX = 3;
	int TY = 3;
	dim3 blockSize(TX, TY); // Equivalent to dim3 blockSize(TX, TY, 1);
	int bx = (nCols_R + blockSize.x - 1) / blockSize.x;
	int by = (nRows_S + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(bx, by);
    orthogonalizeSignals << <gridSize, blockSize >> > (d_S, d_O, d_result);
    //matrixMul << <numBlocks, threadsPerBlock >> > (d_R, d_S, d_result);
    cudaDeviceSynchronize();

    // Copy the result back to the CPU
    cudaMemcpy(result, d_result, nRows_S * nCols_R * sizeof(double), cudaMemcpyDeviceToHost); // Adjusted size for 6x4 result
    cudaDeviceSynchronize();

    // Verify the result
    // You can compare 'result' with the CPU matrix multiplication here
	Matrix resultMat(nRows_S, nCols_R, result);
	resultMat.printMatrix("s*");

    // Free GPU memory
    cudaFree(d_R);
    cudaFree(d_S);
    cudaFree(d_result);

    return 0;
}
