#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

// gpu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "utils/arrays/Linspace.h"
#include "utils/arrays/meshgrid.h"

__constant__  float PI = 3.14159265358979323846;

__global__ void generateGaussian(float* result_gaussian_curves
    , float* modelspace_vf_points_x
    , float* modelspace_vf_points_y
    , float* stimulus_vf_points_x
    , float* stimulus_vf_points_y
    , int nGaussianRows
    , int nGaussianCols
    , int nStimulusRows
    , int nStimulusCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nGaussianRows && col < nGaussianCols)
    {
        float y_mean = modelspace_vf_points_y[row];
        float x_mean = modelspace_vf_points_x[col];
        float sigma = 2.0;  // Standard deviation
        int meanPairIdx = row * nGaussianCols + col;
        int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

        int gaussIdx = currentGaussianCurveStartIdx;
        //printf("Thread: (%d, %d), meanPairIdx: %d,  currentGaussianCurveStartIdx: %d\n", row, col, meanPairIdx, currentGaussianCurveStartIdx);
        for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
        {
            for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
            {
                float y = stimulus_vf_points_y[stim_vf_row];
                float x = stimulus_vf_points_x[stim_vf_col];
                float exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);
                result_gaussian_curves[gaussIdx] = exp(exponent); // / (2 * PI * sigma * sigma);				
                //printf("Thread: (%d, %d), Ux: %f, Uy: %f, result_gaussian_curves[%d]: %.6e\n", col, row, x_mean, y_mean, gaussIdx, result_gaussian_curves[gaussIdx]);

                gaussIdx++;
            }
        }
    }
}

// Function to write Gaussian curves to text files
void writeGaussianCurvesToFile(const std::string& dataFolder
    , float* result_gaussian_curves
    , float* model_vf_points_x
    , float* model_vf_points_y
    , int nModelSpaceGridRows
    , int nModelSpaceGridCols
    , int nStimulusRows
    , int nStimulusCols)
{
    for (int row = 0; row < nModelSpaceGridRows; row++)
    {
        for (int col = 0; col < nModelSpaceGridCols; col++)
        {
            float mean_x = model_vf_points_x[col];
            float mean_y = model_vf_points_y[row];
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
                    float value = result_gaussian_curves[gaussIdx];
                    outfile << value << "\n";
                    gaussIdx++;
                }
            }
            outfile.close();
        }
    }
}


int main()
{
    const int nStimulusRows = 5;
    const int nStimulusCols = 5;    
    const int nStimulsFrames = 2;
    const int nQuadrilateralSpaceGridRows = 3; // could be test space or search space (or model signals space)
    const int nQuadrilateralSpaceGridCols = 3;

    // For GAUSSIAN
    Linspace stimulusVisualFieldPointsLinspace(-9.0, 9.0, nStimulusRows);
    float* stimulus_vf_points_x = stimulusVisualFieldPointsLinspace.GenerateLinspaceArr();
    float* stimulus_vf_points_y = stimulusVisualFieldPointsLinspace.GenerateLinspaceArr();

    // For MEANS (i.e. muX and muY)
    Linspace testspaceVisualFieldPointsLinspace(-9.0, 9.0, nQuadrilateralSpaceGridRows);
    float* testspace_vf_points_x = testspaceVisualFieldPointsLinspace.GenerateLinspaceArr(); // values for muX
    float* testspace_vf_points_y = testspaceVisualFieldPointsLinspace.GenerateLinspaceArr(); // values for muY

    // device
    float* dev_stimulus_vf_points_x = 0;
    float* dev_stimulus_vf_points_y = 0;
    float* dev_testspace_vf_points_x = 0;
    float* dev_testspace_vf_points_y = 0;
    float* dev_result_gaussian_curves = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Memory allocations
    cudaStatus = cudaMalloc((void**)&dev_stimulus_vf_points_x, nStimulusCols * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_stimulus_vf_points_y, nStimulusRows * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_testspace_vf_points_x, nQuadrilateralSpaceGridCols * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_testspace_vf_points_y, nQuadrilateralSpaceGridRows * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_result_gaussian_curves, (nStimulusRows * nStimulusCols)*(nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Memory tranfers - host to device
    cudaStatus = cudaMemcpy(dev_stimulus_vf_points_x, stimulus_vf_points_x, nStimulusCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_stimulus_vf_points_y, stimulus_vf_points_y, nStimulusRows * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_testspace_vf_points_x, testspace_vf_points_x, nQuadrilateralSpaceGridCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_testspace_vf_points_y, testspace_vf_points_y, nQuadrilateralSpaceGridRows * sizeof(float), cudaMemcpyHostToDevice);   
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Kernel
    int TX = 3;
    int TY = 3;
    dim3 blockSize(TX, TY); // Equivalent to dim3 blockSize(TX, TY, 1);
    int bx = (nQuadrilateralSpaceGridCols + blockSize.x - 1) / blockSize.x;
    int by = (nQuadrilateralSpaceGridRows + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);
    generateGaussian <<<gridSize, blockSize>>> (dev_result_gaussian_curves
        , dev_testspace_vf_points_x
        , dev_testspace_vf_points_y
        , dev_stimulus_vf_points_x
        , dev_stimulus_vf_points_y
        , nQuadrilateralSpaceGridRows
        , nQuadrilateralSpaceGridCols
        , nStimulusRows
        , nStimulusCols);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy the results back to the host
    float* result_gaussian_curves_host = new float[nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols * nStimulusRows * nStimulusCols];
    cudaStatus = cudaMemcpy(result_gaussian_curves_host, dev_result_gaussian_curves, (nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols) * (nStimulusRows * nStimulusCols) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Call the function to write Gaussian curves to text files
    writeGaussianCurvesToFile("D:/code/sid-git/fmri/local-extracted-datasets/gpu-tests/resulting-gaussian-curves/"
        , result_gaussian_curves_host
        , testspace_vf_points_x
        , testspace_vf_points_y
        , nQuadrilateralSpaceGridRows
        , nQuadrilateralSpaceGridCols
        , nStimulusRows
        , nStimulusCols
    );

    // Cleanup
    delete[] result_gaussian_curves_host;


  Error:
    cudaFree(dev_stimulus_vf_points_x);
    cudaFree(dev_stimulus_vf_points_y);
    cudaFree(dev_testspace_vf_points_x);
    cudaFree(dev_testspace_vf_points_y);
    cudaFree(dev_result_gaussian_curves);

    return cudaStatus;
}
