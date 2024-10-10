#include "../header/ModelSignals.h"
#include "../../utils/arrays/Linspace.h"
#include <Eigen/Dense>
#include "../../utils/matrix/Matrix.h"

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

// gpu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


ModelSignals::ModelSignals(double* flattenedHRFConvolvedStimulusData
	, int nQuadrilateralSpaceGridRows
	, int nQuadrilateralSpaceGridCols
	, int nStimulusRows
	, int nStimulusCols
	, int nStimulusFrames
	, double sigma)
	: _flattenedHRFConvolvedStimulusData(flattenedHRFConvolvedStimulusData),
	_nQuadrilateralSpaceGridRows(nQuadrilateralSpaceGridRows),
	_nQuadrilateralSpaceGridCols(nQuadrilateralSpaceGridCols),
	_nStimulusRows(nStimulusRows),
	_nStimulusCols(nStimulusCols),
	_nStimulusFrames(nStimulusFrames)
	, _sigma(sigma)
{
	_singleFlattenedGaussianCurvelength = _nStimulusRows * _nStimulusCols;
	_nTotalGaussianCurvesPerStimulusFrame = _nQuadrilateralSpaceGridRows * _nQuadrilateralSpaceGridCols;
	_h_result_columns_summation = new double[_nTotalGaussianCurvesPerStimulusFrame * _nStimulusFrames];
}

ModelSignals::~ModelSignals()
{
	delete[] _h_result_columns_summation;
}

void ModelSignals::computeTrends(int ndct, int nStimulusFrames, double *trends)
{
	int tf = nStimulusFrames; // timepoints
	const double PI = 3.14159265358979323846;

	for (int i = 0; i < tf; i++)
	{
		double tc = 2.0 * PI * i / (tf - 1);
		for (int j = 0; j < ndct; j++)
		{
			double val = std::cos(tc * (j * 0.5));
			trends[i * ndct + j] = val;
		}
	}
}

bool ModelSignals::generateModelSignals()
{
	cudaError_t cudaStatus;

	// Gaussian curves
	double* dev_result_gaussian_curves = computeModelCurves();

	// Apply Stimulus Mask
	double* dev_stimulusMaskAppliedGaussianCurves = applyStimulusMask(dev_result_gaussian_curves);
		
	// Compute signal Timecourses	
	double* d_timeCourses = computeRowwiseSum(dev_stimulusMaskAppliedGaussianCurves);

	// Compute trends
	int nDCT = 3;
	int ndct = 2 * nDCT + 1;
	double* trends = new double[_nStimulusFrames * ndct];
	computeTrends(ndct, _nStimulusFrames, trends);

	// QR decomposition of trends
	Matrix trendsMat(_nStimulusFrames, ndct, trends);
	int nRowsThinQ = _nStimulusFrames;
	int nColsThinQ = _nStimulusFrames;
	double* R = trendsMat.qrDecomposition(nRowsThinQ, nColsThinQ); // NOTE: <----Returning Signs flipped

	// Orthonormaliazation of signal Timecourses
	double* d_orthoSignals = orthonormalizeSignals(d_timeCourses, R);


	// Get results - Device to Host		
	cudaStatus = cudaMemcpy(_h_result_columns_summation, d_orthoSignals, _nTotalGaussianCurvesPerStimulusFrame * _nStimulusFrames * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

Error:
	freeDeviceMemoryWrapper(d_timeCourses);
	freeDeviceMemoryWrapper(dev_result_gaussian_curves);
	freeDeviceMemoryWrapper(dev_stimulusMaskAppliedGaussianCurves);
	delete[] trends;
	return true;
}


double* ModelSignals::getResults()
{
	return _h_result_columns_summation;
}