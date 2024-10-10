extern "C"
{
	__global__ void computeGaussianAndStimulusMultiplicationKernel(double* result_stimulus_multiplied_gaussian_curves
/*	, double* flattened_3d_stimulusData
	, double* flattened_gaussian_curves_data
	, int singleGaussianCurveLength
	, int nTotalGaussianCurvesPerStimulusFrame
	, int nStimulusFrames
	, int nStimulusRows
	, int nStimulusCols*/)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int frame = blockIdx.z * blockDim.z + threadIdx.z;

	//if (y < nTotalGaussianCurvesPerStimulusFrame && x < singleGaussianCurveLength && frame < nStimulusFrames)
	//{
	//	int stimIdx = frame * (nStimulusRows * nStimulusCols) + x; // MIND that sitmIdx is not using the term "(y * nStimulusCols + x)" because each frame of the stimulus is considered to be a flattened 1d array
	//	int gaussIdx = (y * singleGaussianCurveLength + x);
	//	double stimulusCellValue = flattened_3d_stimulusData[stimIdx];
	//	double gaussianCurveValue = flattened_gaussian_curves_data[gaussIdx];
	//	double mulitplicationResult = stimulusCellValue * gaussianCurveValue;
	//	int resultIdx = (frame * singleGaussianCurveLength * nTotalGaussianCurvesPerStimulusFrame) + (y * singleGaussianCurveLength + x);

	//	// Print array values for debugging in a single statement
	//	//printf("Thread: (%d, %d, %d), x: %d, y: %d, frame: %d, stimulusCellValue[%d]: %.6f, gaussianCurveValue: %.6e, result: %.6e at [%d]\n", threadIdx.x, threadIdx.y, threadIdx.z, x, y, frame, stimIdx ,stimulusCellValue, gaussianCurveValue, mulitplicationResult, resultIdx);

	//	result_stimulus_multiplied_gaussian_curves[resultIdx] = mulitplicationResult;

	//}
}
}

