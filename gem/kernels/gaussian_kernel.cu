extern "C"
{
	// Send (ux, uy and sigma) in a single flattened array
	__global__ void gc_using_args_arrays_cuda_Kernel(
		double* result_gaussian_curves,
		double* prfPointsArgsFlatArr,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,	
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		int prfPointIdx = blockIdx.x * blockDim.x + threadIdx.x;

		// // // Debug: Test if the multi-dimensional pRF params array is being passed correctly
		// // if (prfPointIdx == 0)
		// // {
		// // 	for(int i=0; i<numTotalGaussianCurves; i++)
		// // 	{
		// // 		printf("prfPointsArgsFlatArr[%d]: %f, %f, %f\n", i, prfPointsArgsFlatArr[i*num_dimensions], prfPointsArgsFlatArr[i*num_dimensions+1], prfPointsArgsFlatArr[i*num_dimensions+2]);
		// // 	}
		// // }

		if (prfPointIdx < numTotalGaussianCurves)
		{
			// int argsIdx = prfPointIdx * num_dimensions;
			// double y_mean = prfPointsArgsFlatArr[argsIdx];
			// double x_mean = prfPointsArgsFlatArr[argsIdx + 1];
			// double sigma = prfPointsArgsFlatArr[argsIdx + 2];

			double x_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions];
			double y_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 1];
			double sigma = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 2];

			// // printf("prfPointsArgsFlatArr[%d]: %f, %f, %f\n", prfPointIdx, prfPointsArgsFlatArr[prfPointIdx*num_dimensions], prfPointsArgsFlatArr[prfPointIdx*num_dimensions+1], prfPointsArgsFlatArr[prfPointIdx*num_dimensions+2]);

			// // printf("prfPointsArgsFlatArr[%d]: %f, %f, %f\n", prfPointIdx, y_mean, x_mean, sigma);

			// int meanPairIdx = argsIdx; 
			// int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size
			int currentGaussianCurveStartIdx = prfPointIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

			//printf("Thread: (%d, %d, %d), [ux, uy, sigma]: (%f, %f, %f) curve no.: %d,  currentGaussianCurveStartIdx: %d\n", row, col, frame, x_mean, y_mean, sigma, meanPairIdx, currentGaussianCurveStartIdx);

			int gaussIdx = currentGaussianCurveStartIdx;
			//printf("Thread: (%d, %d), meanPairIdx: %d,  currentGaussianCurveStartIdx: %d\n", row, col, meanPairIdx, currentGaussianCurveStartIdx);			
			for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
			{				
				for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
				{
					double y = stimulus_vf_points_y[stim_vf_row];
					double x = stimulus_vf_points_x[stim_vf_col];


					double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);


					result_gaussian_curves[gaussIdx] = exp(exponent);

					gaussIdx++;
				}
			}
		}
	}

	// Send (ux, uy and sigma) directly as arrays
	__global__ void dgc_dx_using_args_arrays_cuda_Kernel(
		double* result_Dx_gaussian_curves,
		double* prfPointsArgsFlatArr,		
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		int prfPointIdx = blockIdx.x * blockDim.x + threadIdx.x;

		// int argsIdx = blockIdx.x * blockDim.x + threadIdx.x;
		// if (argsIdx < numTotalGaussianCurves)
		if (prfPointIdx < numTotalGaussianCurves)
		{			
			int argsIdx = prfPointIdx * num_dimensions;
			double x_mean = prfPointsArgsFlatArr[argsIdx];
			double y_mean = prfPointsArgsFlatArr[argsIdx + 1];
			double sigma = prfPointsArgsFlatArr[argsIdx + 2];

			int meanPairIdx = argsIdx; 			
			int currentGaussianCurveStartIdx = prfPointIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

			int gaussIdx = currentGaussianCurveStartIdx;
			
			for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
			{
				for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
				{
					double y = stimulus_vf_points_y[stim_vf_row];
					double x = stimulus_vf_points_x[stim_vf_col];


					double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);

					// Derivatives					
					result_Dx_gaussian_curves[gaussIdx] = ((x - x_mean) / (sigma * sigma)) * exp(exponent);

					gaussIdx++;
				}
			}
		}
	}

	// Send (ux, uy and sigma) directly as arrays
	__global__ void dgc_dy_using_args_arrays_cuda_Kernel(
		double* result_Dy_gaussian_curves,		
		double* prfPointsArgsFlatArr,		
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		int prfPointIdx = blockIdx.x * blockDim.x + threadIdx.x;

		if (prfPointIdx < numTotalGaussianCurves)
		{
			int argsIdx = prfPointIdx * num_dimensions;
			double x_mean = prfPointsArgsFlatArr[argsIdx];
			double y_mean = prfPointsArgsFlatArr[argsIdx + 1];
			double sigma = prfPointsArgsFlatArr[argsIdx + 2];

			int meanPairIdx = argsIdx; 
			// int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size
			int currentGaussianCurveStartIdx = prfPointIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

			int gaussIdx = currentGaussianCurveStartIdx;
			
			for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
			{
				for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
				{
					double y = stimulus_vf_points_y[stim_vf_row];
					double x = stimulus_vf_points_x[stim_vf_col];


					double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);

					// Derivatives					
					result_Dy_gaussian_curves[gaussIdx] = ((y - y_mean) / (sigma * sigma)) * exp(exponent);

					gaussIdx++;
				}
			}
		}
	}

	// Send (ux, uy and sigma) directly as arrays
	__global__ void dgc_dsigma_using_args_arrays_cuda_Kernel(
		double* result_Dsigma_gaussian_curves,
		double* prfPointsArgsFlatArr,		
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		int prfPointIdx = blockIdx.x * blockDim.x + threadIdx.x;

		if (prfPointIdx < numTotalGaussianCurves)
		{
			int argsIdx = prfPointIdx * num_dimensions;
			double x_mean = prfPointsArgsFlatArr[argsIdx];
			double y_mean = prfPointsArgsFlatArr[argsIdx + 1];
			double sigma = prfPointsArgsFlatArr[argsIdx + 2];

			int meanPairIdx = argsIdx; 
			// int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size
			int currentGaussianCurveStartIdx = prfPointIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

			int gaussIdx = currentGaussianCurveStartIdx;
			
			for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
			{
				for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
				{
					double y = stimulus_vf_points_y[stim_vf_row];
					double x = stimulus_vf_points_x[stim_vf_col];


					double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);

					// Derivatives
					result_Dsigma_gaussian_curves[gaussIdx] = (((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (sigma * sigma * sigma)) * exp(exponent); //NOTE: removed "minus" sign

					gaussIdx++;
				}
			}
		}
	}

}

