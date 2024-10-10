//IMPORTANT NOTE: This kernel is not used anymore. The actual kernel is present in the "oprf/hpc/kernels"

extern "C"
{
	__global__ void generateGaussianKernel(
		double* result_gaussian_curves,
		double* result_Dx_gaussian_curves,
		double* result_Dy_gaussian_curves,
		double* result_Dsigma_gaussian_curves,
		double* modelspace_vf_points_x,
		double* modelspace_vf_points_y,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int nGaussianRows,
		int nGaussianCols,
		int nStimulusRows,
		int nStimulusCols,
		double sigma)
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
					double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);
					result_gaussian_curves[gaussIdx] = exp(exponent); // / (2 * PI * sigma * sigma);									
					//printf("Thread: (%d, %d), Ux: %f, Uy: %f, result_gaussian_curves[%d]: %.6e\n", col, row, x_mean, y_mean, gaussIdx, result_gaussian_curves[gaussIdx]);

					// Derivatives
					result_Dx_gaussian_curves[gaussIdx] = ((x - x_mean) / (sigma * sigma)) * exp(exponent);
					result_Dy_gaussian_curves[gaussIdx] = ((y - y_mean) / (sigma * sigma)) * exp(exponent);


					//result_Dsigma_gaussian_curves[gaussIdx] = -(((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (sigma * sigma * sigma)) * exp(-(((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma)));
					result_Dsigma_gaussian_curves[gaussIdx] = -(((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (sigma * sigma * sigma)) * exp(exponent);

					gaussIdx++;
				}
			}
		}
	}

	__global__ void generateGaussianWithSigmaKernel(
		double* result_gaussian_curves,
		double* result_Dx_gaussian_curves,
		double* result_Dy_gaussian_curves,
		double* result_Dsigma_gaussian_curves,
		double* modelspace_vf_points_x,
		double* modelspace_vf_points_y,
		double* modelspace_vf_points_sigma,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int nGaussianRows,
		int nGaussianCols,
		int nGaussianFrames,
		int nStimulusRows,
		int nStimulusCols
		)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int frame = blockIdx.z * blockDim.z + threadIdx.z;

		if (row < nGaussianRows && col < nGaussianCols && frame < nGaussianFrames)
		{
			double y_mean = modelspace_vf_points_y[row];
			double x_mean = modelspace_vf_points_x[col];
			double sigma = modelspace_vf_points_sigma[frame];

			int meanPairIdx = (frame * (nGaussianRows * nGaussianCols)) + (row * nGaussianCols + col);
			int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

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
					
					
					result_gaussian_curves[gaussIdx] = exp(exponent); // / (2 * PI * sigma * sigma);				
					//printf("Thread: (%d, %d), Ux: %f, Uy: %f, result_gaussian_curves[%d]: %.6e\n", col, row, x_mean, y_mean, gaussIdx, result_gaussian_curves[gaussIdx]);

					// Derivatives
					result_Dx_gaussian_curves[gaussIdx] = ((x - x_mean) / (sigma * sigma)) * exp(exponent);
					result_Dy_gaussian_curves[gaussIdx] = ((y - y_mean) / (sigma * sigma)) * exp(exponent);
					result_Dsigma_gaussian_curves[gaussIdx] = -(((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (sigma * sigma * sigma)) * exp(exponent);

					gaussIdx++;
				}
			}
		}
	}

	__global__ void generateTestSpaceGaussianWithSigmaKernel(
		double* result_gaussian_curves,
		double* modelspace_vf_points_x,
		double* modelspace_vf_points_y,
		double* modelspace_vf_points_sigma,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int nGaussianRows,
		int nGaussianCols,
		int nGaussianFrames,
		int nStimulusRows,
		int nStimulusCols
		)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int frame = blockIdx.z * blockDim.z + threadIdx.z;

		if (row < nGaussianRows && col < nGaussianCols && frame < nGaussianFrames)
		{
			double y_mean = modelspace_vf_points_y[row];
			double x_mean = modelspace_vf_points_x[col];
			double sigma = modelspace_vf_points_sigma[frame];

			int meanPairIdx = (frame * (nGaussianRows * nGaussianCols)) + (row * nGaussianCols + col);
			int currentGaussianCurveStartIdx = meanPairIdx * (nStimulusCols * nStimulusRows); // (nStimulusCols * nStimulusRows) = single Gaussian curve size

			int gaussIdx = currentGaussianCurveStartIdx;
			//printf("Thread: (%d, %d), meanPairIdx: %d,  currentGaussianCurveStartIdx: %d\n", row, col, meanPairIdx, currentGaussianCurveStartIdx);
			for (int stim_vf_row = 0; stim_vf_row < nStimulusRows; stim_vf_row++)
			{
				for (int stim_vf_col = 0; stim_vf_col < nStimulusCols; stim_vf_col++)
				{
					double y = stimulus_vf_points_y[stim_vf_row];
					double x = stimulus_vf_points_x[stim_vf_col];


					double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);


					result_gaussian_curves[gaussIdx] = exp(exponent); // / (2 * PI * sigma * sigma);				
					//printf("Thread: (%d, %d), Ux: %f, Uy: %f, result_gaussian_curves[%d]: %.6e\n", col, row, x_mean, y_mean, gaussIdx, result_gaussian_curves[gaussIdx]);

					gaussIdx++;
				}
			}
		}
	}
}

