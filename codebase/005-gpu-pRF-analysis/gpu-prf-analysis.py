# Add my "oprf" package path
import sys
sys.path.append("D:/code/sid-git/fmri/")

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cProfile
import os

# Local Imports
#from config.config import Configuration as config
from oprf.standard.prf_stimulus import Stimulus
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator

###########################################--------Variables------------###############################################################
# Variables
search_space_rows = 101
search_space_cols = 101
stim_width = 101
stim_height = 101
stim_frames = 301
sigma = np.double(2)
total_gaussian_curves_per_stim_frame = search_space_rows * search_space_cols
single_gaussian_curve_length = stim_width * stim_height

###########################################--------Stimulus------------###############################################################
# HRF Curve
hrf_t = np.linspace(0, 30, 31)
hrf_curve = spm_hrf_compat(hrf_t)
stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2]))
stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)

###########################################--------Create Simulated Signals------------###############################################################
def generate_noisy_signals(org_signals_along_columns):
    noisy_signals = org_signals_along_columns + cp.random.normal(0, 0.01, size=org_signals_along_columns.shape)
    return noisy_signals



###########################################---------------------------###############################################################
def get_gaussian_kernel(kernel_filename, kernel_name):
    # Get the path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the CUDA kernel file
    kernel_file_path = os.path.join(script_dir, 'kernels/gaussian_kernel.cu')

    # Load the CUDA kernel file
    with open(kernel_file_path, 'r') as kernel_file:
        kernel_code = kernel_file.read()

    # Compile the kernel code using CuPy
    kernel = cp.RawKernel(kernel_code, 'generateGaussianKernel')

    return kernel

def gpu_compute():
    ###########################################--------Compute Gaussian Curves------------###############################################################
    # Gaussian curves
    stim_xx_gpu = cp.linspace(-9, +9, stim_width)
    stim_yy_gpu = cp.linspace(-9, +9, stim_height)

    search_space_xx_gpu = cp.linspace(-9, +9, search_space_rows)
    search_space_yy_gpu = cp.linspace(-9, +9, search_space_cols)

    ## gpu - Gaussian
    #stim_xx_gpu = cp.asarray(stim_xx)
    #stim_yy_gpu = cp.asarray(stim_yy)
    #search_space_xx_gpu = cp.asarray(search_space_xx)
    #search_space_yy_gpu = cp.asarray(search_space_yy)
    result_flat_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * stim_width * stim_height), dtype=cp.float64)

    # Define CUDA kernel
    block_dim_1 = (32, 32)
    bx1 = int((search_space_cols + block_dim_1[0] - 1) / block_dim_1[0])
    by1 = int((search_space_rows + block_dim_1[1] - 1) / block_dim_1[1])
    grid_dim_1 = (bx1, by1)
    gaussian_kernel = get_gaussian_kernel('gaussian_kernel.cu', 'generateGaussianKernel')
    gaussian_kernel(grid_dim_1, block_dim_1, (
        result_flat_gaussian_curves_data_gpu,
        search_space_xx_gpu, 
        search_space_yy_gpu,
        stim_xx_gpu,
        stim_yy_gpu,
        search_space_rows,
        search_space_cols, 
        stim_width,
        stim_height,
        sigma
    ))

    # # Verify results - plt.imshow(reshaped_result[:, :, 10])
    # Get results - Gaussian Curves
    # result_flat_gaussian_curves_data_cpu = cp.asnumpy(result_flat_gaussian_curves_data_gpu)
    # reshaped_result = np.reshape(result_flat_gaussian_curves_data_cpu, (101, 101, total_gaussian_curves), order='F')

    ###########################################--------Mask Gaussian Curves (Not Working)------------###############################################################
    # result_stimulus_multiplied_gaussian_curves_gpu = cp.zeros((single_gaussian_curve_length * total_gaussian_curves_per_stim_frame * stim_frames ), dtype=cp.float64)
    # stimulus_flat_data_gpu = cp.asarray(stimulus.resampled_hrf_convolved_data.flatten('F'))
    # block_dim_2 = (32, 32, 1)
    # bx2 = int((single_gaussian_curve_length + block_dim_2[0] - 1) / block_dim_2[0])
    # by2 = int((total_gaussian_curves_per_stim_frame + block_dim_2[1] - 1) / block_dim_2[1])
    # bz2 = int((stim_frames + block_dim_2[2] -1 ) / block_dim_2[2])
    # grid_dim_2 = (bx2, by2, bz2)
    # apply_stimulus_mask_kernel = get_gaussian_kernel('apply_stimulus_mask_kernel.cu', 'computeGaussianAndStimulusMultiplicationKernel')
    # apply_stimulus_mask_kernel(grid_dim_2, block_dim_2, (
    #     result_stimulus_multiplied_gaussian_curves_gpu
    #     # ,
    #     # stimulus_flat_data_gpu,
    #     # result_flat_gaussian_curves_data_gpu, # computed above
    #     # single_gaussian_curve_length,
    #     # total_gaussian_curves_per_stim_frame,
    #     # stim_frames,
    #     # stim_height,
    #     # stim_width
    # ))



    ###########################################--------Gaussian Curves to Timeseries (Shortend)------------###############################################################
    # reshape gpu gaussian curuves row-wise
    nRows_gaussian_curves_matrix = search_space_rows * search_space_cols
    nCols_gaussian_curves_matrix = stim_height * stim_width
    gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
    # test_gc = (cp.asnumpy(gaussian_curves_rowmajor_gpu[0, :])).reshape((stim_height, stim_width))

    stimulus_flat_data_gpu = cp.asarray(stimulus.resampled_hrf_convolved_data.flatten('F'))
    stimulus_data_columnmajor_gpu = cp.reshape(stimulus_flat_data_gpu, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame
    # test_stim = (cp.asnumpy(stimulus_data_columnmajor_gpu[:, 10])).reshape((stim_height, stim_width))

    timeseries_rowmajor_gpu = cp.zeros((search_space_rows * search_space_cols, stim_frames), dtype=cp.float64)
    cp.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu, timeseries_rowmajor_gpu)
    # test_tc = (cp.asnumpy(timeseries_rowmajor_gpu[1, :]))

    ###########################################--------Regressors------------###############################################################
    nDCT = 3
    ndct = 2 * nDCT + 1
    trends = np.zeros((stim_frames, np.max([np.sum(ndct), 1])))        

    tc = np.linspace(0, 2.*np.pi, stim_frames)[:, None]        
    trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))

    q, r = np.linalg.qr(trends) # QR decomposition
    q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0

    R_gpu = cp.asarray(q)
    O_gpu = (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))

    # orthogonalization of signals/timecourses (present along the columns)
    S_signals_columnmajor_gpu = cp.dot(O_gpu, timeseries_rowmajor_gpu.T)
    # test_orthogonalized_tc = (cp.asnumpy(signals_columnmajor_gpu[:, 1]))

    # nomalization of signals
    #column_norms = cp.linalg.norm(signals_columnmajor_gpu, axis=0)
    S_signals_columnmajor_gpu = S_signals_columnmajor_gpu / (cp.linalg.norm(S_signals_columnmajor_gpu, axis=0)) # now we have orthonormal signals (along columns)
    # test_normalized_orthogonalized_tc = (cp.asnumpy(signals_columnmajor_gpu[:, 1]))


    ###########################################--------Simulated Noisy Signals------------###############################################################
    Y_noisy_signals_gpu = generate_noisy_signals(S_signals_columnmajor_gpu)
    # noisy_tc = (cp.asnumpy(noisy_signals_gpu[:, 1]))

    ###########################################--------Projection Squared------------###############################################################    
    best_fit_proj_gpu = cp.argmax(((Y_noisy_signals_gpu.T @ S_signals_columnmajor_gpu)**2 ), axis=0)






    # Synchronize and release memory
    cp.cuda.Device().synchronize()


###################################-----------Main()---------------------------------####################
cProfile.run('gpu_compute()', sort='cumulative')
print("done")




