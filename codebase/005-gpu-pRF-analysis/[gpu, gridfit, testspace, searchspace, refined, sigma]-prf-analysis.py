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
from compute_inverse_matrix_M_with_sigma import Grids2MpInv, get_block_indices_new, get_block_indices_with_Sigma

###########################################--------Variables------------###############################################################
# Variables
num_noisy_signals_per_signal = 1
search_space_rows = 51
search_space_cols = 51
search_space_frames = 1 #<-----------when we have variable SIGMA
test_space_rows = 3
test_space_cols = 3
test_space_frames = 1
stim_width = 101
stim_height = 101
# sigma = np.double(2)
total_gaussian_curves_per_stim_frame = search_space_rows * search_space_cols
single_gaussian_curve_length = stim_width * stim_height

# search space
search_space_xx = np.linspace(-15, +15, search_space_cols)
search_space_yy = np.linspace(-15, +15, search_space_rows)
search_space_sigma_range = np.linspace(2, 3, search_space_frames) # 0.5 to 1.5

# test space
test_space_xx = np.linspace(-6, +6, test_space_rows)
test_space_yy = np.linspace(-6, +6, test_space_cols)
test_space_sigma_range = np.linspace(2, 2, test_space_frames) ## 0.6 to 1

# Gaussian curves
x_range_gpu = cp.linspace(-9, +9, stim_width)
y_range_gpu = cp.linspace(-9, +9, stim_height)

mu_X_grid, mu_Y_grid, sigma_grid = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range) 
test_mu_X_grid, test_mu_Y_grid = np.meshgrid(test_space_xx, test_space_yy) 
arr_2d_location_inv_M_cpu = Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid) ##<<<<--------Compute pre-define matrix M

###########################################--------Stimulus------------###############################################################
# HRF Curve
hrf_t = np.arange(0, 31, 1) # np.linspace(0, 30, 31)
hrf_curve = spm_hrf_compat(hrf_t)
stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2]))
stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
stim_frames = stimulus.resampled_hrf_convolved_data.shape[2] 

###########################################--------FUNCTIONS------------###############################################################
def create_cofficients_matrices_A_and_B(coefficients):
    # coefficients = [ a11, a22, a33, a12, a13, a23, b1, b2,  b3] <----MIND that a22 is at second position
    '''   
        A =[[a11    a12     a13]
            [a12    a22     a23]
            [a13    a23     a33]]      
    ''' 
    A = np.array([
            [coefficients[0], coefficients[3], coefficients[4]],
            [coefficients[3], coefficients[1], coefficients[5]],
            [coefficients[4], coefficients[5], coefficients[2]]
        ])
    
    B = np.array([coefficients[6], coefficients[7], coefficients[8]])
    return A, B

# indexing convention: (y, x, z) = (row, col, frame) = (uy, ux, sigma)
def flatIdx2ThreeDimIndices(flatIdx, nRows, nCols, nFrames):
    flatIdx = np.atleast_1d(flatIdx)  # Convert to NumPy array to handle multiple values or a single value
    frame = flatIdx // (nRows * nCols)
    flatIdx = flatIdx % (nRows * nCols)
    row = flatIdx // nCols
    col = flatIdx % nCols
    return row, col, frame


def threeDimIndices2FlatIdx(threeDimIdx, nRows, nCols):
    #threeDimIdx = np.atleast_3d(threeDimIdx)  # Convert to NumPy array to handle multiple values or a single value
    row, col, frame = threeDimIdx.T  # Transpose to unpack rows and columns
    flatIdx = frame * (nRows * nCols) + (row * nCols + col)
    return flatIdx



##--------Create Simulated Signals
def generate_noisy_signals(org_signals_along_columns, synthesis_ratio):
    noise_std=0.01
    #noisy_signals = org_signals_along_columns + cp.random.normal(0, 0.01, size=org_signals_along_columns.shape)
    #return noisy_signals
    num_signals, signal_length = org_signals_along_columns.shape
    noisy_signals = cp.tile(org_signals_along_columns, (1, synthesis_ratio))
    noise = cp.random.normal(0, noise_std, size=(num_signals, synthesis_ratio * signal_length))
    noisy_signals += noise
    
    return noisy_signals

##--------CUDA Gaussian Kernel
def get_gaussian_related_kernel(kernel_filename, kernel_name):
    # Get the path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the CUDA kernel file
    kernel_file_path = os.path.join(script_dir, 'kernels/gaussian_kernel.cu')

    # Load the CUDA kernel file
    with open(kernel_file_path, 'r') as kernel_file:
        kernel_code = kernel_file.read()

    # Compile the kernel code using CuPy
    kernel = cp.RawKernel(kernel_code, kernel_name)

    return kernel

###########################################------------------PROGRAM------------###############################################################
def plot_element(e_gpu, Y_signals_gpu, y_idx, search_space_xx, search_space_yy, refined_X, actual_X):
    # Calculate ee and ee_8_gpu (assuming other variables are defined elsewhere)
    ee = (e_gpu)**2
    ee_8_gpu = ee[y_idx, :]  # element row, all columns
    ee_8_cpu = cp.asnumpy(ee_8_gpu)
    
    # Create a figure and plot the contour
    plt.figure()
    plt.contour(search_space_xx, search_space_yy, ee_8_cpu.reshape((len(search_space_yy), len(search_space_xx))))

    # Plot refined_X
    x1, y1 = refined_X
    plt.scatter(x1, y1, color='red', marker='o', label=f'Computed ({x1:.2f}, {y1:.2f})')  # Rounded to 2 decimal places
    
    # Plot actual_X 
    x2, y2 = actual_X
    plt.scatter(x2, y2, color='blue', marker='s', label=f'Real ({x2}, {y2})')  # Rounded to 2 decimal places

    # Include y_idx in the title
    plt.title(f'Plot for y_idx: {y_idx:.2f}')

    plt.legend()
    
    # Display the plot
    plt.show(block=False)
    
    print('done')

def plot_function(refined_X, muX_grid, muY_grid, A, B, search_space_xx, search_space_yy):
    X = np.asarray(refined_X)
    # e = X.T @ (A @ X) + B@X

    fex = []
    for muY in search_space_yy:
        for muX in search_space_xx:
            X = np.asarray([muX, muY])
            e = X.T @ (A @ X) + B@X
            fex.append(e)

    # Convert fex to a NumPy array for reshaping
    fex = np.array(fex)

    # Create a figure and plot the contour
    plt.figure()
    plt.contour(search_space_xx, search_space_yy, fex.reshape((len(search_space_yy), len(search_space_xx))))

    # Plot refined_X
    x1, y1 = refined_X
    plt.scatter(x1, y1, color='red', marker='o', label=f'Computed ({x1:.2f}, {y1:.2f})')  # Rounded to 2 decimal places

    plt.legend()
    
    # Display the plot
    # plt.show(block=False)
    return plt.gca()
    

temp_X , temp_Y = np.meshgrid(search_space_xx, search_space_yy)
def gpu_compute():
    ###########################################--------Compute Gaussian Curves------------###############################################################
    #search_space_xx_gpu = cp.linspace(-9, +9, search_space_cols)
    #search_space_yy_gpu = cp.linspace(-9, +9, search_space_rows)

    ## gpu - Gaussian
    #stim_xx_gpu = cp.asarray(stim_xx)
    #stim_yy_gpu = cp.asarray(stim_yy)
    search_space_xx_gpu = cp.asarray(search_space_xx)
    search_space_yy_gpu = cp.asarray(search_space_yy)
    search_space_sigmas_gpu = cp.asarray(search_space_sigma_range)  
    result_flat_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)
    result_flat_Dmu_x_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)
    result_flat_Dmu_y_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)
    result_flat_Dsigma_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)

    # test space
    test_space_xx_gpu = cp.asarray(test_space_xx)
    test_space_yy_gpu = cp.asarray(test_space_yy)
    test_space_sigmas_gpu = cp.asarray(test_space_sigma_range)
    test_flat_gaussian_curves_data_gpu = cp.zeros((test_space_rows * test_space_cols * test_space_frames * stim_width * stim_height), dtype=cp.float64)

    # Define CUDA kernel
    block_dim_1 = (32, 32, 1)
    bx1 = int((search_space_cols + block_dim_1[0] - 1) / block_dim_1[0])
    by1 = int((search_space_rows + block_dim_1[1] - 1) / block_dim_1[1])
    bz1 = int((search_space_frames + block_dim_1[2] - 1) / block_dim_1[2])
    grid_dim_1 = (bx1, by1, bz1)

    # Gaussian related kernels
    gaussian_kernel = get_gaussian_related_kernel('gaussian_kernel.cu', 'generateGaussianWithSigmaKernel')

    # Launch Gaussian related kernels
    #---compute gaussian curves
    gaussian_kernel(grid_dim_1, block_dim_1, (
        result_flat_gaussian_curves_data_gpu,
        result_flat_Dmu_x_gaussian_curves_data_gpu,
        result_flat_Dmu_y_gaussian_curves_data_gpu,
        result_flat_Dsigma_gaussian_curves_data_gpu,
        search_space_xx_gpu, 
        search_space_yy_gpu,
        search_space_sigmas_gpu,
        x_range_gpu,
        y_range_gpu,
        search_space_rows,
        search_space_cols, 
        search_space_frames, 
        stim_width,
        stim_height
    ))

    #---compute gaussian curves - TEST SPACE
    test_space_gaussian_kernel = get_gaussian_related_kernel('gaussian_kernel.cu', 'generateTestSpaceGaussianWithSigmaKernel')
    block_dim_2 = (32, 32, 1)
    bx2 = int((test_space_cols + block_dim_2[0] - 1) / block_dim_2[0])
    by2 = int((test_space_rows + block_dim_2[1] - 1) / block_dim_2[1])
    bz2 = int((test_space_frames + block_dim_2[2] - 1) / block_dim_2[2])
    grid_dim_2 = (bx2, by2, bz2)
    test_space_gaussian_kernel(grid_dim_2, block_dim_2, (
        test_flat_gaussian_curves_data_gpu,        
        test_space_xx_gpu, 
        test_space_yy_gpu,
        test_space_sigmas_gpu,
        x_range_gpu,
        y_range_gpu,
        test_space_rows,
        test_space_cols, 
        test_space_frames,
        stim_width,
        stim_height))

    # # Verify results - plt.imshow(reshaped_result[:, :, 10])
    # Get results - Gaussian Curves
    # result_flat_gaussian_curves_data_cpu = cp.asnumpy(result_flat_gaussian_curves_data_gpu)
    # reshaped_result = np.reshape(result_flat_gaussian_curves_data_cpu, (101, 101, total_gaussian_curves_per_stim_frame), order='F')
    # Dx = cp.asnumpy(result_flat_Dmu_x_gaussian_curves_data_gpu)
    # reshaped_result_Dx = np.reshape(Dx, (101, 101, total_gaussian_curves_per_stim_frame), order='F')
    # plt.imshow(reshaped_result_Dx [:, :, 10])

    ###########################################--------Gaussian Curves to Timeseries (Shortend)------------###############################################################
    # reshape gpu gaussian curuves row-wise
    nRows_gaussian_curves_matrix = search_space_rows * search_space_cols * search_space_frames
    nCols_gaussian_curves_matrix = stim_height * stim_width
    gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
    dx_gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_Dmu_x_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix))
    dy_gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_Dmu_y_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix))
    dsigma_gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_Dsigma_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix))
    test_gc = (cp.asnumpy(gaussian_curves_rowmajor_gpu[0, :])).reshape((stim_height, stim_width))

    stimulus_flat_data_gpu = cp.asarray(stimulus.resampled_hrf_convolved_data.flatten('F'))
    stimulus_data_columnmajor_gpu = cp.reshape(stimulus_flat_data_gpu, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame
    test_stim = (cp.asnumpy(stimulus_data_columnmajor_gpu[:, 10])).reshape((stim_height, stim_width))

    # S_rowmajor_gpu = cp.zeros((search_space_rows * search_space_cols, stim_frames), dtype=cp.float64)
    # cp.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu, S_rowmajor_gpu)
    S_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    dS_dx_rowmajor_gpu = cp.dot(dx_gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    dS_dy_rowmajor_gpu = cp.dot(dy_gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    dS_dsigma_rowmajor_gpu = cp.dot(dsigma_gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    test_tc = (cp.asnumpy(S_rowmajor_gpu[1, :]))

    # test space
    # reshape gpu gaussian curuves row-wise
    test_nRows_gaussian_curves_matrix = test_space_rows * test_space_cols * test_space_frames
    test_nCols_gaussian_curves_matrix = stim_height * stim_width
    test_gaussian_curves_rowmajor_gpu = cp.reshape(test_flat_gaussian_curves_data_gpu, (test_nRows_gaussian_curves_matrix, test_nCols_gaussian_curves_matrix))
    test_Y_rowmajor_gpu = cp.dot(test_gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)

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

    # orthogonalization + nomalization of signals/timecourses (present along the columns)
    S_star_columnmajor_gpu = cp.dot(O_gpu, S_rowmajor_gpu.T)
    S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns) 
    S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals

    dS_star_dx_columnmajor_gpu = cp.dot(O_gpu, dS_dx_rowmajor_gpu.T)
    dS_star_dy_columnmajor_gpu = cp.dot(O_gpu, dS_dy_rowmajor_gpu.T)    
    dS_star_dsigma_columnmajor_gpu = cp.dot(O_gpu, dS_dsigma_rowmajor_gpu.T)    
    
    # dS_prime_dx_columnmajor_gpu = dS_star_dx_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu.T @ dS_star_dx_columnmajor_gpu).diagonal())
    # dS_prime_dy_columnmajor_gpu = dS_star_dy_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu.T @ dS_star_dy_columnmajor_gpu).diagonal())
    # dS_prime_dsigma_columnmajor_gpu = dS_star_dsigma_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu.T @ dS_star_dsigma_columnmajor_gpu).diagonal())
    dS_prime_dx_columnmajor_gpu = dS_star_dx_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dx_columnmajor_gpu).sum(axis=0))
    dS_prime_dy_columnmajor_gpu = dS_star_dy_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dy_columnmajor_gpu).sum(axis=0))
    dS_prime_dsigma_columnmajor_gpu = dS_star_dsigma_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dsigma_columnmajor_gpu).sum(axis=0))


    # test_orthogonalized_tc = (cp.asnumpy(signals_columnmajor_gpu[:, 1]))


    ###########################################--------Simulated Noisy Signals------------###############################################################
    test_space_along_columns = test_Y_rowmajor_gpu.T
    test_space_along_columns = test_space_along_columns / (((test_space_along_columns ** 2).sum(axis=0)) ** (-1/2)) # normalization
    Y_signals_gpu = generate_noisy_signals(test_space_along_columns, synthesis_ratio=1)    
    # noisy_tc = (cp.asnumpy(noisy_signals_gpu[:, 1]))

    ###########################################--------Projection Squared------------###############################################################    
    e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
    de_dx_full_gpu = 2 * e_gpu * (Y_signals_gpu.T @ dS_prime_dx_columnmajor_gpu)
    de_dy_full_gpu = 2 * e_gpu * (Y_signals_gpu.T @ dS_prime_dy_columnmajor_gpu)
    de_dsigma_full_gpu = 2 * e_gpu * (Y_signals_gpu.T @ dS_prime_dsigma_columnmajor_gpu)
    best_fit_proj_gpu = cp.argmax(e_gpu **2, axis=1)     #<<<<----find the max. element's index for the rows along their columns (that's why axis=1)
    

    ###########################################--------Refined Search------------###############################################################          
    # Y_signals_cpu = cp.asnumpy(Y_signals_gpu)
    # S_prime_columnmajor_cpu = cp.asnumpy(S_prime_columnmajor_gpu)
    # dx_S_prime_columnmajor_cpu = cp.asnumpy(dx_S_prime_columnmajor_gpu)
    # dy_S_prime_columnmajor_cpu = cp.asnumpy(dy_S_prime_columnmajor_gpu)
    # e_cpu = cp.asnumpy(e_gpu)
    best_fit_proj_cpu = cp.asnumpy(best_fit_proj_gpu)
    de_dx_full_cpu = cp.asnumpy(de_dx_full_gpu)
    de_dy_full_cpu = cp.asnumpy(de_dy_full_gpu)
    de_dsigma_full_cpu = cp.asnumpy(de_dsigma_full_gpu)

    # TEST: plot errors for the measured/simulated signal no. 8
    if(False):        
        ee = (e_gpu)**2
        ee_8_gpu = ee[int(((Y_signals_gpu.T).shape[0]) // 4), :] # 8th row i.e. 8th signal, all columns
        ee_8_cpu = cp.asnumpy(ee_8_gpu)
        plt.figure()
        plt.contour(search_space_xx, search_space_yy , ee_8_cpu.reshape((len(search_space_yy),len(search_space_xx))))
        # plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(search_space_xx, search_space_yy, ee_8_cpu.reshape((len(search_space_yy),len(search_space_xx))),cmap='viridis', edgecolor='none')
        print('done')

    for y_idx in range(Y_signals_gpu.shape[1]):
        best_s_idx = best_fit_proj_cpu[y_idx]
        best_s_3d_idx = flatIdx2ThreeDimIndices(best_s_idx, search_space_rows, search_space_cols, search_space_frames)

        # get the current signal and its neighboring signals indices
        block_3d_indices = get_block_indices_with_Sigma(row=best_s_3d_idx[0], col=best_s_3d_idx[1], frame=best_s_3d_idx[2], nRows=search_space_rows, nCols=search_space_cols, nFrames=search_space_frames, distance=1) #get_block_indices_new(row=best_s_2d_idx[0], col=best_s_2d_idx[1], nRows=search_space_rows, nCols=search_space_cols, distance=1)
        block_flat_indices = (threeDimIndices2FlatIdx(threeDimIdx=block_3d_indices, nRows=search_space_rows, nCols=search_space_cols)).astype(int)

        # compute the coffeficients
        #...get the pre-computed Mp Inverse matrix (already containing information about the neighbors)
        MpInv = arr_2d_location_inv_M_cpu[best_s_idx]        
        
        #...compute the de/dx, de/dy and de/dsigma vectors# 
        de_dx_vec_cpu = de_dx_full_cpu[y_idx, block_flat_indices]
        de_dy_vec_cpu = de_dy_full_cpu[y_idx, block_flat_indices]
        de_dsigma_vec_cpu = de_dsigma_full_cpu[y_idx, block_flat_indices]
        de_dX_cpu = (np.vstack( (de_dx_vec_cpu, de_dy_vec_cpu, de_dsigma_vec_cpu) )).ravel(order = 'F') # <<<----MIND, it's capital X in de_dX. Capital X means the complete vector i.e. [ux1, uy1, ux2, uy2, ux3, uy3...]
        coefficients = MpInv@de_dX_cpu
        A, B = create_cofficients_matrices_A_and_B(coefficients)
        refined_X = -0.5 * (np.linalg.inv(A) @ B)
        actual_X = (mu_X_grid[best_s_3d_idx[0], best_s_3d_idx[1], best_s_3d_idx[2]]
                    , mu_Y_grid[best_s_3d_idx[0], best_s_3d_idx[1], best_s_3d_idx[2]]
                    , sigma_grid[best_s_3d_idx[0], best_s_3d_idx[1], best_s_3d_idx[2]])

        # plot_element(e_gpu ** 2, Y_signals_gpu, y_idx, search_space_xx, search_space_yy, refined_X, actual_X)
        # # plot_element(de_dx_full_cpu, Y_signals_gpu, y_idx, search_space_xx, search_space_yy, refined_X, actual_X)
        # # plot_element(de_dy_full_cpu, Y_signals_gpu, y_idx, search_space_xx, search_space_yy, refined_X, actual_X)

        f = plt.figure()
        quiver = plt.gca().quiver(temp_X , temp_Y, -de_dx_full_cpu[y_idx], -de_dy_full_cpu[y_idx], angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5))
        
        # # block_meshgrid = np.meshgrid()
        # ax = plot_function(refined_X, mu_X_grid, mu_Y_grid, A, B, search_space_xx, search_space_yy)

        # actual_X1 = [(mu_X_grid[a[0], a[1], a[2]], mu_Y_grid[a[0], a[1], a[2]], sigma_grid[a[0], a[1], a[2]]) for a in block_3d_indices.astype(int)]
        # xx, yy = zip(*actual_X1)
        # plt.plot(actual_X[0][0], actual_X[1][0], 'bx')
        # quiver = ax.quiver(xx , yy, de_dx_vec_cpu, de_dy_vec_cpu, angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5))
        print('done')


    # Synchronize and release memory
    cp.cuda.Device().synchronize()


# ###################################-----------Main()---------------------------------####################
cProfile.run('gpu_compute()', sort='cumulative')
print("done")




