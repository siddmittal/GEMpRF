# Add my "oprf" package path
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import nibabel as nib
import cProfile
import os
import json

# Config
from config.oprf_config import ConfigurationWrapper as cfg
cfg.load_configuration()

# Local Imports
import sys
sys.path.append(cfg.path_to_append)

from oprf.standard.prf_stimulus import Stimulus
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator
from compute_inverse_matrix_M_with_sigma import Grids2MpInv, get_block_indices_with_Sigma

###########################################--------Variables------------###############################################################
# Variables
num_noisy_signals_per_signal = 1
search_space_rows = int(cfg.search_space["nRows"])
search_space_cols = int(cfg.search_space["nCols"])
search_space_frames = int(cfg.search_space["nSigma"]) #<-----------SIGMA

# ...stimulus
stim_width = int(cfg.stimulus["width"])
stim_height = int(cfg.stimulus["height"])
stim_frames = int(cfg.stimulus["num_frames"])
total_gaussian_curves_per_stim_frame = search_space_rows * search_space_cols
single_gaussian_curve_length = stim_width * stim_height

# ...search space
search_space_xx = np.linspace(-float(cfg.search_space["visual_field"]), float(cfg.search_space["visual_field"]), int(cfg.search_space["nCols"]))
search_space_yy = np.linspace(-float(cfg.search_space["visual_field"]), float(cfg.search_space["visual_field"]), int(cfg.search_space["nRows"]))
search_space_sigma_range = np.linspace(float(cfg.search_space["min_sigma"]), float(cfg.search_space["max_sigma"]), int(cfg.search_space["nSigma"])) # 0.5 to 1.5

# ...Gaussian curves
x_range_cpu = np.linspace(-9, +9, int(cfg.stimulus["width"]))
y_range_cpu = np.linspace(-9, +9, int(cfg.stimulus["height"]))
x_range_gpu = cp.asarray(x_range_cpu)  # cp.linspace(-9, +9, stim_width)
y_range_gpu = cp.asarray(y_range_cpu) # cp.linspace(-9, +9, stim_height)

mu_X_grid, mu_Y_grid, sigma_grid = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range) 
arr_2d_location_inv_M_cpu = Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid) ##<<<<--------Compute pre-define matrix M

###########################################--------Stimulus------------###############################################################
# HRF Curve
hrf_t = np.arange(0, 31, 1) # np.linspace(0, 30, 31)
# hrf_curve = spm_hrf_compat(hrf_t)
hrf_curve = np.array([0, 0.0055, 0.1137, 0.4239, 0.7788, 0.9614, 0.9033, 0.6711, 0.3746, 0.1036, -0.0938, -0.2065, -0.2474, -0.2388, -0.2035, -0.1590, -0.1161, -0.0803, -0.0530, -0.0336, -0.0206, -0.0122, -0.0071, -0.0040, -0.0022, -0.0012, -0.0006, -0.0003, -0.0002, -0.0001, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000]) # mrVista Values
stimulus = Stimulus(cfg.stimulus["filepath"], size_in_degrees=float(cfg.stimulus["visual_field"]))
stimulus.compute_resample_stimulus_data((stim_height, stim_width, stim_frames)) #stimulus.org_data.shape[2]
stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
stim_frames = stimulus.resampled_hrf_convolved_data.shape[2] 

###########################################--------Data------------###############################################################
# load the BOLD response data
bold_response_img = nib.load(cfg.measured_data["filepath"])
Y_signals_cpu = bold_response_img.get_fdata()

# reshape the BOLD response data to 2D
Y_signals_cpu = Y_signals_cpu.reshape(-1, Y_signals_cpu.shape[-1])
num_voxels, num_timepoints = Y_signals_cpu.shape

# just to make them column vectors
Y_signals_cpu = Y_signals_cpu.T

if(stim_frames > num_timepoints):
    num_stim_frames_to_be_deleted = stim_frames - num_timepoints
    stimulus.resampled_hrf_convolved_data = np.delete(stimulus.resampled_hrf_convolved_data, num_stim_frames_to_be_deleted, axis=2)
    stim_frames = stimulus.resampled_hrf_convolved_data.shape[2]
    #Y_signals_cpu = np.concatenate([Y_signals_cpu, np.zeros((num_voxels, 1))], axis=1)  

## send data to gpu
#Y_signals_gpu = cp.asarray(Y_signals_cpu)


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
    kernel_file_path = os.path.join(script_dir, 'kernels', kernel_filename) # os.path.join(script_dir, kernel_filename)

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
    

def gpu_compute():
    ###########################################--------Compute Gaussian Curves------------###############################################################

    ## gpu - Gaussian
    search_space_xx_gpu = cp.asarray(search_space_xx)
    search_space_yy_gpu = cp.asarray(search_space_yy)
    search_space_sigmas_gpu = cp.asarray(search_space_sigma_range)  
    result_flat_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)
    result_flat_Dmu_x_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)
    result_flat_Dmu_y_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)
    result_flat_Dsigma_gaussian_curves_data_gpu = cp.zeros((search_space_rows * search_space_cols * search_space_frames * stim_width * stim_height), dtype=cp.float64)


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

    ###########################################--------Gaussian Curves to Timeseries (Shortend)------------###############################################################
    # reshape gpu gaussian curuves row-wise
    nRows_gaussian_curves_matrix = search_space_rows * search_space_cols * search_space_frames
    nCols_gaussian_curves_matrix = stim_height * stim_width
    gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
    dx_gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_Dmu_x_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix))
    dy_gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_Dmu_y_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix))
    dsigma_gaussian_curves_rowmajor_gpu = cp.reshape(result_flat_Dsigma_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix))
    #test_gc = (cp.asnumpy(gaussian_curves_rowmajor_gpu[0, :])).reshape((stim_height, stim_width))

    stimulus_flat_data_gpu = cp.asarray(stimulus.resampled_hrf_convolved_data.flatten('F'))
    stimulus_data_columnmajor_gpu = cp.reshape(stimulus_flat_data_gpu, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame
    #test_stim = (cp.asnumpy(stimulus_data_columnmajor_gpu[:, 10])).reshape((stim_height, stim_width))
    
    S_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    dS_dx_rowmajor_gpu = cp.dot(dx_gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    dS_dy_rowmajor_gpu = cp.dot(dy_gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    dS_dsigma_rowmajor_gpu = cp.dot(dsigma_gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    #test_tc = (cp.asnumpy(S_rowmajor_gpu[1, :]))

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
    
    dS_prime_dx_columnmajor_gpu = dS_star_dx_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dx_columnmajor_gpu).sum(axis=0))
    dS_prime_dy_columnmajor_gpu = dS_star_dy_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dy_columnmajor_gpu).sum(axis=0))
    dS_prime_dsigma_columnmajor_gpu = dS_star_dsigma_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dsigma_columnmajor_gpu).sum(axis=0))

    # test_orthogonalized_tc = (cp.asnumpy(signals_columnmajor_gpu[:, 1]))


    ###########################################--------Simulated/Measured Signals------------###############################################################    
    #Y_signals_gpu = generate_noisy_signals(test_space_along_columns, synthesis_ratio=1)    
    # send data to gpu
    Y_signals_gpu = cp.asarray(Y_signals_cpu)


    ###########################################--------Projection Squared------------###############################################################    
    e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
    de_dx_full_gpu = 2 * e_gpu * (Y_signals_gpu.T @ dS_prime_dx_columnmajor_gpu)
    de_dy_full_gpu = 2 * e_gpu * (Y_signals_gpu.T @ dS_prime_dy_columnmajor_gpu)
    de_dsigma_full_gpu = 2 * e_gpu * (Y_signals_gpu.T @ dS_prime_dsigma_columnmajor_gpu)
    best_fit_proj_gpu = cp.argmax(e_gpu **2, axis=1)     #<<<<----find the max. element's index for the rows along their columns (that's why axis=1)
    
    #######################-------------------------- Synchronize and release memory--------#######################################################
    cp.cuda.Device().synchronize()

    ###########################################--------Refined Search------------###############################################################          
    results = []
    best_fit_proj_cpu = cp.asnumpy(best_fit_proj_gpu)
    de_dx_full_cpu = cp.asnumpy(de_dx_full_gpu)
    de_dy_full_cpu = cp.asnumpy(de_dy_full_gpu)
    de_dsigma_full_cpu = cp.asnumpy(de_dsigma_full_gpu)

    for y_idx in range(Y_signals_cpu.shape[1]):
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

        results.append(y_idx)
        results[y_idx] = refined_X
        # results.append(f"{y_idx}: {refined_X}")

    return results

# Generate Gaussian - CPU
def generate_2d_gaussian(meshgrid_X, meshgrid_Y, mean_x, mean_y, sigma):                    
    mean_corrdinates = [mean_x, mean_y]        
    Z = np.exp(-(meshgrid_X - mean_corrdinates[0])**2 / (2 * sigma**2)) * np.exp(-(meshgrid_Y - mean_corrdinates[1])**2 / (2 * sigma**2))
    return Z

def args2jsonEntry(muX, muY, sigma, r2, signal):
    json_entry = {
                "Centerx0": muX,
                "Centery0": muY,
                "Theta": 0,
                "sigmaMajor": sigma,
                "sigmaMinor": 0,
                "R2": r2,
                "modelpred": signal.tolist()
            }
    return json_entry

# GLM fitting: compute y_hat
def get_y_hat(y, s):

    s /= s.max()

    if s.ndim == 1:
        # Add back a singleton axis
        # otherwise stacking will give an error
        s = s[:, np.newaxis]

    # Predictors
    nPolynomials = 3
    # intercept = np.ones((model_signal.size, 1))
    trends = np.vstack([np.linspace(0, 1, len(y)) ** i for i in range(nPolynomials)]).T
    X = np.hstack((s, trends))
    betas = np.linalg.inv(X.T @ X) @ X.T @ y

    #Estimated y (y_hat): We will compute the MSE between this estimated "y_hat" and the observed/measured "y" (voxel_signal)
    y_hat = X@betas

    return y_hat, betas, trends

# R2 variance explained
def compute_r2_variance_explained_results(refined_matching_results):
    r2_arr = []
    json_data_results_with_r2 = []

    stim_data = stimulus.resampled_hrf_convolved_data.flatten('F')
    stim_data = cp.reshape(stim_data, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame
    # test_stim_data = (stim_data[:, 10]).reshape((stim_height, stim_width))

    mesh_X, mesh_Y = np.meshgrid(x_range_cpu, y_range_cpu)

    # default r2 value
    for i in range((len(refined_matching_results))):
        #print(f'processing-{i}...')
        y = Y_signals_cpu[:, i] #column vector
        muX, muY, sigma = refined_matching_results[i]
        gc = generate_2d_gaussian(mesh_X, mesh_Y, muX, muY, sigma) # mu_X_grid[:, :, 0] and mu_Y_grid[:, :, 0], because we have 3D meshgrids now due to variable sigma so, we are using only the first frame of the X and Y meshgrids
        s = gc.flatten() @ stim_data
        if (all(element == 0 for element in s)):
            r2 = -2
            json_entry = args2jsonEntry(muX, muY, sigma, r2, s)    
            json_data_results_with_r2.append(json_entry)  
            continue

        y_hat, betas, trends = get_y_hat(y, s)     
        if betas[0]>0:
            # r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)) # WORKED
            r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - trends@betas[1:]) ** 2))

            # r2 = 1 - ( np.sqrt(y_hat^2) / np.sqrt((y-trends@betas[1:])^2) )
            #r2 = 1- (np.sum(np.square(y - y_hat)) / np.sum(np.square(y)))
        else:
            r2 = -1

        json_entry = args2jsonEntry(muX, muY, sigma, r2, s)    
        json_data_results_with_r2.append(json_entry)    

        # only TEST
        r2_arr.append(r2)

    return json_data_results_with_r2


# ###################################-----------Main()---------------------------------####################
def main():
    refined_matching_results = gpu_compute()
    json_data = compute_r2_variance_explained_results(refined_matching_results)
    
    # Convert the list of dictionaries to a JSON string
    json_string = json.dumps(json_data, indent=4)

    # Write the JSON data to the file
    with open(cfg.result_file_path, 'w') as json_file:
        json_file.write(json_string)


if __name__ == "__main__":
    cProfile.run('main()', sort='cumulative')


# # cProfile.run('gpu_compute()', sort='cumulative')
# refined_matching_results = gpu_compute()
# json_data = compute_r2_variance_explained_results(refined_matching_results)

# # Convert the list of dictionaries to a JSON string
# json_string = json.dumps(json_data, indent=4)

# # Write the JSON data to the file
# with open(cfg.result_file_path, 'w') as json_file:
#     json_file.write(json_string)

# # # write to a file
# # with open(cfg.result_file_path, "w") as file:    
# #     for result in refined_matching_results:
# #         file.write(result + "\n")

print("done")




