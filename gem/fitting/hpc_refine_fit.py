import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cupy as cp

# gem
from gem.space.PRFSpace import PRFSpace

from gem.fitting.hpc_coefficient_matrix import CoefficientMatrix
from gem.utils.hpc_cupy_utils import HpcUtils as Utils

# debugging purpose
import nibabel as nib
import os

class RefineFit:
    @classmethod
    def get_refined_fit_results(cls, prf_space : PRFSpace, num_Y_signals, best_fit_proj_cpu, arr_2d_location_inv_M_cpu, e_full, de_dtheta_list_cpu):        
        #NOTE: for DEBUG info, look into the old-gem-files folder
        ONLY_SIGNLE_SIGNAL = False
        if(num_Y_signals == 1):
            ONLY_SIGNLE_SIGNAL = True

        de_dtheta_list_cpu = np.array(de_dtheta_list_cpu).squeeze()
        num_params = len(de_dtheta_list_cpu)
        results = np.zeros((num_Y_signals, num_params), dtype=np.float64)

        Fex_results = []
        # perform refine search
        for y_idx in range(num_Y_signals):
            best_s_idx = best_fit_proj_cpu[y_idx]            
            block_flat_indices = prf_space.multi_dim_points_neighbours_flat_indices[best_s_idx]            

            # NOTE: in case of validating the pRF points, the number of total multi-dimensional points used to compute model signals are changed. 
            # However, the neighbours indices represent the indices for the full multi-points array. 
            # Therefore, we need to map the indices corresponding to full-array to indices corresponding to the validated points array.
            block_flat_indices = prf_space.get_full_2_validated_indices(block_flat_indices) 

            # compute the coffeficients
            #...get the pre-computed Mp Inverse matrix (already containing information about the neighbors)
            MpInv = arr_2d_location_inv_M_cpu[best_s_idx] # NOTE: This is the correct way for the program, but for debugging, I am not using it and computing it again       
        
            #...compute the de/dx, de/dy and de/dsigma vectors# 
            e_vec = e_full[y_idx, block_flat_indices] # NOTE: ** 2 is required if we are taking error term as (yts)^2      
            if len(e_vec) == 1: # i.e. no other neighbours
                vec = e_vec.squeeze()
            else:              
                vec = np.vstack((e_vec.squeeze())).squeeze()   
                non_NaN_e_row_indices = cls.get_non_nan_row_indices(vec)         
            for theta in range(num_params):
                if(ONLY_SIGNLE_SIGNAL):     
                    vec = np.vstack((vec, (de_dtheta_list_cpu[theta, block_flat_indices].T)[0]))
                else:
                    vec = np.vstack((vec, (de_dtheta_list_cpu[theta, y_idx, block_flat_indices].T)[0]))

            vec = vec[:, non_NaN_e_row_indices]
            vec = vec.T.reshape(-1)

            # in case of concatenation runs, we have `vec` as cupy array
            if type(vec) is cp.ndarray:
                vec = cp.asnumpy(vec)

            # compute non_nan_indices for M matrix            
            if len(non_NaN_e_row_indices) != len(block_flat_indices): # i.e. some of the indices are NaN
                non_NaN_e_row_indices = cp.asnumpy(non_NaN_e_row_indices)
                good_indices_M = []
                num_linear_equations = num_params + 1
                for i in non_NaN_e_row_indices:
                    good_indices_M.extend(range(i * num_linear_equations, (i + 1) * num_linear_equations))
                MpInv = MpInv[:, good_indices_M]
                                
            # compute the coefficients
            coefficients = MpInv@vec                        
            A, B, C = CoefficientMatrix.create_cofficients_matrices_A_B_and_C(coefficients)
            try:
                refined_params_vec = np.linalg.solve(2*A, -1 * B) # solve for X the derivative equation, 2AX + B = 0 ==> 2AX = -B                
            except:
                refined_params_vec = np.array([np.nan, np.nan, np.nan])
                
            # update results    
            results[y_idx, :] = refined_params_vec
            
            fex = cls.compute_fx(refined_params_vec, A, B, C)
            Fex_results.append(fex)

        return results, Fex_results             

    @classmethod
    def get_non_nan_row_indices(cls, input_array):
        package = None
        if type(input_array) is cp.ndarray:
            package = cp
        elif type(input_array) is np.ndarray:
            package = np
        
        if input_array.ndim == 1:
            return package.where(~package.isnan(input_array))[0]
        else:
            return package.where(~package.isnan(input_array).any(axis=1))[0]


    @classmethod
    def compute_fx(cls, refined_X, A, B, C):
        # e = X.T @ (A @ X) + B@X + C

        X = np.asarray(refined_X)

        # X = np.asarray([muX, muY, sigma])
        fex = X.T @ (A @ X) + B@X + C

        return fex    
    
    @classmethod
    def compute_fx_and_gradients(cls, refined_X, A, B, C):
        # e = X.T @ (A @ X) + B@X + C

        X = np.asarray(refined_X)

        # X = np.asarray([muX, muY, sigma])
        fex = X.T @ (A @ X) + B@X + C
        gradient_fex = 2 * (A @ X) + B

        return fex, gradient_fex

    #############################################---------------------Debug related code------------------------------######################################################
    #############################################---------------------Debug related code------------------------------######################################################
    #########----for DEBUG
    @classmethod
    def save_data_to_nifiti(cls, selected_y_signal_idx, X_3d, Y_3d, Sigma_3d, selected_signal_e, selected_signal_de_dx, selected_signal_de_dy, selected_signal_de_dsigma, A, B, C):
        dir_path = r'D:\results\gradients-test\saved-debug-data'        

        # dimensions
        nRows, nCols, nSigma = X_3d.shape

        # affine transform matrix
        dx = X_3d[0, 1, 0] - X_3d[0, 0, 0] 
        dy = Y_3d[1, 0, 0] - Y_3d[0, 0, 0] 
        ds = Sigma_3d[0, 0, 1] - Sigma_3d[0, 0, 0] 
        Ox = X_3d[0, 0, 0]
        Oy = Y_3d[0, 0, 0]
        Os = Sigma_3d[0, 0, 0]
        affine_mat = np.array([[dx, 0, 0, Ox], 
                               [0, dy, 0, Oy], 
                               [0, 0, ds, Os], 
                               [0, 0, 0 , 1]])

        # data = np.ones((32, 32, 15, 100), dtype=np.float64)
        # img = nib.Nifti1Image(data, np.eye(4))
        # img.set_data_dtype(np.dtype(np.float64))
        # nib.save(img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error.nii.gz')))

        # save error data
        # plt.figure(); plt.gca().contour(X_3d[:, :, 0] , Y_3d[:, :, 0] , coarse_error[:, :, 0])
        coarse_error = (selected_signal_e.reshape((nRows, nCols, nSigma), order='F'))
        coarse_error_img = nib.Nifti1Image(coarse_error, affine=affine_mat)
        coarse_error_img.set_data_dtype(np.dtype(np.float64))
        nib.save(coarse_error_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_coarse_error.nii.gz')))

        # gradients data
        de_dx = (selected_signal_de_dx.reshape((nRows, nCols, nSigma), order='F'))
        de_dy = (selected_signal_de_dy.reshape((nRows, nCols, nSigma), order='F'))
        de_dsigma = (selected_signal_de_dsigma.reshape((nRows, nCols, nSigma), order='F'))
        gradients = np.array([de_dx, de_dy, de_dsigma])  
        gradients  = np.moveaxis(gradients, 0, -1)
        gradients_img = nib.Nifti1Image(gradients, affine=affine_mat)
        gradients_img.set_data_dtype(np.dtype(np.float64))
        nib.save(gradients_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error_gradients.nii.gz')))

        # quad fitting error e = f(x) AND its gradient (gradients_fex = 2AX + B)
        fex_error = []
        gradients_fex = []
        fex_dx = []        
        fex_dy = []        
        fex_dsigma = []        
        for sigma  in range(nSigma):
            for row in range(nRows):
                for col in range(nCols):                                    
                            X = np.array([X_3d[row, col, sigma], Y_3d[row, col, sigma], Sigma_3d[row, col, sigma]])
                            fex, grad_fex = cls.compute_fx_and_gradients(X, A, B, C)
                            fex_error.append(fex)
                            # gradients_fex.append(grad_fex)
                            fex_dx.append(grad_fex[0])
                            fex_dy.append(grad_fex[1])
                            fex_dsigma.append(grad_fex[2])

        # save f(x)
        fex_error = (np.array(fex_error)).reshape((nRows, nCols, nSigma), order='F')                        
        fex_img = nib.Nifti1Image(fex_error, affine=affine_mat)
        fex_img.set_data_dtype(np.dtype(np.float64))
        nib.save(fex_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error_fex_weighted.nii.gz')))

        # save gradients_fex = 2AX + B    
        fex_dx = np.array([fex_dx])
        fex_dy = np.array([fex_dy])
        fex_dsigma= np.array([fex_dsigma])
        dfx_dx = (fex_dx.reshape((nRows, nCols, nSigma), order='F'))
        dfx_dy = (fex_dy.reshape((nRows, nCols, nSigma), order='F'))
        dfx_dsigma = (fex_dsigma.reshape((nRows, nCols, nSigma), order='F'))
        gradients_fex = np.array([dfx_dx, dfx_dy, dfx_dsigma])
        gradients_fex = np.moveaxis(gradients_fex, 0, -1)
        grad_fex_img = nib.Nifti1Image(gradients_fex, affine=affine_mat)
        grad_fex_img.set_data_dtype(np.dtype(np.float64))
        nib.save(grad_fex_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error_gradients_fex_weighted.nii.gz')))        

        print()

    @classmethod
    def get_all_debug_info_error_terms_after_refinement(cls, refined_matching_results, Y_signals_gpu, O_gpu, stimulus, stim_height, stim_width, x_range_cpu, y_range_cpu):        
        refined_matching_results_arr = np.array(refined_matching_results)
        muX_arr, muY_arr, sigma_arr = refined_matching_results_arr[: , 0], refined_matching_results_arr[: , 1], refined_matching_results_arr[: , 2]
        
        gaussian_cuda_module = Utils.get_raw_module('gaussian_using_arrays_kernel.cu')
        gc_kernel = gaussian_cuda_module.get_function("gc_using_args_arrays_cuda_Kernel")
        dgc_dx_using_args_arrays_cuda_Kernel = gaussian_cuda_module.get_function("dgc_dx_using_args_arrays_cuda_Kernel")
        dgc_dy_using_args_arrays_cuda_Kernel = gaussian_cuda_module.get_function("dgc_dy_using_args_arrays_cuda_Kernel")
        dgc_dsigma_using_args_arrays_cuda_Kernel = gaussian_cuda_module.get_function("dgc_dsigma_using_args_arrays_cuda_Kernel")

        # initialize result curves
        total_num_gc = len(refined_matching_results)
        result_gc_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)
        result_dgc_dx_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)
        result_dgc_dy_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)
        result_dgc_dsigma_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)

        # kernel grid
        block_dim = (32, 1, 1)
        bx = int((total_num_gc + block_dim[0] - 1) / block_dim[0])
        by = 1
        bz = 1
        grid_dim = (bx, by, bz)

        # launch kernel - gc
        gc_kernel(grid_dim, block_dim, (
        result_gc_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # launch kernel - dgc_dx
        dgc_dx_using_args_arrays_cuda_Kernel(grid_dim, block_dim, (
        result_dgc_dx_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # launch kernel - dgc_dy
        dgc_dy_using_args_arrays_cuda_Kernel(grid_dim, block_dim, (
        result_dgc_dy_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # launch kernel - dgc_dsigma
        dgc_dsigma_using_args_arrays_cuda_Kernel(grid_dim, block_dim, (
        result_dgc_dsigma_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # timecourses
        stim_data = stimulus.get_flattened_columnmajor_stimulus_data_gpu()
        nRows_gaussian_curves_matrix = total_num_gc
        nCols_gaussian_curves_matrix = stim_height * stim_width  

        #.....reshape GC
        gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
        dgc_dx_rowmajor_gpu = cp.reshape(result_dgc_dx_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) 
        dgc_dy_rowmajor_gpu = cp.reshape(result_dgc_dy_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) 
        dgc_dsigma_rowmajor_gpu = cp.reshape(result_dgc_dsigma_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) 
        
        #....compute timecourses
        refined_S_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stim_data)
        refined_dS_dx_rowmajor_gpu = cp.dot(dgc_dx_rowmajor_gpu, stim_data)
        refined_dS_dy_signals_rowmajor_gpu = cp.dot(dgc_dy_rowmajor_gpu, stim_data)
        refined_dS_dsigma_signals_rowmajor_gpu = cp.dot(dgc_dsigma_rowmajor_gpu, stim_data)

        #.....orthonormalize
        S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu = cls._get_orthonormalize_refined_signals_for_debug(O_gpu, 
                                                                                                                                                                     refined_S_rowmajor_gpu, 
                                                                                                                                                                     refined_dS_dx_rowmajor_gpu, 
                                                                                                                                                                     refined_dS_dy_signals_rowmajor_gpu, 
                                                                                                                                                                     refined_dS_dsigma_signals_rowmajor_gpu)

        # S        
        e_gpu = cls._compute_refined_error_term_for_debug(O_gpu, Y_signals_gpu, S_prime_columnmajor_gpu)
   
        # dS_dx        
        de_dx_full_gpu = cls._compute_refined_derivative_error_term_cpu_for_debug(O_gpu, Y_signals_gpu, dS_prime_dx_columnmajor_gpu, e_gpu)

        # dS_dy
        de_dy_full_gpu = cls._compute_refined_derivative_error_term_cpu_for_debug(O_gpu, Y_signals_gpu, dS_prime_dy_columnmajor_gpu, e_gpu)
        
        # dS_dsigma
        de_dsigma_full_gpu = cls._compute_refined_derivative_error_term_cpu_for_debug(O_gpu, Y_signals_gpu, dS_prime_dsigma_columnmajor_gpu, e_gpu)
        
        # cpu
        e_cpu = cp.asnumpy(e_gpu)
        de_dx_full_cpu = cp.asnumpy(de_dx_full_gpu)
        de_dy_full_cpu = cp.asnumpy(de_dy_full_gpu)
        de_dsigma_full_cpu = cp.asnumpy(de_dsigma_full_gpu)

        return e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu
    
    @classmethod
    def get_error_terms_after_refinement(cls, refined_matching_results, Y_signals_gpu, O_gpu, stimulus, stim_height, stim_width, x_range_gpu, y_range_gpu):        
        refined_matching_results_arr = np.array(refined_matching_results)
        muX_arr, muY_arr, sigma_arr = refined_matching_results_arr[: , 0], refined_matching_results_arr[: , 1], refined_matching_results_arr[: , 2]
        
        gaussian_cuda_module = Utils.get_raw_module('gaussian_using_arrays_kernel.cu')
        gc_kernel = gaussian_cuda_module.get_function("gc_using_args_arrays_cuda_Kernel")

        # initialize result curves
        total_num_gc = len(refined_matching_results)
        result_gc_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)

        # kernel grid
        block_dim = (32, 1, 1)
        bx = int((total_num_gc + block_dim[0] - 1) / block_dim[0])
        by = 1
        bz = 1
        grid_dim = (bx, by, bz)

        # launch kernel - gc
        gc_kernel(grid_dim, block_dim, (
        result_gc_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        x_range_gpu,
        y_range_gpu,        
        stim_height,
        stim_width,
        total_num_gc))

        # timecourses
        stim_data = stimulus.get_flattened_columnmajor_stimulus_data_gpu()
        nRows_gaussian_curves_matrix = total_num_gc
        nCols_gaussian_curves_matrix = stim_height * stim_width  

        #.....reshape GC
        gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
        
        #....compute timecourses
        refined_S_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stim_data)

        #.....orthonormalize
        S_prime_columnmajor_gpu = cls._get_orthonormalize_refined_signals(O_gpu, refined_S_rowmajor_gpu)

        # S        
        e_gpu = cls._compute_refined_error_term_for_debug(O_gpu, Y_signals_gpu, S_prime_columnmajor_gpu)
   
        # cpu
        e_cpu = cp.asnumpy(e_gpu)

        return e_cpu
    
    @classmethod
    def _get_orthonormalize_refined_signals_for_debug(cls, O_gpu, S_rowmajor_gpu, dS_dx_rowmajor_gpu, dS_dy_rowmajor_gpu, dS_dsigma_rowmajor_gpu):
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
        
        return S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu   

    @classmethod
    def _get_orthonormalize_refined_signals(cls, O_gpu, S_rowmajor_gpu):
        # orthogonalization + nomalization of signals/timecourses (present along the columns)
        S_star_columnmajor_gpu = cp.dot(O_gpu, S_rowmajor_gpu.T)
        S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns) 
        S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals
    
        return S_prime_columnmajor_gpu

    @classmethod
    def _compute_refined_error_term_for_debug(cls, O_gpu, Y_signals_gpu, S_prime_columnmajor_gpu):
        e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
        return e_gpu
    
    @classmethod
    def _compute_refined_derivative_error_term_cpu_for_debug(cls, O_gpu, Y_signals_gpu, dS_prime_dtheta_columnmajor_gpu, e_gpu):
        de_dtheta_gpu = (Y_signals_gpu.T @ dS_prime_dtheta_columnmajor_gpu)
        de_dtheta_cpu = cp.asnumpy(de_dtheta_gpu)
        return de_dtheta_cpu