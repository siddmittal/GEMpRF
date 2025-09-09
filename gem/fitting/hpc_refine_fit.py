import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cupy as cp

# gem
from gem.space.PRFSpace import PRFSpace

from gem.fitting.hpc_coefficient_matrix import CoefficientMatrix
from gem.utils.hpc_cupy_utils import HpcUtils as Utils
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

# debugging purpose
import nibabel as nib
import os

class RefineFit:
    padded_arr_2d_location_inv_M = None
    padded_multi_dim_points_neighbours_flat_indices = None

    @classmethod
    def _prepare_padded_arrays(cls, arr_2d_location_inv_M_cpu, prf_space, on_gpu):
        """Prepare padded arrays for arr_2d_location_inv_M and multi_dim_points_neighbours."""
        pkg = (np, cp)[on_gpu]

        # --- 1a) Prepare arr_2d_location_inv_M ---
        arr_2d_location_inv_M_cpu_list = arr_2d_location_inv_M_cpu
        N = len(arr_2d_location_inv_M_cpu_list)
        R = arr_2d_location_inv_M_cpu_list[0].shape[0]
        cols = np.array([a.shape[1] for a in arr_2d_location_inv_M_cpu_list], dtype=int)
        max_cols = int(cols.max())

        if (cols == max_cols).all():
            padded_arr_cpu = np.stack(arr_2d_location_inv_M_cpu_list, axis=0)
        else:
            # vectorized padding using insert
            cumsum_cols = np.cumsum(cols)
            pad_lens = max_cols - cols
            where_to_pad = np.repeat(cumsum_cols, pad_lens) if pad_lens.sum() > 0 else np.array([], dtype=int)

            padded_rows = []
            for r in range(R):
                row_concat = np.concatenate([a[r] for a in arr_2d_location_inv_M_cpu_list])
                if where_to_pad.size:
                    row_padded = np.insert(row_concat, where_to_pad, np.nan)
                else:
                    row_padded = row_concat
                padded_rows.append(row_padded.reshape(N, max_cols))
            padded_arr_cpu = np.stack(padded_rows, axis=1)

        # --- 1b) Prepare multi_dim_points_neighbours_flat_indices ---
        neigh_list = [np.asarray(a).ravel() for a in prf_space.multi_dim_points_neighbours_flat_indices]
        lens = np.array([a.size for a in neigh_list], dtype=int)
        max_len = int(lens.max())

        if (lens == max_len).all():
            padded_neigh_cpu = np.stack(neigh_list, axis=0)
        else:
            cumsum_lens = np.cumsum(lens)
            pad_lens2 = max_len - lens
            where_to_pad2 = np.repeat(cumsum_lens, pad_lens2) if pad_lens2.sum() > 0 else np.array([], dtype=int)
            all_concat = np.concatenate(neigh_list)
            padded_all = np.insert(all_concat, where_to_pad2, -1) if where_to_pad2.size else all_concat
            padded_neigh_cpu = padded_all.reshape(N, max_len)

        # --- Convert to GPU if requested ---
        if on_gpu:
            cls.padded_arr_2d_location_inv_M = cp.asarray(padded_arr_cpu)
            cls.padded_multi_dim_points_neighbours_flat_indices = cp.asarray(padded_neigh_cpu.astype(np.int64))[:, :, None]
        else:
            cls.padded_arr_2d_location_inv_M = padded_arr_cpu
            cls.padded_multi_dim_points_neighbours_flat_indices = padded_neigh_cpu.astype(np.int64)[:, :, None]


    @classmethod
    def get_refined_fit_results(cls, prf_space: PRFSpace, num_Y_signals, best_fit_proj,
                                arr_2d_location_inv_M_cpu, e_full, de_dtheta_3darr):
        on_gpu = isinstance(e_full, cp.ndarray)
        pkg = (np, cp)[on_gpu]

        # ---------- Step 1: Padded arrays (keep your optimized version) ----------
        if cls.padded_arr_2d_location_inv_M is None or cls.padded_multi_dim_points_neighbours_flat_indices is None:
            cls._prepare_padded_arrays(arr_2d_location_inv_M_cpu, prf_space, on_gpu)

        # ---------- Step 2: Gather indices ----------
        all_block_flat_indices = cls.padded_multi_dim_points_neighbours_flat_indices[best_fit_proj].squeeze()
        all_block_flat_indices_cpu = pkg.asnumpy(all_block_flat_indices) if on_gpu else all_block_flat_indices
        all_validated_block_flat_indices_cpu = prf_space.get_full_2_validated_indices(all_block_flat_indices_cpu, invalid_key_value=-1).reshape(all_block_flat_indices_cpu.shape)
        all_validated_block_flat_indices = pkg.asarray(all_validated_block_flat_indices_cpu)

        # ---------- Step 3: Gather MpInv ----------
        all_MpInv = cls.padded_arr_2d_location_inv_M[best_fit_proj]

        # ---------- Step 4-6: Compute coefficients in helper ----------
        # coefficients = MpInv@vec
        coefficients = cls._compute_coefficients(pkg, num_Y_signals, e_full, de_dtheta_3darr,
                                                all_validated_block_flat_indices, all_MpInv, on_gpu)

        # ---------- Step 7: Build A, B, C ----------
        A, B, C = CoefficientMatrix.create_cofficients_matrices_A_B_and_C_vectorized(coefficients)

        # ---------- Step 8: Solve system ----------
        refined_params_vecs_gpu = pkg.einsum('fij,fj->fi', pkg.linalg.pinv(2*A), -B) # as some of the A matrices might be singular so using pinv instead of inv

        return refined_params_vecs_gpu, None


    # ----------------- Helper functions -----------------
    @classmethod
    def _compute_coefficients(cls, pkg, num_Y_signals, e_full, de_dtheta_3darr, validated_indices, MpInv, on_gpu):
        """Compute coefficients with intermediate arrays freed early."""
        # Transpose de_dtheta to shape (num_Y_signals, num_models, num_params)
        de_dtheta_transposed = de_dtheta_3darr.transpose(1, 2, 0)
        e_full_expanded = e_full[:, :, pkg.newaxis]

        # Concatenate along last dim
        combined = pkg.concatenate([e_full_expanded, de_dtheta_transposed], axis=2) # shape: (num_Y_signals, num_models_signals, num_params+1)

        # Gather vecs
        # About validated_indices ...
        # ...it has shape (number_of_signals, max_possible_neighbours_per_prf)
        # ...it stores neighbor indices for each pRF
        # ...since some pRFs have fewer neighbors, validated_indices is padded with -1 for missing entries
        # ...during indexing, -1 is temporarily replaced (e.g., with 0) and then masked to NaN
        # vecs shape: (num_Y_signals, max_neighbors, num_params+1)
        vecs = combined[pkg.arange(num_Y_signals)[:, None], validated_indices.clip(min=0)] # replaces -1 with 0 temporarily, because negative indices would index from the end in Python
        vecs = pkg.where(validated_indices[..., None] == -1, pkg.nan, vecs) # Mask invalid indices

        # Flatten vecs
        flattened_vecs = vecs.reshape(vecs.shape[0], -1)

        # Free intermediate arrays early
        del combined, vecs
        # if on_gpu:
        #     cp._default_memory_pool.free_all_blocks()

        # Set NaNs to 0 so that we can do matrix multiplication and the values do not affect the result
        MpInv_masked = pkg.nan_to_num(MpInv, nan=0.0)
        vecs_masked = pkg.nan_to_num(flattened_vecs, nan=0.0)
        del flattened_vecs
        # if on_gpu:
        #     cp._default_memory_pool.free_all_blocks()

        # Compute coefficients
        coefficients = pkg.einsum('fvi,fi->fv', MpInv_masked, vecs_masked) # coefficients = MpInv@vec
        del MpInv_masked, vecs_masked
        # if on_gpu:
        #     cp._default_memory_pool.free_all_blocks()

        return coefficients

    @classmethod
    def get_refined_fit_results_simpler_padded_arrays(cls, prf_space : PRFSpace, num_Y_signals, best_fit_proj, arr_2d_location_inv_M_cpu, e_full, de_dtheta_3darr):  
        """This is just a leftover function which contains the simpler version of padded arrays creation. It is kept here just for reference."""
        on_gpu = isinstance(e_full, cp.ndarray)
  
        pkg = (np, cp)[isinstance(e_full, cp.ndarray)]
        # ---------- Step 1: Create padded arrays ON GPU ----------
        if cls.padded_arr_2d_location_inv_M is None or cls.padded_multi_dim_points_neighbours_flat_indices is None:
            # Pre-convert all arrays to GPU once before the loop
            arr_2d_location_inv_M_list = [pkg.asarray(a) for a in arr_2d_location_inv_M_cpu] if on_gpu else arr_2d_location_inv_M_cpu
            multi_dim_points_neighbours_flat_indices_list = [pkg.asarray(a) for a in prf_space.multi_dim_points_neighbours_flat_indices] if on_gpu else prf_space.multi_dim_points_neighbours_flat_indices

            num_total_model_signals = len(arr_2d_location_inv_M_cpu)
            num_rows_arr_2d_location_inv_M_cpu = arr_2d_location_inv_M_cpu[0].shape[0]
            num_cols_arr_2d_location_inv_M_cpu = max(arr.shape[1] for arr in arr_2d_location_inv_M_cpu)
            num_rows_multi_dim_points_neighbours_flat_indices = max(
                arr.shape[0] for arr in prf_space.multi_dim_points_neighbours_flat_indices
            )
            num_cols_multi_dim_points_neighbours_flat_indices = 1  # since these are 1D arrays

            # Allocate padded arrays directly on GPU
            cls.padded_arr_2d_location_inv_M = pkg.full(
                (num_total_model_signals, num_rows_arr_2d_location_inv_M_cpu, num_cols_arr_2d_location_inv_M_cpu),
                pkg.nan, dtype=pkg.float64
            )

            cls.padded_multi_dim_points_neighbours_flat_indices = pkg.full(
                (num_total_model_signals, num_rows_multi_dim_points_neighbours_flat_indices, num_cols_multi_dim_points_neighbours_flat_indices),
                -1, dtype=pkg.int64
            )

            # Fill GPU arrays
            for i in range(num_total_model_signals):
                cls.padded_arr_2d_location_inv_M[i, :arr_2d_location_inv_M_cpu[i].shape[0], :arr_2d_location_inv_M_cpu[i].shape[1]] = arr_2d_location_inv_M_list[i]
                cls.padded_multi_dim_points_neighbours_flat_indices[i, :prf_space.multi_dim_points_neighbours_flat_indices[i].shape[0], :prf_space.multi_dim_points_neighbours_flat_indices[i].shape[1]] = multi_dim_points_neighbours_flat_indices_list[i]

        # ---------- Step 2: Gather indices ----------
        all_block_flat_indices = cls.padded_multi_dim_points_neighbours_flat_indices[best_fit_proj].squeeze()
        all_block_flat_indices_cpu = pkg.asnumpy(all_block_flat_indices) if on_gpu else all_block_flat_indices
        all_validated_block_flat_indices_cpu = prf_space.get_full_2_validated_indices(all_block_flat_indices_cpu, invalid_key_value=-1).reshape(all_block_flat_indices_cpu.shape)
        all_validated_block_flat_indices = pkg.asarray(all_validated_block_flat_indices_cpu)

        # ---------- Step 3: Gather MpInv ----------
        all_MpInv = cls.padded_arr_2d_location_inv_M[best_fit_proj]

        # ---------- Step 4: Prepare de_dtheta + e_full ----------
        de_dtheta_transposed = de_dtheta_3darr.transpose(1, 2, 0)  # -> (num_Y_signals, num_models, num_params)
        e_full_expanded = e_full[:, :, pkg.newaxis]  # (num_Y_signals, num_models, 1)

        combined = pkg.concatenate([e_full_expanded, de_dtheta_transposed], axis=2)

        # ---------- Step 5: Gather vecs ----------
        vecs = combined[
            pkg.arange(num_Y_signals)[:, None],
            all_validated_block_flat_indices.clip(min=0)
        ]
        vecs = pkg.where(all_validated_block_flat_indices[..., None] == -1, pkg.nan, vecs)

        flattened_vecs = vecs.reshape(vecs.shape[0], -1)  # (num_Y_signals, (num_params+1)*max_neighbors)

        # ---------- Step 6: Compute coefficients ----------
        MpInv_masked = pkg.nan_to_num(all_MpInv, nan=0.0)
        vecs_masked = pkg.nan_to_num(flattened_vecs, nan=0.0)

        coefficients = pkg.einsum('fvi,fi->fv', MpInv_masked, vecs_masked)

        # ---------- Step 7: Build A, B, C ----------
        # Assuming this function supports CuPy input
        A, B, C = CoefficientMatrix.create_cofficients_matrices_A_B_and_C_vectorized(coefficients)

        # ---------- Step 8: Solve system ----------
        refined_params_vecs_gpu = pkg.einsum('fij,fj->fi', pkg.linalg.pinv(2*A), -B)

        return refined_params_vecs_gpu, None    


    @classmethod
    def get_refined_fit_results_cpu_loop_based(cls, prf_space : PRFSpace, num_Y_signals, best_fit_proj_cpu, arr_2d_location_inv_M_cpu, e_full, de_dtheta_list_cpu):  
        """This is a leftover function which contains the CPU version of the refine fit. It is kept here just for reference."""    
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
            block_flat_indices = prf_space.get_full_2_validated_indices(block_flat_indices, invalid_key_value=None)

            # compute the coffeficients
            #...get the pre-computed Mp Inverse matrix (already containing information about the neighbors)
            MpInv = arr_2d_location_inv_M_cpu[best_s_idx] # NOTE: This is the correct way for the program, but for debugging, I am not using it and computing it again       
        
            #...compute the de/dx, de/dy and de/dsigma vectors# 
            if type(e_full) is cp.ndarray:
                default_gpu_id = ggm.get_instance().default_gpu_id
                with cp.cuda.Device(default_gpu_id):
                    e_vec = e_full[y_idx, block_flat_indices] # NOTE: ** 2 is required if we are taking error term as (yts)^2    
            else:
                e_vec = e_full[y_idx, block_flat_indices]  
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
                with cp.cuda.Device(default_gpu_id):
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