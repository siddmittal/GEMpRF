import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cupy as cp

from oprf.hpc.hpc_matrix_operations import MatrixOps
from oprf.hpc.hpc_coefficient_matrix import CoefficientMatrix
from oprf.hpc.hpc_coefficient_matrix_NUMBA import ParallelComputedCoefficientMatrix
from oprf.hpc.hpc_cupy_utils import Utils

# debugging purpose
import nibabel as nib
import os

class RefineFit:
    @classmethod
    def get_refined_fit_results(cls, num_Y_signals, best_fit_proj_cpu, arr_2d_location_inv_M_cpu, e_full, de_dx_full, de_dy_full, de_dsigma_full, search_space_rows, search_space_cols, search_space_frames
                                , search_space_xx
                                , search_space_yy
                                , search_space_sigma_range
                                ):
        
        # # # only for debugging
        # temp_X , temp_Y, temp_sigma = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range) # NOTE: to be deleted
        # temp_X_2D , temp_Y_2D = np.meshgrid(search_space_xx, search_space_yy) # NOTE: to be deleted
        # temp_X , temp_Y = np.meshgrid(search_space_xx, search_space_yy) # NOTE: to be deleted
        # temp_sigma_x , temp_sigma = np.meshgrid(search_space_xx, search_space_sigma_range) # NOTE: to be deleted                

        results = []
        Fex_results = []
        # results_line_intersection = []
        # perform refine search
        for y_idx in range(num_Y_signals):
            best_s_idx = best_fit_proj_cpu[y_idx]            
            best_s_3d_idx = MatrixOps.flatIdx2ThreeDimIndices(best_s_idx, search_space_rows, search_space_cols, search_space_frames)

            # get the current signal and its neighboring signals indices
            block_3d_indices = ParallelComputedCoefficientMatrix.Wrapper_get_block_indices_with_Sigma(row= best_s_3d_idx[0]
                                                            , col=best_s_3d_idx[1]
                                                            , frame=best_s_3d_idx[2]
                                                            , nRows=search_space_rows
                                                            , nCols=search_space_cols
                                                            , nFrames=search_space_frames
                                                            , distance=1) #get_block_indices_new(row=best_s_2d_idx[0], col=best_s_2d_idx[1], nRows=search_space_rows, nCols=search_space_cols, distance=1)
            # block_3d_indices = CoefficientMatrix.get_block_indices_with_Sigma(row=best_s_3d_idx[0]
            #                                                 , col=best_s_3d_idx[1]
            #                                                 , frame=best_s_3d_idx[2]
            #                                                 , nRows=search_space_rows
            #                                                 , nCols=search_space_cols
            #                                                 , nFrames=search_space_frames
            #                                                 , distance=1) #get_block_indices_new(row=best_s_2d_idx[0], col=best_s_2d_idx[1], nRows=search_space_rows, nCols=search_space_cols, distance=1)
            block_flat_indices = (MatrixOps.threeDimIndices2FlatIdx(threeDimIdx=block_3d_indices, nRows=search_space_rows, nCols=search_space_cols)).astype(int)

            # compute the coffeficients
            #...get the pre-computed Mp Inverse matrix (already containing information about the neighbors)
            MpInv = arr_2d_location_inv_M_cpu[best_s_idx]        
        
            #...compute the de/dx, de/dy and de/dsigma vectors# 
            # e_vec_cpu = e_full_cpu[y_idx, block_flat_indices] ** 2 # NOTE: ** 2 is required if we are taking error term as (yts)^2
            e_vec = e_full[y_idx, block_flat_indices]
            de_dx_vec = de_dx_full[y_idx, block_flat_indices]
            de_dy_vec = de_dy_full[y_idx, block_flat_indices]
            de_dsigma_vec = de_dsigma_full[y_idx, block_flat_indices]
            vec = (np.vstack( (e_vec, de_dx_vec, de_dy_vec, de_dsigma_vec) )).ravel(order = 'F') # <<<----MIND, it's capital X in de_dX. Capital X means the complete vector i.e. [ux1, uy1, ux2, uy2, ux3, uy3...]
            if isinstance(vec, cp.ndarray): # in case of concatenated runs, the "vec" is on GPU
                vec = cp.asnumpy(vec)

            coefficients = MpInv@vec
            # A, B = CoefficientMatrix.create_cofficients_matrices_A_and_B(coefficients)
            A, B, C = CoefficientMatrix.create_cofficients_matrices_A_B_and_C(coefficients)
            try:
                #refined_params_vec = -0.5 * (np.linalg.inv(A) @ B)
                refined_params_vec = np.linalg.solve(2*A, -1 * B) # solve for X the derivative equation, 2AX + B = 0 ==> 2AX = -B
                
                # for DEBUG
                if(False):
                    coarse_gradients = [e_full[y_idx, best_s_idx], de_dx_full[y_idx, best_s_idx], de_dy_full[y_idx, best_s_idx], de_dsigma_full[y_idx, best_s_idx]]
                    dummy2 = 0

                    coarse_x_idx = best_s_3d_idx[0][0]
                    y_idx = best_s_3d_idx[1][0]
                    sigma_idx = best_s_3d_idx[2][0]

                    x = temp_X[coarse_x_idx, y_idx]
                    y = temp_Y[coarse_x_idx, y_idx]
                    sigma = search_space_sigma_range[sigma_idx]

                    best_coarse = [x, y, sigma]

                    # compute refined gradients
                    refined_x = refined_params_vec[0]
                    refined_y = refined_params_vec[1]
                    refined_s = refined_params_vec[2]
                    x_mean = x # because x is directly coming from the meshgrid, which contains mean values
                    y_mean = y 
                    exponent = -((refined_x - x_mean) * (refined_x - x_mean) + (refined_y - y_mean) * (refined_x - y_mean)) / (2 * refined_s * refined_s)
                    result_Dsigma_gaussian_curves = -(((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (sigma * sigma * sigma)) * exp(exponent)                

                if(False):
                    # debugging
                    best_coarse_sigma_idx = best_s_3d_idx[2][0]
                    ibcs = best_coarse_sigma_idx
                    l_ibcs = ibcs - 1 # left nighbor                    
                    r_ibcs = ibcs + 1 # right neighbor

                    f = plt.figure()

                    # dx, dy
                    quiver = plt.gca().quiver(temp_X , temp_Y, -((de_dx_full[y_idx])[((l_ibcs-1)*51*51):(l_ibcs*51*51)]).reshape(51, 51), -((de_dy_full[y_idx])[((l_ibcs-1)*51*51):(l_ibcs*51*51)]).reshape(51, 51), angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5), color='g')
                    quiver = plt.gca().quiver(temp_X , temp_Y, -((de_dx_full[y_idx])[((ibcs-1)*51*51):(ibcs*51*51)]).reshape(51, 51), -((de_dy_full[y_idx])[((ibcs-1)*51*51):(ibcs*51*51)]).reshape(51, 51), angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5), color='r')
                    quiver = plt.gca().quiver(temp_X , temp_Y, -((de_dx_full[y_idx])[((r_ibcs-1)*51*51):(r_ibcs*51*51)]).reshape(51, 51), -((de_dy_full[y_idx])[((r_ibcs-1)*51*51):(r_ibcs*51*51)]).reshape(51, 51), angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5), color='b')
                    
                    # # 3D quiver plot
                    # temp2_X , temp2_Y, temp2_Z = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range) # NOTE: to be deleted
                    # plt.figure().add_subplot(projection='3d')
                    # plt.gca().quiver(temp2_X , temp2_Y, temp2_Z, -de_dx_full_cpu[y_idx].reshape(51, 51, 16), -de_dy_full_cpu[y_idx].reshape(51, 51, 16), -de_dsigma_full_cpu[y_idx].reshape(51, 51, 16), length=0.1, normalize=True)
                    
                    dummy = 0
                # if(False and (y_idx == 789 or y_idx == 881 or y_idx == 958 or y_idx == 342 or y_idx == 345)): # 342 and 345 are good indices
                # if(True and (y_idx == 4553 or y_idx == 4965 or y_idx == 20027)): # 342 and 345 are good indices
                #if(True and (y_idx == 1273 or y_idx == 3483 or y_idx == 5176 or y_idx == 6512)): # 1273 is good index
                # if(False and (y_idx == 1273 or y_idx == 1667 or y_idx == 5505 \
                #              or y_idx == 3483 or y_idx == 5176 or y_idx == 6512 or y_idx == 13658 or y_idx == 14059)): # 1273, 1667 and 5505 are good indices, and 3483, 5176, 6512, 13658, 14059 are low error worsening cases

                if (False):
                # if (y_idx == 11 or y_idx == 176 or y_idx == 459 or y_idx == 503 or y_idx == 1763): # small dataset, bad locations
                # if (y_idx == 252 or y_idx == 257 or y_idx == 333 or y_idx == 334 or y_idx == 341 or y_idx == 345 or y_idx == 445 or y_idx == 446 or y_idx == 449): # small dataset, good locations
                    coarse_gradients = [e_full[y_idx, best_s_idx], de_dx_full[y_idx, best_s_idx], de_dy_full[y_idx, best_s_idx], de_dsigma_full[y_idx, best_s_idx]]

                    coarse_x_idx = best_s_3d_idx[0]
                    coarse_y_idx = best_s_3d_idx[1]
                    sigma_idx = best_s_3d_idx[2]

                    x = temp_X[coarse_x_idx, coarse_y_idx, sigma_idx]
                    y = temp_Y[coarse_x_idx, coarse_y_idx, sigma_idx]
                    sigma = search_space_sigma_range[sigma_idx]

                    coarse_estimations = [x, y, sigma]

                    # compute refined gradients
                    refined_x = refined_params_vec[0]
                    refined_y = refined_params_vec[1]
                    refined_s = refined_params_vec[2]
                    x_mean = x # because x is directly coming from the meshgrid, which contains mean values
                    y_mean = y 
                    refined_estimations = [refined_x, refined_y, refined_s]

                    # debugging
                    best_coarse_sigma_idx = best_s_3d_idx[2]
                    ibcs = best_coarse_sigma_idx
                    l_ibcs = ibcs - 1 # left nighbor                    
                    r_ibcs = ibcs + 1 # right neighbor

                    # # contour plot
                    # plt.figure()
                    # plt.gca().contour(temp_X_2D , temp_Y_2D, ((e_full[y_idx])[((ibcs)*51*51):((ibcs+1)*51*51)]).reshape(51, 51))
                    # plt.plot(x,y,'bo') 
                    # plt.plot(refined_x,refined_y,'gx')
                    # plt.title(f'idx: {y_idx}')
                    
                    # gradients plot
                    # dx, dy
                    # f = plt.figure()
                    # quiver = plt.gca().quiver(temp_X_2D , temp_Y_2D, ((de_dx_full[y_idx])[((l_ibcs)*51*51):((l_ibcs+1)*51*51)]).reshape(51, 51), ((de_dy_full[y_idx])[((l_ibcs-1)*51*51):(l_ibcs*51*51)]).reshape(51, 51), angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5), color='g')
                    # quiver = plt.gca().quiver(temp_X_2D , temp_Y_2D, ((de_dx_full[y_idx])[((ibcs)*51*51):((ibcs+1)*51*51)]).reshape(51, 51), ((de_dy_full[y_idx])[((ibcs-1)*51*51):(ibcs*51*51)]).reshape(51, 51), angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5), color='r')
                    # quiver = plt.gca().quiver(temp_X_2D , temp_Y_2D, ((de_dx_full[y_idx])[((r_ibcs)*51*51):((r_ibcs+1)*51*51)]).reshape(51, 51), ((de_dy_full[y_idx])[((r_ibcs-1)*51*51):(r_ibcs*51*51)]).reshape(51, 51), angles='xy', scale_units='xy', norm=plt.Normalize(-5, 5), color='b')
                    # plt.plot(x,y,'bo') 
                    # plt.plot(refined_x,refined_y,'gx') 

                    # # gradients debugging - plotting only gradients
                    # plt.figure()
                    # plt.gca().quiver(np.arange(0,27), e_vec, np.arange(0,27), de_dx_vec, color='r')
                    # plt.gca().quiver(np.arange(0,27), e_vec, np.arange(0,27), de_dx_vec, color='r')
                    # plt.gca().quiver(np.arange(0,27), e_vec, np.arange(0,27), de_dy_vec, color='g')
                    # plt.gca().quiver(np.arange(0,27), e_vec, np.arange(0,27), de_dsigma_vec, color='b')
                    # plt.plot(e_vec)
                    # best_e_vec = e_full[y_idx, best_s_idx]
                    # fex = cls.compute_fx(refined_params_vec, A, B, C) 
                    # plt.axhline(best_e_vec, color='r')
                    # plt.axhline(fex, color='g')
                    # plt.title(f'y_idx = {y_idx}')

                    # # debugging - compute f(x) at all X-values in the neighbourhood using the coefficients
                    # fex_whole_block = []
                    # for vec_idx in range(len(block_3d_indices)):
                    #     myVec = np.array([temp_X[block_3d_indices[vec_idx][0], block_3d_indices[vec_idx][1], block_3d_indices[vec_idx][2]],
                    #                      temp_Y[block_3d_indices[vec_idx][0], block_3d_indices[vec_idx][1], block_3d_indices[vec_idx][2]],
                    #                      temp_sigma[block_3d_indices[vec_idx][0], block_3d_indices[vec_idx][1], block_3d_indices[vec_idx][2]]])
                    #     myFex = cls.compute_fx(myVec, A, B, C)                                                
                    #     fex_whole_block.append(myFex)
                    # fex_whole_block = np.array(fex_whole_block)
                    # plt.figure()
                    # plt.title(f'y_idx = {y_idx}')
                    # plt.plot(fex_whole_block)
                    # plt.plot(e_vec)
                    # best_e_vec = e_full[y_idx, best_s_idx]
                    # plt.axhline(best_e_vec, color='r')
                    # plt.axhline(fex, color='g')
                     

                    # # # store all Gradients related debugging info to a Nifti file
                    # cls.save_data_to_nifiti(selected_y_signal_idx=y_idx, X_3d=temp_X, Y_3d=temp_Y, Sigma_3d=temp_sigma, selected_signal_e=e_full[y_idx, :], 
                    #                         selected_signal_de_dx=de_dx_full[y_idx, :],
                    #                         selected_signal_de_dy=de_dy_full[y_idx, :],
                    #                         selected_signal_de_dsigma=de_dsigma_full[y_idx, :],
                    #                         A=A, B=B, C=C)
                                        
                    d = 0

            except:
                refined_params_vec = np.array([np.nan, np.nan, np.nan])
                
            results.append(y_idx)
            results[y_idx] = refined_params_vec
            
            fex = cls.compute_fx(refined_params_vec, A, B, C)
            Fex_results.append(fex)

            # # Method-2: Line–line intersection                        
            # mu_X_vec = temp_X[block_3d_indices[:, 0], block_3d_indices[:, 1], block_3d_indices[:, 2]]
            # mu_Y_vec = temp_Y[block_3d_indices[:, 0], block_3d_indices[:, 1], block_3d_indices[:, 2]]
            # sigma_vec = temp_sigma[block_3d_indices[:, 0], block_3d_indices[:, 1], block_3d_indices[:, 2]]

            # num_dimensions = 3            
            # G = (np.column_stack( (de_dx_vec, de_dy_vec, de_dsigma_vec) ))#.ravel(order = 'F')            
            # G_norm = np.linalg.norm(G, axis=1)
            # G_norm = np.maximum(G_norm, 1e-6)
            # N = G / G_norm[:,None]
                        
            # X = (np.column_stack( (mu_X_vec, mu_Y_vec, sigma_vec) )) #block_3d_indices
            # R = np.sum([ np.eye(num_dimensions) - np.outer(n, n)      for n    in     N    ], axis=0) 
            # q = np.sum([(np.eye(num_dimensions) - np.outer(n, n)) @ x for n, x in zip(N, X)], axis=0)
            # m_est = np.linalg.solve(R, q)
            # results_line_intersection.append(m_est)



        return results, Fex_results # results_line_intersection #Fex_results            

        # return results, Fex_results
    
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