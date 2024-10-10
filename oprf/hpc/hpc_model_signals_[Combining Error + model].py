
# import numpy as np
# import cupy as cp

# # Stimulus Class
# from oprf.standard.prf_stimulus import Stimulus

# # Receptive Field Response Class
# from oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse

# # Utilities
# from oprf.hpc.hpc_cupy_utils import Utils as gpu_utils
# from oprf.hpc.hpc_grid_fit import GridFit


# class ModelSignals:
#     def __init__(self, model_type, model_nRows, model_nCols, model_nFrames, model_visual_field, model_min_sigma, model_max_sigma,  stim_width, stim_height, stim_visual_field):
#         # model space
#         self.model_type = model_type
#         self.model_nCols = model_nRows
#         self.model_nRows = model_nCols
#         self.model_nFrames = model_nFrames
#         self.model_visual_field = model_visual_field
#         self.model_min_sigma = model_min_sigma
#         self.model_max_sigma = model_max_sigma

#         # stimulus related
#         self.stim_visual_field = stim_visual_field
#         self.stim_width = stim_width
#         self.stim_height = stim_height

#         # gaussian
#         self.total_gaussian_curves_per_stim_frame = model_nRows * model_nCols
#         self.single_gaussian_curve_length = stim_width * stim_height

#         # CUDA kernel launch config
#         self.block_dim = None
#         self.grid_dim = None

#     def set_kernel_config(self):
#         self.block_dim = (32, 32, 1)
#         bx = int((self.model_nRows + self.block_dim[0] - 1) / self.block_dim[0])
#         by = int((self.model_nRows + self.block_dim[1] - 1) / self.block_dim[1])
#         bz = int((self.model_nFrames + self.block_dim[2] - 1) / self.block_dim[2])
#         self.grid_dim = (bx, by, bz)


#     def _compute_timecourse(self, kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu):
#         if self.block_dim is None or self.grid_dim is None:
#             self.set_kernel_config()

#         # Launch Gaussian related kernels
#         #---compute gaussian curves
#         result_gc_curves_gpu = cp.zeros((self.model_nRows * self.model_nCols * self.model_nFrames * self.stim_width * self.stim_height), dtype=cp.float64)
#         kernel(self.grid_dim, self.block_dim, (
#         result_gc_curves_gpu,            
#         self.search_space_xx_gpu, # 51 points  (grid dimesion) between -13.5 to +13.5
#         self.search_space_yy_gpu,
#         self.search_space_sigma_range_gpu,
#         x_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
#         y_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
#         self.model_nRows, # search space grid nRows = 51
#         self.model_nCols, # search space grid nCols = 51
#         self.model_nFrames, # search space number of sigma values = 16
#         self.stim_width,
#         self.stim_height))
          
#         #--------Gaussian Curves to Timeseries
#         nRows_gaussian_curves_matrix = self.model_nRows * self.model_nCols * self.model_nFrames
#         nCols_gaussian_curves_matrix = self.stim_height * self.stim_width  

#         gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
#         signal_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)

#         return signal_rowmajor_gpu

#     def _compute_timecourses_chunk(self, kernel, x_range_gpu, y_range_gpu, sigma,  stimulus_data_columnmajor_gpu):
#         if self.block_dim is None or self.grid_dim is None:
#             self.set_kernel_config()

#         # Launch Gaussian related kernels
#         # compute all possible gaussian curves for each sigma value        
#         sigma_array_gpu = cp.asarray(sigma)
        
#         #---compute gaussian curves
#         result_gc_curves_gpu = cp.zeros((self.model_nRows * self.model_nCols * 1 * self.stim_width * self.stim_height), dtype=cp.float64)
#         kernel(self.grid_dim, self.block_dim, (
#         result_gc_curves_gpu,            
#         self.search_space_xx_gpu, # 51 points  (grid dimesion) between -13.5 to +13.5
#         self.search_space_yy_gpu,
#         sigma_array_gpu,
#         x_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
#         y_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
#         self.model_nRows, # search space grid nRows = 51
#         self.model_nCols, # search space grid nCols = 51
#         1, # nGaussianFrames = 1 because we are processing one sigma at a time
#         self.stim_width,
#         self.stim_height))
        
#         #--------Gaussian Curves to Timeseries
#         nRows_gaussian_curves_matrix = self.model_nRows * self.model_nCols * 1
#         nCols_gaussian_curves_matrix = self.stim_height * self.stim_width  

#         gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
#         signal_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)

#         return signal_rowmajor_gpu

#     def _compute_timecourse_batches(self, kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu):
#         if self.block_dim is None or self.grid_dim is None:
#             self.set_kernel_config()

#         single_timecourse_length = stimulus_data_columnmajor_gpu.shape[1]
#         batch_size = self.model_nRows * self.model_nCols
#         all_model_signals_rowmajor_gpu = cp.zeros((self.model_nRows * self.model_nCols * self.model_nFrames, single_timecourse_length), dtype=cp.float64)

#         # Launch Gaussian related kernels
#         # compute all possible gaussian curves for each sigma value
#         for i in range(len(self.search_space_sigma_range_gpu)):
#             sigma = self.search_space_sigma_range_gpu[i]
#             all_model_signals_rowmajor_gpu[i * batch_size: (i + 1) * batch_size, :] = self._compute_timecourses_chunk(kernel, x_range_gpu, y_range_gpu, sigma,  stimulus_data_columnmajor_gpu)                        

#         return all_model_signals_rowmajor_gpu

#     def _get_orthonormalize_signals(self, O_gpu, S_rowmajor_gpu, dS_dx_rowmajor_gpu, dS_dy_rowmajor_gpu, dS_dsigma_rowmajor_gpu):
#         # orthogonalization + nomalization of signals/timecourses (present along the columns)
#         S_star_columnmajor_gpu = cp.dot(O_gpu, S_rowmajor_gpu.T)
#         S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns) 
#         S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals

#         dS_star_dx_columnmajor_gpu = cp.dot(O_gpu, dS_dx_rowmajor_gpu.T)
#         dS_star_dy_columnmajor_gpu = cp.dot(O_gpu, dS_dy_rowmajor_gpu.T)    
#         dS_star_dsigma_columnmajor_gpu = cp.dot(O_gpu, dS_dsigma_rowmajor_gpu.T)    
    
#         dS_prime_dx_columnmajor_gpu = dS_star_dx_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dx_columnmajor_gpu).sum(axis=0))
#         dS_prime_dy_columnmajor_gpu = dS_star_dy_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dy_columnmajor_gpu).sum(axis=0))
#         dS_prime_dsigma_columnmajor_gpu = dS_star_dsigma_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dsigma_columnmajor_gpu).sum(axis=0))

#         # test_orthogonalized_tc = (cp.asnumpy(signals_columnmajor_gpu[:, 1]))        
        
#         return S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu

#     def get_orthonormal_signals(self, stimulus_data_columnmajor_gpu, O_gpu):
#         # ...model space (search/test)
#         self.search_space_xx_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nCols)
#         self.search_space_yy_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nRows)
#         self.search_space_sigma_range_gpu = cp.linspace(self.model_min_sigma, self.model_max_sigma, self.model_nFrames) # 0.5 to 1.5

#         # gaussian curves x and y range
#         x_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_width)
#         y_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_height)

#         # get CUDA kernels
#         gaussian_cuda_module = gpu_utils.get_raw_module('gaussian_kernel.cu')
#         gc_kernel = gaussian_cuda_module.get_function("gc_cuda_Kernel") # gpu_utils.get_raw_kernel('gaussian_kernel.cu', 'gc_cuda_Kernel')
#         dgc_dx_kernel = gaussian_cuda_module.get_function('dgc_dx_cuda_Kernel')
#         dgc_dy_kernel = gaussian_cuda_module.get_function('dgc_dy_cuda_Kernel')
#         dgc_dsigma_kernel = gaussian_cuda_module.get_function('dgc_dsigma_cuda_Kernel')

#         # compute gaussian curves and timecourses
#         S_rowmajor_gpu = self._compute_timecourse(gc_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         dS_dx_rowmajor_gpu = self._compute_timecourse(dgc_dx_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         dS_dy_rowmajor_gpu = self._compute_timecourse(dgc_dy_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         dS_dsigma_rowmajor_gpu = self._compute_timecourse(dgc_dsigma_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         # gpu_utils.print_gpu_memory_stats()        
#         #test_tc = (cp.asnumpy(S_rowmajor_gpu[1, :]))        

#         S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu = self._get_orthonormalize_signals(O_gpu
#                                                                                                                                                               , S_rowmajor_gpu
#                                                                                                                                                               , dS_dx_rowmajor_gpu
#                                                                                                                                                               , dS_dy_rowmajor_gpu
#                                                                                                                                                               , dS_dsigma_rowmajor_gpu)

#         return S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu

#     def get_orthonormal_signals_in_batches(self, stimulus_data_columnmajor_gpu, O_gpu):
#         # ...model space (search/test)
#         self.search_space_xx_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nCols)
#         self.search_space_yy_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nRows)
#         self.search_space_sigma_range_gpu = cp.linspace(self.model_min_sigma, self.model_max_sigma, self.model_nFrames) # 0.5 to 1.5

#         # gaussian curves x and y range
#         x_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_width)
#         y_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_height)

#         # get CUDA kernels
#         gaussian_cuda_module = gpu_utils.get_raw_module('gaussian_kernel.cu')
#         gc_kernel = gaussian_cuda_module.get_function("gc_cuda_Kernel") # gpu_utils.get_raw_kernel('gaussian_kernel.cu', 'gc_cuda_Kernel')
#         dgc_dx_kernel = gaussian_cuda_module.get_function('dgc_dx_cuda_Kernel')
#         dgc_dy_kernel = gaussian_cuda_module.get_function('dgc_dy_cuda_Kernel')
#         dgc_dsigma_kernel = gaussian_cuda_module.get_function('dgc_dsigma_cuda_Kernel')

#         # compute gaussian curves and timecourses
#         S_rowmajor_gpu = self._compute_timecourse_batches(gc_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         dS_dx_rowmajor_gpu = self._compute_timecourse_batches(dgc_dx_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         dS_dy_rowmajor_gpu = self._compute_timecourse_batches(dgc_dy_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         dS_dsigma_rowmajor_gpu = self._compute_timecourse_batches(dgc_dsigma_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
#         # gpu_utils.print_gpu_memory_stats()        
#         #test_tc = (cp.asnumpy(S_rowmajor_gpu[1, :]))        

#         S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu = self._get_orthonormalize_signals(O_gpu
#                                                                                                                                                               , S_rowmajor_gpu
#                                                                                                                                                               , dS_dx_rowmajor_gpu
#                                                                                                                                                               , dS_dy_rowmajor_gpu
#                                                                                                                                                               , dS_dsigma_rowmajor_gpu)

#         return S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu
    

#     # def _orthonormalize_model_signal(self, O_gpu, S_rowmajor_gpu):
#     #     # orthogonalization + nomalization of signals/timecourses (present along the columns)
#     #     S_star_columnmajor_gpu = cp.dot(O_gpu, S_rowmajor_gpu.T)
#     #     S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns) 
#     #     S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals
        
#     #     return S_prime_columnmajor_gpu, S_star_columnmajor_gpu, S_star_S_star_invroot_gpu
    
#     def _orthonormalize_derivative_model_signal(self, O_gpu, dS_rowmajor_gpu, S_star_columnmajor_gpu, S_star_S_star_invroot_gpu):
#         # orthogonalization + nomalization of derivative signals/timecourses (present along the columns)
#         dS_star_columnmajor_gpu = cp.dot(O_gpu, dS_rowmajor_gpu.T)
#         del dS_rowmajor_gpu # save GPU memory

#         dS_prime_columnmajor_gpu = dS_star_columnmajor_gpu * S_star_S_star_invroot_gpu - \
#             (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * \
#             ((S_star_columnmajor_gpu * dS_star_columnmajor_gpu).sum(axis=0))
        
#         return dS_prime_columnmajor_gpu

#     # nice nice
#     def _compute_derivative_error_term(self, drivative_kernel, 
#                                        x_range_gpu, 
#                                        y_range_gpu, 
#                                        stimulus_data_columnmajor_gpu, 
#                                        y_signals_batch, 
#                                        S_star_columnmajor_gpu, 
#                                        S_star_S_star_invroot_gpu, O_gpu):
#         # derivative timecourse
#         dS_rowmajor_gpu = self._compute_timecourse_batches(drivative_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)

#         # orthogonalization + nomalization
#         dS_prime_columnmajor_gpu = self._orthonormalize_derivative_model_signal(O_gpu, dS_rowmajor_gpu, S_star_columnmajor_gpu, S_star_S_star_invroot_gpu)

#         # error term 
#         de_gpu = GridFit.compute_error_term(Y_signals_gpu=y_signals_batch, S_prime_columnmajor_gpu=dS_prime_columnmajor_gpu)

#         return de_gpu



#     # nice
#     def _compute_model_signals_and_orthonormalize_in_batch(self, kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu, O_gpu):
#         if self.block_dim is None or self.grid_dim is None:
#             self.set_kernel_config()

#         single_timecourse_length = stimulus_data_columnmajor_gpu.shape[1]
#         batch_size = self.model_nRows * self.model_nCols
#         all_model_signals_rowmajor_gpu = cp.zeros((self.model_nRows * self.model_nCols * self.model_nFrames, single_timecourse_length), dtype=cp.float64)

#         # Launch Gaussian related kernels
#         # compute all possible gaussian curves for each sigma value
#         for i in range(len(self.search_space_sigma_range_gpu)):
#             sigma = self.search_space_sigma_range_gpu[i]
#             all_model_signals_rowmajor_gpu[i * batch_size: (i + 1) * batch_size, :] = self._compute_timecourses_chunk(kernel, x_range_gpu, y_range_gpu, sigma,  stimulus_data_columnmajor_gpu)                        

#         S_star_columnmajor_gpu = cp.dot(O_gpu, all_model_signals_rowmajor_gpu.T)
#         S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns) 
#         S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals

#         return S_star_columnmajor_gpu, S_star_S_star_invroot_gpu, S_prime_columnmajor_gpu
    
#     def get_error_terms_in_batches(self, y_signals_batch, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu, O_gpu):
#         # ...model space (search/test)
#         self.search_space_xx_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nCols)
#         self.search_space_yy_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nRows)
#         self.search_space_sigma_range_gpu = cp.linspace(self.model_min_sigma, self.model_max_sigma, self.model_nFrames) # 0.5 to 1.5

#         # gaussian curves x and y range
#         x_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_width)
#         y_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_height)

#         # get CUDA kernels
#         gaussian_cuda_module = gpu_utils.get_raw_module('gaussian_kernel.cu')
#         gc_kernel = gaussian_cuda_module.get_function("gc_cuda_Kernel") # gpu_utils.get_raw_kernel('gaussian_kernel.cu', 'gc_cuda_Kernel')
#         dgc_dx_kernel = gaussian_cuda_module.get_function('dgc_dx_cuda_Kernel')
#         dgc_dy_kernel = gaussian_cuda_module.get_function('dgc_dy_cuda_Kernel')
#         dgc_dsigma_kernel = gaussian_cuda_module.get_function('dgc_dsigma_cuda_Kernel')

#         # ERROR Terms
#         # ...e
#         # ...get orthonormalized model signals batch 
#         S_star_columnmajor_gpu, S_star_S_star_invroot_gpu, S_prime_columnmajor_gpu = self._compute_model_signals_and_orthonormalize_in_batch(gc_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu, O_gpu)
#         e_gpu = GridFit.compute_error_term(Y_signals_gpu=y_signals_batch, S_prime_columnmajor_gpu=S_prime_columnmajor_gpu)
#         del S_prime_columnmajor_gpu # to save memory


#         # ...de/dtheta
#         # ...compute error terms (includes computation of derivative model signal, orthonomalization and error term calculation)
#         de_dx_gpu = self._compute_derivative_error_term(dgc_dx_kernel, 
#                                        x_range_gpu, 
#                                        y_range_gpu, 
#                                        stimulus_data_columnmajor_gpu, 
#                                        y_signals_batch, 
#                                        S_star_columnmajor_gpu, 
#                                        S_star_S_star_invroot_gpu, O_gpu)
        
#         de_dy_gpu = self._compute_derivative_error_term(dgc_dy_kernel, 
#                                        x_range_gpu, 
#                                        y_range_gpu, 
#                                        stimulus_data_columnmajor_gpu, 
#                                        y_signals_batch, 
#                                        S_star_columnmajor_gpu, 
#                                        S_star_S_star_invroot_gpu, O_gpu)
        

#         de_dsigma_gpu = self._compute_derivative_error_term(dgc_dsigma_kernel, 
#                                        x_range_gpu, 
#                                        y_range_gpu, 
#                                        stimulus_data_columnmajor_gpu, 
#                                        y_signals_batch, 
#                                        S_star_columnmajor_gpu, 
#                                        S_star_S_star_invroot_gpu, O_gpu)


#         return e_gpu, de_dx_gpu, de_dy_gpu, de_dsigma_gpu