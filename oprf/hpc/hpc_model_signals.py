import numpy as np
import cupy as cp

# Stimulus Class
from oprf.standard.prf_stimulus import Stimulus

# Receptive Field Response Class
from oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse

# Utilities
from oprf.hpc.hpc_cupy_utils import Utils as gpu_utils


class ModelSignals:
    def __init__(self, model_type, model_nRows, model_nCols, model_nFrames, model_visual_field, model_min_sigma, model_max_sigma,  stim_width, stim_height, stim_visual_field):
        # model space
        self.model_type = model_type
        self.model_nCols = model_nRows
        self.model_nRows = model_nCols
        self.model_nFrames = model_nFrames
        self.model_visual_field = model_visual_field
        self.model_min_sigma = model_min_sigma
        self.model_max_sigma = model_max_sigma

        # stimulus related
        self.stim_visual_field = stim_visual_field
        self.stim_width = stim_width
        self.stim_height = stim_height

        # gaussian
        self.total_gaussian_curves_per_stim_frame = model_nRows * model_nCols
        self.single_gaussian_curve_length = stim_width * stim_height

        # CUDA kernel launch config
        self.block_dim = None
        self.grid_dim = None

    def set_kernel_config(self):
        self.block_dim = (32, 32, 1)
        bx = int((self.model_nRows + self.block_dim[0] - 1) / self.block_dim[0])
        by = int((self.model_nRows + self.block_dim[1] - 1) / self.block_dim[1])
        bz = int((self.model_nFrames + self.block_dim[2] - 1) / self.block_dim[2])
        self.grid_dim = (bx, by, bz)

    def _compute_timecourse(self, kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu):
        if self.block_dim is None or self.grid_dim is None:
            self.set_kernel_config()

        # Launch Gaussian related kernels
        #---compute gaussian curves
        result_gc_curves_gpu = cp.zeros((self.model_nRows * self.model_nCols * self.model_nFrames * self.stim_width * self.stim_height), dtype=cp.float64)
        kernel(self.grid_dim, self.block_dim, (
        result_gc_curves_gpu,
        self.search_space_xx_gpu, # 51 points  (grid dimesion) between -13.5 to +13.5
        self.search_space_yy_gpu,
        self.search_space_sigma_range_gpu,
        x_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
        y_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
        self.model_nRows, # search space grid nRows = 51
        self.model_nCols, # search space grid nCols = 51
        self.model_nFrames, # search space number of sigma values = 16
        self.stim_width,
        self.stim_height))
        
        #--------Gaussian Curves to Timeseries
        nRows_gaussian_curves_matrix = self.model_nRows * self.model_nCols * self.model_nFrames
        nCols_gaussian_curves_matrix = self.stim_height * self.stim_width

        gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
        signal_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)

        return signal_rowmajor_gpu

    def _compute_timecourses_chunk(self, kernel, x_range_gpu, y_range_gpu, sigma_array_gpu, sigma_batch_size,  stimulus_data_columnmajor_gpu, gpu_idx):
        if self.block_dim is None or self.grid_dim is None:
            self.set_kernel_config()

        # Launch Gaussian related kernels
        # compute all possible gaussian curves for each sigma value
        with cp.cuda.Device(gpu_idx):
            #---compute gaussian curves
            if gpu_idx == 0:
                result_gc_curves_gpu = cp.zeros((self.model_nRows * self.model_nCols * sigma_batch_size * self.stim_width * self.stim_height), dtype=cp.float64)
                kernel(self.grid_dim, self.block_dim, (
                result_gc_curves_gpu,
                self.search_space_xx_gpu, # 51 points  (grid dimesion) between -13.5 to +13.5
                self.search_space_yy_gpu,
                sigma_array_gpu,
                x_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
                y_range_gpu, # 101 points  (stimulus dimension) between -9 to +9
                self.model_nRows, # search space grid nRows = 51
                self.model_nCols, # search space grid nCols = 51
                sigma_batch_size,
                self.stim_width,
                self.stim_height,
                gpu_idx))
            else:
                result_gc_curves_gpu = cp.zeros((self.model_nRows * self.model_nCols * sigma_batch_size * self.stim_width * self.stim_height), dtype=cp.float64)
                kernel(self.grid_dim, self.block_dim, (
                result_gc_curves_gpu,
                cp.array(self.search_space_xx_gpu), # 51 points  (grid dimesion) between -13.5 to +13.5
                cp.array(self.search_space_yy_gpu),
                cp.array(sigma_array_gpu),
                cp.array(x_range_gpu), # 101 points  (stimulus dimension) between -9 to +9
                cp.array(y_range_gpu), # 101 points  (stimulus dimension) between -9 to +9
                self.model_nRows, # search space grid nRows = 51
                self.model_nCols, # search space grid nCols = 51
                sigma_batch_size,
                self.stim_width,
                self.stim_height,
                gpu_idx))


            #--------Gaussian Curves to Timeseries
            nRows_gaussian_curves_matrix = self.model_nRows * self.model_nCols * sigma_batch_size #1
            nCols_gaussian_curves_matrix = self.stim_height * self.stim_width

            gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC

            # we need to put the stimulus data on the correct GPU device
            signal_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, cp.array(stimulus_data_columnmajor_gpu))

        return signal_rowmajor_gpu

    def _compute_timecourse_batches(self, kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu):
        if self.block_dim is None or self.grid_dim is None:
            self.set_kernel_config()

        # results batches
        result_batches = []            

        single_timecourse_length = stimulus_data_columnmajor_gpu.shape[1]
        num_signals_per_sigma = self.model_nRows * self.model_nCols # i.e. the number of signals per slice
        
        # Launch Gaussian related kernels
        # compute all possible gaussian curves for each sigma value
        num_gpus = gpu_utils.get_number_of_gpus() # NOTE: sigma batch can be decided based on the number of GPUs....then we know which sigma-batch is allocated on which GPU
        total_sigmas = len(self.search_space_sigma_range_gpu)
        per_gpu_assigned_sigma_batch_size = np.max([1, int(total_sigmas / num_gpus)])
        per_gpu_batch_required_mem = gpu_utils.gpu_mem_required_in_gb( num_elements= (self.model_nRows * self.model_nCols * per_gpu_assigned_sigma_batch_size) * (self.stim_width * self.stim_height))
        single_sigma_grid_slice_required_bytes = gpu_utils.gpu_mem_required_in_gb( num_elements= (self.model_nRows * self.model_nCols) * (self.stim_width * self.stim_height)) * (1024 ** 3)        
        for gpu_idx in range(num_gpus): # NOTE: later on, we will specify the GPU numbers in the config, and read the value from there
            signal_rowmajor_batch_current_gpu = None              

            with cp.cuda.Device(gpu_idx):
                # check if we can process a whole batch on the selected GPU-device, if not, then try to divide the assigned batch into smaller chunks
                signal_rowmajor_batch_current_gpu = cp.zeros((self.model_nRows * self.model_nCols * per_gpu_assigned_sigma_batch_size, single_timecourse_length), dtype=cp.float64)                                
                available_bytes = gpu_utils.device_available_mem_bytes(device_id=gpu_idx)                                
                possible_sigma_batch_size = int(available_bytes / (single_sigma_grid_slice_required_bytes * 2)) # additional "2" because, we will also be needing memory for the signal timecourses so, better be safe !!!
                if possible_sigma_batch_size < 1:
                    raise ValueError(f"Not enough GPU memory available on device-{gpu_idx}.\nAvailable (Gigabytes) = {available_bytes / (1024 ** 3)}, Required (Gigabytes) = {(single_sigma_grid_slice_required_bytes * 2) / (1024 ** 3)}.\nGEM-pRF cannot compute further model signals !!!")
                selected_gpu_possible_sigma_chunk_size = per_gpu_assigned_sigma_batch_size if per_gpu_assigned_sigma_batch_size <= possible_sigma_batch_size else possible_sigma_batch_size                
                                
                # process even smaller chunks of batches
                chunked_sigma_arr = self.search_space_sigma_range_gpu[gpu_idx * per_gpu_assigned_sigma_batch_size : gpu_idx * per_gpu_assigned_sigma_batch_size + per_gpu_assigned_sigma_batch_size]                       
                for sigma_chunk_idx in range(0, per_gpu_assigned_sigma_batch_size, selected_gpu_possible_sigma_chunk_size): # (0, total_sigma, batch_size)                            
                    sigma_values_arr_gpu = chunked_sigma_arr[sigma_chunk_idx : sigma_chunk_idx + selected_gpu_possible_sigma_chunk_size] # self.search_space_sigma_range_gpu[from_sigma_idx : to_sigma_idx]
                    chunk_size = len(sigma_values_arr_gpu)
                    chunk_signal_rowmajor_batch_gpu = self._compute_timecourses_chunk(kernel, x_range_gpu, y_range_gpu, sigma_values_arr_gpu, chunk_size, stimulus_data_columnmajor_gpu, gpu_idx)            
                    signal_rowmajor_batch_current_gpu[sigma_chunk_idx * num_signals_per_sigma: (sigma_chunk_idx + chunk_size) * num_signals_per_sigma, :] = chunk_signal_rowmajor_batch_gpu    

                result_batches.append(signal_rowmajor_batch_current_gpu)

        return result_batches

    def _get_orthonormalize_signals(self, O_gpu, S_rowmajor_gpu, dS_dx_rowmajor_gpu, dS_dy_rowmajor_gpu, dS_dsigma_rowmajor_gpu):
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

    def _get_orthonormalize_signals_multi_gpu(self, O_gpu, S_rowmajor_gpu_batches, dS_dx_rowmajor_gpu_batches, dS_dy_rowmajor_gpu_batches, dS_dsigma_rowmajor_gpu_batches):
        # resulting batches
        S_prime_columnmajor_gpu_batches = []
        dS_prime_dx_columnmajor_gpu_result_batches = []
        dS_prime_dy_columnmajor_gpu_result_batches = []
        dS_prime_dsigma_columnmajor_gpu_result_batches = []

        # process each batch (present on different GPU) seperately
        num_batches = len(S_rowmajor_gpu_batches)
        for batch_idx in range(num_batches):
            device_id = S_rowmajor_gpu_batches[batch_idx].device.id
            S_rowmajor_gpu_batch = S_rowmajor_gpu_batches[batch_idx]
            dS_dx_rowmajor_gpu_batch = dS_dx_rowmajor_gpu_batches[batch_idx]
            dS_dy_rowmajor_gpu_batch = dS_dy_rowmajor_gpu_batches[batch_idx]
            dS_dsigma_rowmajor_gpu_batch = dS_dsigma_rowmajor_gpu_batches[batch_idx]

            with cp.cuda.Device(device_id):                        
                O_gpu_current_device = O_gpu if device_id == 0 else cp.array(O_gpu)

                # orthogonalization + nomalization of signals/timecourses (present along the columns)
                S_star_columnmajor_gpu = cp.dot(O_gpu_current_device, S_rowmajor_gpu_batch.T)
                S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns)
                S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals

                dS_star_dx_columnmajor_gpu = cp.dot(O_gpu_current_device, dS_dx_rowmajor_gpu_batch.T)
                dS_star_dy_columnmajor_gpu = cp.dot(O_gpu_current_device, dS_dy_rowmajor_gpu_batch.T)
                dS_star_dsigma_columnmajor_gpu = cp.dot(O_gpu_current_device, dS_dsigma_rowmajor_gpu_batch.T)

                dS_prime_dx_columnmajor_gpu = dS_star_dx_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dx_columnmajor_gpu).sum(axis=0))
                dS_prime_dy_columnmajor_gpu = dS_star_dy_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dy_columnmajor_gpu).sum(axis=0))
                dS_prime_dsigma_columnmajor_gpu = dS_star_dsigma_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dsigma_columnmajor_gpu).sum(axis=0))
                
                # collect result batches
                S_prime_columnmajor_gpu_batches.append(S_prime_columnmajor_gpu)
                dS_prime_dx_columnmajor_gpu_result_batches.append(dS_prime_dx_columnmajor_gpu)
                dS_prime_dy_columnmajor_gpu_result_batches.append(dS_prime_dy_columnmajor_gpu)
                dS_prime_dsigma_columnmajor_gpu_result_batches.append(dS_prime_dsigma_columnmajor_gpu)


        return S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_result_batches, dS_prime_dy_columnmajor_gpu_result_batches, dS_prime_dsigma_columnmajor_gpu_result_batches # S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu

    def get_orthonormal_signals(self, stimulus_data_columnmajor_gpu, O_gpu):
        # ...model space (search/test)
        self.search_space_xx_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nCols)
        self.search_space_yy_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nRows)
        self.search_space_sigma_range_gpu = cp.linspace(self.model_min_sigma, self.model_max_sigma, self.model_nFrames) # 0.5 to 1.5

        # gaussian curves x and y range
        x_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_width)
        y_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_height)

        # get CUDA kernels
        gaussian_cuda_module = gpu_utils.get_raw_module('gaussian_kernel.cu')
        gc_kernel = gaussian_cuda_module.get_function("gc_cuda_Kernel") # gpu_utils.get_raw_kernel('gaussian_kernel.cu', 'gc_cuda_Kernel')
        dgc_dx_kernel = gaussian_cuda_module.get_function('dgc_dx_cuda_Kernel')
        dgc_dy_kernel = gaussian_cuda_module.get_function('dgc_dy_cuda_Kernel')
        dgc_dsigma_kernel = gaussian_cuda_module.get_function('dgc_dsigma_cuda_Kernel')

        # compute gaussian curves and timecourses
        S_rowmajor_gpu = self._compute_timecourse(gc_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
        dS_dx_rowmajor_gpu = self._compute_timecourse(dgc_dx_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
        dS_dy_rowmajor_gpu = self._compute_timecourse(dgc_dy_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
        dS_dsigma_rowmajor_gpu = self._compute_timecourse(dgc_dsigma_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
        # gpu_utils.print_gpu_memory_stats()
        #test_tc = (cp.asnumpy(S_rowmajor_gpu[1, :]))

        S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu = self._get_orthonormalize_signals(O_gpu
                                                                                                                                                              , S_rowmajor_gpu
                                                                                                                                                              , dS_dx_rowmajor_gpu
                                                                                                                                                              , dS_dy_rowmajor_gpu
                                                                                                                                                              , dS_dsigma_rowmajor_gpu)

        return S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu

    def get_orthonormal_signals_in_batches(self, stimulus_data_columnmajor_gpu, O_gpu):
        # ...model space (search/test)
        self.search_space_xx_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nCols)
        self.search_space_yy_gpu = cp.linspace(-self.model_visual_field, self.model_visual_field, self.model_nRows)
        self.search_space_sigma_range_gpu = cp.linspace(self.model_min_sigma, self.model_max_sigma, self.model_nFrames) # 0.5 to 1.5

        # gaussian curves x and y range
        x_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_width)
        y_range_gpu = cp.linspace(-self.stim_visual_field, self.stim_visual_field, self.stim_height)

        # get CUDA kernels
        gaussian_cuda_module = gpu_utils.get_raw_module('gaussian_kernel.cu')
        gc_kernel = gaussian_cuda_module.get_function("gc_cuda_Kernel") # gpu_utils.get_raw_kernel('gaussian_kernel.cu', 'gc_cuda_Kernel')
        dgc_dx_kernel = gaussian_cuda_module.get_function('dgc_dx_cuda_Kernel')
        dgc_dy_kernel = gaussian_cuda_module.get_function('dgc_dy_cuda_Kernel')
        dgc_dsigma_kernel = gaussian_cuda_module.get_function('dgc_dsigma_cuda_Kernel')

        # compute gaussian curves and timecourses
        S_rowmajor_gpu_batches = self._compute_timecourse_batches(gc_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
        dS_dx_rowmajor_gpu_batches = self._compute_timecourse_batches(dgc_dx_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
        dS_dy_rowmajor_gpu_batches = self._compute_timecourse_batches(dgc_dy_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)
        dS_dsigma_rowmajor_gpu_batches = self._compute_timecourse_batches(dgc_dsigma_kernel, x_range_gpu, y_range_gpu, stimulus_data_columnmajor_gpu)


        S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches = self._get_orthonormalize_signals_multi_gpu(O_gpu
                                                                                                                                                              , S_rowmajor_gpu_batches
                                                                                                                                                              , dS_dx_rowmajor_gpu_batches
                                                                                                                                                              , dS_dy_rowmajor_gpu_batches
                                                                                                                                                              , dS_dsigma_rowmajor_gpu_batches)

        return S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches # S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu