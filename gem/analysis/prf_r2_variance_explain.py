import numpy as np
import cupy as cp
from gem.tools.json_file_operations import JsonMgr
from gem.utils.hpc_cupy_utils import HpcUtils as Utils
from gem.model import prf_model
from gem.signals.signal_synthesizer import SignalSynthesizer
from gem.model.prf_model import GaussianModelParams
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

class R2:
    @classmethod
    def compute_r2_variance_explained_results(cls, refined_matching_results, Y_signals_cpu, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames):
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
            gc = cls.generate_2d_gaussian(mesh_X, mesh_Y, muX, muY, sigma) # mu_X_grid[:, :, 0] and mu_Y_grid[:, :, 0], because we have 3D meshgrids now due to variable sigma so, we are using only the first frame of the X and Y meshgrids
            s = gc.flatten() @ stim_data
            if (all(element == 0 for element in s)):
                r2 = -2
                json_entry = JsonMgr.args2jsonEntry(muX, muY, sigma, r2, s)    
                json_data_results_with_r2.append(json_entry)  
                continue

            y_hat, betas, trends = cls.get_y_hat(y, s)     
            if betas[0]>0:
                # r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)) # WORKED
                r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - trends@betas[1:]) ** 2))
            else:
                r2 = -1

            json_entry = JsonMgr.args2jsonEntry(muX, muY, sigma, r2, s)    
            json_data_results_with_r2.append(json_entry)    

        return json_data_results_with_r2

    # Generate Gaussian - CPU
    @classmethod
    def generate_2d_gaussian(cls, meshgrid_X, meshgrid_Y, mean_x, mean_y, sigma):                    
        mean_corrdinates = [mean_x, mean_y]        
        Z = np.exp(-(meshgrid_X - mean_corrdinates[0])**2 / (2 * sigma**2)) * np.exp(-(meshgrid_Y - mean_corrdinates[1])**2 / (2 * sigma**2))
        return Z

    # GLM fitting: compute y_hat
    @classmethod
    def get_y_hat(cls, y, s):

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


###############################################----------------------New R2 approach---------------------------------############################
    @classmethod
    def get_r2_new_method_with_epsilon_as_Fx(cls, Y_signals_gpu, O_gpu, epsilon_results_cpu):    
        r2 = []
        R_Rt_gpu = cp.eye(O_gpu.shape[0]) - O_gpu

        num_y_signals = len(epsilon_results_cpu)
        for yIdx in range (num_y_signals):
            y = Y_signals_gpu[:, yIdx]
            yty = y.T @ y
            ystar = O_gpu @ y
            ystarT_ystar_inv = (ystar.T @ ystar) ** (-1)            
            e = (float(epsilon_results_cpu[yIdx])) ** 2 #<----------------------------------
            yt_RRt_y = (y.T @ R_Rt_gpu) @ y
            r2_gpu = 1 - ((yty - e - yt_RRt_y) * ystarT_ystar_inv)
            r2.append(float(r2_gpu))

        return r2   

    @classmethod
    def get_r2_new_method_with_epsilon_as_yTs(cls, Y_signals_gpu, O_gpu, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames):    
        r2_results = []
        R_Rt_gpu = cp.eye(O_gpu.shape[0]) - O_gpu

        stim_data = stimulus.stimulus_data_gpu
        
        # compute Gaussian Curves for the Refined parameters
        total_num_gc = len(refined_matching_results)
        refined_matching_results_arr = np.array(refined_matching_results)
        muX_arr, muY_arr, sigma_arr = refined_matching_results_arr[: , 0], refined_matching_results_arr[: , 1], refined_matching_results_arr[: , 2]
        all_gc_gpu = cls._compute_refined_params_gaussian_curves(muX_arr, muY_arr, sigma_arr, x_range_cpu, y_range_cpu, total_num_gc, stim_height, stim_width)
        #....gaussian Curves to Timeseries
        nRows_gaussian_curves_matrix = total_num_gc
        nCols_gaussian_curves_matrix = stim_height * stim_width  

        gaussian_curves_rowmajor_gpu = cp.reshape(all_gc_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
        refined_signals_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stim_data)

        num_y_signals = len(refined_matching_results)
        for yIdx in range (num_y_signals):
            y = Y_signals_gpu[:, yIdx]
            yty = y.T @ y
            ystar = O_gpu @ y
            ystarT_ystar_inv = (ystar.T @ ystar) ** (-1)

            if all(np.isnan(refined_matching_results[yIdx])): # sometimes the y-signal itself is completely zero and in those scenerio, we set the refined parameters as (nan, nan, nan) in the refinement step
                r2 = -1
            else:
                # compute error-term
                s = refined_signals_rowmajor_gpu[yIdx, :] # gc.flatten() @ stim_data
                if (all(element == 0 for element in s)):
                    r2 = -2 
                else:      
                    s_star = O_gpu @ s
                    s_prime = s_star * (( s_star.T @ s_star ) ** (-1/2))
                    e = (y.T @ s_prime) ** 2 #<----------------------------------
                    yt_RRt_y = (y.T @ R_Rt_gpu) @ y
                    r2 = 1 - ((yty - e - yt_RRt_y) * ystarT_ystar_inv)

            r2_results.append(float(r2))

        return r2_results        

    @classmethod
    def get_r2_num_den_method_with_epsilon_as_yTs(cls, Y_signals_gpu, O_gpu, refined_matching_results, refined_S_cpu):  
        with cp.cuda.Device(ggm.get_instance().default_gpu_id):  
            numerators_gpu , denominators_gpu = cls.get_r2_numerator_denominator_terms(Y_signals_gpu, O_gpu, refined_matching_results, refined_S_cpu)
            r2_results_gpu = 1 - (numerators_gpu / denominators_gpu)

            return cp.asnumpy(r2_results_gpu)   

    @classmethod # R2 numerator = (yty - e - yt_RRt_y)    
    def get_r2_numerator_denominator_terms(cls, Y_signals_gpu, O_gpu, refined_matching_results, refined_signals_cpu : np.ndarray): 
        with cp.cuda.Device(ggm.get_instance().default_gpu_id):
            refined_signals_gpu = cp.asarray(refined_signals_cpu) 
            num_y_signals = len(refined_matching_results)
            r2_numerator_results_gpu = cp.zeros(num_y_signals, dtype=float)
            r2_denominator_results_gpu = cp.zeros(num_y_signals, dtype=float)

            R_Rt_gpu = cp.eye(O_gpu.shape[0]) - O_gpu

            # # # compute signals for the refined paramerters
            # # refined_prf_multi_dim_points_gpu = cp.array(refined_matching_results)
            # # refined_S_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_gpu=refined_prf_multi_dim_points_gpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE)            
            # # refined_signals_rowmajor_gpu = refined_S_batches[0]


            num_y_signals = len(refined_matching_results)
            for yIdx in range (num_y_signals):
                y = Y_signals_gpu[:, yIdx]
                yty = y.T @ y
                ystar = O_gpu @ y
                # ystarT_ystar_inv = (ystar.T @ ystar) ** (-1)
                ystarT_ystar = (ystar.T @ ystar)

                if all(np.isnan(refined_matching_results[yIdx])): # sometimes the y-signal itself is completely zero and in those scenerio, we set the refined parameters as (nan, nan, nan) in the refinement step
                    # r2 = -1
                    num = 2
                    den = 1
                else:
                    # compute error-term
                    s = refined_signals_gpu[yIdx, :] # gc.flatten() @ stim_data
                    if (all(element == 0 for element in s)):
                        # r2 = -2 
                        num = 3
                        den = 1
                    else:      
                        s_star = O_gpu @ s
                        s_prime = s_star * (( s_star.T @ s_star ) ** (-1/2))
                        e = (y.T @ s_prime) ** 2 #<----------------------------------
                        yt_RRt_y = (y.T @ R_Rt_gpu) @ y
                        # r2 = 1 - ((yty - e - yt_RRt_y) * ystarT_ystar_inv)
                        num = (yty - e - yt_RRt_y)
                        den = ystarT_ystar

                r2_numerator_results_gpu[yIdx] = float(num)
                r2_denominator_results_gpu[yIdx] = float(den)

            return r2_numerator_results_gpu, r2_denominator_results_gpu


    #########
    @classmethod
    def _compute_refined_params_gaussian_curves(cls, muX, muY, sigma, x_range_cpu, y_range_cpu, total_num_gc, stim_height, stim_width):
        muX_arr = np.array([muX])
        muY_arr = np.array([muY])
        sigma_arr = np.array([sigma])
        
        gaussian_cuda_module = Utils.get_raw_module('gaussian_kernel.cu')
        gc_kernel = gaussian_cuda_module.get_function("gc_using_args_arrays_cuda_Kernel")
        result_gc_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)

        # kernel grid
        block_dim = (32, 1, 1)
        bx = int((total_num_gc + block_dim[0] - 1) / block_dim[0])
        by = 1
        bz = 1
        grid_dim = (bx, by, bz)

        # launch kernel
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

        return result_gc_curves_gpu

    # # @classmethod
    # # def format_in_json_format(cls, r2_results, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames):
    # #     json_data_results_with_r2 = []

    # #     # # NOTE: old code commented out below
    # #     # stim_data = stimulus.resampled_hrf_convolved_data.flatten('F')
    # #     # stim_data = cp.reshape(stim_data, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame
    # #     stim_data = stimulus.stimulus_data_cpu


    # #     # test_stim_data = (stim_data[:, 10]).reshape((stim_height, stim_width))

    # #     mesh_X, mesh_Y = np.meshgrid(x_range_cpu, y_range_cpu)

    # #     # default r2 value
    # #     for i in range((len(refined_matching_results))):
    # #         r2 = float(r2_results[i])
            
    # #         # compute corresponding signal, s
    # #         muX, muY, sigma = refined_matching_results[i]
    # #         gc = cls.generate_2d_gaussian(mesh_X, mesh_Y, muX, muY, sigma) # mu_X_grid[:, :, 0] and mu_Y_grid[:, :, 0], because we have 3D meshgrids now due to variable sigma so, we are using only the first frame of the X and Y meshgrids
    # #         s = gc.flatten() @ stim_data
            
    # #         json_entry = JsonMgr.args2jsonEntry(muX, muY, sigma, r2, s)    
    # #         json_data_results_with_r2.append(json_entry)    

    # #     return json_data_results_with_r2    


    @classmethod
    def format_in_json_format(cls, r2_results, refined_matching_results, refined_signal_timecourses, refined_signals_present = True):
        json_data_results_with_r2 = []

        # default r2 value
        for i in range((len(refined_matching_results))):
            r2 = float(r2_results[i])            
            muX, muY, sigma = refined_matching_results[i]    
            if refined_signals_present:
                refined_signal_timecourse = refined_signal_timecourses[i]
            else:        
                refined_signal_timecourse = np.array([None])
            json_entry = JsonMgr.args2jsonEntry(muX, muY, sigma, r2, refined_signal_timecourse)    
            json_data_results_with_r2.append(json_entry)    

        return json_data_results_with_r2   