# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Siddharth Mittal",
"@Desc    :   None",     
"""

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

class GridFit:
    @classmethod
    def compute_error_term(cls, Y_signals_gpu, S_prime_columnmajor_gpu):
        e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
        return e_gpu
    
    @classmethod
    def _compute_derivative_error_term(cls, isResultOnGPU, Y_signals_gpu, dS_prime_dtheta_columnmajor_gpu, e_gpu):
        de_dtheta_gpu = (Y_signals_gpu.T @ dS_prime_dtheta_columnmajor_gpu)
        # de_dtheta_cpu = cp.asnumpy(de_dtheta_gpu)
        return de_dtheta_gpu if isResultOnGPU == True else cp.asnumpy(de_dtheta_gpu)

    @classmethod
    def _compute_best_projection_fit(cls, e_gpu):
        best_fit_proj_gpu = cp.nanargmax(e_gpu, axis=1)     #<<<<----find the max. element's index for the rows along their columns (that's why axis=1)
        return best_fit_proj_gpu

    @classmethod    
    def _get_error_terms_from_batches(cls, isResultOnGPU, Y_signals_gpu, S_prime_cm_gpu_batches, dS_prime_dtheta_cm_gpu_batches_list):   
        default_gpu_id = ggm.get_instance().default_gpu_id     
        num_batches = len(S_prime_cm_gpu_batches)
        if dS_prime_dtheta_cm_gpu_batches_list is None:
            num_theta_params = 0
        else:
            num_theta_params = max(0, len(dS_prime_dtheta_cm_gpu_batches_list))

        # compute total signals present across all the batches
        total_model_signals = 0
        for i in range(num_batches):
            total_model_signals = total_model_signals + S_prime_cm_gpu_batches[i].shape[1] # i.e.  number of columns: bacuse each signal is present across a single column

        # allocate memory to hold results on default device
        with cp.cuda.Device(default_gpu_id):
            total_y_signals = Y_signals_gpu.shape[1]
            e_result_gpu = cp.zeros((total_y_signals, total_model_signals), dtype=cp.float64)

            # derivative error terms result
            if num_theta_params > 0:
                if isResultOnGPU:
                    # shape: (num_theta_params, total_y_signals, total_model_signals)
                    de_dtheta_batches_3darr = cp.zeros((num_theta_params, total_y_signals, total_model_signals), dtype=cp.float64)
                else:
                    de_dtheta_batches_3darr = np.zeros((num_theta_params, total_y_signals, total_model_signals), dtype=np.float64)
            else:
                de_dtheta_batches_3darr = None

        # process each batch individually and store the results
        column_idx = 0
        for batch_idx in range(num_batches):
            device_id = S_prime_cm_gpu_batches[batch_idx].device.id 
            num_signals_current_batch = S_prime_cm_gpu_batches[batch_idx].shape[1]    

            # allocate chunks
            # chunk_e_result_gpu  = cp.zeros((total_y_signals, num_signals_current_batch), dtype=cp.float64)  # NOTE: Commented on March 2024, because it's not needed
            chunk_de_dtheta_gpu = [None] * num_theta_params                
            with cp.cuda.Device(device_id):
                current_device_Y_signals = Y_signals_gpu if device_id == default_gpu_id else cp.array(Y_signals_gpu)
                chunk_e_result_gpu = cls.compute_error_term(current_device_Y_signals, S_prime_cm_gpu_batches[batch_idx])
                chunk_e_result_gpu[cp.isinf(chunk_e_result_gpu) & (chunk_e_result_gpu > 0)] = -cp.inf # replace +inf with -inf so that indices with "inf" are not seleted as best fit at the argmax() step

                # derivates --- on GPU/CPU
                for theta in range(num_theta_params):
                    chunk_de_dtheta_gpu[theta] = cls._compute_derivative_error_term(isResultOnGPU, current_device_Y_signals, dS_prime_dtheta_cm_gpu_batches_list[theta][batch_idx], chunk_e_result_gpu)

            # NOTE: dealing with the warning message about array allocated on a different device (Peer Access)
            for theta in range(num_theta_params):
                if isResultOnGPU: # i.e. in case of concatenated runs, we want the errors to stay on GPU
                    with cp.cuda.Device(default_gpu_id):                    
                        de_dtheta_batches_3darr[theta, :, column_idx : column_idx + num_signals_current_batch] = chunk_de_dtheta_gpu[theta] if chunk_de_dtheta_gpu[theta].device.id == default_gpu_id else cp.asarray(chunk_de_dtheta_gpu[theta])
                else:
                    de_dtheta_batches_3darr[theta, :, column_idx : column_idx + num_signals_current_batch] = chunk_de_dtheta_gpu[theta]

            # finally, get the "error" result back to device-0
            with cp.cuda.Device(default_gpu_id):
                if device_id == default_gpu_id:
                    e_result_gpu[:, column_idx : column_idx + num_signals_current_batch] = chunk_e_result_gpu
                else:
                    e_result_gpu[:, column_idx : column_idx + num_signals_current_batch] = cp.array(chunk_e_result_gpu)                              

            # Note: update index               
            column_idx = column_idx + num_signals_current_batch
        
        return e_result_gpu, de_dtheta_batches_3darr

    @classmethod    
    def _get_only_error_terms_from_batches(cls, Y_signals_gpu, S_prime_columnmajor_gpu_batches):        
        num_batches = len(S_prime_columnmajor_gpu_batches)

        # compute total signals present across all the batches
        total_model_signals = 0
        for i in range(num_batches):
            total_model_signals = total_model_signals + S_prime_columnmajor_gpu_batches[i].shape[1] # i.e.  number of columns: bacuse each signal is present across a single column

        # allocate memory to hold results on default device (i.e. device-0)
        total_y_signals = Y_signals_gpu.shape[1]
        e_result_gpu = cp.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
        
        # process each batch individually and store the results
        column_idx = 0
        for batch_idx in range(num_batches):
            device_id = S_prime_columnmajor_gpu_batches[batch_idx].device.id 
            num_signals_current_batch = S_prime_columnmajor_gpu_batches[batch_idx].shape[1]    

            # allocate chunks
            chunk_e_result_gpu  = cp.zeros((total_y_signals, num_signals_current_batch), dtype=cp.float64)       
            with cp.cuda.Device(device_id):
                current_device_Y_signals = Y_signals_gpu if device_id == 0 else cp.array(Y_signals_gpu)
                chunk_e_result_gpu = cls.compute_error_term(current_device_Y_signals, S_prime_columnmajor_gpu_batches[batch_idx])
                
            
            # finally, get the "error" result back to device-0
            with cp.cuda.Device(0):
                if device_id == 0:
                    e_result_gpu[:, column_idx : column_idx + num_signals_current_batch] = chunk_e_result_gpu
                else:
                    e_result_gpu[:, column_idx : column_idx + num_signals_current_batch] = cp.array(chunk_e_result_gpu)                              

            # Note: update index               
            column_idx = column_idx + num_signals_current_batch
        
        return e_result_gpu

    @classmethod
    # Note: all ORTHOGONALIZED signals "S" are assumed along Columns    
    def get_error_terms(cls, isResultOnGPU, Y_signals_gpu, S_prime_cm_batches_gpu, dS_prime_dtheta_cm_batches_list_gpu):
        # nomalization of signals/timecourses (present along the columns)

        # all in one
        e_gpu, de_dtheta_3darr_gpu = cls._get_error_terms_from_batches(isResultOnGPU, Y_signals_gpu, S_prime_cm_batches_gpu, dS_prime_dtheta_cm_batches_list_gpu)

        with cp.cuda.Device(ggm.get_instance().default_gpu_id):
            e_gpu[cp.isinf(e_gpu) & (e_gpu > 0)] = -cp.inf # replace +inf with -inf so that indices with "inf" are not seleted as best fit

            # best fit
            # best_fit_proj_cpu = np.nanargmax(cp.asnumpy(e_gpu), axis=1) #cls._compute_best_projection_fit(e_gpu)
            best_fit_proj_cpu = cp.asnumpy(cp.nanargmax(e_gpu, axis=1)) #  optimized version of the above line
        
            ###---return results on CPU or GPU
            # GPU result
            if isResultOnGPU:
                return cp.asarray(best_fit_proj_cpu), e_gpu, de_dtheta_3darr_gpu
            
            # CPU result
            else:            
                best_fit_proj_cpu = cp.asnumpy(best_fit_proj_cpu)
                e_cpu = cp.asnumpy(e_gpu)

                return best_fit_proj_cpu, e_cpu, cp.asnumpy(de_dtheta_3darr_gpu)
                
    @classmethod
    # Note: all ORTHOGONALIZED signals "S" are assumed along Columns
    def get_only_error_terms(cls, isResultOnGPU, Y_signals_gpu, S_prime_cm_batches_gpu, dS_prime_dtheta_cm_batches_list_gpu): # NOTE: "dS_prime_dtheta_cm_batches_list_gpu" is placed to maintain same signature as "get_error_terms"
        best_fit_proj_cpu, e_gpu, _ = cls.get_error_terms(isResultOnGPU, Y_signals_gpu, S_prime_cm_batches_gpu, None)
                
        # # best fit
        # best_fit_proj_cpu = np.nanargmax(cp.asnumpy(e_gpu), axis=1) #cls._compute_best_projection_fit(e_gpu)
    
        ###---return results on CPU or GPU
        # GPU result
        if isResultOnGPU:            
            return ggm.get_instance().execute_cupy_func_on_default(cupy_func = cp.asarray, cupy_func_args=(best_fit_proj_cpu,)), e_gpu, None
        
        # CPU result
        else:            
            best_fit_proj_cpu = cp.asnumpy(best_fit_proj_cpu)
            e_cpu = cp.asnumpy(e_gpu)

            return best_fit_proj_cpu, e_cpu, None

