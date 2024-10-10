import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os
import cupy as cp

class GridFit:
    @classmethod
    def compute_error_term(cls, Y_signals_gpu, S_prime_columnmajor_gpu):
        e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
        return e_gpu
    
########################----------------------######################
    # ORIGINAL - considering error term (e) is squared---------------------------------------#################
    #@classmethod
    #def _compute_derivative_error_term_cpu(cls, O_gpu, Y_signals_gpu, dS_prime_dtheta_columnmajor_gpu, e_gpu):
    #    de_dtheta_gpu = 2 * e_gpu * (Y_signals_gpu.T @ dS_prime_dtheta_columnmajor_gpu)
    #    de_dtheta_cpu = cp.asnumpy(de_dtheta_gpu)
    #    return de_dtheta_cpu

    #@classmethod
    #def _compute_best_projection_fit(cls, e_gpu):
    #    best_fit_proj_gpu = cp.argmax(e_gpu **2, axis=1)     #<<<<----find the max. element's index for the rows along their columns (that's why axis=1)        
    #    return best_fit_proj_gpu
########################----------------------######################

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
    def _get_error_terms_from_batches(cls, isResultOnGPU, Y_signals_gpu, S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches):        
        num_batches = len(S_prime_columnmajor_gpu_batches)

        # compute total signals present across all the batches
        total_model_signals = 0
        for i in range(num_batches):
            total_model_signals = total_model_signals + S_prime_columnmajor_gpu_batches[i].shape[1] # i.e.  number of columns: bacuse each signal is present across a single column

        # allocate memory to hold results on default device (i.e. device-0)
        total_y_signals = Y_signals_gpu.shape[1]
        e_result_gpu = cp.zeros((total_y_signals, total_model_signals), dtype=cp.float64)

        if isResultOnGPU:
            de_dx_result = cp.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
            de_dy_result = cp.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
            de_sigma_result = cp.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
        else:
            de_dx_result = np.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
            de_dy_result = np.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
            de_sigma_result = np.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
        
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

                # derivates --- on CPU
                de_dx_result[:, column_idx : column_idx + num_signals_current_batch] = cls._compute_derivative_error_term(isResultOnGPU, current_device_Y_signals, dS_prime_dx_columnmajor_gpu_batches[batch_idx], chunk_e_result_gpu)
                de_dy_result[:, column_idx : column_idx + num_signals_current_batch] = cls._compute_derivative_error_term(isResultOnGPU, current_device_Y_signals, dS_prime_dy_columnmajor_gpu_batches[batch_idx], chunk_e_result_gpu)
                de_sigma_result[:, column_idx : column_idx + num_signals_current_batch] = cls._compute_derivative_error_term(isResultOnGPU, current_device_Y_signals, dS_prime_dsigma_columnmajor_gpu_batches[batch_idx], chunk_e_result_gpu)                                
            
            # finally, get the "error" result back to device-0
            with cp.cuda.Device(0):
                if device_id == 0:
                    e_result_gpu[:, column_idx : column_idx + num_signals_current_batch] = chunk_e_result_gpu
                else:
                    e_result_gpu[:, column_idx : column_idx + num_signals_current_batch] = cp.array(chunk_e_result_gpu)                              

            # Note: update index               
            column_idx = column_idx + num_signals_current_batch
        
        return e_result_gpu, de_dx_result, de_dy_result, de_sigma_result


    @classmethod
    # Note: all ORTHOGONALIZED signals "S" are assumed along Columns
    #def get_error_terms(cls, O_gpu, Y_signals_gpu, S_star_columnmajor_gpu, dS_star_dx_columnmajor_gpu, dS_star_dy_columnmajor_gpu, dS_star_dsigma_columnmajor_gpu):
    def get_error_terms(cls, isResultOnGPU, O_gpu, Y_signals_gpu, S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu, multi_gpu_batching):
        # nomalization of signals/timecourses (present along the columns)

        if multi_gpu_batching is False:
            # S
            e_gpu = cls.compute_error_term(Y_signals_gpu, S_prime_columnmajor_gpu)
    
            # dS_dx        
            de_dx_full_gpu = cls._compute_derivative_error_term(isResultOnGPU, Y_signals_gpu, dS_prime_dx_columnmajor_gpu, e_gpu)

            # dS_dy        
            de_dy_full_gpu = cls._compute_derivative_error_term(isResultOnGPU, Y_signals_gpu, dS_prime_dy_columnmajor_gpu, e_gpu)
            
            # dS_dsigma
            de_dsigma_full_gpu = cls._compute_derivative_error_term(isResultOnGPU, Y_signals_gpu, dS_prime_dsigma_columnmajor_gpu, e_gpu)
            
            # best fit
            best_fit_proj_gpu = cls._compute_best_projection_fit(e_gpu)
        
        else:
            # all in one
            e_gpu, de_dx_full_gpu, de_dy_full_gpu, de_dsigma_full_gpu = cls._get_error_terms_from_batches(isResultOnGPU, Y_signals_gpu, S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu)
                
        # best fit
        best_fit_proj_gpu = cls._compute_best_projection_fit(e_gpu)
    
        ###---return results on CPU or GPU
        # CPU result
        if isResultOnGPU:
            return best_fit_proj_gpu, e_gpu, de_dx_full_gpu, de_dy_full_gpu, de_dsigma_full_gpu
        
        # GPU result
        else:            
            best_fit_proj_cpu = cp.asnumpy(best_fit_proj_gpu)
            e_cpu = cp.asnumpy(e_gpu)
            de_dx_full_cpu = cp.asnumpy(de_dx_full_gpu)
            de_dy_full_cpu = cp.asnumpy(de_dy_full_gpu)
            de_dsigma_full_cpu = cp.asnumpy(de_dsigma_full_gpu)

            return best_fit_proj_cpu, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu
        

