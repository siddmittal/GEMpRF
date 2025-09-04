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
    def get_r2_num_den_method_with_epsilon_as_yTs(cls, Y_signals_gpu, O_gpu, refined_matching_results, refined_S_cpu):  
        with cp.cuda.Device(ggm.get_instance().default_gpu_id):  
            numerators_gpu , denominators_gpu = cls.get_r2_numerator_denominator_terms(Y_signals_gpu, O_gpu, refined_matching_results, refined_S_cpu)
            r2_results_gpu = 1 - (numerators_gpu / denominators_gpu)

            return cp.asnumpy(r2_results_gpu)   

    @classmethod # R2 numerator = (yty - e - yt_RRt_y)    
    def get_r2_numerator_denominator_terms(cls, Y_signals_gpu, O_gpu, refined_matching_results, refined_signals_cpu: np.ndarray): 
        with cp.cuda.Device(ggm.get_instance().default_gpu_id):
            refined_signals_gpu = cp.asarray(refined_signals_cpu)
            num_y_signals = Y_signals_gpu.shape[1]

            R_Rt_gpu = cp.eye(O_gpu.shape[0]) - O_gpu

            # Compute yty for each signal
            yty = cp.sum(Y_signals_gpu ** 2, axis=0)

            # Compute ystar and ystarT_ystar
            ystar = O_gpu @ Y_signals_gpu
            ystarT_ystar = cp.sum(ystar ** 2, axis=0)

            # Check for NaN refined results
            nan_mask = np.isnan(refined_matching_results).all(axis=1)
            nan_mask = cp.asarray(nan_mask)

            # Check for zero signals
            s = refined_signals_gpu
            zero_mask = np.all(refined_signals_cpu == 0, axis=1)
            zero_mask = cp.asarray(zero_mask)

            # Compute s_star
            s_star = O_gpu @ s.T  # shape: [timepoints, num_signals]

            # Normalize s_star to get s_prime
            s_prime = s_star * (cp.sum(s_star ** 2, axis=0) ** (-0.5)) 

            # Compute error-term e = (y.T @ s_prime)^2
            e = cp.sum(Y_signals_gpu * s_prime, axis=0) ** 2 #<----------------------------------

            # Compute yt_RRt_y
            # # yt_RRt_y = cp.sum(Y_signals_gpu * (R_Rt_gpu @ Y_signals_gpu), axis=0)
            yt_R = Y_signals_gpu * (R_Rt_gpu @ Y_signals_gpu)
            yt_RRt_y = cp.sum(yt_R, axis=0)

            # Compute final numerator and denominator
            num = yty - e - yt_RRt_y
            den = ystarT_ystar

            # Apply masks for nan/zero
            num = cp.where(nan_mask, 2.0, num)
            num = cp.where(zero_mask & ~nan_mask, 3.0, num)
            den = cp.where(nan_mask | zero_mask, 1.0, den)

            return num, den

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