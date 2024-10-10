# Add my "oprf" package path
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cProfile
import os
import datetime
import threading
import queue

# paths
import re

# Config
from config.oprf_config import ConfigurationWrapper as cfg
cfg.load_configuration()

# Local Imports
import sys
sys.path.append(cfg.path_to_append)

from oprf.hpc.hpc_cupy_utils import Utils as gpu_utils
from oprf.standard.prf_stimulus import Stimulus
from oprf.hpc.hpc_coefficient_matrix import CoefficientMatrix
from oprf.hpc.hpc_coefficient_matrix_NUMBA import ParallelComputedCoefficientMatrix
from oprf.hpc.hpc_model_signals import ModelSignals
from oprf.hpc.hpc_observed_signals import ObservedData, DataType
from oprf.hpc.hpc_orthogonalization import OrthoMatrix
from oprf.hpc.hpc_grid_fit import GridFit
from oprf.hpc.hpc_refine_fit import RefineFit
from oprf.analysis.prf_r2_variance_explain import R2
from oprf.tools.json_file_operations import JsonMgr
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator

###########################################--------Variables------------###############################################################
# gpu_utils.print_gpu_memory_stats()
# mem = gpu_utils.device_available_mem_bytes(device_id=2)

# Variables
num_noisy_signals_per_signal = 1
search_space_rows = int(cfg.search_space["nRows"])
search_space_cols = int(cfg.search_space["nCols"])
search_space_frames = int(cfg.search_space["nSigma"]) #<-----------SIGMA

# ...stimulus
stim_width = int(cfg.stimulus["width"])
stim_height = int(cfg.stimulus["height"])
total_gaussian_curves_per_stim_frame = search_space_rows * search_space_cols
single_gaussian_curve_length = stim_width * stim_height

# ...search space
search_space_xx = np.linspace(-float(cfg.search_space["visual_field"]), float(cfg.search_space["visual_field"]), int(cfg.search_space["nCols"]))
search_space_yy = np.linspace(-float(cfg.search_space["visual_field"]), float(cfg.search_space["visual_field"]), int(cfg.search_space["nRows"]))
search_space_sigma_range = np.linspace(float(cfg.search_space["min_sigma"]), float(cfg.search_space["max_sigma"]), int(cfg.search_space["nSigma"])) # 0.5 to 1.5

# ...Gaussian curves
x_range_cpu = np.linspace(-float(cfg.stimulus["visual_field"]), +float(cfg.stimulus["visual_field"]), int(cfg.stimulus["width"]))
y_range_cpu = np.linspace(-float(cfg.stimulus["visual_field"]), +float(cfg.stimulus["visual_field"]), int(cfg.stimulus["height"]))
x_range_gpu = cp.asarray(x_range_cpu)  # cp.linspace(-9, +9, stim_width)
y_range_gpu = cp.asarray(y_range_cpu) # cp.linspace(-9, +9, stim_height)



###########################################--------Stimulus------------###############################################################
# HRF Curve
hrf_t = np.arange(0, 31, 1) # np.linspace(0, 30, 31)
# hrf_curve = spm_hrf_compat(hrf_t)
hrf_curve = np.array([0, 0.0055, 0.1137, 0.4239, 0.7788, 0.9614, 0.9033, 0.6711, 0.3746, 0.1036, -0.0938, -0.2065, -0.2474, -0.2388, -0.2035, -0.1590, -0.1161, -0.0803, -0.0530, -0.0336, -0.0206, -0.0122, -0.0071, -0.0040, -0.0022, -0.0012, -0.0006, -0.0003, -0.0002, -0.0001, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000]) # mrVista Values
# stimulus = Stimulus(cfg.stimulus["filepath"], size_in_degrees=float(cfg.stimulus["visual_field"]))
# stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2])) #stimulus.org_data.shape[2]
# stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
# stim_frames = stimulus.resampled_hrf_convolved_data.shape[2]

###########################################------------------PROGRAM------------###############################################################
def gpu_compute_model_signals(O_gpu, stimulus):
    ##--------Compute Gaussian Curves, and Gaussian Curves to Timeseries
    # gc class based
    model_signals = ModelSignals(model_type='quadrilateral'
                                     , model_nRows=search_space_rows
                                     , model_nCols =search_space_cols
                                     , model_nFrames=search_space_frames
                                     , model_visual_field = float(cfg.search_space["visual_field"])
                                     , model_min_sigma = float(cfg.search_space["min_sigma"])
                                     , model_max_sigma = float(cfg.search_space["max_sigma"])
                                     , stim_width = stim_width
                                     , stim_height = stim_height
                                     , stim_visual_field = float(cfg.stimulus["visual_field"]))

    # stimulus data on GPU
    stimulus_data_columnmajor_gpu = stimulus.get_flattened_columnmajor_stimulus_data_gpu()

    #---Orthogonalization + Nomalization
    S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches = model_signals.get_orthonormal_signals_in_batches(O_gpu=O_gpu
                                                                                                                                                           , stimulus_data_columnmajor_gpu= stimulus_data_columnmajor_gpu)

    return S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches

def gpu_fitting(O_gpu, arr_2d_location_inv_M_cpu, Y_signals_gpu, Y_signals_cpu, S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches):
    #--------send data to gpu
    # Y_signals_gpu = cp.asarray(Y_signals_cpu)

    #------- Synchronize and release memory
    cp.cuda.Device().synchronize()

    # get the grid-fit results, and error-terms (plus its derivatives)
    best_fit_proj_cpu, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu = GridFit.get_error_terms(O_gpu=O_gpu
                                                                                                    , isResultOnGPU = False
                                                                                                    , Y_signals_gpu=Y_signals_gpu
                                                                                                    , S_prime_columnmajor_gpu=S_prime_columnmajor_gpu_batches
                                                                                                    , dS_prime_dx_columnmajor_gpu=dS_prime_dx_columnmajor_gpu_batches
                                                                                                    , dS_prime_dy_columnmajor_gpu=dS_prime_dy_columnmajor_gpu_batches
                                                                                                    , dS_prime_dsigma_columnmajor_gpu=dS_prime_dsigma_columnmajor_gpu_batches
                                                                                                    , multi_gpu_batching = True
                                                                                                    )

    # perform refine search
    refined_results, Fex_results = RefineFit.get_refined_fit_results(Y_signals_cpu.shape[1]
                                                        , best_fit_proj_cpu
                                                        , arr_2d_location_inv_M_cpu
                                                        , e_cpu
                                                        , de_dx_full_cpu
                                                        , de_dy_full_cpu
                                                        , de_dsigma_full_cpu
                                                        , search_space_rows
                                                        , search_space_cols
                                                        , search_space_frames)

    return refined_results, Fex_results

# ###################################-----------Main()---------------------------------####################
def compute_error_terms(Y_signals_batch_gpu, 
                        O_gpu, 
                        S_prime_columnmajor_gpu_batches, 
                        dS_prime_dx_columnmajor_gpu_batches, 
                        dS_prime_dy_columnmajor_gpu_batches, 
                        dS_prime_dsigma_columnmajor_gpu_batches):    
    
    # grid and refine fitting
    # get the grid-fit results, and error-terms (plus its derivatives)
    best_fit_proj_cpu, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu = GridFit.get_error_terms(O_gpu=O_gpu
                                                                                                , isResultOnGPU = True
                                                                                                , Y_signals_gpu=Y_signals_batch_gpu
                                                                                                , S_prime_columnmajor_gpu=S_prime_columnmajor_gpu_batches
                                                                                                , dS_prime_dx_columnmajor_gpu=dS_prime_dx_columnmajor_gpu_batches
                                                                                                , dS_prime_dy_columnmajor_gpu=dS_prime_dy_columnmajor_gpu_batches
                                                                                                , dS_prime_dsigma_columnmajor_gpu=dS_prime_dsigma_columnmajor_gpu_batches
                                                                                                , multi_gpu_batching = True
                                                                                                )

    return best_fit_proj_cpu, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu


def execute_Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid, result_queue):    
    arr_2d_location_inv_M_cpu = ParallelComputedCoefficientMatrix.Wrapper_Grids2MpInv_numba(mu_X_grid, mu_Y_grid, sigma_grid) ##<<<<--------Compute pre-define matrix M
    result_queue.put(arr_2d_location_inv_M_cpu)

##########------Main()
def main():
    # compute M_inv
    mu_X_grid, mu_Y_grid, sigma_grid = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range)

    # compute MpInv matrix on another thread
    result_queue = queue.Queue()
    MpInv_thread = threading.Thread(target=execute_Grids2MpInv, args=(mu_X_grid, mu_Y_grid, sigma_grid, result_queue))
    MpInv_thread.start()

    # build all stimulus structures
    stimulus_paths = cfg.concatenated_runs['stimulus']
    num_stimulus = len(stimulus_paths)
    stimulus_arr = []
    for stim_idx in range(num_stimulus):
        # load
        stimulus = Stimulus(stimulus_paths[stim_idx], size_in_degrees=float(cfg.stimulus["visual_field"]))
        stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2])) #stimulus.org_data.shape[2]
        stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
        stim_frames = stimulus.resampled_hrf_convolved_data.shape[2]
        stimulus_arr.append(stimulus)

    #...get Orthogonalization matrices
    arr_O_gpu = []    
    for stim_idx in range(num_stimulus):
        ortho_matrix = OrthoMatrix(nDCT=3, num_frame_stimulus=stim_frames)
        O_gpu = ortho_matrix.get_orthogonalization_matrix() # (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))
        arr_O_gpu.append(O_gpu)

    # model signals - depends on the stimulus, if same stimuus is used then, the only one set of model signals is required
    observed_datasets = cfg.concatenated_runs['measured_data']
    num_runs = len(observed_datasets)
    arr_S_prime_columnmajor_gpu_batches = []
    arr_dS_prime_dx_columnmajor_gpu_batches = []
    arr_dS_prime_dy_columnmajor_gpu_batches = []
    arr_dS_prime_dsigma_columnmajor_gpu_batches = []
    for stim_idx in range(num_stimulus):
        S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches = gpu_compute_model_signals(O_gpu=arr_O_gpu[stim_idx], stimulus=stimulus_arr[stim_idx])
        arr_S_prime_columnmajor_gpu_batches.append(S_prime_columnmajor_gpu_batches)
        arr_dS_prime_dx_columnmajor_gpu_batches.append(dS_prime_dx_columnmajor_gpu_batches)
        arr_dS_prime_dy_columnmajor_gpu_batches.append(dS_prime_dy_columnmajor_gpu_batches)
        arr_dS_prime_dsigma_columnmajor_gpu_batches.append(dS_prime_dsigma_columnmajor_gpu_batches)

    # get M-inverse matrix
    MpInv_thread.join()
    if not result_queue.empty():
        arr_2d_location_inv_M_cpu = result_queue.get()

    # load observed datasets on CPU
    arr_Y_signals_cpu = []
    for data_idx in range(num_runs):
        dataset_filepath = observed_datasets[data_idx]  
        y_data = ObservedData(data_type=DataType.measured_data)
        Y_signals_cpu = y_data.get_y_signals(dataset_filepath)
        arr_Y_signals_cpu.append(Y_signals_cpu)    

    ###################
    # process Y-BATCHES
    ###################    
    json_data = None    
    total_y_signals = Y_signals_cpu.shape[1]
    num_batches = cfg.measured_data["batches"]
    batch_size = int(total_y_signals / num_batches)
    for current_batch_idx in range(0, total_y_signals, batch_size):    
        # go through all datasets and compute error terms for each run
        # arr_e_cpu = None #cp.empty((num_runs, batch_size, num_signals)) #[]
        arr_e_cpu = []
        arr_de_dx_full_cpu = []
        arr_de_dy_full_cpu = []
        arr_de_dsigma_full_cpu = []                
        for data_idx in range(num_runs):        
            
            # current Y-BATCH, for current dataset
            Y_signals_batch_gpu = cp.asarray((arr_Y_signals_cpu[data_idx])[:, current_batch_idx: current_batch_idx + batch_size])
            Y_signals_batch_cpu = (arr_Y_signals_cpu[data_idx])[:, current_batch_idx: current_batch_idx + batch_size]
            num_Y_signals_in_batch = Y_signals_batch_cpu.shape[1]

            model_signals_idx = 0 if num_stimulus == 1 else data_idx
            _, e_gpu, de_dx_full_gpu, de_dy_full_gpu, de_dsigma_full_gpu = compute_error_terms(Y_signals_batch_gpu, # <------------------------
                                                                                            arr_O_gpu[model_signals_idx],
                                                                                            arr_S_prime_columnmajor_gpu_batches[model_signals_idx],
                                                                                            arr_dS_prime_dx_columnmajor_gpu_batches[model_signals_idx],
                                                                                            arr_dS_prime_dy_columnmajor_gpu_batches[model_signals_idx],
                                                                                            arr_dS_prime_dsigma_columnmajor_gpu_batches[model_signals_idx])
            if arr_e_cpu is None:
                arr_e_cpu = cp.empty((num_runs, e_gpu.shape[0], e_gpu.shape[1]))
            
            #arr_e_cpu[data_idx] = e_gpu #.append(e_gpu)
            arr_e_cpu.append(e_gpu)
            arr_de_dx_full_cpu.append(de_dx_full_gpu)
            arr_de_dy_full_cpu.append(de_dy_full_gpu)
            arr_de_dsigma_full_cpu.append(de_dsigma_full_gpu)

        # current Y-BATCH concatenated error terms
        # concatenated_e_gpu = np.sum(arr_e_cpu, axis=0)
        concatenated_e_gpu = cp.add(*arr_e_cpu)
        
        # current Y-BATCH concatenated best fit
        best_fit_proj_gpu = np.nanargmax(concatenated_e_gpu, axis=1)

        #  current Y-BATCH refine fit
        refined_matching_results, _ = RefineFit.get_refined_fit_results(Y_signals_batch_cpu.shape[1]
                                                            , cp.asnumpy(best_fit_proj_gpu)
                                                            , arr_2d_location_inv_M_cpu
                                                            , cp.add(*arr_e_cpu) # send overall error terms
                                                            , cp.add(*arr_de_dx_full_cpu)
                                                            , cp.add(*arr_de_dy_full_cpu)
                                                            , cp.add(*arr_de_dsigma_full_cpu)
                                                            , search_space_rows
                                                            , search_space_cols
                                                            , search_space_frames)    

        # current Y-BATCH compute concatenated R2        
        numerators_gpu = cp.empty((num_runs, num_Y_signals_in_batch))
        denominators_gpu = cp.empty((num_runs, num_Y_signals_in_batch))
        #....compute the numerator and denominator terms for each run's dataset individually using the above computed Refinement results. The O_gpu signals depend on the Stimulus so, send the correct one !!!
        for data_idx in range(num_runs):
            model_signals_idx = 0 if num_stimulus == 1 else data_idx
            _, _, stim_frames = stimulus_arr[data_idx].resampled_hrf_convolved_data.shape
            num_gpu, den_gpu = R2.get_r2_numerator_denominator_terms(Y_signals_batch_gpu, arr_O_gpu[model_signals_idx], refined_matching_results, x_range_cpu, y_range_cpu, stimulus_arr[model_signals_idx], stim_height, stim_width)
            numerators_gpu[data_idx] = num_gpu
            denominators_gpu[data_idx] = den_gpu
        
        ## ...compute overall r2 for current Y-BATCH
        r2_numerator_term = cp.sum(numerators_gpu, axis=0)
        r2_inverse_term = (cp.sum(denominators_gpu, axis=0)) ** (-1)
        r2_result_batch = cp.where(r2_numerator_term>0, 1 - r2_numerator_term * r2_inverse_term, r2_numerator_term) 
        batch_json_data = R2.format_in_json_format(r2_result_batch, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames)
        if json_data is None:
            json_data = batch_json_data
        else:
            json_data += batch_json_data

    # write JSON data to file
    #JsonMgr.write_to_file(filepath=cfg.result_file_path, data=json_data)            
    result_file = f"/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/smittal/multi-gpu/results/2024-01-26_sub-sidtest_ses-001_task-bar_run-avg_hemi-L_estimates{cfg.results['custom_filename_postfix']}.json"
    JsonMgr.write_to_file(filepath=result_file, data=json_data)

if __name__ == "__main__":
    main()
    #cProfile.run('main()', sort='cumulative')

print("Analysis done")




