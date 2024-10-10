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
stimulus = Stimulus(cfg.stimulus["filepath"], size_in_degrees=float(cfg.stimulus["visual_field"]))
stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2])) #stimulus.org_data.shape[2]
stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
stim_frames = stimulus.resampled_hrf_convolved_data.shape[2]

###########################################------------------PROGRAM------------###############################################################
def gpu_compute_model_signals(O_gpu):
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
    best_fit_proj_cpu, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu = GridFit.get_error_terms(isResultOnGPU=False,
                                                                                                           O_gpu=O_gpu,
                                                                                                           Y_signals_gpu=Y_signals_gpu,
                                                                                                           S_prime_columnmajor_gpu=S_prime_columnmajor_gpu_batches,
                                                                                                           dS_prime_dx_columnmajor_gpu=dS_prime_dx_columnmajor_gpu_batches,
                                                                                                           dS_prime_dy_columnmajor_gpu=dS_prime_dy_columnmajor_gpu_batches,
                                                                                                           dS_prime_dsigma_columnmajor_gpu=dS_prime_dsigma_columnmajor_gpu_batches,
                                                                                                           multi_gpu_batching=True
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
def load_and_evaluate_measured_data(dataset_filepath, result_base_path, O_gpu, arr_2d_location_inv_M_cpu, S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches):
    # measured data
    measured_data = [dataset_filepath] # cfg.measured_data['filepaths']
    result_base_path = result_base_path #cfg.results['basepath']
    for data_idx in range(len(measured_data)):
        # result name
        file = os.path.basename(measured_data[data_idx])
        filename = (file.split("."))[0]
        filename = (str(str(datetime.date.today()) + '_') if cfg.results['prepend_date'] == "True" else '') + filename
        result_file = result_base_path + '/' + filename.replace("bold", "estimates") + cfg.results['custom_filename_postfix'] + ".json"

        # y-signals
        y_data = ObservedData(data_type=DataType.measured_data)
        Y_signals_cpu = y_data.get_y_signals(measured_data[data_idx])

        # final json data
        json_data = None

        # process bathches
        total_y_signals = Y_signals_cpu.shape[1]
        num_batches = cfg.measured_data["batches"]
        batch_size = int(total_y_signals / num_batches)
        for current_batch_idx in range(0, total_y_signals, batch_size):
            Y_signals_batch_gpu = cp.asarray(Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size])
            Y_signals_batch_cpu = Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size]

            # grid and refine fitting
            refined_matching_results, Fex_results = gpu_fitting(O_gpu, arr_2d_location_inv_M_cpu, Y_signals_batch_gpu, Y_signals_batch_cpu, S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches)

            # verfication R2
            ## r2 with e as yts
            r2_results = R2.get_r2_num_den_method_with_epsilon_as_yTs(Y_signals_batch_gpu, O_gpu, refined_matching_results, x_range_cpu, y_range_cpu, stimulus)
            # r2_results = R2.get_OPTIMIZED_r2_new_method_with_epsilon_as_yTs(Y_signals_batch_gpu, O_gpu, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames)
            batch_json_data = R2.format_in_json_format(r2_results, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames)
            if json_data is None:
                json_data = batch_json_data
            else:
                json_data += batch_json_data

        # write JSON data to file
        #JsonMgr.write_to_file(filepath=cfg.result_file_path, data=json_data)
        JsonMgr.write_to_file(filepath=result_file, data=json_data)
        #JsonMgr.log_string(filepath='/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/hcpret/BIDS/derivatives/prfanalyze-gem/analysis-01/log-gem.txt', message=f'computed: {dataset_filepath}')

    return

def execute_Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid, result_queue):    
    arr_2d_location_inv_M_cpu = ParallelComputedCoefficientMatrix.Wrapper_Grids2MpInv_numba(mu_X_grid, mu_Y_grid, sigma_grid) ##<<<<--------Compute pre-define matrix M
    result_queue.put(arr_2d_location_inv_M_cpu)

def main():
    # compute M_inv
    mu_X_grid, mu_Y_grid, sigma_grid = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range)

    # NOTE
    # arr_2d_location_inv_M_cpu = CoefficientMatrix.Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid) ##<<<<--------Compute pre-define matrix M    

    # compute MpInv matrix on another thread
    result_queue = queue.Queue()
    MpInv_thread = threading.Thread(target=execute_Grids2MpInv, args=(mu_X_grid, mu_Y_grid, sigma_grid, result_queue))
    MpInv_thread.start()

    #...get Orthogonalization matrix
    ortho_matrix = OrthoMatrix(nDCT=3, num_frame_stimulus=stim_frames)
    O_gpu = ortho_matrix.get_orthogonalization_matrix() # (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))

    # model signals
    S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches = gpu_compute_model_signals(O_gpu=O_gpu)

    # get M-inverse matrix
    MpInv_thread.join()
    if not result_queue.empty():
        arr_2d_location_inv_M_cpu = result_queue.get()

    # go through all datasets
    task = cfg.stimulus['task'] #'task-bars1'
    directory_path = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/hcpret/BIDS/derivatives/prfprepare/analysis-01"
    # directory_path = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/02/BIDS"
    for folder_name in sorted(os.listdir(directory_path)):
        if folder_name.startswith('sub-') and os.path.isdir(os.path.join(directory_path, folder_name)):
            for ses_value in ["001","002", "003"]:
                dataset_folder_path = os.path.join(directory_path, folder_name + f'/ses-{ses_value}/func/') # sample_path = "Y:/data/hcpret/BIDS/derivatives/prfprepare/analysis-01/sub-100610/ses-001/func"                
                
                # compute result base path
                pattern = re.compile(r'/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/hcpret/BIDS/derivatives/prfprepare/analysis-01/(.*?)/func') #  regular expression pattern
                # pattern = re.compile(r'/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/02/BIDS/(.*?)/func') #  regular expression pattern
                match = pattern.search(dataset_folder_path)        
                if match:
                    extracted_part = match.group(1)            
                    result_base_path = os.path.join('/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/hcpret/BIDS/derivatives/prfanalyze-gem/analysis-01/', extracted_part)
                    # result_base_path = os.path.join('/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/02/BIDS/derivatives/prfanalyze-gem/analysis-02/', extracted_part)
                    # print(result_base_path)

                # dataset to process
                # Iterate through all files in the specified directory
                for file_name in os.listdir(dataset_folder_path):
                    file_path = os.path.join(dataset_folder_path, file_name)
                    
                    # Check if the file contains 'task-bars1' in its name
                    if task in file_name and file_name.endswith('.nii.gz') and os.path.isfile(file_path):                
                        dataset_filepath = file_path
                        
                        # compute and save results
                        load_and_evaluate_measured_data(dataset_filepath, result_base_path, O_gpu, arr_2d_location_inv_M_cpu, S_prime_columnmajor_gpu_batches, dS_prime_dx_columnmajor_gpu_batches, dS_prime_dy_columnmajor_gpu_batches, dS_prime_dsigma_columnmajor_gpu_batches)


    

if __name__ == "__main__":
    main()
    #cProfile.run('main()', sort='cumulative')

print("HCP done")




