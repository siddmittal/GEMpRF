import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cProfile
import os
import datetime
from enum import Enum

# Config
import sys
sys.path.append(r'/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/smittal/dgx-versions/fmri/')
sys.path.append(r'/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/smittal/dgx-versions/fmri/codebase')
sys.path.append(r'D:\code\sid-git\fmri\codebase')
from all_configs.config_manager import ConfigurationWrapper as cfg
from all_configs.config_manager import RunType
cfg.load_configuration(run_type=RunType.ANALYSIS)

# Local Imports
import sys
sys.path.append(cfg.path_to_append)

# # # NOTE: Test-------------------------------####################################
# # from gem.data.bids_handler import BidsHandler
# # files_info = BidsHandler.get_input_filepaths(cfg.bids)
# # result_files = BidsHandler.inputpath2resultpath(cfg.bids, files_info[0])


# gem
from gem.model.prf_model import PRFModel
from gem.model.prf_model import GaussianModelParams
from gem.model.prf_gaussian_model import PRFGaussianModel
from gem.space.PRFSpace import PRFSpace
from gem.fitting.hpc_grid_fit import GridFit
from gem.fitting.hpc_refine_fit import RefineFit
from gem.analysis.prf_analysis import PRFAnalysis
from gem.space.coefficient_matrix import CoefficientMatix
from gem.utils.hpc_cupy_utils import Utils as gpu_utils
from gem.model.selected_prf_model import SelectedPRFModel
from gem.signals.signal_synthesizer import SignalSynthesizer
from gem.model.prf_stimulus import Stimulus
from gem.utils.logger import Logger
from gem.analysis.prf_r2_variance_explain import R2
from gem.data.observed_data import ObservedData, DataSource
from gem.signals.orthogonalization_matrix import OrthoMatrix


# oprf
# from oprf.hpc.hpc_cupy_utils import Utils as gpu_utils
# from oprf.standard.prf_stimulus import Stimulus
# from oprf.hpc.hpc_coefficient_matrix import CoefficientMatrix
# from oprf.hpc.hpc_coefficient_matrix_NUMBA import ParallelComputedCoefficientMatrix
# from oprf.hpc.hpc_model_signals import ModelSignals
# from oprf.hpc.hpc_orthogonalization import OrthoMatrix
# from oprf.hpc.hpc_refine_fit import RefineFit
# # from oprf.analysis.prf_r2_variance_explain import R2
# from oprf.hpc.hpc_search_space import SearchSpace
# from oprf.hpc.hpc_matrix_operations import MatrixOps
from oprf.tools.json_file_operations import JsonMgr
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator


class SpaceTypeEnum(Enum):
    TEST = 1
    SEARCH = 2

###########################################--------Variables------------###############################################################
# Variables
num_noisy_signals_per_signal = 1
search_space_rows = int(cfg.search_space["nRows"])
search_space_cols = int(cfg.search_space["nCols"])
search_space_frames = int(cfg.search_space["nSigma"]) #<-----------SIGMA

# noise
noisy_signals_space_rows = int(cfg.test_space["nRows"])
noisy_signals_space_cols = int(cfg.test_space["nCols"])
noisy_signals_space_frames = int(cfg.test_space["nSigma"]) #<-----------SIGMA

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
x_range_gpu = cp.asarray(x_range_cpu)
y_range_gpu = cp.asarray(y_range_cpu)


###########################################--------Stimulus------------###############################################################
# HRF Curve
hrf_t = np.arange(0, 31, 1) # np.linspace(0, 30, 31)
# hrf_curve = spm_hrf_compat(hrf_t)
hrf_curve = np.array([0, 0.0055, 0.1137, 0.4239, 0.7788, 0.9614, 0.9033, 0.6711, 0.3746, 0.1036, -0.0938, -0.2065, -0.2474, -0.2388, -0.2035, -0.1590, -0.1161, -0.0803, -0.0530, -0.0336, -0.0206, -0.0122, -0.0071, -0.0040, -0.0022, -0.0012, -0.0006, -0.0003, -0.0002, -0.0001, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000]) # mrVista Values
stimulus = Stimulus(cfg.stimulus["filepath"], size_in_degrees=float(cfg.stimulus["visual_field"]), stim_config = cfg.stimulus)

stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2])) #stimulus.org_data.shape[2]
stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
stim_frames = stimulus.resampled_data.shape[2]

###########################################------------------PROGRAM------------###############################################################
def compute_orthonormalized_signals(O_gpu, prf_space : PRFSpace, prf_model : PRFModel, stimulus : Stimulus):
    S_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE)            
    dS_dx_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.X0)   
    dS_dy_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.Y0)
    dS_dsigma_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.SIGMA)

    orthonormalized_S_cm_gpu_batches, orthonormalized_dervatives_signals_batches_list = SignalSynthesizer.orthonormalize_modelled_signals(O_gpu=O_gpu, 
                                                                                                                                            model_signals_rm_batches= S_batches, 
                                                                                                                                            dS_dtheta_rm_batches_list = [dS_dx_batches, dS_dy_batches, dS_dsigma_batches] ) 

    return orthonormalized_S_cm_gpu_batches, orthonormalized_dervatives_signals_batches_list


##########################---------------Main-----------------################################################################################################
import threading
import queue

# For new Irregular XY points distribution
def execute_Grids2MpInv_NewMethod(prf_space : PRFSpace, result_queue):    
    prf_space.compute_multidim_points_neighbours()        
    arr_2d_location_inv_M = CoefficientMatix.Wrapper_Grids2MpInv_numba(prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_vf_neighbours)
    result_queue.put(arr_2d_location_inv_M)    

def get_prf_spatial_points():
    x_mesh, y_mesh = np.meshgrid(search_space_xx, search_space_yy) # NOTE: (col, row)
    spatial_points = np.column_stack((y_mesh.ravel(), x_mesh.ravel())) # (col i.e. x, row i.e. y)
    
    return spatial_points

def get_refined_signals_cpu(refined_prf_params_YX : np.ndarray, prf_model : PRFModel, stimulus : Stimulus):
    refined_S_batches_gpu = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=refined_prf_params_YX, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE)            
    
    refined_S_cpu = []
    # refined signal batches could be present on different GPUs
    for batch_idx in range(len(refined_S_batches_gpu)):
        device_id = refined_S_batches_gpu[batch_idx].device.id
        with cp.cuda.Device(device_id):
            refined_signal_batch_cpu = cp.asnumpy(refined_S_batches_gpu[batch_idx])
            refined_S_cpu.append(refined_signal_batch_cpu)
    
    refined_S_cpu = np.concatenate(refined_S_cpu, axis=0)
    # refined_S_cpu = cp.asnumpy(cp.concatenate(refined_S_batches_gpu, axis=0))

    return refined_S_cpu

def get_valid_refined_data(refined_matching_results_XY, Y_signals_gpu, O_gpu, prf_model, stimulus, coarse_e_cpu,  best_fit_proj_cpu , coarse_pRF_estimations):
    refined_prf_points_YX = refined_matching_results_XY
    refined_prf_points_YX[:, [0, 1]] = refined_prf_points_YX[:, [1, 0]] # the CUDA code expected the (row, col) i.e. (y, x) convention

    # refined S batches
    refined_S_batches_gpu = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=refined_prf_points_YX, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE)            

    # refined S' batches
    orthonormalized_S_cm_gpu_batches, _ = SignalSynthesizer.orthonormalize_modelled_signals(O_gpu=O_gpu,
                                                                                            model_signals_rm_batches=refined_S_batches_gpu,
                                                                                            dS_dtheta_rm_batches_list=[])
    # refined error
    _, refined_e_cpu, _ = GridFit.get_error_terms(isResultOnGPU=False, 
                                                                Y_signals_gpu=Y_signals_gpu, 
                                                                S_prime_cm_batches_gpu=orthonormalized_S_cm_gpu_batches, 
                                                                dS_prime_dtheta_cm_batches_list_gpu=[])

    # ...get the locations where the errors are getting worse (ideally (refined - coarse) should be >0)
    coarse_error_vector = coarse_e_cpu[np.arange(len(coarse_e_cpu)), best_fit_proj_cpu]
    refined_error_vector = np.diagonal(refined_e_cpu)
    worsened_error_y_signal_indices = np.argwhere((~np.isnan(refined_error_vector - coarse_error_vector)) & (refined_error_vector - coarse_error_vector < 0))

    # keep the coarse pRF parameters where the refined esitmations got worse
    refined_prf_points_YX[worsened_error_y_signal_indices, :] = coarse_pRF_estimations[worsened_error_y_signal_indices, :]

    # swap back to (X, Y) convention
    refined_matching_results_XY = refined_prf_points_YX
    refined_matching_results_XY[:, [0, 1]] = refined_matching_results_XY[:, [1, 0]] 

    return refined_matching_results_XY



def get_bids_measured_data_info():
    measured_data_info = []
    data1 = ('01', '002', '001', os.join(cfg.bids['basepath'], 'sub-002', 'ses-001', 'func', 'sub-002_ses-001_task-prf_run-01_bold.nii.gz'))
    data2 = ('01', '002', '001', os.join(cfg.bids['basepath'], 'sub-002', 'ses-001', 'func', 'sub-002_ses-001_task-prf_run-01_bold.nii.gz'))
    measured_data_info.append(data1)
    measured_data_info.append(data2)

    return measured_data_info

###############
def main(prf_spatial_points):
    selected_prf_model = SelectedPRFModel.Gaussian

    # define PRF Points
    # ...create additional dimensions
    # # y_mesh, x_mesh = np.meshgrid(search_space_yy, search_space_xx, indexing='ij') # NOTE: (row, col)
    # # spatial_points = np.column_stack((y_mesh.ravel(), x_mesh.ravel()))

    # Get INPUT points
    # NOTE: swap x and y because the program works with (row, col) i.e. (y, x) convention
    spatial_points = prf_spatial_points
    spatial_points[:, [0, 1]] = spatial_points[:, [1, 0]] 

    additional_dimensions = PRFSpace.make_extra_dimensions(search_space_sigma_range)
    prf_space = PRFSpace(spatial_points, additional_dimensions=additional_dimensions)
    prf_space.convert_spatial_to_multidim()
    # prf_points.compute_neighbours_for_each_multi_dimensional_point()    

    # M-Matrix using new method (Irregular XY points distribution)    
    result_queue = queue.Queue()    
    MpInv_thread = threading.Thread(target=execute_Grids2MpInv_NewMethod, args=(prf_space, result_queue))
    MpInv_thread.start()
    # # # time.sleep(30)

    #...get Orthogonalization matrix
    ortho_matrix = OrthoMatrix(nDCT=3, num_frame_stimulus=stim_frames)
    O_gpu = ortho_matrix.get_orthogonalization_matrix() # (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))

    #...compute Model Signals
    if selected_prf_model == SelectedPRFModel.Gaussian:            
        prf_analysis = PRFAnalysis(prf_space=prf_space, stimulus=stimulus) # to hold all the information about this analysis run
        prf_model = PRFGaussianModel()        

        # compute all signals
        prf_analysis.orthonormalized_S_batches, prf_analysis.orthonormalized_dS_dtheta_batches_list = compute_orthonormalized_signals(O_gpu=O_gpu, 
                                                                                                                                      prf_space= prf_space, 
                                                                                                                                      prf_model= prf_model, 
                                                                                                                                      stimulus= stimulus)  
        Logger.print_green_message("model signals computed...", print_file_name=False)

    # get M-inverse matrix
    MpInv_thread.join()
    if not result_queue.empty():
        arr_2d_location_inv_M_cpu = result_queue.get()

    # measured data
    measured_data = None
    measured_data_info = None
    if cfg.bids['@enable'] == "False":
        measured_data = cfg.fixed_paths['measured_data_filepath']['filepath']
    else:
        measured_data_info = get_bids_measured_data_info()
    # # measured_data = cfg.measured_data['filepaths']
    result_base_path = cfg.results['basepath']
    for data_idx in range(len(measured_data)):
        # result name
        file = os.path.basename(measured_data[data_idx])
        filename = (file.split("."))[0]
        filename = (str(str(datetime.date.today()) + '_') if cfg.results['prepend_date'] == "True" else '') + filename
        result_file = result_base_path + filename.replace("bold", "estimates") + cfg.results['custom_filename_postfix'] + ".json"

        # y-signals
        y_data = ObservedData(data_source=DataSource.measured_data)
        Y_signals_cpu = y_data.get_y_signals(measured_data[data_idx])

        # final json data
        json_data = None

        # process bathches
        total_y_signals = Y_signals_cpu.shape[1]
        num_batches = int(cfg.measured_data["batches"])
        batch_size = int(total_y_signals / num_batches)
        for current_batch_idx in range(0, total_y_signals, batch_size):
            Y_signals_batch_gpu = cp.asarray(Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size])
            Y_signals_batch_cpu = Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size]

            # error
            # prf_analysis.error_e = (Y_signals_batch_gpu.T @ dS_prime_dtheta_columnmajor_gpu)
            best_fit_proj_cpu, e_cpu, de_dtheta_list_cpu = GridFit.get_error_terms(isResultOnGPU=False, 
                                                                Y_signals_gpu=Y_signals_batch_gpu, 
                                                                S_prime_cm_batches_gpu=prf_analysis.orthonormalized_S_batches, 
                                                                dS_prime_dtheta_cm_batches_list_gpu=prf_analysis.orthonormalized_dS_dtheta_batches_list)
            
            Logger.print_green_message(f"error computed for batch {current_batch_idx} - {current_batch_idx + min(batch_size, total_y_signals-current_batch_idx) }...", print_file_name=False)

            # NOTE: RefineFit produces results in (X, Y) format
            # perform refine search, the obtained refined results will be in the (X, Y) format
            num_Y_signals = Y_signals_batch_cpu.shape[1]
            refined_matching_results_XY, Fex_results = RefineFit.get_refined_fit_results(
                prf_space,
                num_Y_signals,
                best_fit_proj_cpu,
                arr_2d_location_inv_M_cpu,
                e_cpu,
                de_dtheta_list_cpu,
                search_space_rows,
                search_space_cols,
                search_space_frames, search_space_xx,
                search_space_yy,
                search_space_sigma_range)            

            # validate if the refined pRF estimations are really improving the error value, and for the pRF points where error is getting worse, keep the coarse pRF estimations
            coarse_pRF_estimations = prf_space.multi_dim_points_cpu[best_fit_proj_cpu]
            valid_refined_prf_points_XY = get_valid_refined_data(refined_matching_results_XY, Y_signals_gpu=Y_signals_batch_gpu,
                                                                 O_gpu=O_gpu, prf_model=prf_model, stimulus=stimulus, coarse_e_cpu=e_cpu, best_fit_proj_cpu=best_fit_proj_cpu, coarse_pRF_estimations=coarse_pRF_estimations)

            # compute timecourses for refined pRF estimated params
            valid_refined_prf_points_YX = valid_refined_prf_points_XY
            valid_refined_prf_points_YX[:, [0, 1]] = valid_refined_prf_points_YX[:, [1, 0]] # the CUDA code expected the (row, col) i.e. (y, x) convention
            valid_refined_S_cpu = get_refined_signals_cpu(valid_refined_prf_points_YX, prf_model, stimulus) # NOTE: need to be replaced with VALID refined data

            Logger.print_red_message("RESULTS CORRECT? Make sure to use set correct Stimulus_VF and <visual_field> = 9", print_file_name=False)

            # compute Variance Explained
            r2_results = R2.get_r2_num_den_method_with_epsilon_as_yTs(Y_signals_batch_gpu, O_gpu, refined_matching_results_XY, x_range_cpu, y_range_cpu, stimulus, refined_signals_gpu = valid_refined_S_cpu, prf_model= prf_model)
                                
            # format results to JSON                
            batch_json_data = R2.format_in_json_format( r2_results, refined_matching_results_XY, valid_refined_S_cpu)                    
            
            if json_data is None:
                json_data = batch_json_data
            else:
                json_data += batch_json_data

        # write JSON data to file
        #JsonMgr.write_to_file(filepath=cfg.result_file_path, data=json_data)
        JsonMgr.write_to_file(filepath=result_file, data=json_data)


    print("Done")

    




# run the main function
if __name__ == "__main__":
    prf_spatial_points = get_prf_spatial_points()
    main(prf_spatial_points)