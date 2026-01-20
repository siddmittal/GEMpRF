# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Medical University of Vienna",
"@Desc    :   None",
        
"""
import math
import shutil
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import cupy as cp
import threading
import queue
import os
import datetime
import cProfile
import pstats
import pandas as pd
import time
import datetime
from contextlib import nullcontext

# gem
from gem.model.prf_model import PRFModel
from gem.model.prf_model import GaussianModelParams
from gem.model.prf_gaussian_model import PRFGaussianModel
from gem.space.PRFSpace import PRFSpace
from gem.fitting.hpc_grid_fit import GridFit
from gem.fitting.hpc_refine_fit import RefineFit
from gem.analysis.prf_analysis import PRFAnalysis
from gem.space.coefficient_matrix import CoefficientMatix
from gem.utils.hpc_cupy_utils import HpcUtils as gpu_utils
from gem.model.selected_prf_model import SelectedPRFModel
from gem.signals.signal_synthesizer import SignalSynthesizer
from gem.model.prf_stimulus import Stimulus
from gem.utils.logger import Logger
from gem.utils.gem_write_to_file import GemWriteToFile
from gem.analysis.prf_r2_variance_explain import R2
from gem.data.observed_data import ObservedData, DataSource
from gem.signals.orthogonalization_matrix import OrthoMatrix
from gem.data.bids_handler import GemBidsHandler
from gem.data.gem_bids_concatenation_data_info import BidsConcatenationDataInfo
from gem.data.gem_stimulus_file_info import StimulusFileInfo
from gem.tools.json_file_operations import JsonMgr
from gem.utils.gem_gpu_manager import GemGpuManager as ggm
from gem.utils.logger import Logger
from gem.utils.gem_h5_file_handler import H5FileManager
from gem.signals.hrf_generator import spm_hrf_compat

class GEMpRFAnalysis:
    __selected_prf_model = SelectedPRFModel.NoneType

    @classmethod
    def get_hrf_curve(cls, cfg, stimulus : Stimulus):
        if cfg.optional_analysis_params['enable'] and cfg.optional_analysis_params['hrf']['use_from_file']:
            hrf_curve = H5FileManager.get_key_value(cfg.optional_analysis_params['filepath'], cfg.optional_analysis_params['hrf']['key'])
            if hrf_curve is None:
                Logger.print_red_message(f"Could not load HRF curve from file: {cfg.optional_analysis_params['filepath']} with key: {cfg.optional_analysis_params['hrf']['key']}", print_file_name=False)
                sys.exit(1)
        else:
            # get TR
            if cfg.default_hrf["TR"] is None:                
                TR = stimulus.Header['pixdim'][4]  # now we need to get TR from stimulus # assuming 4th dimension is time
                Logger.print_yellow_message(f"\nSetting HRF 't' step value to stimulus ({stimulus.StimulusTaskName}) TR: {TR:.3f} seconds.", print_file_name=False)
                cfg.default_hrf["t"] = (*cfg.default_hrf["t"][:2], TR)
            else:
                TR = cfg.default_hrf["TR"]
            
            # spm "t" (start, stop, step/TR)
            cfg.default_hrf["t"] = (*cfg.default_hrf["t"], TR)

            # generate HRF curve using SPM parameters
            hrf_params = (np.arange(*cfg.default_hrf["t"]),
                          cfg.default_hrf["peak_delay"], 
                          cfg.default_hrf["under_shoot_delay"], 
                          cfg.default_hrf["peak_disp"], 
                          cfg.default_hrf["under_disp"], 
                          cfg.default_hrf["peak_to_undershoot"], 
                          cfg.default_hrf["normalize"])
            hrf_curve = spm_hrf_compat(*hrf_params)
        return hrf_curve

    @classmethod
    def load_config(cls, config_filepath : str = None):        
        from gem.configs.config_manager import ConfigurationWrapper as cfg
        cfg.load_configuration(run_type=None, config_filepath=config_filepath)

        return cfg

    @classmethod
    def load_stimulus(cls, cfg, stimulus_info : StimulusFileInfo = None)-> Stimulus:
        # ...stimulus
        stim_width = int(cfg.stimulus["width"])
        stim_height = int(cfg.stimulus["height"])  
        binarize = True if cfg.stimulus["binarization"].get("@enable") == "True" else False
        binarize_threshold = float(cfg.stimulus["binarization"].get("@threshold"))
        high_temporal_resolution_info = cfg.stimulus_high_temporal_resolution if cfg.stimulus_high_temporal_resolution['enable'] else None

        stimulus = Stimulus(os.path.join(stimulus_info.stimulus_dir, 
                                         stimulus_info.stimulus_filename), 
                                         size_in_degrees=float(cfg.stimulus["visual_field"]), 
                                         stim_config = cfg.stimulus, 
                                         binarize=binarize, 
                                         binarize_threshold=binarize_threshold,
                                         high_temporal_resolution_info=high_temporal_resolution_info,
                                         stimulus_task_name=stimulus_info.stimulus_task)

        # get HRF curve
        hrf_curve = cls.get_hrf_curve(cfg, stimulus)
        GemWriteToFile.get_instance().write_array_to_h5(hrf_curve, variable_path=['hrf'], append_to_existing_variable=False)

        stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2])) #stimulus.org_data.shape[2]
        stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
        GemWriteToFile.get_instance().write_array_to_h5(stimulus.resampled_data, variable_path=[f'stimulus', f'{stimulus_info.stimulus_task}', 'resampled_data'], append_to_existing_variable=False)
        GemWriteToFile.get_instance().write_array_to_h5(stimulus.stimulus_data_cpu, variable_path=[f'stimulus', f'{stimulus_info.stimulus_task}', 'stimulus_data_hrf_convolved'], append_to_existing_variable=False)

        return stimulus
        
    @classmethod
    def get_prf_spatial_points(cls, cfg)-> np.ndarray:
        search_space_xx = np.linspace(-float(cfg.default_spatial_grid["visual_field_radius"]), float(cfg.default_spatial_grid["visual_field_radius"]), int(cfg.default_spatial_grid["num_horizontal_prfs"])) # nCols
        search_space_yy = np.linspace(-float(cfg.default_spatial_grid["visual_field_radius"]), float(cfg.default_spatial_grid["visual_field_radius"]), int(cfg.default_spatial_grid["num_vertical_prfs"])) # nRows
        x_mesh, y_mesh = np.meshgrid(search_space_xx, search_space_yy) # NOTE: (col, row)
        spatial_points_xy = np.column_stack((y_mesh.ravel(), x_mesh.ravel())) # (col i.e. x, row i.e. y)        
        return spatial_points_xy
    
    @classmethod
    def get_additional_dimensions(cls, cfg, selected_prf_model : SelectedPRFModel):
        if selected_prf_model == SelectedPRFModel.GAUSSIAN:
            if cfg.optional_analysis_params['enable'] and cfg.optional_analysis_params['sigmas']['use_from_file']: # Get user defined custom values for sigma from H5 file
                search_space_sigma_range = H5FileManager.get_key_value(filepath=cfg.optional_analysis_params['filepath'], key = cfg.optional_analysis_params['sigmas']['key'])
                if search_space_sigma_range is None:
                    Logger.print_red_message(f"Could not load sigma range from file: {cfg.optional_analysis_params['filepath']} with key: {cfg.optional_analysis_params['sigmas']['key']}", print_file_name=False)
                    sys.exit(1)
            else: # if user hasn't specifed anything, use default values
                search_space_sigma_range = np.linspace(float(cfg.default_sigmas['min_sigma']), float(cfg.default_sigmas['max_sigma']), int(cfg.default_sigmas['num_sigmas'])) # 0.5 to 1.5

            additional_dimensions = PRFSpace.make_extra_dimensions(search_space_sigma_range)
        else:
            raise ValueError("Invalid PRF Model")

        return additional_dimensions

    @classmethod
    def execute_Grids2MpInv_NewMethod(cls, prf_space : PRFSpace, result_queue):    
        prf_space.compute_multidim_points_neighbours()        
        arr_2d_location_inv_M = CoefficientMatix.Wrapper_Grids2MpInv_numba(prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_vf_neighbours)
        result_queue.put(arr_2d_location_inv_M) 

    @classmethod
    def get_selected_prf_model(cls, cfg):
        if cfg.pRF_model_details['model'] == "2d_gaussian":
            cls.__selected_prf_model = SelectedPRFModel.GAUSSIAN
        else:
            raise ValueError("Invalid PRF Model")

        return cls.__selected_prf_model

    @classmethod    
    def compute_orthonormalized_signals(cls, O_gpu, prf_space : PRFSpace, prf_model : PRFModel, stimulus : Stimulus, cfg, stimulus_task_name : str = None):
        # model signals
        S_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE, cfg=cfg)

        subcat = f"{stimulus_task_name}" if stimulus_task_name is not None else ""

        # model derivatives signals
        dS_dtheta_batches_list = []
        if cfg.refine_fitting_enabled:
            num_theta = prf_model.num_dimensions
            for theta_idx in range(num_theta):
                dS_dtheta_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams(theta_idx), cfg=cfg)
                dS_dtheta_batches_list.append(dS_dtheta_batches)
                GemWriteToFile.get_instance().write_array_to_h5(dS_dtheta_batches, variable_path=[f'model', f'{subcat}', f'model_signals_derivative_d{theta_idx}'], append_to_existing_variable=False)

        # Orthonormalized model + derivatives signals
        orthonormalized_S_cm_gpu_batches, orthonormalized_dervatives_signals_batches_list = SignalSynthesizer.orthonormalize_modelled_signals(O_gpu=O_gpu, 
                                                                                                                                        model_signals_rm_batches= S_batches, 
                                                                                                                                        dS_dtheta_rm_batches_list = dS_dtheta_batches_list)
        # Write debug info
        GemWriteToFile.get_instance().write_array_to_h5(S_batches, variable_path=[f'model', f'{subcat}', 'model_signals'], append_to_existing_variable=False)  
        GemWriteToFile.get_instance().write_array_to_h5(orthonormalized_S_cm_gpu_batches, variable_path=[f'model', f'{subcat}', 'orthonormalized_model_signals'], append_to_existing_variable=False)  
        if orthonormalized_dervatives_signals_batches_list is not None:
            for theta_idx in range(len(orthonormalized_dervatives_signals_batches_list)):
                GemWriteToFile.get_instance().write_array_to_h5(orthonormalized_dervatives_signals_batches_list[theta_idx], variable_path=[f'model', f'{subcat}', f'orthonormalized_model_signals_derivative_d{theta_idx}'], append_to_existing_variable=False)
        return orthonormalized_S_cm_gpu_batches, orthonormalized_dervatives_signals_batches_list    

    @classmethod
    def get_single_run_data_files_info(cls, cfg):
        # List of input measured data filepaths
        measured_data_list = None
        if cfg.bids['@enable'] == "True":
            measured_data_info_list = GemBidsHandler.get_input_filepaths(bids_config=cfg.bids, stimuli_dir_path= cfg.stimulus['directory'])
            measured_data_list = (lambda x: np.array(x)[:, 0] if x else np.array([]))(measured_data_info_list)  # extract only filepaths from the list of tuples
        else:
            measured_data_list = cfg.fixed_paths['measured_data_filepath']['filepath']
            if isinstance(measured_data_list, str):
                measured_data_list = [measured_data_list] # Ensure it is always an array
            
        # List of result filepaths
        result_filepaths_list = []
        for data_idx in range(len(measured_data_list)):
            if cfg.bids['@enable'] == "True":
                result_file =  GemBidsHandler.inputpath2resultpath(cfg.bids, measured_data_info_list[data_idx], analysis_id=cfg.bids["results_anaylsis_id"]["#text"])                
            else:
                file = os.path.basename(measured_data_list[data_idx])
                filename = (file.split("."))[0]
                filename = (str(str(datetime.date.today()) + '_') if cfg.results['prepend_date'] == "True" else '') + filename
                result_base_path = cfg.results['basepath']                
                custom_postfix = cfg.results['custom_filename_postfix'] if cfg.results['custom_filename_postfix'] is not None else ""
                result_file = os.path.join(result_base_path, filename.replace("bold", "estimates")+ custom_postfix + ".json")                
            
            # append to list
            result_filepaths_list.append(result_file)
        
        return measured_data_list, result_filepaths_list

    @classmethod
    def get_concatenated_runs_data_files_info(cls, cfg):
        # List of input measured data filepaths
        if cfg.bids['@enable'] == "True":
            measured_data_info_list = GemBidsHandler.get_input_filepaths(bids_config=cfg.bids, stimuli_dir_path= cfg.stimulus['directory'])
        else:
            raise ValueError("Invalid Configuration: Concatenation runs are only supported for BIDS data")
            
        # compute result filepaths for the concatenated items
        num_specified_concatenated_items = len(cfg.bids.get("concatenated").get("concatenate_item"))
        num_found_concatenated_items = len(measured_data_info_list)
        if num_specified_concatenated_items != num_found_concatenated_items:
            raise ValueError(f"Number of specified concatenated items ({num_specified_concatenated_items}) does not match the number of found concatenated items ({num_found_concatenated_items})")

        # Making sure that each measured_data_info_list is sorted based on the filepath
        for sublist in measured_data_info_list:            
            sublist.sort(key=lambda x: x[0]) # Sort each sublist based on the filepath (sublist[i][0])

        required_concatenations_info = []
        for items_to_be_concatenated_info in zip(*measured_data_info_list):
            input_filepaths = [item[0] for item in items_to_be_concatenated_info]
            data_info_dictionaries_list = [item[1] for item in items_to_be_concatenated_info]    
            stimulus_info = [item[2] for item in items_to_be_concatenated_info]            
            # print(input_filepaths)
            # print(data_info_dictionaries_list)
            concatenation_result_info = BidsConcatenationDataInfo.compare_and_merge_data_info_dicts(data_info_dictionaries_list)
            concatenated_result_filename = GemBidsHandler.get_concatenated_result_filepath(cfg.bids, input_filepaths[0], concatenation_result_info)
            concatenation_data_info = BidsConcatenationDataInfo(input_filepaths, data_info_dictionaries_list, stimulus_info, concatenation_result_info, concatenated_result_filename)
            required_concatenations_info.append(concatenation_data_info)
        
        return required_concatenations_info

    @classmethod
    def get_refined_signals_cpu(cls, refined_prf_params_XY : np.ndarray, prf_model : PRFModel, stimulus : Stimulus, cfg):
        refined_S_batches_gpu = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=cp.asnumpy(refined_prf_params_XY), points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE, cfg=cfg)            
        
        refined_S_cpu = []
        # refined signal batches could be present on different GPUs
        for batch_idx in range(len(refined_S_batches_gpu)):
            device_id = refined_S_batches_gpu[batch_idx].device.id
            with cp.cuda.Device(device_id):
                refined_signal_batch_cpu = cp.asnumpy(refined_S_batches_gpu[batch_idx])
                refined_S_cpu.append(refined_signal_batch_cpu)
        
        refined_S_cpu = np.concatenate(refined_S_cpu, axis=0)
        # # refined_S_cpu = cp.asnumpy(cp.concatenate(refined_S_batches_gpu, axis=0))

        return refined_S_cpu

    @classmethod
    def get_valid_refined_data(cls, refined_matching_results_XY, Y_signals_gpu, O_gpu, prf_model, stimulus, coarse_e,  best_fit_proj , coarse_pRF_estimations, cfg):
        # refined S batches
        refined_S_batches_gpu = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=cp.asnumpy(refined_matching_results_XY), points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE, cfg=cfg)            

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
        coarse_error_vector = coarse_e[np.arange(len(coarse_e)), best_fit_proj]
        coarse_error_vector = cp.asnumpy(coarse_error_vector) if isinstance(coarse_error_vector, cp.ndarray) else coarse_error_vector
        refined_error_vector = np.diagonal(refined_e_cpu)
        diff = refined_error_vector - coarse_error_vector
        worsened_error_y_signal_indices = np.argwhere((~np.isnan(diff)) & (diff < 0))

        # keep the coarse pRF parameters where the refined esitmations got worse
        refined_matching_results_XY[worsened_error_y_signal_indices, :] = coarse_pRF_estimations[worsened_error_y_signal_indices, :]

        return refined_matching_results_XY

    @classmethod
    def get_pRF_estimations(cls, cfg, O_gpu, prf_space, prf_model, stimulus, prf_analysis, arr_2d_location_inv_M_cpu, measured_data_filepath):
        valid_refined_prf_points_XY = None
        r2_results = None
        valid_refined_S_cpu = None

         # y-signals
        y_data = ObservedData(data_source=DataSource.measured_data)
        Y_signals_cpu = y_data.get_y_signals(measured_data_filepath)

        # process bathches
        Y_signals_cpu = Y_signals_cpu[:, None] if Y_signals_cpu.ndim == 1 else Y_signals_cpu # in case only one signal is present

        # exit if number of timepoints in y-signals and stimulus do not match
        if Y_signals_cpu.shape[0] != (stimulus.NumFrames, stimulus.NumFramesDownsampled)[stimulus.HighTemporalResolutionEnabled]:
            Logger.print_red_message(f"Number of timepoints in measured fMRI data ({Y_signals_cpu.shape[0]}) and stimulus ({stimulus.NumFrames}) do not match for file: {measured_data_filepath}", print_file_name=False)
            sys.exit(1)

        total_y_signals = Y_signals_cpu.shape[1]
        num_batches = int(cfg.measured_data["batches"])
        batch_size = max(1, int(total_y_signals / num_batches)) # to deal with the situation of only one y_signal, which will result in batch_size = 0
        for current_batch_idx in range(0, total_y_signals, batch_size):
            Y_signals_batch_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=(Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size],))
            Y_signals_batch_cpu = Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size]

            # error
            # prf_analysis.error_e = (Y_signals_batch_gpu.T @ dS_prime_dtheta_columnmajor_gpu)
            error_term_computation_func = GridFit.get_error_terms if cfg.refine_fitting_enabled else GridFit.get_only_error_terms
            best_fit_proj, e, de_dtheta_3darr = error_term_computation_func(isResultOnGPU=((cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled) | (not cfg.refine_fitting_enabled)),
                                                                           Y_signals_gpu=Y_signals_batch_gpu,
                                                                           S_prime_cm_batches_gpu=prf_analysis.orthonormalized_S_batches,
                                                                           dS_prime_dtheta_cm_batches_list_gpu=prf_analysis.orthonormalized_dS_dtheta_batches_list)
            
            # Logger.print_green_message(f"error computed for batch {current_batch_idx} - {current_batch_idx + min(batch_size, total_y_signals-current_batch_idx) }...", print_file_name=False)

            # NOTE: RefineFit produces results in (X, Y) format
            # perform refine search, the obtained refined results will be in the (X, Y) format
            if cfg.refine_fitting_enabled:
                num_Y_signals = Y_signals_batch_cpu.shape[1]
                refined_matching_results_XY, Fex_results = RefineFit.get_refined_fit_results(prf_space,
                                                                                             num_Y_signals,
                                                                                             best_fit_proj,
                                                                                             arr_2d_location_inv_M_cpu,
                                                                                             e,
                                                                                             de_dtheta_3darr)
        
            # NOTE: The coarse_estimation values are in XY format (i.e. (col, row) format)
            coarse_pRF_estimations = (prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_gpu)[((cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled) | (not cfg.refine_fitting_enabled))][best_fit_proj]
            
            # validate if the refined pRF estimations are really improving the error value, and for the pRF points where error is getting worse, keep the coarse pRF estimations
            if cfg.refine_fitting_enabled:
                valid_refined_prf_points_XY_batch = GEMpRFAnalysis.get_valid_refined_data(refined_matching_results_XY,
                                                                                          Y_signals_gpu=Y_signals_batch_gpu,
                                                                                          O_gpu=O_gpu,
                                                                                          prf_model=prf_model,
                                                                                          stimulus=stimulus,
                                                                                          coarse_e=e,
                                                                                          best_fit_proj=best_fit_proj,
                                                                                          coarse_pRF_estimations=coarse_pRF_estimations,
                                                                                          cfg=cfg)
            else:
                valid_refined_prf_points_XY_batch = coarse_pRF_estimations

            # compute timecourses for refined pRF estimated params
            valid_refined_S_cpu_batch = GEMpRFAnalysis.get_refined_signals_cpu(valid_refined_prf_points_XY_batch, prf_model, stimulus, cfg)

            # compute Variance Explained
            r2_results_batch = R2.get_r2_num_den_method_with_epsilon_as_yTs(Y_signals_batch_gpu, O_gpu, valid_refined_prf_points_XY_batch, valid_refined_S_cpu_batch).reshape(-1, 1)

            # concatenate the batch results
            if current_batch_idx == 0:
                valid_refined_prf_points_XY = valid_refined_prf_points_XY_batch
                valid_refined_S_cpu = valid_refined_S_cpu_batch
                r2_results = r2_results_batch
            else:
                valid_refined_prf_points_XY = np.concatenate((valid_refined_prf_points_XY, valid_refined_prf_points_XY_batch), axis = 0)                    
                valid_refined_S_cpu = np.concatenate((valid_refined_S_cpu, valid_refined_S_cpu_batch), axis = 0)
                r2_results = np.concatenate((r2_results, r2_results_batch), axis = 0)            

        return valid_refined_prf_points_XY, r2_results, valid_refined_S_cpu

    ##########################################################---------------------------------RUN---------------------------------################################################
    @classmethod
    def concatenated_run(cls, cfg, prf_model, prf_space):
        # cfg = GEMpRFAnalysis.load_config(config_filepath=config_filepath) # load default config
        default_gpu_id = ggm.get_instance().default_gpu_id
        refinefit_on_gpu = cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled
        
        # data info
        required_concatenations_info = cls.get_concatenated_runs_data_files_info(cfg)

        if len(required_concatenations_info) == 0:
            Logger.print_red_message("No data files found. Please check the specified paths in your XML configuration file. Aborting now...", print_file_name=False)
            return

        # NOTE: ----------------- COMMON VARIABLES
        arr_2d_location_inv_M_cpu = None

        # M-Matrix
        result_queue = queue.Queue()    
        MpInv_thread = threading.Thread(target=cls.execute_Grids2MpInv_NewMethod, args=(prf_space, result_queue))
        MpInv_thread.start()

        # dictionary to hold all the stimulus-task specific data
        task_specific_data_dict = {}
        class TaskSpecificData:
            def __init__(self, stimulus, O_gpu, prf_analysis):
                self.stimulus = stimulus
                self.O_gpu = O_gpu
                self.prf_analysis = prf_analysis

        # NOTE: Compute TASK-SPECIFIC data, such as Stimulus, O_gpu, and Prediction signals (and their derivatives) for each stimulus
        for concatenate_block_info in required_concatenations_info:
            # NOTE: ----------------- STIMULUS SPECIFIC VARIABLES: load all the required stimulus for each participating input data in the concatenation
            for single_stimulus_info in concatenate_block_info.all_stimuli_info:
                stimulus_task_name = single_stimulus_info.stimulus_task 
                if stimulus_task_name not in task_specific_data_dict: 
                    task_specific_stimulus = GEMpRFAnalysis.load_stimulus(cfg, single_stimulus_info)
                    #...get Orthogonalization matrix
                    # NOTE: use the correct stimulus as the number of frames could be different!!!!!!!!!!!!!!
                    ortho_matrix_dim = task_specific_stimulus.NumFrames if (not task_specific_stimulus.HighTemporalResolutionEnabled) else task_specific_stimulus.NumFramesDownsampled
                    ortho_matrix = OrthoMatrix(nDCT=cfg.nDCT, num_frame_stimulus=ortho_matrix_dim) 
                    O_gpu = ortho_matrix.get_orthogonalization_matrix()
                    GemWriteToFile.get_instance().write_array_to_h5(O_gpu, variable_path=[f'model', f'{stimulus_task_name}', 'orthogonalization_matrix'], append_to_existing_variable=False)

                    prf_analysis = PRFAnalysis(prf_space=prf_space, stimulus=task_specific_stimulus) # to hold all the information about this analysis run,  # NOTE: PRFAnalysis class will be helpful for the concatenation runs, where you can store the results with different stimulus in corresponding objects (i.e. prf_analysis)                              
                    prf_analysis.orthonormalized_S_batches, prf_analysis.orthonormalized_dS_dtheta_batches_list = cls.compute_orthonormalized_signals(O_gpu=O_gpu, 
                                                                                                                                                prf_space= prf_space, 
                                                                                                                                                prf_model= prf_model, 
                                                                                                                                                stimulus= task_specific_stimulus,
                                                                                                                                                cfg = cfg,
                                                                                                                                                stimulus_task_name=stimulus_task_name) 

                    # add to dictionary
                    task_specific_data = TaskSpecificData(task_specific_stimulus, O_gpu, prf_analysis)
                    task_specific_data_dict[stimulus_task_name] = task_specific_data
                    # task_specific_data_dict[stimulus_task] = (stimulus, O_gpu, prf_analysis)
            
        # get M-inverse matrix
        if arr_2d_location_inv_M_cpu is None:
            MpInv_thread.join()
            if not result_queue.empty():
                arr_2d_location_inv_M_cpu = result_queue.get()     

        # NOTE: Process each Concatenation Block
        class YSignalsInfo:
            def __init__(self, Y_signals_cpu, task_name):
                self.Y_signals_cpu = Y_signals_cpu
                self.task_name = task_name

        counter = 0
        for concatenate_block_info in required_concatenations_info:            
            counter += 1
            json_data = None   
            # Collect Y-Signals
            start_time = time.time()
            arr_Y_signals_cpu = []
            num_concatenation_items = len(concatenate_block_info.filepaths_to_be_concatenated) 
            for concat_item_idx in range(num_concatenation_items):
                input_data_filepath = concatenate_block_info.filepaths_to_be_concatenated[concat_item_idx]
                task_name = concatenate_block_info.input_data_info_to_be_concatenated[concat_item_idx].get("task")
                if not os.path.exists(input_data_filepath):
                    raise ValueError(f"Input source file does not exist: {input_data_filepath}", print_file_name=False)

                Logger.print_green_message(f"Processing-{counter}/{len(required_concatenations_info)} data file: {input_data_filepath}", print_file_name=False)
                measured_data_filepath = input_data_filepath
                  
                # y-signals
                y_data = ObservedData(data_source=DataSource.measured_data)
                Y_signals_cpu = y_data.get_y_signals(measured_data_filepath)
                y_signals_info = YSignalsInfo(Y_signals_cpu, task_name) 
                arr_Y_signals_cpu.append(y_signals_info)    
                # arr_Y_signals_cpu.append((Y_signals_cpu, task_name))    
        
            ###################
            # process Y-BATCHES
            ###################               
            # json_data = None   
            total_y_signals = arr_Y_signals_cpu[0].Y_signals_cpu.shape[1]
            num_batches = int(cfg.measured_data["batches"])
            batch_size = int(total_y_signals / num_batches)
            for current_batch_idx in range(0, total_y_signals, batch_size):    
                # go through all datasets and compute error terms for each run
                # arr_e_cpu = None #cp.empty((num_runs, batch_size, num_signals)) #[]
                arr_e_list = []
                arr_de_dtheta_full_list = []
                Y_signals_batch_gpu_list = []
                for concat_item_idx in range(num_concatenation_items):                                
                    # current Y-BATCH, for current dataset
                    Y_signals_batch_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=((arr_Y_signals_cpu[concat_item_idx].Y_signals_cpu)[:, current_batch_idx: current_batch_idx + batch_size],))                                        
                    Y_signals_batch_gpu_list.append(Y_signals_batch_gpu)            
                    Y_signals_batch_cpu = (arr_Y_signals_cpu[concat_item_idx].Y_signals_cpu)[:, current_batch_idx: current_batch_idx + batch_size]
                    num_Y_signals_in_batch = Y_signals_batch_cpu.shape[1] # this is just the number of Y-signals in the current batch, it is independent of the task-name
                    current_data_task = arr_Y_signals_cpu[concat_item_idx].task_name
                    _, e_gpu, de_dtheta_full_list = GridFit.get_error_terms(isResultOnGPU=((cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled) | (not cfg.refine_fitting_enabled)),
                                                                                Y_signals_gpu=Y_signals_batch_gpu,
                                                                                S_prime_cm_batches_gpu=task_specific_data_dict[current_data_task].prf_analysis.orthonormalized_S_batches,
                                                                                dS_prime_dtheta_cm_batches_list_gpu=task_specific_data_dict[current_data_task].prf_analysis.orthonormalized_dS_dtheta_batches_list)
                
                    arr_e_list.append(e_gpu)
                    arr_de_dtheta_full_list.append(de_dtheta_full_list)

                # NOTE: process this batch of concatenation block
                # ...sum up the error terms (e and de_dtheta) for all the runs
                # ...current Y-BATCH concatenated error terms                                            
                xp = cp if refinefit_on_gpu else np
                ctx = cp.cuda.Device(default_gpu_id) if refinefit_on_gpu else nullcontext()
                with ctx:                                        
                    # ...sum up e terms
                    arr_e = xp.stack(arr_e_list, axis=0)
                    concatenated_e = xp.sum(arr_e, axis=0)                    

                    if cfg.refine_fitting_enabled:
                        # ...sum up de_dtheta terms
                        arr_de_dtheta_full = xp.stack(arr_de_dtheta_full_list, axis=0) #shape = (num items to concatenate, num params, num y signals, num model signals)
                        concatenated_de_dtheta = xp.sum(arr_de_dtheta_full, axis=0)

                    best_fit_proj = xp.nanargmax(concatenated_e, axis=1)# current Y-BATCH concatenated best fit                    

                #  current Y-BATCH refine fit
                refined_matching_results_XY = None
                coarse_pRF_estimations = None
                if cfg.refine_fitting_enabled:
                    refined_matching_results_XY, _ = RefineFit.get_refined_fit_results(prf_space=prf_space,
                                                                                    num_Y_signals=num_Y_signals_in_batch,
                                                                                    best_fit_proj=best_fit_proj,
                                                                                    arr_2d_location_inv_M_cpu=arr_2d_location_inv_M_cpu,
                                                                                    e_full=concatenated_e,  # send overall error terms
                                                                                    de_dtheta_3darr=concatenated_de_dtheta)
                else:
                    coarse_pRF_estimations = (prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_gpu)[((cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled) | (not cfg.refine_fitting_enabled))][best_fit_proj]
                                        
                # # # NOTE NOTE NOTE: Validate the refined results!!!!!!!!!!! STEP MISSING: because for this, we need to compute the results with the all stimuli used for different tasks, it will waste a lot of time.
                # # coarse_pRF_estimations = prf_space.multi_dim_points_cpu[best_fit_proj_cpu]
                # # valid_refined_prf_points_XY_batch = GEMpRFAnalysis.get_valid_refined_data(refined_matching_results_XY, 
                # #                                                                           Y_signals_gpu=Y_signals_batch_gpu, # NOTE: here we would need to pass the list of signals for all concatenated_item
                # #                                                                           O_gpu=O_gpu, 
                # #                                                                           prf_model=prf_model, 
                # #                                                                           stimulus=stimulus,  # NOTE: here we would need to pass the list of stimuli required for each concatenated_item
                # #                                                                           coarse_e_cpu=e_cpu, 
                # #                                                                           best_fit_proj_cpu=best_fit_proj_cpu, 
                # #                                                                           coarse_pRF_estimations=coarse_pRF_estimations)

                # Final results 
                valid_refined_prf_points_XY_batch = (coarse_pRF_estimations, refined_matching_results_XY)[cfg.refine_fitting_enabled]

                valid_refined_S_cpu_batch_list = []
                for concat_item_idx in range(num_concatenation_items):
                    stimulus_task_name = arr_Y_signals_cpu[concat_item_idx].task_name
                    task_specific_stimulus = task_specific_data_dict[stimulus_task_name].stimulus
                    valid_refined_S_cpu_batch = GEMpRFAnalysis.get_refined_signals_cpu(valid_refined_prf_points_XY_batch, prf_model, task_specific_stimulus, cfg)
                    valid_refined_S_cpu_batch_list.append(valid_refined_S_cpu_batch)

                # current Y-BATCH compute concatenated R2        
                numerators_gpu = cp.empty((num_concatenation_items, num_Y_signals_in_batch))
                denominators_gpu = cp.empty((num_concatenation_items, num_Y_signals_in_batch))
                # ...compute the numerator and denominator terms for each run's dataset individually using the above computed Refinement results. 
                # ...The O_gpu signals depend on the Stimulus so, send the correct one !!!
                for concat_item_idx in range(num_concatenation_items):
                    stimulus_task_name = arr_Y_signals_cpu[concat_item_idx].task_name
                    task_specific_stimulus = task_specific_data_dict[stimulus_task_name].stimulus
                    task_specific_O_gpu = task_specific_data_dict[stimulus_task_name].O_gpu
                    num_gpu, den_gpu = R2.get_r2_numerator_denominator_terms(Y_signals_batch_gpu_list[concat_item_idx], 
                                                                             task_specific_O_gpu, 
                                                                             valid_refined_prf_points_XY_batch, 
                                                                             valid_refined_S_cpu_batch_list[concat_item_idx])
                    numerators_gpu[concat_item_idx] = num_gpu
                    denominators_gpu[concat_item_idx] = den_gpu

                ## ...compute overall r2 for current Y-BATCH
                r2_numerator_term = cp.sum(numerators_gpu, axis=0)
                r2_inverse_term = (cp.sum(denominators_gpu, axis=0)) ** (-1)
                r2_result_batch = cp.where(r2_numerator_term>0, 1 - r2_numerator_term * r2_inverse_term, r2_numerator_term) 
                batch_json_data = R2.format_in_json_format(r2_result_batch, valid_refined_prf_points_XY_batch, None, refined_signals_present=False)
                if json_data is None:
                    json_data = batch_json_data
                else:
                    json_data += batch_json_data

                # print ("Refined fitting done...")

            # NOTE: Write the full results of the current concatenation block to file
            JsonMgr.write_to_file(filepath=concatenate_block_info.concatenation_result_filepath, data=json_data)   

            Logger.print_green_message(f"Results written to file: {concatenate_block_info.concatenation_result_filepath}", print_file_name=False)

            # end time
            end_time = time.time()
            iteration_time = end_time - start_time
            # iteration_times.append(iteration_time)
            print(f"Time taken for this analysis: {iteration_time}\n")
        
        print

    @classmethod
    def individual_run(cls, cfg, prf_model, prf_space):
        # time
        start_time = time.time()

        # data info
        measured_data_list, result_filepaths_list = cls.get_single_run_data_files_info(cfg)
        if len(measured_data_list) == 0:
            Logger.print_red_message("No data files found. Please check the specified paths in your XML configuration file. Aborting now...", print_file_name=False)
            return

        GemWriteToFile.get_instance().write_array_to_h5(np.array(measured_data_list), variable_path=['input_data', 'measured_data_list'], append_to_existing_variable=False)

        # stimulus
        if cfg.bids['@enable'] == "True":
            stimulus_info = GemBidsHandler.get_stimulus_info(stimulus_dir = cfg.stimulus['directory'], stimulus_name = cfg.bids['individual']['task'])
        else:
            stimulus_info = GemBidsHandler.get_non_bids_stimulus_info(cfg)
        stimulus = GEMpRFAnalysis.load_stimulus(cfg, stimulus_info)

        # M-Matrix
        if cfg.refine_fitting_enabled:
            result_queue = queue.Queue()    
            MpInv_thread = threading.Thread(target=cls.execute_Grids2MpInv_NewMethod, args=(prf_space, result_queue))
            MpInv_thread.start()

        #...get Orthogonalization matrix
        ortho_matrix_dim = stimulus.NumFrames if (not stimulus.HighTemporalResolutionEnabled) else stimulus.NumFramesDownsampled
        ortho_matrix = OrthoMatrix(nDCT=cfg.nDCT, num_frame_stimulus=ortho_matrix_dim)
        O_gpu = ortho_matrix.get_orthogonalization_matrix() # (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))
        GemWriteToFile.get_instance().write_array_to_h5(O_gpu, variable_path=['model', 'orthogonalization_matrix'], append_to_existing_variable=False)


        #...compute Model Signals
        prf_analysis = PRFAnalysis(prf_space=prf_space, stimulus=stimulus) # to hold all the information about this analysis run,  # NOTE: PRFAnalysis class will be helpful for the concatenation runs, where you can store the results with different stimulus in corresponding objects (i.e. prf_analysis)                              
        prf_analysis.orthonormalized_S_batches, prf_analysis.orthonormalized_dS_dtheta_batches_list = cls.compute_orthonormalized_signals(O_gpu=O_gpu, 
                                                                                                                                    prf_space= prf_space, 
                                                                                                                                    prf_model= prf_model, 
                                                                                                                                    stimulus= stimulus,
                                                                                                                                    cfg = cfg)  
        Logger.print_green_message("model signals computed...", print_file_name=False)

        #...get M-inverse matrix
        arr_2d_location_inv_M_cpu = None
        if cfg.refine_fitting_enabled:
            inv_mat_join_start_time = datetime.datetime.now()            
            MpInv_thread.join()
            if not result_queue.empty():
                arr_2d_location_inv_M_cpu = result_queue.get()
            Logger.print_green_message(f"Time taken to compute M-inverse matrix: {datetime.datetime.now() - inv_mat_join_start_time}\n", print_file_name=False)

        # # end time
        # end_time = time.time()
        # iteration_time = end_time - start_time     
        # print(iteration_time)  
        # iteration_times = []       
        
        # pRF Estimations                
        file_processed_counter = 1             
        data_src = []             
        # data_idx = 0  
        # for i in range(10):
        for data_idx in range(len(measured_data_list)):
            # check if input file exists
            if not os.path.exists(measured_data_list[data_idx]):
                Logger.print_red_message(f"Input source file does not exist: {measured_data_list[data_idx]}", print_file_name=False)
                continue

            start_time = time.time()
            Logger.print_green_message(f"Processing file ({file_processed_counter}/{len(measured_data_list)}): {measured_data_list[data_idx]}", print_file_name=False)
            valid_refined_prf_points_XY, r2_results, valid_refined_S_cpu = GEMpRFAnalysis.get_pRF_estimations(cfg, O_gpu, prf_space, prf_model, stimulus, prf_analysis, arr_2d_location_inv_M_cpu, measured_data_list[data_idx])
            # profiler.disable()
            # stats = pstats.Stats(profiler, stream=profile_stream)
            # stats.strip_dirs().sort_stats("cumulative").print_stats(20)  # Top 20 most time-consuming calls
            # print(profile_stream.getvalue())


            # format results to JSON                
            # json_data = R2.format_in_json_format( r2_results, valid_refined_prf_points_XY, valid_refined_S_cpu)   # NOTE: use this line if you want to print the refined signals in the JSON file
            json_data = R2.format_in_json_format( r2_results, valid_refined_prf_points_XY, None, refined_signals_present=False)                    
            
            # write results to file
            JsonMgr.write_to_file(filepath=result_filepaths_list[data_idx], data=json_data)

            # information
            Logger.print_green_message(f"Results written to file: {result_filepaths_list[data_idx]}", print_file_name=False)
            data_src.append(measured_data_list[data_idx])  
            file_processed_counter += 1

            # end time
            end_time = time.time()
            iteration_time = end_time - start_time
            # iteration_times.append(iteration_time)
            print(f"Time taken for this analysis: {iteration_time}\n")

            # write the time taken for each iteration
            # csv_filepath = r"D:\results\gem-paper-simulated-data\analysis\05\BIDS\derivatives\time_records\v2_iteration_times_151x151x16.csv"
            # csv_filepath = r"/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/05/BIDS/derivatives/time_records/v2_iteration_times_151x151x16--RefinefitScipy.csv"
            # df = pd.DataFrame({'DataSrc': data_src, 'Time (seconds)': iteration_times})
            # df.to_csv(csv_filepath, index=False)
        print ("All files processed...")

      

    @classmethod
    def run(cls, cfg, prf_model, prf_space):
        # Run the analysis (Concatenation or Individual Run)
        if cfg.bids['@enable'] == "True" and cfg.bids['@run_type'].lower() == "concatenated":                
            GEMpRFAnalysis.concatenated_run(cfg, prf_model, prf_space)
        else:
            GEMpRFAnalysis.individual_run(cfg, prf_model, prf_space)        

        return 0


# # ################################################---------------------------------MAIN---------------------------------################################################
# # # run the main function
# # if __name__ == "__main__":    
# #     start_time = datetime.datetime.now()

# #     print ("Running the GEM pRF Analysis...")
# #     # from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
# #     # GEMpRFAnalysis.run()    
# #     # cProfile.run('GEMpRFAnalysis.run()', sort='cumulative')

# #     # Run the profiling
# #     # profiler = cProfile.Profile()
# #     # profiler.enable()

# #     config_filepath = os.path.join(os.path.dirname(__file__), '..', 'configs', 'analysis_configs', 'analysis_config.xml')

# #     # config_filepath = r'D:\code\sid-git\fmri\gem\configs\default_config\new_concatenationDummyTest_config.xml'
# #     # GEMpRFAnalysis.concatenated_run(config_filepath)
# #     print("Starting GEM analysis...")
# #     GEMpRFAnalysis.run(config_filepath)
# #     # profiler.disable()

# #     # print time taken
# #     print(f"Complete Time taken: {datetime.datetime.now() - start_time}")

# #     # # Specify the file name to save the profiling results
# #     # output_file = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/05/BIDS/derivatives/prfanalyze-gem/analysis-05/sub-100000/ses-0n0/profiling_results.txt'

# #     # # Dump the profiling statistics to the specified file
# #     # with open(output_file, 'w') as f:
# #     #     stats = pstats.Stats(profiler, stream=f)
# #     #     stats.sort_stats('cumulative')
# #     #     stats.print_stats()