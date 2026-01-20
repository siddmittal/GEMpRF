# -*- coding: utf-8 -*-
"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2023-2025, Medical University of Vienna",
"@Desc    :   None",
        
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
from enum import Enum
import pynvml
import cProfile
import pstats
import io
import datetime
import numpy as np

# gem imports
from gem.model.prf_gaussian_model import PRFGaussianModel
from gem.space.PRFSpace import PRFSpace
from gem.model.selected_prf_model import SelectedPRFModel
from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
from gem.utils.gem_gpu_manager import GemGpuManager as ggm
from gem.utils.gem_write_to_file import GemWriteToFile
from gem.utils.logger import Logger
from gem.utils.gem_h5_file_handler import H5FileManager

# gemprf wrapper import
from gemprf import __version__

class SelectedProgram(Enum):
    GEMAnalysis = 0
    GEMSimulations = 1

def set_system_gpus(max_available_gpus):
    all_gpus_list = list(range(max_available_gpus))
    all_gpus_str = ','.join(map(str, all_gpus_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus_str

def manage_gpus(cfg, max_available_gpus):
    try:
        custom_default_gpu_id = int(cfg.gpu.get("default_gpu"))
    except Exception:
        raise ValueError(
            "Invalid GPU configuration: the 'default_gpu' value must be an integer. \nSee https://gemprf.github.io/")

    # Optional additional GPUs
    additional_node = cfg.gpu.get("additional_available_gpus")

    if additional_node and "gpu" in additional_node:
        additional_gpus = additional_node["gpu"]

        # Handle single <gpu> vs multiple <gpu>
        if not isinstance(additional_gpus, list):
            additional_gpus = [additional_gpus]

        other_available_gpus = sorted({int(g) for g in additional_gpus if int(g) != custom_default_gpu_id})
    else:
        other_available_gpus = []
        max_available_gpus > 1 and Logger.print_blue_message("Note: Multi-GPU is supported, but none were specified. Using the specified default GPU.", print_file_name=False)

    user_specified_gpus_list = [custom_default_gpu_id] + other_available_gpus

    # Sanity check
    if not all(0 <= gpu_id < max_available_gpus for gpu_id in user_specified_gpus_list):
        raise ValueError(f"GPU IDs must be in range [0, {max_available_gpus - 1}].")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, user_specified_gpus_list))


def run_selected_program(selected_program, config_filepath):
    cfg = GEMpRFAnalysis.load_config(config_filepath=config_filepath) # load default config

    # NOTE: match versions
    config_file_version = cfg.config_data["@version"]
    if config_file_version != __version__:
        Logger.print_red_message(
            f"Version mismatch: config={config_file_version}, GEMpRF={__version__}. "
            "\nDownload matching versions at: https://gemprf.github.io/",
            print_file_name=False
        )
        sys.exit(1)        
    print(f"GEMpRF version: {__version__}")

    # read user defined spatial points, if any
    spatial_points_xy = None
    if cfg.optional_analysis_params['enable'] and cfg.optional_analysis_params['spatial_grid_xy']['use_from_file']:
        spatial_points_xy = H5FileManager.get_key_value(filepath=cfg.optional_analysis_params['filepath'], key = cfg.optional_analysis_params['spatial_grid_xy']['key'])
        if spatial_points_xy is None:
            Logger.print_red_message(f"Could not load spatial grid points from file: {cfg.optional_analysis_params['filepath']} with key: {cfg.optional_analysis_params['spatial_grid_xy']['key']}", print_file_name=False)
            sys.exit(1)

    # GPUs management
    #...get max. number of available GPUs without using cuda
    pynvml.nvmlInit()
    max_available_gpus = pynvml.nvmlDeviceGetCount() #cp.cuda.runtime.getDeviceCount()
    pynvml.nvmlShutdown()
    try:
        manage_gpus(cfg, max_available_gpus)
    except ValueError as e:
        Logger.print_red_message(f"GPU config error: {e} Utilizing all available GPUs...", print_file_name=False)
        set_system_gpus(max_available_gpus)
    _ = ggm(default_gpu_id = 0) # Here "0" reresents the index in "os.environ["CUDA_VISIBLE_DEVICES"]", which will automatically be the correct one

    # copy the config file to the results folder, decide to overwrite or not the existing analysis results diretory
    if selected_program == SelectedProgram.GEMAnalysis:
        if cfg.bids['@enable'] == "True":
            result_dir = os.path.join(cfg.bids.get("basepath"), "derivatives", "prfanalyze-gem", f'analysis-{cfg.bids["results_anaylsis_id"]["#text"]}')
        else:
            result_dir = cfg.fixed_paths['results']['basepath']
        if os.path.exists(result_dir) and cfg.bids["results_anaylsis_id"].get("@overwrite").lower() == "false":
            shutil.move(result_dir, f'{result_dir}_backup-{datetime.datetime.now():%Y%m%d-%H%M%S}')
        shutil.copy(config_filepath, result_dir) if os.makedirs(result_dir, exist_ok=True) is None else None

    # ...prf spatial points
    if spatial_points_xy is None:
        if selected_program == SelectedProgram.GEMAnalysis:
            spatial_points_yx = GEMpRFAnalysis.get_prf_spatial_points(cfg) # this will return the spatial points in (row/y, col/x) format

        # convert spatial points from (row, col) to (col, row)
        spatial_points_xy = spatial_points_yx
        spatial_points_xy = spatial_points_yx[:, [1, 0]]

    # selected pRF model
    selected_prf_model = GEMpRFAnalysis.get_selected_prf_model(cfg)                
    if selected_prf_model == SelectedPRFModel.GAUSSIAN:             
        prf_model = PRFGaussianModel(visual_field_radius= np.max(spatial_points_xy)) 

    # additional dimensions
    if selected_program == SelectedProgram.GEMAnalysis:
        additional_dimensions = GEMpRFAnalysis.get_additional_dimensions(cfg, selected_prf_model)
        
    prf_space = PRFSpace(spatial_points_xy, additional_dimensions=additional_dimensions)
    prf_space.convert_spatial_to_multidim()
    prf_space.keep_validated_sampling_points(prf_model.get_validated_sampling_points_indices) # update the computed multi-dimensional points array

    # NOTE
    _ = GemWriteToFile(result_dir = result_dir, debugging_enabled=cfg.write_debug_info) # initialize the GemWriteToFile singleton instance
    GemWriteToFile.get_instance().write_array_to_h5(prf_space.multi_dim_points_cpu, variable_path=['model', 'prf_grid'], append_to_existing_variable=False)

    # run the pRF analysis
    if selected_program == SelectedProgram.GEMAnalysis:             
        GEMpRFAnalysis.run(cfg, prf_model, prf_space)
        
# run the main function
def init_setup(config_filepath = None):    
    # NOTE: Select the program to run
    selected_program = SelectedProgram.GEMAnalysis

    print ("Running the GEM pRF Analysis...")     

    run_selected_program(selected_program, config_filepath)

if __name__ == "__main__":
    __import__("utils.assert_no_gemprf").assert_no_gemprf.check_gemprf_not_installed()
    default_config = os.path.join(os.path.dirname(__file__), 'configs', 'analysis_configs', 'analysis_config.xml')
    print(f"\033[38;5;208mThe program will try to use the default config filepath:\n\033[0m  {default_config}")
    choice = input("\033[38;5;208m\nDo you want to continue with this default config? [y/N]: \033[0m").strip().lower()
    if choice != 'y':
        print("Aborting program.")
        sys.exit(0)    
    init_setup(default_config)