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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import shutil
from enum import Enum
import pynvml
import cProfile
import pstats
import io
import datetime

# gem imports
from gem.model.prf_gaussian_model import PRFGaussianModel
from gem.space.PRFSpace import PRFSpace
from gem.model.selected_prf_model import SelectedPRFModel
from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
from gem.utils.gem_gpu_manager import GemGpuManager as ggm
from gem.utils.logger import Logger

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
        additional_gpus = cfg.gpu["additional_available_gpus"]['gpu']
        other_available_gpus = sorted(set(int(gpu_id) for gpu_id in additional_gpus if int(gpu_id) != custom_default_gpu_id))        
    except (ValueError) as e:
        raise ValueError("All GPU IDs must be integers.")

    user_specified_gpus_list = [custom_default_gpu_id] + other_available_gpus

    # Ensure all specified GPU IDs are within available range
    if not all(0 <= gpu_id < max_available_gpus for gpu_id in user_specified_gpus_list):
        raise ValueError(f"GPU IDs must be less than the number of available GPUs ({max_available_gpus}).")

    all_gpus_str = ','.join(map(str, user_specified_gpus_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus_str


def run_selected_program(selected_program, config_filepath, spatial_points_xy = None):
    cfg = GEMpRFAnalysis.load_config(config_filepath=config_filepath) # load default config

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
        result_dir = os.path.join(cfg.bids.get("basepath"), "derivatives", "prfanalyze-gem", f'analysis-{cfg.bids["results_anaylsis_id"]["#text"]}')
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
        prf_model = PRFGaussianModel(visual_field_radius= float(cfg.search_space["visual_field"])) 

    # additional dimensions
    if selected_program == SelectedProgram.GEMAnalysis:
        additional_dimensions = GEMpRFAnalysis.get_additional_dimensions(cfg, selected_prf_model)
        
    prf_space = PRFSpace(spatial_points_xy, additional_dimensions=additional_dimensions)
    prf_space.convert_spatial_to_multidim()
    prf_space.keep_validated_sampling_points(prf_model.get_validated_sampling_points_indices) # update the computed multi-dimensional points array

    # run the pRF analysis
    if selected_program == SelectedProgram.GEMAnalysis:             
        GEMpRFAnalysis.run(cfg, prf_model, prf_space)
        

def wm():
    Logger.print_red_message("""
        ============================================================
        GEM pRF Analysis - Unreleased Version for Testing Only
        ============================================================

        @Author   : Siddharth Mittal
        @Version  : 1.0
        @Contact  : siddharth.mittal@meduniwien.ac.at
        @License  : (C) Copyright 2024, Medical University of Vienna

        WARNING:
        This is an unreleased version for testing purposes.
        Its distribution is strictly prohibited.

        ============================================================
    """, 
    print_file_name=False)



################################################---------------------------------MAIN---------------------------------################################################
# run the main function
def run():      
    wm()
          
    # NOTE: Select the program to run
    # selected_program = SelectedProgram.GEMAnalysis
    selected_program = SelectedProgram.GEMAnalysis

    print ("Running the GEM pRF Analysis...")
    # from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
    # GEMpRFAnalysis.run()    
    # cProfile.run('GEMpRFAnalysis.run()', sort='cumulative')

    # # Run the profiling
    # profiler = cProfile.Profile()
    # profiler.enable()

    # Check if the user has provided a config filepath
    has_user_provided_config = False
    try:
        config_filepath = sys.argv[1]
        has_user_provided_config = True
        if selected_program == SelectedProgram.GEMSimulations:
            pass
    except IndexError:
        pass

    # ...Set config paths
    if not has_user_provided_config:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if selected_program == SelectedProgram.GEMSimulations:
            pass
        elif selected_program == SelectedProgram.GEMAnalysis:
            # config_filepath = os.path.join(os.path.dirname(__file__), 'gem', 'configs', 'analysis_configs', 'analysis_config - VerifyChanges.xml')
            # config_filepath = os.path.join(os.path.dirname(__file__), 'gem', 'configs', 'analysis_configs', 'analysis_config_retcomp17BIDS_wedgeLR-VerfiyDefaultGPU.xml')
            config_filepath = os.path.join(os.path.dirname(__file__), 'gem', 'configs', 'analysis_configs', 'analysis_config.xml')
        
    # GEMSimulationsRunner.run(config_filepath, isSimulation=IS_SIMULATION)
    run_selected_program(selected_program, config_filepath)

    # profiler.disable()

    # # Specify the file name to save the profiling results
    # output_file = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paperD-simulated-data/analysis/05/BIDS/derivatives/prfanalyze-gem/analysis-05/sub-100000/ses-0n0/profiling_results.txt'#
    # output_file = "/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/smittal/simulations/stimuli_comparison/profileing.txt"

    # # Dump the profiling statistics to the specified file
    # with open(output_file, 'w') as f:
    #     stats = pstats.Stats(profiler, stream=f)
    #     stats.sort_stats('cumulative')
    #     stats.print_stats()

if __name__ == "__main__":
    run()
    