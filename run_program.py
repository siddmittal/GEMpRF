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
import cProfile
import pstats
import io

# gem imports
from gem.model.prf_gaussian_model import PRFGaussianModel
from gem.space.PRFSpace import PRFSpace
from gem.model.selected_prf_model import SelectedPRFModel
from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

class SelectedProgram(Enum):
    GEMAnalysis = 0
    GEMSimulations = 1

def run_selected_program(selected_program, config_filepath, spatial_points_xy = None):
    cfg = GEMpRFAnalysis.load_config(config_filepath=config_filepath) # load default config
    _ = ggm(default_gpu_id = int(cfg.gpu['default_gpu']))

    # copy the config file to the results folder
    result_dir = os.path.join(cfg.bids.get("basepath"), "derivatives", "prfanalyze-gem", f'analysis-{cfg.bids.get("results_anaylsis_id")}')
    shutil.copy(config_filepath, result_dir) if os.makedirs(result_dir, exist_ok=True) is None else None

    # ...prf spatial points
    if spatial_points_xy is None:
        if selected_program == SelectedProgram.GEMAnalysis:
            spatial_points_yx = GEMpRFAnalysis.get_prf_spatial_points(cfg) # this will return the spatial points in (row/y, col/x) format
        elif selected_program == SelectedProgram.GEMSimulations:
            spatial_points_yx = GEMSimulationsRunner.get_prf_spatial_points(cfg)

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
    elif selected_program == SelectedProgram.GEMSimulations:
        additional_dimensions = GEMSimulationsRunner.get_additional_dimensions(cfg, selected_prf_model)
        
    prf_space = PRFSpace(spatial_points_xy, additional_dimensions=additional_dimensions)
    prf_space.convert_spatial_to_multidim()
    prf_space.keep_validated_sampling_points(prf_model.get_validated_sampling_points_indices) # update the computed multi-dimensional points array

    # run the pRF analysis
    if selected_program == SelectedProgram.GEMAnalysis:             
        GEMpRFAnalysis.run(cfg, prf_model, prf_space)
    elif selected_program == SelectedProgram.GEMSimulations:
        GEMSimulationsRunner.run(cfg, prf_model, prf_space)
        

################################################---------------------------------MAIN---------------------------------################################################
# run the main function
if __name__ == "__main__":    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    selected_program = SelectedProgram.GEMAnalysis
    print ("Running the GEM pRF Analysis...")
    # from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
    # GEMpRFAnalysis.run()    
    # cProfile.run('GEMpRFAnalysis.run()', sort='cumulative')

    # # Run the profiling
    # profiler = cProfile.Profile()
    # profiler.enable()

    # ...simulations
    if selected_program == SelectedProgram.GEMSimulations:
        from raven.simulation.run.run_gem_simulations import GEMSimulationsRunner
        # NOTE: KDE computation Configs
        config_filepath = os.path.join(script_dir, 'raven', 'configs', 'simulations_configs', '0002_kde_generation-voxel-394-specific_config.xml')
        # config_filepath = os.path.join(script_dir, '..', 'configs', 'simulations_configs', '0001_kde_generation_config.xml') # simulations_config.xml # david_001_hcp_stimuli_simulations_config # 0001_kde_generation_config
        
        # NOTE: Stimuli Comparison Configs
        # config_filepath = os.path.join(script_dir, '..', 'configs', 'stimuli_comparison_configs', 'num-001_dataset-hcpret_task-bars_purpose-stimulicomparison_remarks-david.xml') 
        # config_filepath = os.path.join(script_dir, '..', 'configs', 'stimuli_comparison_configs', 'num-002_dataset-hcpret_task-wedgering_purpose-stimulicomparison_remarks-david.xml') 
        # config_filepath = os.path.join(script_dir, '..', 'configs', 'stimuli_comparison_configs', 'num-003_dataset-hcpret_task-barwedgering_purpose-stimulicomparison_remarks-david.xml')     
    elif selected_program == SelectedProgram.GEMAnalysis:
        config_filepath = os.path.join(os.path.dirname(__file__), 'gem', 'configs', 'analysis_configs', 'analysis_config - VerifyChanges.xml')
        
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