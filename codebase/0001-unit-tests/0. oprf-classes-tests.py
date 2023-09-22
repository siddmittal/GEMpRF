
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import math
from matplotlib.widgets import Slider, Button

# Local Imports
from codebase.oprf.standard.prf_stimulus import Stimulus
from codebase.oprf.analysis.prf_histograms import Histograms
from codebase.oprf.analysis.prf_histograms import LocationHistogram
from codebase.oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator
from codebase.oprf.external.DeepRF import data_synthetic as deeprf_data_synthetic # DeepRF module
from codebase.oprf.standard.prf_quadrilateral_signals_space import QuadrilateralSignalsSpace
from codebase.oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse
from codebase.oprf.standard.prf_ordinary_least_square import OLS
from codebase.oprf.analysis.prf_synthetic_data_generator import SynthesizedDataGenerator
from codebase.oprf.analysis.prf_synthetic_data_generator import NoiseLevels

#################
def RunProgram():
    # HRF Curve
    hrf_t = np.linspace(0, 30, 31)
    hrf_curve = spm_hrf_compat(hrf_t)

    # Load stimulus
    #stimulus = Stimulus("../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.compute_resample_stimulus_data((101, 101, stimulus.org_data.shape[2]))
    stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)

    # Search space
    nRows_search_space = 9
    nCols_search_space = 9
    sigma_search_space = 2
    search_space = QuadrilateralSignalsSpace(grid_nRows=nRows_search_space
                    , grid_nCols=nCols_search_space
                    , sigma=sigma_search_space
                    , stimulus=stimulus)
    search_space.generate_model_responses()

    # Compute Test Space...
    # ...define space
    nRows_test_space = 11
    nCols_test_space = 11
    sigma_test_space = 2
    test_space = QuadrilateralSignalsSpace(grid_nRows=nRows_test_space
                    , grid_nCols=nCols_test_space
                    , sigma=sigma_test_space
                    , stimulus=stimulus)
    test_space.generate_model_responses()

    # ...define desired noise levels
    desired_low_freq_noise_level = 0.1
    desired_system_noise_level = 0.1
    desired_physiological_noise_level = 0.1
    desired_task_noise_level = 0.1
    desired_temporal_noise_level = 0.1
    noise_levels = NoiseLevels(desired_low_freq_noise_level
                                    , desired_system_noise_level
                                    , desired_physiological_noise_level
                                    , desired_task_noise_level
                                    , desired_temporal_noise_level)

    #...synthesize noisy signals
    synthesis_ratio = 100
    data_synthesizer = SynthesizedDataGenerator(noise_levels=noise_levels, source_data=test_space.data, synthesis_ratio=synthesis_ratio, TR=1)
    noisy_synthetic_data = data_synthesizer.generate_synthetic_data() #generate_synthetic_data_With_noise_models()

    # Data matching...
    #...collect search space timecourses (against which we are going to do the matching)
    search_space_timecourses = []    
    for row in range(nRows_search_space):
        for col in range(nCols_search_space):
            search_space_timecourses.append(search_space.data[row][col].timecourse)      

    #...collect test space timecourses (which we are going to match)
    synthesized_data_timecourses = []    
    index = 0
    for row in range(nRows_test_space):
        for col in range(nCols_test_space):         
            for times in range(synthesis_ratio):  
                synthesized_data_timecourses.append(data_synthesizer.data[index].timecourse)
                index = index + 1  

    #...compute best fits for all the location
    ols = OLS(modelled_signals=search_space_timecourses
        , measured_signals=synthesized_data_timecourses
        )
    best_fit_info = ols.compute_proj_squared()

    # Histogram
    histograms = Histograms(search_space=search_space
                            , noisy_test_data = test_space.data
                            , nRows_search_space=nRows_search_space
                            , nCols_search_space=nCols_search_space
                            , best_fit_info_wrt_search_space=best_fit_info, synthesis_ratio=synthesis_ratio)
    histograms.on_test()
    
                          
    print("done!")

if __name__ == "__main__":
    RunProgram()

