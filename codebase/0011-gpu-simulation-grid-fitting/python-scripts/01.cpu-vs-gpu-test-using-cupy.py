
import numpy as np


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

##############################---Program---#######################
def RunProgram():
    # HRF Curve
    hrf_t = np.linspace(0, 30, 31)
    hrf_curve = spm_hrf_compat(hrf_t)

    # Load stimulus
    stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.compute_resample_stimulus_data((101, 101, stimulus.org_data.shape[2]))
    stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)

    # Search space
    nRows_search_space = 101
    nCols_search_space = 101
    sigma_search_space = 2
    search_space = QuadrilateralSignalsSpace(grid_nRows=nRows_search_space
                    , grid_nCols=nCols_search_space
                    , sigma=sigma_search_space
                    , stimulus=stimulus)
    search_space.generate_model_responses() # cpu
    search_space.gpu_cupy_generate_model_responses() # gpu
                          
    print("done!")

if __name__ == "__main__":
    RunProgram()

