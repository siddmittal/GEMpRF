# Add my "oprf" package path
import sys
sys.path.append("D:/code/sid-git/fmri/")

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import math
from matplotlib.widgets import Slider, Button

# Local Imports
from oprf.standard.prf_stimulus import Stimulus
from oprf.analysis.prf_histograms import Histograms
from oprf.analysis.prf_histograms import LocationHistogram
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator
from oprf.external.DeepRF import data_synthetic as deeprf_data_synthetic # DeepRF module
from oprf.standard.prf_quadrilateral_signals_space import QuadrilateralSignalsSpace
from oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse
from oprf.standard.prf_ordinary_least_square import OLS
from oprf.analysis.prf_synthetic_data_generator import SynthesizedDataGenerator
from oprf.analysis.prf_synthetic_data_generator import NoiseLevels

#################
def RunProgram():
    # HRF Curve
    hrf_t = np.linspace(0, 30, 31)
    hrf_curve = spm_hrf_compat(hrf_t)

    # Dummy Stimulus data
    # Define the dimensions
    nStimulusFrames = 2
    nStimulusRows = 5
    nStimulusCols = 5

    # Load stimulus
    #stimulus = Stimulus("../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.compute_resample_stimulus_data((5, 5, 2))       
    dummyHRFConvolvedStimulusData = np.array([[[1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]],

       [[1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]],

       [[1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]],

       [[1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]],

       [[1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]]])	     
    stimulus.resampled_hrf_convolved_data = dummyHRFConvolvedStimulusData

    # Search space
    nRows_search_space = 3
    nCols_search_space = 3
    sigma_search_space = 2
    search_space = QuadrilateralSignalsSpace(grid_nRows=nRows_search_space
                    , grid_nCols=nCols_search_space
                    , sigma=sigma_search_space
                    , stimulus=stimulus)
    search_space.generate_model_responses()
                       
    print("done!")

if __name__ == "__main__":
    RunProgram()

