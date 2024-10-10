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
    nQuadrilateralSpaceGridRows = 2
    nQuadrilateralSpaceGridCols = 3
    nStimulusRows = 5 
    nStimulusCols = 5
    nStimulusFrames = 4
    sigma = 2

    # Load stimulus
    #stimulus = Stimulus("../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.compute_resample_stimulus_data((5, 5, 4))     
    stimulus.compute_hrf_convolved_stimulus_data(hrf_curve)  

    flattenedHRFConvolvedStimulusData =  np.array([

		# frame - 1
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,


		# frame - 2
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		# frame - 3
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,

		# frame - 4
		0, 0, 1, 0, 0,
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0,
		0, 0, 1, 0, 0,

	])

    dummyHRFConvolvedStimulusData = flattenedHRFConvolvedStimulusData.reshape((5, 5, 4), order='F') # Stimulus - 4 frames, 5 x 5 size
    stimulus.resampled_hrf_convolved_data = dummyHRFConvolvedStimulusData

    # Search space
    search_space = QuadrilateralSignalsSpace(grid_nRows=nQuadrilateralSpaceGridRows
                    , grid_nCols=nQuadrilateralSpaceGridCols
                    , sigma=sigma
                    , stimulus=stimulus)
    search_space.generate_model_responses()

        # Data matching...
    #...collect search space timecourses (against which we are going to do the matching)
    search_space_timecourses = []    
    for row in range(nQuadrilateralSpaceGridRows):
        for col in range(nQuadrilateralSpaceGridCols):
            search_space_timecourses.append(search_space.data[row][col].timecourse)  

    #...compute best fits for all the location
    ols = OLS(modelled_signals=search_space_timecourses
        , measured_signals=search_space_timecourses
        )
    best_fit_info = ols.compute_proj_squared()

    print("done!")


if __name__ == "__main__":
    RunProgram()

