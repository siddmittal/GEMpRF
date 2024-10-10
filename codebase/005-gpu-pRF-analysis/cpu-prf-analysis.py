# Add my "oprf" package path
import sys
sys.path.append("D:/code/sid-git/fmri/")

import numpy as np
import matplotlib.pyplot as plt
import cProfile
import os

# Local Imports
#from config.config import Configuration as config
from oprf.standard.prf_stimulus import Stimulus
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator

###########################################--------Variables------------###############################################################
# Variables
search_space_rows = 201
search_space_cols = 201
stim_width = 101
stim_height = 101
stim_frames = 301
sigma = np.double(2)
total_gaussian_curves_per_stim_frame = search_space_rows * search_space_cols
single_gaussian_curve_length = stim_width * stim_height

###########################################--------Stimulus------------###############################################################
# HRF Curve
hrf_t = np.linspace(0, 30, 31)
hrf_curve = spm_hrf_compat(hrf_t)
stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2]))
stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)

def cpu_compute():
    ###########################################--------Compute Gaussian Curves------------###############################################################
    # Gaussian curves
    stim_xx = np.linspace(-9, +9, stim_width)
    stim_yy = np.linspace(-9, +9, stim_height)

    search_space_xx = np.linspace(-9, +9, search_space_rows)
    search_space_yy = np.linspace(-9, +9, search_space_cols)

    range_xx, range_yy = np.meshgrid(stim_xx, stim_yy)    
    range_x = range_xx.flatten()
    range_y = range_yy.flatten()

    mu_x, mu_y = np.meshgrid(search_space_xx, search_space_yy)
    mean_x = mu_x.flatten()
    mean_y = mu_y.flatten()

    result_flat_gaussian_curves_data_gpu = np.exp(-((range_x[:,None] - mean_x[None,:])**2 + (range_y[:,None] - mean_y[None,:])**2) / (2*sigma**2))

    ###########################################--------Gaussian Curves to Timeseries (Shortend)------------###############################################################
    # reshape gpu gaussian curuves row-wise
    nRows_gaussian_curves_matrix = search_space_rows * search_space_cols
    nCols_gaussian_curves_matrix = stim_height * stim_width
    gaussian_curves_rowmajor_gpu = np.reshape(result_flat_gaussian_curves_data_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
    test_gc = ((gaussian_curves_rowmajor_gpu[0, :])).reshape((stim_height, stim_width))

    stimulus_flat_data_gpu = stimulus.resampled_hrf_convolved_data.flatten('F')
    stimulus_data_columnmajor_gpu = np.reshape(stimulus_flat_data_gpu, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame
    test_stim = ((stimulus_data_columnmajor_gpu[:, 10])).reshape((stim_height, stim_width))

    timeseries_rowmajor_gpu = np.dot(gaussian_curves_rowmajor_gpu, stimulus_data_columnmajor_gpu)
    test_tc = ((timeseries_rowmajor_gpu[1, :]))

    ###########################################--------Regressors------------###############################################################
    nDCT = 3
    ndct = 2 * nDCT + 1
    trends = np.zeros((stim_frames, np.max([np.sum(ndct), 1])))        

    tc = np.linspace(0, 2.*np.pi, stim_frames)[:, None]        
    trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))

    q, r = np.linalg.qr(trends) # QR decomposition
    q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0

    R_gpu = q
    O_gpu = (np.eye(stim_frames)  - np.dot(R_gpu, R_gpu.T))
    orthogonalized_signals_columnmajor_gpu = np.dot(O_gpu, timeseries_rowmajor_gpu.T) # orthogonalize each timecourse signal (present along the column)
    test_orthogonalized_tc = ((orthogonalized_signals_columnmajor_gpu[:, 1]))
    print('computed...')


###################################-----------Main()---------------------------------####################
cProfile.run('cpu_compute()', sort='cumulative')
print("done")




