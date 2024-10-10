# Add my "oprf" package path
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cProfile
from scipy.fft import fft, ifft

# Config
from config.oprf_config import ConfigurationWrapper as cfg
cfg.load_configuration()

# Local Imports
import sys
sys.path.append(cfg.path_to_append)

from oprf.hpc.hpc_cupy_utils import Utils as gpu_utils
from oprf.standard.prf_stimulus import Stimulus
from oprf.hpc.hpc_coefficient_matrix import CoefficientMatrix
from oprf.hpc.hpc_model_signals import ModelSignals
from oprf.hpc.hpc_observed_signals import ObservedData, DataType
from oprf.hpc.hpc_orthogonalization import OrthoMatrix
from oprf.hpc.hpc_grid_fit import GridFit
from oprf.hpc.hpc_refine_fit import RefineFit
from oprf.analysis.prf_r2_variance_explain import R2
from oprf.tools.json_file_operations import JsonMgr
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator

###########################################--------Variables------------###############################################################
gpu_utils.print_gpu_memory_stats()

# Variables
num_noisy_signals_per_signal = 1
search_space_rows = int(cfg.search_space["nRows"])
search_space_cols = int(cfg.search_space["nCols"])
search_space_frames = int(cfg.search_space["nSigma"]) #<-----------SIGMA

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
x_range_gpu = cp.asarray(x_range_cpu)  # cp.linspace(-9, +9, stim_width)
y_range_gpu = cp.asarray(y_range_cpu) # cp.linspace(-9, +9, stim_height)



###########################################--------Stimulus------------###############################################################
# HRF Curve
hrf_t = np.arange(0, 31, 1) # np.linspace(0, 30, 31)
# hrf_curve = spm_hrf_compat(hrf_t)
hrf_curve = np.array([0, 0.0055, 0.1137, 0.4239, 0.7788, 0.9614, 0.9033, 0.6711, 0.3746, 0.1036, -0.0938, -0.2065, -0.2474, -0.2388, -0.2035, -0.1590, -0.1161, -0.0803, -0.0530, -0.0336, -0.0206, -0.0122, -0.0071, -0.0040, -0.0022, -0.0012, -0.0006, -0.0003, -0.0002, -0.0001, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000]) # mrVista Values
stimulus = Stimulus(cfg.stimulus["filepath"], size_in_degrees=float(cfg.stimulus["visual_field"]))
stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2])) #stimulus.org_data.shape[2]
stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
stim_frames = stimulus.resampled_hrf_convolved_data.shape[2] 

###########################################------------------PROGRAM------------###############################################################
def gpu_compute_model_signals(O_gpu):
    ##--------Compute Gaussian Curves, and Gaussian Curves to Timeseries
    # gc class based
    model_signals = ModelSignals(model_type='quadrilateral'
                                     , model_nRows=search_space_rows
                                     , model_nCols =search_space_cols
                                     , model_nFrames=search_space_frames
                                     , model_visual_field = float(cfg.search_space["visual_field"])
                                     , model_min_sigma = float(cfg.search_space["min_sigma"])
                                     , model_max_sigma = float(cfg.search_space["max_sigma"])
                                     , stim_width = stim_width
                                     , stim_height = stim_height
                                     , stim_visual_field = float(cfg.stimulus["visual_field"]))

    stimulus_flat_data_gpu = cp.asarray(stimulus.resampled_hrf_convolved_data.flatten('F'))
    stimulus_data_columnmajor_gpu = cp.reshape(stimulus_flat_data_gpu, (stim_height * stim_width, stim_frames), order='F') # each column contains a flat stimulus frame
    #test_stim = (cp.asnumpy(stimulus_data_columnmajor_gpu[:, 10])).reshape((stim_height, stim_width))

    #---Orthogonalization + Nomalization
    S_star_columnmajor_gpu, dS_star_dx_columnmajor_gpu, dS_star_dy_columnmajor_gpu, dS_star_dsigma_columnmajor_gpu = model_signals.get_orthonormal_signals(O_gpu=O_gpu
                                                                                                                                                           , stimulus_data_columnmajor_gpu= stimulus_data_columnmajor_gpu)

    return S_star_columnmajor_gpu, dS_star_dx_columnmajor_gpu, dS_star_dy_columnmajor_gpu, dS_star_dsigma_columnmajor_gpu

def gpu_fitting(O_gpu, arr_2d_location_inv_M_cpu, Y_signals_gpu, Y_signals_cpu, S_star_columnmajor_gpu, dS_star_dx_columnmajor_gpu, dS_star_dy_columnmajor_gpu, dS_star_dsigma_columnmajor_gpu):    
    #--------send data to gpu    
    # Y_signals_gpu = cp.asarray(Y_signals_cpu)

    #------- Synchronize and release memory
    cp.cuda.Device().synchronize()
    
    # get the grid-fit results, and error-terms (plus its derivatives)
    best_fit_proj_cpu, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu = GridFit.get_error_terms(O_gpu=O_gpu
                                                                                                           , isResultOnGPU = False
                                                                                                    , Y_signals_gpu=Y_signals_gpu
                                                                                                    , S_star_columnmajor_gpu=S_star_columnmajor_gpu
                                                                                                    , dS_star_dx_columnmajor_gpu=dS_star_dx_columnmajor_gpu
                                                                                                    , dS_star_dy_columnmajor_gpu=dS_star_dy_columnmajor_gpu
                                                                                                    , dS_star_dsigma_columnmajor_gpu=dS_star_dsigma_columnmajor_gpu
                                                                                                    , multi_gpu_batching = True
                                                                                                    )

    # perform refine search
    refined_results, Fex_results = RefineFit.get_refined_fit_results(Y_signals_cpu.shape[1]
                                                        , best_fit_proj_cpu
                                                        , arr_2d_location_inv_M_cpu
                                                        , e_cpu
                                                        , de_dx_full_cpu
                                                        , de_dy_full_cpu
                                                        , de_dsigma_full_cpu
                                                        , search_space_rows
                                                        , search_space_cols
                                                        , search_space_frames)

    return refined_results, Fex_results

# Function to apply a bandpass filter
def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
    n = len(signal)
    fft_result = fft(signal)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # Create a binary mask for the frequencies within the desired range
    mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
    
    # Apply the mask to the FFT result
    filtered_fft = fft_result * mask
    
    # Inverse FFT to get back to the time domain
    filtered_signal = ifft(filtered_fft)
    
    return np.real(filtered_signal)

# ###################################-----------Main()---------------------------------####################
def main():
    # compute M_inv
    mu_X_grid, mu_Y_grid, sigma_grid = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range) 
    arr_2d_location_inv_M_cpu = CoefficientMatrix.Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid) ##<<<<--------Compute pre-define matrix M

    # y-signals
    y_data = ObservedData(data_type=DataType.measured_data)
    Y_signals_cpu = y_data.get_y_signals(cfg.measured_data)

    # TEST - APPLYING FREQUENCY BANDPASS FILTER
    # frequency filtered y-signals
    Y_signals_cpu_filtered = np.copy(Y_signals_cpu)
    for i in range(Y_signals_cpu.shape[1]):
        Y_signals_cpu_filtered[:, i] = bandpass_filter(Y_signals_cpu[:, i], low_cutoff=2, high_cutoff=15, sampling_rate=300)

    Y_signals_gpu = cp.asarray(Y_signals_cpu_filtered) # cp.asarray(Y_signals_cpu)

    #...get Orthogonalization matrix
    ortho_matrix = OrthoMatrix(nDCT=3, num_frame_stimulus=stim_frames)
    O_gpu = ortho_matrix.get_orthogonalization_matrix() # (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))

    # grid and refine fitting
    S_star_columnmajor_gpu, dS_star_dx_columnmajor_gpu, dS_star_dy_columnmajor_gpu, dS_star_dsigma_columnmajor_gpu = gpu_compute_model_signals(O_gpu=O_gpu)
    refined_matching_results, Fex_results = gpu_fitting(O_gpu, arr_2d_location_inv_M_cpu, Y_signals_gpu, Y_signals_cpu, S_star_columnmajor_gpu, dS_star_dx_columnmajor_gpu, dS_star_dy_columnmajor_gpu, dS_star_dsigma_columnmajor_gpu)

    # verfication R2
    # json_data = R2.compute_r2_variance_explained_results(refined_matching_results, Y_signals_cpu, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames, O_gpu = O_gpu, Y_signals_gpu = Y_signals_gpu)
    # JsonMgr.write_to_file(filepath=cfg.result_file_path, data=json_data)

    # r2 with e as f(x)
    #r2_results = R2.get_r2_new_method_with_epsilon_as_Fx(Y_signals_gpu, O_gpu, epsilon_results_cpu = Fex_results)
    #json_data = R2.format_in_json_format(r2_results, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames)    
    #JsonMgr.write_to_file(filepath=cfg.result_file_path, data=json_data)

    ## r2 with e as yts
    r2_results = R2.get_r2_num_den_method_with_epsilon_as_yTs(Y_signals_gpu, O_gpu, refined_matching_results, x_range_cpu, y_range_cpu, stimulus)
    json_data = R2.format_in_json_format(r2_results, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames)    
    JsonMgr.write_to_file(filepath=cfg.result_file_path, data=json_data)

if __name__ == "__main__":
    cProfile.run('main()', sort='cumulative')

print("done")




