# Add my "oprf" package path
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cProfile
import os
import datetime
import pandas as pd

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
from oprf.hpc.hpc_matrix_operations import MatrixOps

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

    # stimulus data on GPU
    stimulus_data_columnmajor_gpu = stimulus.get_flattened_columnmajor_stimulus_data_gpu()

    #---Orthogonalization + Nomalization
    S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu = model_signals.get_orthonormal_signals(O_gpu=O_gpu
                                                                                                                                                           , stimulus_data_columnmajor_gpu= stimulus_data_columnmajor_gpu)

    return S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu

def gpu_fitting(O_gpu, arr_2d_location_inv_M_cpu, Y_signals_gpu, Y_signals_cpu, S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu):    
    #--------send data to gpu    
    # Y_signals_gpu = cp.asarray(Y_signals_cpu)

    #------- Synchronize and release memory
    cp.cuda.Device().synchronize()
    
    # get the grid-fit results, and error-terms (plus its derivatives)
    best_fit_proj_cpu, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu = GridFit.get_error_terms(O_gpu=O_gpu
                                                                                                           , isResultOnGPU = False
                                                                                                    , Y_signals_gpu=Y_signals_gpu
                                                                                                    , S_prime_columnmajor_gpu=S_prime_columnmajor_gpu
                                                                                                    , dS_prime_dx_columnmajor_gpu=dS_prime_dx_columnmajor_gpu
                                                                                                    , dS_prime_dy_columnmajor_gpu=dS_prime_dy_columnmajor_gpu
                                                                                                    , dS_prime_dsigma_columnmajor_gpu=dS_prime_dsigma_columnmajor_gpu
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
                                                        , search_space_frames
                                                        , search_space_xx
                                                        , search_space_yy
                                                        , search_space_sigma_range)

    return refined_results, Fex_results, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu

# ###################################-----------Main()---------------------------------####################
def add_to_dataframe(dataframe, df_idx, type, param_name, param_value):
    dataframe.loc[df_idx] = [type, param_name, param_value]
    df_idx = df_idx + 1
    return df_idx

def main():
    # compute M_inv
    mu_X_grid, mu_Y_grid, sigma_grid = np.meshgrid(search_space_xx, search_space_yy, search_space_sigma_range) 
    arr_2d_location_inv_M_cpu = CoefficientMatrix.Grids2MpInv(mu_X_grid, mu_Y_grid, sigma_grid) ##<<<<--------Compute pre-define matrix M

    # y-signals
    y_data = ObservedData(data_type=DataType.measured_data)
    Y_signals_cpu = y_data.get_y_signals((cfg.measured_data['filepaths'])[0])
    Y_signals_gpu = cp.asarray(Y_signals_cpu)

    #...get Orthogonalization matrix
    ortho_matrix = OrthoMatrix(nDCT=3, num_frame_stimulus=stim_frames)
    O_gpu = ortho_matrix.get_orthogonalization_matrix() # (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))

    # grid and refine fitting
    S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu = gpu_compute_model_signals(O_gpu=O_gpu)
    refined_matching_results, Fex_results, e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu = gpu_fitting(O_gpu, arr_2d_location_inv_M_cpu, Y_signals_gpu, Y_signals_cpu, S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu)


    # Test errors
    refine_e_full_cpu, refine_de_dx_full_cpu, refine_de_dy_full_cpu, refine_de_dsigma_full_cpu = RefineFit.get_all_debug_info_error_terms_after_refinement(refined_matching_results = refined_matching_results, 
                                                                                                  Y_signals_gpu = Y_signals_gpu, 
                                                                                                  O_gpu = O_gpu, 
                                                                                                  stimulus = stimulus, 
                                                                                                  stim_height = stim_height, 
                                                                                                  stim_width = stim_width, 
                                                                                                  x_range_cpu = x_range_cpu, 
                                                                                                  y_range_cpu = y_range_cpu)

    # create dataframe
    df = pd.DataFrame(columns = ["CoarseEsitmations", "RefinedEstimations", "CoarseGradients", "RefinedGradients"])
    # df = pd.DataFrame(columns = ["type", "param", "value"])
    df_idx = 0
    best_fit_proj_cpu = np.nanargmax(e_cpu, axis=1)
    for y_idx in range(0, Y_signals_cpu.shape[1], 10):
        best_s_idx = best_fit_proj_cpu[y_idx]
        best_s_3d_idx = MatrixOps.flatIdx2ThreeDimIndices(best_s_idx, search_space_rows, search_space_cols, search_space_frames)
    
        #...compute the de/dx, de/dy and de/dsigma vectors#         
        e_vec_cpu = e_cpu[y_idx, best_s_idx]
        de_dx_vec_cpu = de_dx_full_cpu[y_idx, best_s_idx]
        de_dy_vec_cpu = de_dy_full_cpu[y_idx, best_s_idx]
        de_dsigma_vec_cpu = de_dsigma_full_cpu[y_idx, best_s_idx]
        coarse_gradients = [e_vec_cpu, de_dx_vec_cpu, de_dy_vec_cpu, de_dsigma_vec_cpu]

        #...compute REFINED de/dx, de/dy and de/dsigma vectors#         
        best_refined_idx = y_idx
        refine_e_cpu = refine_e_full_cpu[y_idx, best_refined_idx]
        refine_de_dx_cpu = refine_de_dx_full_cpu[y_idx, best_refined_idx]
        refine_de_dy_cpu = refine_de_dy_full_cpu[y_idx, best_refined_idx]
        refine_de_dsigma_cpu = refine_de_dsigma_full_cpu[y_idx, best_refined_idx]
        refined_gradients = [refine_e_cpu, refine_de_dx_cpu, refine_de_dy_cpu, refine_de_dsigma_cpu]

        # coarse estimations...mu_X_grid, mu_Y_grid, sigma_grid
        x_coarse_idx = best_s_3d_idx[0]
        y_coarse_idx = best_s_3d_idx[1]
        sigma_coarse_idx = best_s_3d_idx[2]
        x = mu_X_grid[x_coarse_idx, y_coarse_idx, sigma_coarse_idx]
        y = mu_Y_grid[x_coarse_idx, y_coarse_idx, sigma_coarse_idx]
        sigma = search_space_sigma_range[sigma_coarse_idx]
        coarse_estimations = [x, y, sigma]

        # refined estimations
        refined_matching_results_arr = np.array(refined_matching_results)
        refined_estimations = refined_matching_results_arr[y_idx]

        # # Append Dict as row to DataFrame
        # new_row = {"CoarseEsitmations": coarse_estimations, 
        #            "RefinedEstimations": refined_estimations,
        #            "CoarseGradients": coarse_gradients,
        #            "RefinedGradients": refined_gradients}

        # # df.append(new_row, ignore_index=True)
        # df["CoarseEsitmations"] = [coarse_estimations]
        # df["RefinedEstimations"] = [refined_estimations]
        # df["CoarseGradients"] = [coarse_gradients]
        # df["RefinedGradients"] = [refined_gradients]

        df.loc[y_idx] = [coarse_estimations, refined_estimations, coarse_gradients, refined_gradients]    


        # # for seaborn pltos
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="coarse", param_name="e", param_value=coarse_gradients[0])
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="coarse", param_name="de_dx", param_value=coarse_gradients[1])
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="coarse", param_name="de_dy", param_value=coarse_gradients[2])
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="coarse", param_name="de_dsigma", param_value=coarse_gradients[3])
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="refine", param_name="e", param_value=refined_gradients[0])
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="refine", param_name="de_dx", param_value=refined_gradients[1])
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="refine", param_name="de_dy", param_value=refined_gradients[2])
        # df_idx = add_to_dataframe(dataframe=df, df_idx=df_idx, type="refine", param_name="de_dsigma", param_value=refined_gradients[3])        
        # df.to_excel("D:/results/gradients-test/gradients-new-with-[wihtout-minus-sigma]_fx-51x51x8.xlsx", index=False)  # Set index=False if you don't want to write row indices


    df.to_excel("D:/results/gradients-test/gradients-new-Estimations-[without-minus-sigma]_fx-51x51x8.xlsx", index=False)  # Set index=False if you don't want to write row indices
    print('done')

    # ## r2 with e as yts
    # r2_results = R2.get_r2_new_method_with_epsilon_as_yTs(Y_signals_gpu, O_gpu, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames)
    # json_data = R2.format_in_json_format(r2_results, refined_matching_results, x_range_cpu, y_range_cpu, stimulus, stim_height, stim_width, stim_frames)    
    
    # # result name        
    # file = os.path.basename((cfg.measured_data['filepaths'])[0]) # process only first filename
    # filename = (file.split("."))[0]
    # filename = (str(str(datetime.date.today()) + '_') if cfg.results['prepend_date'] == "True" else '') + filename
    # result_file = cfg.results['basepath'] + filename.replace("bold", "estimates") + cfg.results['custom_filename_postfix'] + ".json"
    
    # # write to file    
    # JsonMgr.write_to_file(filepath=result_file, data=json_data)

if __name__ == "__main__":
    cProfile.run('main()', sort='cumulative')

print("done")




