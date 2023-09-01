
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys
from scipy.ndimage import zoom
import os
from pathlib import Path
from typing import List
import math
from matplotlib.widgets import Slider, Button, TextBox

# HRF Generator
hrf_module_path = (Path(__file__).resolve().parent / '../external-packages/nipy-hrf-generator').resolve()
sys.path.append(str(hrf_module_path))
from hrf_generator_script import spm_hrf_compat

# DeepRF module
deeprf_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../external-packages/DeepRF-main'))
sys.path.append(deeprf_module_path)
from data_synthetic import *
import data_synthetic as deeprf_data_synthetic

class ReceptiveFieldResponse:
    def __init__(self, x, y, timecourse):
        self.x = x
        self.y= y
        self.timecourse = timecourse

class Stimulus:
    def __init__(self, relative_file_path, size_in_degrees):
        # IMPORTANT: The file paths are resolved relative to the current Python script file instead of the current working directory (cwd)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        stimulus_file_path = os.path.join(script_directory, relative_file_path)
        stimulus_img = nib.load(stimulus_file_path)
        self.size_in_degrees = size_in_degrees
        self.data = stimulus_img.get_fdata()

    def resample_stimulus_data(self
                               , resampled_stimulus_shape # e.g. resampled_stimulus_shape = (DESIRED_STIMULUS_SIZE_X, DESIRED_STIMULUS_SIZE_Y, original_stimulus_shape[2])
                               ):  
        original_stimulus_shape = self.data.shape # e.g. (1024, 1024, 1, 300)        
        resampling_factors = (
        resampled_stimulus_shape[0] / (original_stimulus_shape[0] -1),
        resampled_stimulus_shape[1] / (original_stimulus_shape[1] - 1),
        1  # Keep the number of time points unchanged
        )   
        self.data = zoom(self.data.squeeze(), resampling_factors, order=1)
        return self.data

class ModelsGrid:
    def __init__(self
                 , grid_size_x
                 , grid_size_y
                 , sigma
                 , stimulus : Stimulus
                 , hrf_curve
                 ):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.sigma = sigma        
        self.stimulus = stimulus
        self.meshgrid_X = None
        self.meshgrid_Y = None
        self.hrf_curve = hrf_curve
        # self.data = []        
        self.data = np.zeros((grid_size_y, grid_size_x), dtype=ReceptiveFieldResponse)        
            
    def _generate_meshgrid(self):
        x = np.linspace( - self.stimulus.size_in_degrees, + self.stimulus.size_in_degrees, self.stimulus.data.shape[0])
        y = np.linspace( - self.stimulus.size_in_degrees, + self.stimulus.size_in_degrees, self.stimulus.data.shape[1])
        self.meshgrid_X, self.meshgrid_Y  = np.meshgrid(x, y)

    def _generate_2d_gaussian(self, mean_x, mean_y):                    
        mean_corrdinates = [mean_x, mean_y]        
        Z = np.exp(-(self.meshgrid_X - mean_corrdinates[0])**2 / (2 * self.sigma**2)) * np.exp(-(self.meshgrid_Y - mean_corrdinates[1])**2 / (2 * self.sigma**2))
        return Z
    
    # Generate the timecourse for a Gaussian Curve (for a particular pixel location)
    # by taking the Stimulus frames into account
    def _generate_pixel_location_time_course(self, Z):
        time_points = self.stimulus.data.shape[2]
        area_under_gaussian = np.zeros(time_points)

        for t in range(time_points):
            # Use stimulus_frame_data as a binary mask
            mask = (self.stimulus.data[:, :, t] > 0).astype(int)
            
            # Apply mask to the Gaussian curve
            masked_gaussian = Z * mask

            # Compute area under the Gaussian curve after applying the mask
            area_under_gaussian[t] = np.sum(masked_gaussian)
            
        return area_under_gaussian

    def generate_model_responses(self):
        self._generate_meshgrid()
        # Create Expected Responses Grid: Traverse through all locations of the defined grid
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                Z = self._generate_2d_gaussian(mean_x= x, mean_y= y)
                pixel_location_timecourse = self._generate_pixel_location_time_course(Z)
                pixel_location_timecourse /= pixel_location_timecourse.max()
                r_t = np.convolve(pixel_location_timecourse, self.hrf_curve, mode='full')[:len(pixel_location_timecourse)]      
                expected_receptive_field_reponse = ReceptiveFieldResponse(x=x, y=y, timecourse=r_t)  
                #self.data.append(expected_receptive_field_reponse)
                self.data[y][x] = expected_receptive_field_reponse

# define noise levels in percentages
class NoiseLevels:
    def __init__(self
                 , desired_low_freq_noise_level
                 , desired_physiological_noise_level
                 , desired_system_noise_level
                 , desired_task_noise_level
                 , desired_temporal_noise_level):
        self.desired_low_freq_noise_level = desired_low_freq_noise_level
        self.desired_physiological_noise_level = desired_physiological_noise_level
        self.desired_system_noise_level = desired_system_noise_level
        self.desired_task_noise_level = desired_task_noise_level
        self.desired_temporal_noise_level = desired_temporal_noise_level        

class SynthesizedDataGenerator:
    def __init__(self
                 , noise_levels : NoiseLevels
                 , source_data : List[ReceptiveFieldResponse] # data from which new data will be synthesized
                 , synthesis_ratio # how many new timecourses need to be synthesized per given timecourse in the kick_start_data
                 , TR
                 ):
        self.noise_levels = noise_levels
        self.source_data = source_data
        self.synthesis_ratio = synthesis_ratio
        self.TR = TR
        #self.data = []
        self.data = np.zeros(synthesis_ratio*source_data.shape[0]*source_data.shape[1], dtype=ReceptiveFieldResponse)

    def _create_noisy_receptive_field_data(self
                                           , org_single_receptive_field_data : ReceptiveFieldResponse):
        org_timecourse_std = np.std(org_single_receptive_field_data.timecourse)

        # Compute CNRs
        # CNR = std_signal / sigma_noise, where "sigma_noise = desired_noise_level * std_signal"        
        cnr_low_freq = org_timecourse_std / (self.noise_levels.desired_low_freq_noise_level * org_timecourse_std)        
        cnr_physiological = org_timecourse_std / (self.noise_levels.desired_physiological_noise_level * org_timecourse_std)
        cnr_system = org_timecourse_std / (self.noise_levels.desired_system_noise_level * org_timecourse_std)
        cnr_task = org_timecourse_std / (self.noise_levels.desired_task_noise_level * org_timecourse_std)
        cnr_temporal = org_timecourse_std / (self.noise_levels.desired_temporal_noise_level * org_timecourse_std)
        
        # create noises
        low_frequency_noise = deeprf_data_synthetic.LowFrequency(cnr_low_freq, self.TR )
        physiological_noise = deeprf_data_synthetic.Physiological(cnr_physiological, self.TR )
        system_noise = deeprf_data_synthetic.System(cnr_system, np.random.RandomState())
        task_noise = deeprf_data_synthetic.Task(cnr_task, np.random.RandomState())
        temporal_noise = deeprf_data_synthetic.Temporal(cnr_temporal, np.random.RandomState())
        random_generator_y = np.random.RandomState(1258566) # used to generate predictions
        noise = deeprf_data_synthetic.Noise(random_generator_y.rand(5)
                                            , low_frequency_noise
                                            , physiological_noise
                                            , system_noise
                                            , task_noise
                                            , temporal_noise)
        
        # compute noisy timecourse
        noisy_timecourse = noise(org_single_receptive_field_data.timecourse) 
        noisy_receptive_field_response = ReceptiveFieldResponse(x=org_single_receptive_field_data.x
                                                          , y=org_single_receptive_field_data.y
                                                          , timecourse=noisy_timecourse )                                                
        return noisy_receptive_field_response
            
    # def generate_synthetic_data(self):
    #     self.data = [
    #         self._create_noisy_receptive_field_data(org_single_receptive_field_data=src_receptive_field_data)

    #          # type: ReceptiveFieldResponse
    #         for src_receptive_field_data in self.source_data

    #         # each source data would generate multiple synthetic data (as many times as defined by synthesis_ratio)
    #         for _ in range(self.synthesis_ratio)
    #     ]
    #     return self.data

    def generate_synthetic_data(self):
        for times in range(self.synthesis_ratio):
            for x in range (self.source_data.shape[0]):
                for y in range (self.source_data.shape[1]):
                    receptive_field_data = self.source_data[x][y]
                    noisy_data = self._create_noisy_receptive_field_data(receptive_field_data)
                    index = (x*self.source_data.shape[1] + y) + times*self.source_data.shape[0] * self.source_data.shape[1]
                    self.data[index] = noisy_data
                    #self.data[(x*self.source_data.shape[1] + y) + times] = noisy_data

        # self.data = [
        #     self._create_noisy_receptive_field_data(org_single_receptive_field_data=src_receptive_field_data)

        #      # type: ReceptiveFieldResponse
        #     for src_receptive_field_data in self.source_data

        #     # each source data would generate multiple synthetic data (as many times as defined by synthesis_ratio)
        #     for _ in range(self.synthesis_ratio)
        # ]
        return self.data    

# Ordinary Least Squares (OLS)
class OLS:
    def __init__(self
                 , modelled_signals
                 , measured_signals : List[ReceptiveFieldResponse]
                 , type = 'ordinary least squares - simplified'):
        self.modelled_signals = modelled_signals
        self.measured_signals = measured_signals
        self.N = len(modelled_signals[0]) # number of data points in each timecourse
    
    def _make_Trends(self, nDCT=3):
        # Generates trends similar to 'rmMakeTrends' from the 'params' struct in mrVista.    
        # nDCT: Number of discrete cosine transform components.
        
        tf = self.N # this is equal to 300
        ndct = 2 * nDCT + 1
        trends = np.zeros((np.sum(tf), np.max([np.sum(ndct), 1])))        
        
        tc = np.linspace(0, 2.*np.pi, tf)[:, None]        
        trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))

        nTrends = trends.shape[1]
        return trends, nTrends       
    
    def _get_orthogonal_trends(self, trends):
        q, r = np.linalg.qr(trends) # QR decomposition
        q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
        return q
    
    def _compute_orthonormal_model_signals(self, R):
        # Calculate orthogonal projection using O = (I - R @ R^T)
        O = (np.eye(self.N) - R @ R.T) # R: Orthonormal Regressors
        
        # compute orthonormal version of the modelled signals
        total_rows = (len(self.modelled_signals))
        total_cols = len(self.modelled_signals[0])
        orthonormalized_modelled_signals =  np.zeros((total_rows, total_cols)) # pre-allocate array
        for i in range(len(self.modelled_signals)):
            s = self.modelled_signals[i] #s
            s = O @ s #s_prime 
            s /= np.sqrt(s @ s) # orthonormalized_model_signal
            orthonormalized_modelled_signals[i] = s

        return orthonormalized_modelled_signals
    
    # Fitting: Calculate the square of the projection of orthonormal modelled singlas (s_prime) onto targets
    def compute_proj_squared(self):   
        trends, _ = self._make_Trends()
        R = self._get_orthogonal_trends(trends=trends)
        orthonormal_modelled_signals = self._compute_orthonormal_model_signals(R=R)

        # grid of orthonormal modelled signals
        grid_s_prime = np.vstack([timecourse for timecourse in orthonormal_modelled_signals]).T

        # grid of measured/simulated target signals
        grid_y  = np.vstack([timecourse for timecourse in self.measured_signals]).T
        
        # (y^T . s_prime)^2
        proj_squared = (grid_s_prime.T @ grid_y)**2 
        
        # find best matches along the rows (vertical axis) of the array
        best_fit_proj = np.argmax(proj_squared, axis=0)
        return best_fit_proj




##############################-----------PROGRAM-----------------------#################################
# def update_plot(vall):
#     # Get slider values
#     desired_low_freq_noise_level = low_freq_noise_slider.val
#     desired_system_noise_level = system_noise_slider.val
#     desired_physiological_noise_level = physiological_noise_slider.val
#     desired_task_noise_level = task_noise_slider.val
#     desired_temporal_noise_level = temporal_noise_slider.val

    # # Variables
    # nCols_grid = 10
    # nRows_grid = 10
    # synthesis_ratio=3

    # # Load stimulus
    # stimulus = Stimulus("../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
    # stimulus.data = stimulus.resample_stimulus_data((101, 101, stimulus.data.shape[2]))

    # # HRF Curve
    # hrf_t = np.linspace(0,30,31)
    # hrf_curve = spm_hrf_compat(hrf_t) 

    # # Expected responses Grid    
    # grid = ModelsGrid(grid_size_x = nCols_grid
    #              , grid_size_y = nRows_grid
    #              , sigma = 2
    #              , stimulus= stimulus
    #              , hrf_curve = hrf_curve)
    # grid.generate_model_responses()

    # # define desired noise levels
    # noise_levels = NoiseLevels(desired_low_freq_noise_level
    #                            , desired_system_noise_level
    #                            , desired_physiological_noise_level
    #                            , desired_task_noise_level
    #                            , desired_temporal_noise_level)
    
    # # generate synthetic data    
    # data_synthesizer = SynthesizedDataGenerator(noise_levels=noise_levels, source_data = grid.data, synthesis_ratio=synthesis_ratio, TR=1)
    # noisy_synthetic_data = data_synthesizer.generate_synthetic_data()
    
    # # Data matching
    # grid_timecourses = [item.timecourse for row in grid.data for item in row]
    # data_synthesizer_timecourses =[item.timecourse for item in data_synthesizer.data]

    # ols = OLS(modelled_signals=grid_timecourses
    #                            , measured_signals=data_synthesizer_timecourses
    #                            )
    # best_fit_proj = ols.compute_proj_squared()

    # # Analyze fitting results
    # grid_fitting_results = [[0 for _ in range(nCols_grid)] for _ in range(nRows_grid)]
    # for i in range(len(best_fit_proj)):
    #     target_idx = i
    #     target_synthesizer_location = (noisy_synthetic_data[target_idx].x, noisy_synthetic_data[target_idx].y)
    #     flattened_signal_idx = best_fit_proj[i]
    #     signal_idx_x = math.floor(flattened_signal_idx / nCols_grid)
    #     signal_idx_y = flattened_signal_idx - (signal_idx_x * nCols_grid)
    #     matched_signal_location = (grid.data[signal_idx_x][signal_idx_y].x, grid.data[signal_idx_x][signal_idx_y].y)
    #     if target_synthesizer_location == matched_signal_location:
    #         grid_fitting_results[target_synthesizer_location[1]][target_synthesizer_location[0]] += 1  # Note the swap of indices

    # # Normalize results
    # grid_fitting_results = np.array(grid_fitting_results) / synthesis_ratio

    # # Test
    # original_signal = grid.data[0][0].timecourse
    # noisy_signal = noisy_synthetic_data[0].timecourse
    # plt.figure(1)
    # plt.plot(original_signal)
    # plt.plot(noisy_signal)
    # plt.title('Original vs. Noisy Signal')
    # plt.xlabel('Time')
    # plt.ylabel('Signal Amplitude')
    
    # # Set the color mapping range
    # vmin = np.min(grid_fitting_results)  # Minimum value for color mapping
    # vmax = np.max(grid_fitting_results)  # Maximum value for color mapping

    # # Create a heatmap plot
    # plt.figure(2)
    # plt.imshow(grid_fitting_results, cmap='hot', origin='lower', extent=[0, nCols_grid, 0, nRows_grid], vmin=vmin, vmax=vmax)

    # # Annotate each box with its value
    # for i in range(nRows_grid):
    #     for j in range(nCols_grid):
    #         plt.text(j + 0.5, i + 0.5, f"{grid_fitting_results[i][j]:.2f}", color='black', ha='center', va='center')

    # # Customize grid lines at integer values
    # plt.xticks(range(nCols_grid))
    # plt.yticks(range(nRows_grid))
    # plt.grid(which='both', color='gray', linestyle='-', linewidth=1)

    # # Add labels    
    # plt.xlabel('Columns')
    # plt.ylabel('Rows')
    # plt.title('Normalized 2D Location Prediction Heatmap')

    # # Show the plot
    # plt.colorbar(label='Normalized Value')
    # plt.show()

    # print('Program finished !')
    

##############################---Sliders---#######################


class NoiseLevelAdjustmentApp:
    def __init__(self):
        self.desired_low_freq_noise_level = 2
        self.desired_system_noise_level = 2
        self.desired_physiological_noise_level = 2
        self.desired_task_noise_level = 2
        self.desired_temporal_noise_level = 2            
        self.nCols_grid = 10
        self.nRows_grid = 10
        self.synthesis_ratio = 3
        self.grid_fitting_results = np.zeros((self.nRows_grid, self.nCols_grid), dtype='float')    

        # Figure
        self.fig, ((self.ax1, self.ax2)) = plt.subplots(1, 2, gridspec_kw={'bottom': 0.2})
        self.fig.text(0.5, 0.96, 'Noise Modelling', ha='center', fontsize=12) # title        
        self.status_label = self.fig.text(0.5, 0.01, 'Status: Ready', ha='center', fontsize=10) # bottom status label

        # Create sliders and "Recompute" button in the main figure
        self.low_freq_noise_slider_ax = self.fig.add_axes([0.15, 0.01, 0.25, 0.02])
        self.low_freq_noise_slider = Slider(self.low_freq_noise_slider_ax, 'Low Freq Noise', 0, 10, valinit=self.desired_low_freq_noise_level)

        self.system_noise_slider_ax = self.fig.add_axes([0.15, 0.04, 0.25, 0.02])
        self.system_noise_slider = Slider(self.system_noise_slider_ax, 'System Noise', 0, 10, valinit=self.desired_system_noise_level)

        self.physiological_noise_slider_ax = self.fig.add_axes([0.15, 0.07, 0.25, 0.02])
        self.physiological_noise_slider = Slider(self.physiological_noise_slider_ax, 'Physiological Noise', 0, 10, valinit=self.desired_physiological_noise_level)

        self.task_noise_slider_ax = self.fig.add_axes([0.15, 0.10, 0.25, 0.02])
        self.task_noise_slider = Slider(self.task_noise_slider_ax, 'Task Noise', 0, 10, valinit=self.desired_task_noise_level)

        self.temporal_noise_slider_ax = self.fig.add_axes([0.15, 0.13, 0.25, 0.02])
        self.temporal_noise_slider = Slider(self.temporal_noise_slider_ax, 'Temporal Noise', 0, 10, valinit=self.desired_temporal_noise_level)

        recompute_button_ax = self.fig.add_axes([0.8, 0.01, 0.1, 0.04])
        recompute_button = Button(recompute_button_ax, 'Recompute')
        recompute_button.on_clicked(self.recompute_button_clicked)

        # Create a figure for the main plot with 2x2 subplots
        grid_fitting_initial_results, org_timecourse_sample, noisy_timecourse = self.compute_probability_density() # intial computation of grid fitting
        self.im = self.ax1.imshow(grid_fitting_initial_results, cmap='hot', origin='lower', extent=[0, self.nCols_grid, 0, self.nRows_grid], vmin=0.0, vmax=1) # heatmap
        self.colorbar = self.fig.colorbar(self.im, ax=self.ax1, label='Normalized Value') # vertical colorbar for heatmap
        self.annotations = [] # for heatmap annotations
        self.org_timecourse_plot, = self.ax2.plot(range(len(org_timecourse_sample)), org_timecourse_sample) # original timecourse plot
        self.noisy_timecourse_plot, = self.ax2.plot(range(len(noisy_timecourse)), noisy_timecourse) # noisy timecourse plot

        # Initial plot
        self.update_plot(None)

        plt.show()    


    def update_status(self, text):
        # Callback function for updating the status label
        self.status_label.set_text(f'Status: {text}')
        plt.draw()

    def compute_probability_density(self):
        self.update_status("computing data...")
        # Load stimulus
        stimulus = Stimulus("../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
        stimulus.data = stimulus.resample_stimulus_data((101, 101, stimulus.data.shape[2]))

        # HRF Curve
        hrf_t = np.linspace(0, 30, 31)
        hrf_curve = spm_hrf_compat(hrf_t)

        # Expected responses Grid
        grid = ModelsGrid(grid_size_x=self.nCols_grid
                          , grid_size_y=self.nRows_grid
                          , sigma=2
                          , stimulus=stimulus
                          , hrf_curve=hrf_curve)
        grid.generate_model_responses()

        # define desired noise levels
        noise_levels = NoiseLevels(self.desired_low_freq_noise_level
                                    , self.desired_system_noise_level
                                    , self.desired_physiological_noise_level
                                    , self.desired_task_noise_level
                                    , self.desired_temporal_noise_level)

        # generate synthetic data
        data_synthesizer = SynthesizedDataGenerator(noise_levels=noise_levels, source_data=grid.data, synthesis_ratio=self.synthesis_ratio, TR=1)
        noisy_synthetic_data = data_synthesizer.generate_synthetic_data()

        # Data matching
        grid_timecourses = [item.timecourse for row in grid.data for item in row]
        data_synthesizer_timecourses = [item.timecourse for item in data_synthesizer.data]

        ols = OLS(modelled_signals=grid_timecourses
                  , measured_signals=data_synthesizer_timecourses
                  )
        best_fit_proj = ols.compute_proj_squared()

        # Analyze fitting results
        grid_fitting_results = [[0 for _ in range(self.nCols_grid)] for _ in range(self.nRows_grid)]
        for i in range(len(best_fit_proj)):
            target_idx = i
            target_synthesizer_location = (noisy_synthetic_data[target_idx].x, noisy_synthetic_data[target_idx].y)
            flattened_signal_idx = best_fit_proj[i]
            signal_idx_x = math.floor(flattened_signal_idx / self.nCols_grid)
            signal_idx_y = flattened_signal_idx - (signal_idx_x * self.nCols_grid)
            matched_signal_location = (grid.data[signal_idx_x][signal_idx_y].x, grid.data[signal_idx_x][signal_idx_y].y)
            if target_synthesizer_location == matched_signal_location:
                grid_fitting_results[target_synthesizer_location[1]][target_synthesizer_location[0]] += 1  # Note the swap of indices

        # Normalize results
        grid_fitting_results = np.array(grid_fitting_results) / self.synthesis_ratio
        self.update_status("Data computed !!!")

        return grid_fitting_results, grid.data[0][0].timecourse, noisy_synthetic_data[0].timecourse

    def update_plot(self, val):
        self.update_status("updating plot...")
        # Get slider values
        self.desired_low_freq_noise_level = self.low_freq_noise_slider.val
        self.desired_system_noise_level = self.system_noise_slider.val
        self.desired_physiological_noise_level = self.physiological_noise_slider.val
        self.desired_task_noise_level = self.task_noise_slider.val
        self.desired_temporal_noise_level = self.temporal_noise_slider.val

        #  re-compute data
        fitting_results, org_tc, noisy_tc = self.compute_probability_density()

        # Update the data in the existing imshow and plot objects
        self.im.set_array(fitting_results)
        self.org_timecourse_plot.set_ydata(org_tc)
        self.noisy_timecourse_plot.set_ydata(noisy_tc)  # Fixed variable name

        # Annotate each box with its value
        for annotation in self.annotations:
            annotation.remove()
        self.annotations = [] 
        nRows_grid, nCols_grid = fitting_results.shape
        for i in range(nRows_grid):
            for j in range(nCols_grid):
                annotation = self.ax1.text(j + 0.5, i + 0.5, f"{fitting_results[i][j]:.2f}", color='black', ha='center', va='center')
                self.annotations.append(annotation)

        # Customize grid lines at integer values
        self.ax1.set_xticks(range(self.nCols_grid))
        self.ax1.set_yticks(range(self.nRows_grid))
        self.ax1.grid(which='both', color='gray', linestyle='-', linewidth=1)

        # Add labels    
        self.ax1.set_xlabel('Columns')
        self.ax1.set_ylabel('Rows')
        self.ax1.set_title('Normalized 2D Location Prediction Heatmap')

        self.update_status("Plots updated !!!")

        # Force canvas update
        self.fig.canvas.draw_idle()


    def recompute_button_clicked(self, event):
        self.update_status("Button clicked")
        self.update_plot(None)
        self.update_status("Recomputation done !!!")

if __name__ == "__main__":
    app = NoiseLevelAdjustmentApp()

