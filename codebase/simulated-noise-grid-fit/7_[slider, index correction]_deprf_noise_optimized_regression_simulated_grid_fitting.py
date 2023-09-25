# Add my "oprf" package path
import sys
sys.path.append("D:/code/sid-git/fmri/")

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

# Local Imports
from oprf.standard.prf_stimulus import Stimulus
from oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse
from oprf.standard.prf_ordinary_least_square import OLS
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator
from oprf.external.DeepRF import data_synthetic as deeprf_data_synthetic # DeepRF module
from oprf.analysis.prf_synthetic_data_generator import NoiseLevels


# This is older version of my Model Grid before I created the newer generic version called "QuadrilateralSignalsSpace"
class ModelsGrid:
    def __init__(self
                 , grid_nRows
                 , grid_nCols
                 , pRF_size_sigma
                 , stimulus : Stimulus
                 , hrf_curve
                 ):
        self.grid_nRows = grid_nRows
        self.grid_nCols = grid_nCols
        self.pRF_size_sigma = pRF_size_sigma        
        self.stimulus = stimulus
        self.gaussian_meshgrid_X = None
        self.gaussian_meshgrid_Y = None
        self.test_meshgrid_X  = None
        self.test_meshgrid_Y = None
        self.hrf_curve = hrf_curve
        self.data = np.zeros((grid_nRows, grid_nCols), dtype=ReceptiveFieldResponse)        
            
    def _generate_meshgrids(self):
        # For Gaussian Meshgrid: This meshgrid must have the same size as the stimulus. The 2D Gaussian curves generated using these X and Y meshgrids are used to model a pRF response
        gaussian_x_values = np.linspace( - self.stimulus.size_in_degrees, + self.stimulus.size_in_degrees, self.stimulus.resampled_data.shape[0])
        gaussian_y_values = np.linspace( - self.stimulus.size_in_degrees, + self.stimulus.size_in_degrees, self.stimulus.resampled_data.shape[1])
        self.gaussian_meshgrid_X, self.gaussian_meshgrid_Y  = np.meshgrid(gaussian_x_values, gaussian_y_values)

        # For Test Locations Grid Meshgrid: These X and Y meshgrids below contains the MEAN POSITIONS for which we would like to generate the "modelled responses". 
        # The size/shape of these meshgrids depends on us i.e. how many model signal we want to generate
        test_grid_x_points = np.linspace(-self.stimulus.size_in_degrees, self.stimulus.size_in_degrees, self.grid_nCols) 
        test_grid_y_points = np.linspace(-self.stimulus.size_in_degrees, self.stimulus.size_in_degrees, self.grid_nRows) 
        self.test_meshgrid_X, self.test_meshgrid_Y = np.meshgrid(test_grid_x_points, test_grid_y_points)

    def _generate_2d_gaussian(self, mean_x, mean_y):                    
        mean_corrdinates = [mean_x, mean_y]        
        Z = np.exp(-(self.gaussian_meshgrid_X - mean_corrdinates[0])**2 / (2 * self.pRF_size_sigma**2)) * np.exp(-(self.gaussian_meshgrid_Y - mean_corrdinates[1])**2 / (2 * self.pRF_size_sigma**2))
        return Z
    
    # Generate the timecourse for a Gaussian Curve (for a particular pixel location)
    # by taking the Stimulus frames into account
    def _generate_pixel_location_time_course(self, Z):
        time_points = self.stimulus.resampled_data.shape[2]
        area_under_gaussian = np.zeros(time_points)

        for t in range(time_points):
            # Use stimulus_frame_data as a binary mask
            mask = (self.stimulus.resampled_data[:, :, t] > 0).astype(int)
            
            # Apply mask to the Gaussian curve
            masked_gaussian = Z * mask

            # Compute area under the Gaussian curve after applying the mask
            area_under_gaussian[t] = np.sum(masked_gaussian)
            
        return area_under_gaussian

    def generate_model_responses(self):
        self._generate_meshgrids()
                
        # Create Expected Responses Grid: Traverse through all locations of the defined grid
        for row in range(self.grid_nRows):
            for col in range(self.grid_nCols):
                Z = self._generate_2d_gaussian(mean_x= self.test_meshgrid_X[row][col], mean_y= self.test_meshgrid_Y[row][col])
                pixel_location_timecourse = self._generate_pixel_location_time_course(Z)
                pixel_location_timecourse /= pixel_location_timecourse.max()
                r_t = np.convolve(pixel_location_timecourse, self.hrf_curve, mode='full')[:len(pixel_location_timecourse)]      
                expected_receptive_field_reponse = ReceptiveFieldResponse(row=row, col=col, timecourse=r_t)  
                self.data[row][col] = expected_receptive_field_reponse                                     

# This is the older version of original "SynthesizedDataGenerator". In this version, I am using DeepRF models but in some of the versions, I am simply using Gaussian Noise
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
        noisy_receptive_field_response = ReceptiveFieldResponse(row=org_single_receptive_field_data.row
                                                          , col=org_single_receptive_field_data.col
                                                          , timecourse=noisy_timecourse )                                                
        return noisy_receptive_field_response
             
    def generate_synthetic_data(self):      
            index = 0  
            for row in range (self.source_data.shape[0]):
                for col in range (self.source_data.shape[1]):
                    for times in range(self.synthesis_ratio):
                        receptive_field_data = self.source_data[row][col]
                        noisy_data = self._create_noisy_receptive_field_data(receptive_field_data)
                        self.data[index] = noisy_data
                        index = index + 1

            return self.data    


##############################---Sliders---#######################
class NoiseLevelAdjustmentApp:
    def __init__(self
                 , stimulus : Stimulus
                 , grid : ModelsGrid):
        self.stimulus = stimulus
        self.grid = grid
        self.desired_low_freq_noise_level = 0.4
        self.desired_system_noise_level = 0.3
        self.desired_physiological_noise_level = 0.1
        self.desired_task_noise_level = 0.2
        self.desired_temporal_noise_level = 0.3            
        self.nCols_grid = grid.grid_nCols
        self.nRows_grid = grid.grid_nRows
        self.synthesis_ratio = 2
        self.grid_fitting_results = np.zeros((self.nRows_grid, self.nCols_grid), dtype='float')    

        # Figure
        self.fig, ((self.ax1, self.ax2)) = plt.subplots(1, 2, gridspec_kw={'bottom': 0.2})
        self.fig.text(0.5, 0.96, 'Noise Modelling', ha='center', fontsize=12) # title        
        self.status_label = self.fig.text(0.5, 0.01, 'Status: Ready', ha='center', fontsize=10) # bottom status label

        # Create sliders and "Recompute" button in the main figure
        self.low_freq_noise_slider_ax = self.fig.add_axes([0.15, 0.01, 0.25, 0.02])
        self.low_freq_noise_slider = Slider(self.low_freq_noise_slider_ax, 'Low Freq Noise', 0.1, 10, valinit=self.desired_low_freq_noise_level)

        self.system_noise_slider_ax = self.fig.add_axes([0.15, 0.04, 0.25, 0.02])
        self.system_noise_slider = Slider(self.system_noise_slider_ax, 'System Noise', 0.1, 10, valinit=self.desired_system_noise_level)

        self.physiological_noise_slider_ax = self.fig.add_axes([0.15, 0.07, 0.25, 0.02])
        self.physiological_noise_slider = Slider(self.physiological_noise_slider_ax, 'Physiological Noise', 0.1, 10, valinit=self.desired_physiological_noise_level)

        self.task_noise_slider_ax = self.fig.add_axes([0.15, 0.10, 0.25, 0.02])
        self.task_noise_slider = Slider(self.task_noise_slider_ax, 'Task Noise', 0.1, 10, valinit=self.desired_task_noise_level)

        self.temporal_noise_slider_ax = self.fig.add_axes([0.15, 0.13, 0.25, 0.02])
        self.temporal_noise_slider = Slider(self.temporal_noise_slider_ax, 'Temporal Noise', 0.1, 10, valinit=self.desired_temporal_noise_level)

        recompute_button_ax = self.fig.add_axes([0.8, 0.01, 0.1, 0.04])
        recompute_button = Button(recompute_button_ax, 'Recompute')
        recompute_button.on_clicked(self.recompute_button_clicked)

        # Create a figure for the main plot with 2x2 subplots
        grid_fitting_initial_results, org_timecourse_sample, noisy_timecourse = self.compute_probability_density() # intial computation of grid fitting
        self.im = self.ax1.imshow(grid_fitting_initial_results, cmap='hot', origin='lower', extent=[(-stimulus.size_in_degrees), (stimulus.size_in_degrees), (-stimulus.size_in_degrees), (stimulus.size_in_degrees)], vmin=0.0, vmax=1) # heatmap
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

        # define desired noise levels
        noise_levels = NoiseLevels(self.desired_low_freq_noise_level
                                    , self.desired_system_noise_level
                                    , self.desired_physiological_noise_level
                                    , self.desired_task_noise_level
                                    , self.desired_temporal_noise_level)

        # generate synthetic data
        data_synthesizer = SynthesizedDataGenerator(noise_levels=noise_levels, source_data=self.grid.data, synthesis_ratio=self.synthesis_ratio, TR=1)
        noisy_synthetic_data = data_synthesizer.generate_synthetic_data()

        # Data matching
        # grid_timecourses = [item.timecourse for row in self.grid.data for item in row]
        # synthesized_data_timecourses = [item.timecourse for item in data_synthesizer.data]
        grid_timecourses = []
        synthesized_data_timecourses = []
        index = 0
        for row in range(self.nRows_grid):
            for col in range(self.nCols_grid):
                grid_timecourses.append(self.grid.data[row][col].timecourse)
                for times in range(self.synthesis_ratio):
                    if(self.grid.data[row][col].row != data_synthesizer.data[index].row or self.grid.data[row][col].col != data_synthesizer.data[index].col):
                        print('Something wrong in the indexing!!!')
                    synthesized_data_timecourses.append(data_synthesizer.data[index].timecourse)
                    index = index + 1

        ols = OLS(modelled_signals=grid_timecourses
                  , measured_signals=synthesized_data_timecourses
                  )
        best_fit_proj = ols.compute_proj_squared()

        # Analyze fitting results
        grid_fitting_results = np.zeros((self.nRows_grid, self.nCols_grid))#[[0 for _ in range(self.nRows_grid)] for _ in range(self.nCols_grid)]
        for i in range(len(best_fit_proj)):
            target_idx = i
            target_synthesizer_location = (noisy_synthetic_data[target_idx].row, noisy_synthetic_data[target_idx].col)
            flattened_signal_idx = best_fit_proj[i]
            signal_idx_row = math.floor(flattened_signal_idx / self.nCols_grid)
            signal_idx_col = flattened_signal_idx - (signal_idx_row * self.nCols_grid)
            matched_signal_location = (self.grid.data[signal_idx_row][signal_idx_col].row, self.grid.data[signal_idx_row][signal_idx_col].col)
            if target_synthesizer_location == matched_signal_location:
                grid_fitting_results[target_synthesizer_location[0]][target_synthesizer_location[1]] += 1

        # Normalize results
        grid_fitting_results = np.array(grid_fitting_results) / self.synthesis_ratio
        self.update_status("Data computed !!!")

        return grid_fitting_results, self.grid.data[0][0].timecourse, noisy_synthetic_data[0].timecourse

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

        # # Annotate each box with its value
        # for annotation in self.annotations:
        #     annotation.remove()
        # self.annotations = [] 
        # nRows_grid, nCols_grid = fitting_results.shape
        # for i in range(nRows_grid):
        #     for j in range(nCols_grid):
        #         # annotation = self.ax1.text(j + 0.5, i + 0.5, f"{fitting_results[i][j]:.2f}", color='black', ha='center', va='center')
        #         annotation = self.ax1.text(j + 0.5, i + 0.5, f"({i},{j})", color='black', ha='center', va='center')
        #         self.annotations.append(annotation)

        # Customize grid lines at integer values
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
    # Load stimulus
    #stimulus = Stimulus("../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.compute_resample_stimulus_data((101, 101, stimulus.org_data.shape[2]))

    # HRF Curve
    hrf_t = np.linspace(0, 30, 31)
    hrf_curve = spm_hrf_compat(hrf_t)
    
    # Expected responses Grid    
    nRows_grid = 11
    nCols_grid = 11
    grid = ModelsGrid(grid_nRows=nRows_grid
                    , grid_nCols=nCols_grid
                    , pRF_size_sigma=2
                    , stimulus=stimulus
                    , hrf_curve=hrf_curve)
    grid.generate_model_responses()

    app = NoiseLevelAdjustmentApp(stimulus, grid)

