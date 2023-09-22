
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
from matplotlib.gridspec import GridSpec

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

##############################---Sliders---#######################
class NoiseLevelAdjustmentApp:
    def __init__(self
                 , stimulus : Stimulus
                 , grid : QuadrilateralSignalsSpace):
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

def RunProgram():
    # HRF Curve
    hrf_t = np.linspace(0, 30, 31)
    hrf_curve = spm_hrf_compat(hrf_t)

    # Load stimulus
    #stimulus = Stimulus("../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
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

    # histograms._compute_histograms(search_space = search_space
    #                               , noisy_test_data=test_space.data
    #                               , best_fit_info_wrt_search_space=best_fit_info
    #                               , synthesis_ratio=synthesis_ratio)
    # histograms.plot_histograms()
    # plt.plot()
    
                          
    print("done!")

if __name__ == "__main__":
    RunProgram()

