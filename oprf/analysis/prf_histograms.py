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


# HRF Generator
from oprf.external.hrf_generator_script import spm_hrf_compat

# Receptive Field Response Class
from oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse

# QuadrilateralSignalsSpace Class
from oprf.standard.prf_quadrilateral_signals_space import QuadrilateralSignalsSpace

# DeepRF module
from oprf.external.DeepRF import data_synthetic as deeprf_data_synthetic

# Stimlus
from oprf.standard.prf_stimulus import Stimulus

class LocationHistogram:
    def __init__(self, location_row, location_col):
        self.location_row = location_row
        self.location_col = location_col
        self.HistData = None

    def set_histogram(self, histogram):
        self.histogram = histogram    

class Histograms:
    def __init__(self
                 , search_space
                 , noisy_test_data : List[ReceptiveFieldResponse]
                 , nRows_search_space
                 , nCols_search_space
                 , best_fit_info_wrt_search_space
                 , synthesis_ratio):   
        # member variables  
        self.search_space = search_space      
        self.nRows_search_space = nRows_search_space
        self.nCols_search_space = nCols_search_space
        self.histograms = []

        # compute histograms
        self._compute_histograms(search_space
                        , noisy_test_data
                        , best_fit_info_wrt_search_space
                        , synthesis_ratio)
                
        # Plotting related
        self.fig = plt.figure()
        gs = GridSpec(1, 2, width_ratios=[1, 1])  # Create a grid with 1 row and 2 columns

        # Create linspace for x and y legends
        self.x_legends = np.linspace(-9, 9, self.nCols_search_space)
        self.y_legends = np.linspace(-9, 9, self.nRows_search_space)

        # Create the 3D subplot on the left
        self.ax1d = self.fig.add_subplot(gs[0], projection='3d')
        self.ax1d.set_title(f"Histogram for Location (0, 0)")

        # Create the 2D subplot on the right
        self.ax2d = self.fig.add_subplot(gs[1])
        self.ax2d.set_title("Click on this 2D subplot")

        # Initialize the initial plot (0, 0) for the 3D subplot
        dummy_hist_data = np.random.rand(nRows_search_space, nCols_search_space)
        self.surface = self.ax1d.plot_surface(self.search_space.custom_space_meshgrid_Y, self.search_space.custom_space_meshgrid_X, dummy_hist_data, cmap='viridis')

        # Create a white 2D array for the 10x10 square with grid lines
        self.square_data = np.ones((self.nRows_search_space, self.nCols_search_space), dtype=int) * 255  # White color (255 in grayscale)

        # Plot the 2D square in the 2D subplot with grid lines
        self.ax2d.imshow(self.square_data, cmap='gray', extent=[0, self.nCols_search_space, 0, self.nRows_search_space], vmin=0, vmax=255, origin='lower', interpolation='none')

        # Set the x and y axis ticks and labels using legends
        self.ax2d.set_xticks(np.arange(self.nCols_search_space), minor=True)
        self.ax2d.set_yticks(np.arange(self.nRows_search_space), minor=True)
        self.ax2d.set_xticklabels([f"{int(legend)}" for legend in self.x_legends])
        self.ax2d.set_yticklabels([f"{int(legend)}" for legend in self.y_legends])
        self.ax2d.set_xlabel("X")
        self.ax2d.set_ylabel("Y")

        self.ax2d.grid(True, color='black', linestyle='--', linewidth=1)  # Display grid lines in black

        # Add hair-cross along the middle of x and y-axis
        self.ax2d.axhline(self.nRows_search_space // 2 + 0.5, color='red', linestyle='--', linewidth=1)  # Horizontal line
        self.ax2d.axvline(self.nCols_search_space // 2 + 0.5, color='red', linestyle='--', linewidth=1)  # Vertical line

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def on_click(self, event):
        if event.inaxes == self.ax2d:  # Check if the click event is on the 2D subplot
            x_legend = event.xdata
            y_legend = event.ydata
            print(f"Clicked on 2D subplot at (legend_x, legend_y) = ({x_legend}, {y_legend})")

            # Convert legend coordinates to row and column information
            col = int(x_legend)
            row = int(y_legend)
            print(f"Clicked on 2D subplot at (legend_x, legend_y) = ({row}, {col})")
            index = row * self.nCols_search_space + col
            if 0 <= index < len(self.histograms):
                # Update the plot data for the 3D subplot
                hist_data = self.histograms[index].HistData
                if hasattr(self, 'surface') and self.surface is not None:
                    self.surface.remove()  # Remove the previous surface plot if it exists
                self.surface = self.ax1d.plot_surface(self.search_space.custom_space_meshgrid_Y, self.search_space.custom_space_meshgrid_X, hist_data, cmap='viridis')
                self.ax1d.set_title(f"Histogram for Location ({col}, {row})")
                self.fig.canvas.draw()  # Redraw the figure

    def on_test(self):
        x_legend = 0
        y_legend = 0
        print(f"Clicked on 2D subplot at (legend_x, legend_y) = ({x_legend}, {y_legend})")

        # Convert legend coordinates to row and column information
        col = int(x_legend)
        row = int(y_legend)
        print(f"Clicked on 2D subplot at (legend_x, legend_y) = ({row}, {col})")
        index = row * self.nCols_search_space + col
        if 0 <= index < len(self.histograms):
            # Update the plot data for the 3D subplot
            hist_data = self.histograms[index].HistData
            if hasattr(self, 'surface') and self.surface is not None:
                self.surface.remove()  # Remove the previous surface plot if it exists
            self.surface = self.ax1d.plot_surface(self.search_space.custom_space_meshgrid_Y, self.search_space.custom_space_meshgrid_X, hist_data, cmap='viridis')
            self.ax1d.set_title(f"Histogram for Location ({col}, {row})")
            self.fig.canvas.draw()  # Redraw the figure                


    def _compute_histograms(self
                           , search_space : QuadrilateralSignalsSpace
                        , noisy_test_data : List[ReceptiveFieldResponse]
                        , best_fit_info_wrt_search_space
                        , synthesis_ratio):
        
        for row in range(self.nRows_search_space):
            for col in range(self.nCols_search_space):
                histogram = LocationHistogram(row, col)
                flat_idx = synthesis_ratio*(row*self.nCols_search_space + col)
                relevant_elements_for_histogram = best_fit_info_wrt_search_space[flat_idx : flat_idx + synthesis_ratio]
                row_coordinates = []
                col_coordinates = []
                for i in range(len(relevant_elements_for_histogram)):
                    best_match_idx_wrt_search_space = relevant_elements_for_histogram[i]
                    row_idx = int(best_match_idx_wrt_search_space/self.nRows_search_space) 
                    col_idx = best_match_idx_wrt_search_space - (row_idx * self.nCols_search_space) 
                    row_coordinates.append(search_space.custom_space_meshgrid_Y[row_idx][col_idx]) # NOTE: note that the row_coordinates is using Meshgrid_Y
                    col_coordinates.append(search_space.custom_space_meshgrid_X[row_idx][col_idx])
                    
                hist2d, _, _ = np.histogram2d(row_coordinates, col_coordinates, bins=(self.nRows_search_space, self.nCols_search_space))                
                row_edges = search_space.custom_space_row_points
                col_edges = search_space.custom_space_col_points
                # plt.imshow(hist2d, cmap='viridis', origin='lower', extent=[row_edges[0], row_edges[-1], col_edges[0], col_edges[-1]])
                # plt.figure().add_subplot(111, projection='3d').plot_surface(search_space.custom_space_meshgrid_Y, search_space.custom_space_meshgrid_X, hist2d, cmap='viridis')
                histogram.HistData = hist2d #[hist2d, row_edges, col_edges]
                self.histograms.append(histogram)
                print("Test")

        print("computed")                  

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
                                           , org_single_receptive_field_data : ReceptiveFieldResponse
                                           , noise : deeprf_data_synthetic.Noise):  
        # compute noisy timecourse
        noisy_timecourse = noise(org_single_receptive_field_data.timecourse) 
        noisy_receptive_field_response = ReceptiveFieldResponse(row=org_single_receptive_field_data.row
                                                          , col=org_single_receptive_field_data.col
                                                          , timecourse=noisy_timecourse )                                                
        return noisy_receptive_field_response
    
    def _create_simple_noisy_receptive_field_data(self
                                           , org_single_receptive_field_data : ReceptiveFieldResponse):
        org_timecourse = org_single_receptive_field_data.timecourse

        noisy_timecourse = (np.random.randn(1, len(org_timecourse)) * 1 + org_timecourse[None,:])[0]        
        # noisy_timecourse = (np.random.randn(1, len(org_timecourse)) * self.noise_levels.desired_temporal_noise_level + org_timecourse[None,:])[0]        
        noisy_receptive_field_response = ReceptiveFieldResponse(row=org_single_receptive_field_data.row
                                                          , col=org_single_receptive_field_data.col
                                                          , timecourse=noisy_timecourse )                                                
        return noisy_receptive_field_response    
    
    def _compute_cnr_values(self, timecourse):
        org_timecourse_std = np.std(timecourse)

        # Compute CNRs
        # CNR = std_signal / sigma_noise, where "sigma_noise = desired_noise_level * std_signal"        
        cnr_low_freq = org_timecourse_std / (self.noise_levels.desired_low_freq_noise_level * org_timecourse_std)        
        cnr_physiological = org_timecourse_std / (self.noise_levels.desired_physiological_noise_level * org_timecourse_std)
        cnr_system = org_timecourse_std / (self.noise_levels.desired_system_noise_level * org_timecourse_std)
        cnr_task = org_timecourse_std / (self.noise_levels.desired_task_noise_level * org_timecourse_std)
        cnr_temporal = org_timecourse_std / (self.noise_levels.desired_temporal_noise_level * org_timecourse_std)

        return cnr_low_freq, cnr_physiological, cnr_system, cnr_task, cnr_temporal
        

    def generate_synthetic_data_With_noise_models(self):    
        # Compute CNRs
        # CNR = std_signal / sigma_noise, where "sigma_noise = desired_noise_level * std_signal"        
        cnr_low_freq = 0.01
        cnr_physiological = 0.01
        cnr_system = 0.01
        cnr_task = 0.01
        cnr_temporal = 0.01
        
        # Initialize noises
        low_frequency_noise = deeprf_data_synthetic.LowFrequency(cnr_low_freq, self.TR )
        physiological_noise = deeprf_data_synthetic.Physiological(cnr_physiological, self.TR )
        system_noise = deeprf_data_synthetic.System(cnr_system, np.random.RandomState())
        task_noise = deeprf_data_synthetic.Task(cnr_task, np.random.RandomState())
        temporal_noise = deeprf_data_synthetic.Temporal(cnr_temporal, np.random.RandomState())
        random_generator_y = np.random.RandomState(1258566) # used to generate predictions        
        
        index = 0  
        debug_timecourse = None
        for row in range (self.source_data.shape[0]):
            for col in range (self.source_data.shape[1]):
                receptive_field_data = self.source_data[row][col]
                cnr_low_freq, cnr_physiological, cnr_system, cnr_task, cnr_temporal = self._compute_cnr_values(receptive_field_data.timecourse)
                low_frequency_noise.CNR = cnr_low_freq
                physiological_noise.CNR = cnr_low_freq
                system_noise.CNR = cnr_low_freq
                task_noise.CNR = cnr_low_freq
                temporal_noise.CNR = cnr_low_freq
                noise = deeprf_data_synthetic.Noise(random_generator_y.rand(5)
                                        , low_frequency_noise
                                        , physiological_noise
                                        , system_noise
                                        , task_noise
                                        , temporal_noise)
                for times in range(self.synthesis_ratio):                                                                                
                    noisy_data = self._create_noisy_receptive_field_data(receptive_field_data, noise=noise)
                    self.data[index] = noisy_data
                    debug_timecourse = noisy_data
                    index = index + 1

        return self.data   
             
    def generate_synthetic_data(self):      
            index = 0  
            debug_timecourse = None
            for row in range (self.source_data.shape[0]):
                for col in range (self.source_data.shape[1]):
                    for times in range(self.synthesis_ratio):
                        receptive_field_data = self.source_data[row][col]
                        noisy_data = self._create_simple_noisy_receptive_field_data(receptive_field_data)
                        self.data[index] = noisy_data
                        debug_timecourse = noisy_data
                        index = index + 1

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
    stimulus = Stimulus("../../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.compute_resample_stimulus_data((101, 101, stimulus.org_data.shape[2]))
    stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)

    # Search space
    nRows_search_space = 51
    nCols_search_space = 51
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

