
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys
from scipy.ndimage import zoom
import os
from pathlib import Path
from typing import List

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
        self.data = []        
            
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
                self.data.append(expected_receptive_field_reponse)

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
        self.data = []

    def _create_noisy_receptive_field_data(self, org_single_receptive_field_data : ReceptiveFieldResponse):
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
        noisy_timecourse = noise(org_single_receptive_field_data.timecourse) 
        noisy_receptive_field_response = ReceptiveFieldResponse(x=org_single_receptive_field_data.x
                                                          , y=org_single_receptive_field_data.y
                                                          , timecourse=noisy_timecourse )                                                
        return noisy_receptive_field_response
    
    def generate_synthetic_data(self):
        for i in range(len(self.source_data)):
            given_receptive_field_data = self.source_data[i] # type: ReceptiveFieldResponse
            noisy_receptive_field_data = self._create_noisy_receptive_field_data(org_single_receptive_field_data=given_receptive_field_data)
            self.data.append(noisy_receptive_field_data)
        return self.data

# Ordinary Least Squares (OLS)
class OLS:
    def __init__(self
                 , modelled_responses_data : List[ReceptiveFieldResponse]
                 , simulated_or_measured_data : List[ReceptiveFieldResponse]
                 , type = 'ordinary least squares - simplified'):
        self.modelled_data = modelled_responses_data
        self.data_to_be_matched = simulated_or_measured_data
        self.N = modelled_responses_data[0].timecourse.shape[0] # number of data points in each timecourse
        self.orthonormal_modelled_signals = []
    
    def make_Trends(self, nDCT=3):
        # Generates trends similar to 'rmMakeTrends' from the 'params' struct in mrVista.    
        # nDCT: Number of discrete cosine transform components.
        
        tf = self.N # this is equal to 300
        ndct = 2 * nDCT + 1
        trends = np.zeros((np.sum(tf), np.max([np.sum(ndct), 1])))        
        
        tc = np.linspace(0, 2.*np.pi, tf)[:, None]        
        trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))

        nTrends = trends.shape[1]
        return trends, nTrends       
    
    def perform_qr_decomposition(self, trends):
        q, r = np.linalg.qr(trends)
        q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
        R = q
        return R
    
    def compute_orthonormal_model_signals(self, R):
        # Calculate orthogonal projection using O = (I - R @ R^T)
        O = (np.eye(self.N) - R @ R.T) # R: Orthonormal Regressors
        
        for i in range(len(self.modelled_data)):
            s = self.modelled_data[i].timecourse #s
            s = O @ s #s_prime 
            s /= np.sqrt(s @ s)
            orthonormalized_model_signal = ReceptiveFieldResponse(x = self.modelled_data[i].x
                                                                  , y=self.modelled_data[i].y
                                                                  , timecourse= s)
            self.orthonormal_modelled_signals.append(orthonormalized_model_signal)
    
    # Calculate squared projection of orthogonal grid onto targets
    def compute_proj_squared(self):        
        grid_ortho = np.vstack([response.timecourse for response in self.orthonormal_modelled_signals]).T
        targets  = np.vstack([response.timecourse for response in self.data_to_be_matched]).T
        proj_squared = (grid_ortho.T @ targets)**2 #I guess, (y^T . s_prime)^2
        best_fit_proj = np.argmax(proj_squared, axis=0)
        return best_fit_proj

  


##############################-----------PROGRAM-----------------------#################################
def run_grid_fit_simulation_program():
    print('Program started...')

    # Load stimulus
    stimulus = Stimulus("../../local-extracted-datasets/sid-prf-fmri-data/task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.data = stimulus.resample_stimulus_data((101, 101, stimulus.data.shape[2]))

    # HRF Curve
    hrf_t = np.linspace(0,30,31)
    hrf_curve = spm_hrf_compat(hrf_t) 

    # Expected responses Grid
    grid = ModelsGrid(grid_size_x = 4
                 , grid_size_y = 4
                 , sigma = 2
                 , stimulus= stimulus
                 , hrf_curve = hrf_curve)
    grid.generate_model_responses()

    # define desired noise levels
    noise_levels = NoiseLevels(desired_low_freq_noise_level=1
                               , desired_system_noise_level=2
                               , desired_physiological_noise_level=3
                               , desired_task_noise_level=4
                               , desired_temporal_noise_level=5)
    
    # generate synthetic data
    data_synthesizer = SynthesizedDataGenerator(noise_levels=noise_levels, source_data = grid.data, synthesis_ratio=1, TR=1)
    noisy_synthetic_data = data_synthesizer.generate_synthetic_data()
    
    # Data matching
    ols = OLS(modelled_responses_data=grid.data
                               , simulated_or_measured_data= data_synthesizer.data)
    trends, _ = ols.make_Trends()
    R = ols.perform_qr_decomposition(trends=trends)        
    ols.compute_orthonormal_model_signals(R=R)
    best_fit_proj = ols.compute_proj_squared()

    print('Program finished !') # plt.plot(((synthetic_data.data)[0]).timecourse)
    

##############################
if __name__ == "__main__":
    run_grid_fit_simulation_program()