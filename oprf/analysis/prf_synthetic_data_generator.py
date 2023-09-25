import numpy as np
from typing import List

# Local Imports
from oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse
from oprf.external.DeepRF import data_synthetic as deeprf_data_synthetic # DeepRF module

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