import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# DeepRF module
deeprf_module_path = (Path(__file__).resolve().parent / '../external-packages/DeepRF-main').resolve()
sys.path.append(str(deeprf_module_path))
from data_synthetic import *
import data_synthetic as deeprf_data_synthetic

# testing DeepRF
TR = 1.2 # seconds
random_seed = 12345  # Choose an appropriate random seed
random_state = np.random.RandomState(random_seed)
low_frequency_noise = deeprf_data_synthetic.LowFrequency(0.9, TR )
physiological_noise = deeprf_data_synthetic.Physiological(0.9, TR)
system_noise = deeprf_data_synthetic.System(0.9, random_state)
task_noise = deeprf_data_synthetic.Task(0.9, random_state)
temporal_noise = deeprf_data_synthetic.Temporal(0.9, random_state)

# Create a Noise instance with specified noise components and amplitudes
random_generator_t = np.random.RandomState(64556) # used to generate random parameters
random_generator_x = np.random.RandomState(23355) # used to generate random signal/noise
random_generator_y = np.random.RandomState(1258566) # used to generate predictions
noise = deeprf_data_synthetic.Noise(random_generator_y.rand(5), low_frequency_noise, physiological_noise, system_noise, task_noise, temporal_noise)

# voxel data
bias = 800.0
delay = 6
percentsignalchange = random_generator_y.normal(3.0, 0.25)
doublegamma = deeprf_data_synthetic.DoubleGamma(TR, delay)
gaussian = deeprf_data_synthetic.Gaussian(1024, 1024, 
            9, 9, 
            0, 0)
signal = deeprf_data_synthetic.Signal(percentsignalchange, bias, doublegamma, gaussian)
voxel = deeprf_data_synthetic.Voxel(noise, signal)


# Create a synthetic fMRI time series (replace this with your actual data)
fmri_data = np.random.normal(0, 1, size=300)

# Apply the noise to the fMRI data using the __call__ method
noisy_fmri_data = noise(fmri_data)

print("done")