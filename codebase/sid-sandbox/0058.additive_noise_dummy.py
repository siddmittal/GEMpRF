import numpy as np

fMRI_model_signals = [] # dummy

# Adjust the desired noise level
noise_std = 0.1  

# Generate white noise samples
num_samples = len(fMRI_model_signals[0]) # length of a single signal
white_noise = np.random.normal(0, noise_std, size=num_samples)

# Add white noise to each fMRI model signal
observed_signals = []
for model_signal in fMRI_model_signals:
    observed_signal = model_signal + white_noise
    observed_signals.append(observed_signal)
