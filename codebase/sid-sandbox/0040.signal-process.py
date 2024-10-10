import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


# IDEAS:
# it would be okay to remove some y-signals because we are ultimately interested in the coverage maps (representing the activated pRFs)
# the neighboring y-signals should also have a correlation as were activated nearly at the same time and since HRF is Linear Time Invariant (LTI), the neighboring y-signals could contain some components of the adjacent signals


def get_ortho_matrix(signal):
    signal_length = len(signal)
    nDCT = 3
    ndct = 2 * 3 + 1
    trends = np.zeros((signal_length, np.max([np.sum(ndct), 1])))        
    tc = np.linspace(0, 2.*np.pi, signal_length)[:, None]        
    trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))
    q, _ = np.linalg.qr(trends) # QR decomposition
    q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
    R = q
    O = (np.eye(signal_length)  - np.dot(R, R.T))

    return O, R


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


##################################---------------Main()-------------------------------------------------------------#####################################
# get y-signals ------->>>>>>> np.nanargmax(r2_results)342 ------>>>>> r2_results[342] = 0.6772895258602416
bold_response_img = nib.load("D:/code/sid-git/fmri/local-extracted-datasets/sid-prf-fmri-data/sub-sidtest_ses-001_task-bar_run-01_hemi-R_bold.nii.gz")
Y_signals_cpu = bold_response_img.get_fdata()

# reshape the BOLD response data to 2D
Y_signals_cpu = Y_signals_cpu.reshape(-1, Y_signals_cpu.shape[-1])

# just to make them column vectors
Y_signals_cpu = Y_signals_cpu.T

best_y = Y_signals_cpu[:, 342]

# de-trend the signal
ortho_matrix, R = get_ortho_matrix(best_y)
orthogonalized_best_y = best_y @ ortho_matrix

# signal smoothing test - looks nice
window_size = 10
smoothed_signal = np.convolve(best_y, np.ones(window_size)/window_size, mode='valid')

# Bandwidth filter
low_cutoff = 0.5  # in Hz
high_cutoff = 1.4  # in Hz
# ....sample rate of the signal (replace with your actual sample rate)
sampling_rate = 300 # 1.0  # for simplicity, assuming one sample per unit time

# Apply the bandpass filter
# filtered_signal = bandpass_filter(best_y, low_cutoff, high_cutoff, sampling_rate)

#### test frequency filtering  (BEST Found values: low_cutoff = 2, high_cutoff = 15)
n = len(best_y)
fft_result = fft(best_y)
frequencies = np.fft.fftfreq(n, d=1/sampling_rate)

# Create a binary mask for the frequencies within the desired range
mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)

# Apply the mask to the FFT result
filtered_fft = fft_result * mask

# Inverse FFT to get back to the time domain
filtered_signal = ifft(filtered_fft)
plt.plot(filtered_signal)




print("Done...")

