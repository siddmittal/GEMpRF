import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import math

# def get_positive_sine_curve(num_points=300):
#     x = np.linspace(0, 2 * np.pi, num_points)
#     frequency = 10
#     amplitude = 1.0
#     y = amplitude * np.sin(frequency * x)
#     y_positive = np.maximum(y, 0)
#     return x, y_positive


def get_positive_sine_curve(total_duration = 300):
    frequency = 0.02 #Two HZ
    delta_t = 1/(5*2*np.pi*frequency)
    num_points = math.floor(total_duration/delta_t)
    t = np.linspace(0, total_duration, num_points)
    sin_t = np.sin(t)
    #sin_t = np.maximum(sin_t, 0) # keeping only positive part

    return t, sin_t

def spm_hrf_compat(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1, p_u_ratio=6, normalize=True):
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    hrf = np.zeros(t.shape, dtype=np.float64)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.sum(hrf)

##----------PROGRAM-------------##
# Get the sine curve
total_duration = 100
t, sine_curve_amplitudes = get_positive_sine_curve(total_duration)

# Get HRF Curve
hrf_t = np.linspace(0, 30, 31)  # Use the same number of points as sine_curve_amplitudes
hrf_curve = spm_hrf_compat(hrf_t)

# Perform convolution
convolution_results = np.convolve(sine_curve_amplitudes[:300], hrf_curve, mode='same')

# Plots
plt.plot(t, sine_curve_amplitudes, label='Sine Curve')
plt.show()
plt.plot(hrf_curve, label='HRF Curve')
plt.show()
#plt.plot(hrf_t[:-len(hrf_curve) + 1], convolution_results, label='Convolution Result')
plt.plot(convolution_results)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Convolution of Sine Curve with HRF')
plt.legend()
plt.grid(True)
plt.show()
