import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

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
# larger_array = np.array([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0])
# smaller_array = np.array([0,1,0])
# plt.plot(larger_array)
# plt.plot(smaller_array)
 
# result_length = larger_array.shape[0] - smaller_array.shape[0] + 1
# convolution_result = np.zeros(result_length)

# for i in range(result_length):
#     convolution_result[i] = np.sum(larger_array[i:i + smaller_array.shape[0]] * smaller_array)

 

# np_convolution_result = np.convolve(larger_array, smaller_array, mode='same')
# plt.plot(np_convolution_result)   
# plt.show()

# print("done!")

t = np.linspace(0, 300, 901)

h = spm_hrf_compat(t)

f, ax = plt.subplots()
ax.plot(t, h)

y = np.zeros_like(h)
i = np.argmin(np.abs(t - 50))
y[i] = 1

#ax.plot(t, y)

yh = np.convolve(y, h, mode='full')[:len(t)]
ax.plot(t, yh)
plt.show()