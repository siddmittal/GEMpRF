#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:35:16 2023

@author: mwoletz
"""

import numpy as np
import scipy as scp
import nibabel as nib
from skimage.transform import rescale
import scipy.stats as sps
import matplotlib.pyplot as plt


# SPMs HRF
def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio = 6,
                   normalize=True,
                  ):
    """ SPM HRF function from sum of two gamma PDFs

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF.
    peak_delay : float, optional
        delay of peak.
    under_delay : float, optional
        delay of undershoot.
    peak_disp : float, optional
        width (dispersion) of peak.
    under_disp : float, optional
        width (dispersion) of undershoot.
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.  SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`.

    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float64)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale = peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale = under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.sum(hrf)


#stim = scp.io.loadmat('/z/fmri/data/stimsim22/BIDS/sourcedata/stimuli/eightbar_standard_tr1.5_images.mat', squeeze_me=True)
# stim = scp.io.loadmat("Y:\data\stimsim22\BIDS\sourcedata\stimuli\eightbar_standard_tr1.5_images.mat", squeeze_me=True)
stim = scp.io.loadmat("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\michi\eightbar_standard_tr1.5_images.mat", squeeze_me=True)

TR = float(stim['params']['tr'])

V = stim['stimulus']['images'].flatten()[0]

t = np.arange(V.shape[-1]) * TR

B = (V != 128).astype(float)

f = 101. / float(B.shape[0])

M = rescale(B, [f, f, 1], anti_aliasing=True) # scp.ndimage.zoom(B, [0.1, 0.1, 1])

stim_extend = 9.

# IMPORTANT
# IMPORTANT
# IMPORTANT
# IMPORTANT
X,Y = np.meshgrid(np.linspace(-stim_extend,stim_extend, M.shape[0]), np.linspace(-stim_extend,stim_extend, M.shape[1]), indexing='ij')

hrf = spm_hrf_compat(t[:int(32/TR)])

mask = M.sum(2) > 0
Xm = X[mask] # the shape of the resulting array Xm will depend on how many True values there are in the mask array
Ym = Y[mask]
Mm = M[mask]

def signal(x):
    pRF = np.exp(- ((Xm-x[0])**2 + (Ym-x[1])**2) / (2 * x[2]**2)) #2d Gaussian
    y = pRF @ Mm
    yh = np.convolve(y, hrf)[:y.shape[0]]
    yh /= yh.max()
    
    return yh

# Test function
def testRun():
    theta_j = (0, 0, 2)
    # theta_j = (stim_extend, 0, 2)
    y = signal(theta_j)

    x_fit, y_fit, sigma_fit = np.meshgrid(np.linspace(-stim_extend/2, stim_extend/2, 51), [0], [2])
    # x_fit, y_fit, sigma_fit = np.meshgrid(np.linspace(-stim_extend*2, stim_extend*2, 201), [0], [2])
    # x_fit, y_fit, sigma_fit = np.meshgrid(np.linspace(-stim_extend*1, stim_extend*6, 201), [0], [2])
    x_fit = x_fit.flatten()
    y_fit = y_fit.flatten()
    sigma_fit = sigma_fit.flatten()

    # since y_fit and sigma_fit are only single elements, so we are going to generate as many timecourse as the number of points in the x_fit, which is 51
    # therefore the shape of fit_timecourses will be (51, 448) i.e. 51 timecourses of length 448 each
    fit_timecourses = np.array([signal((x,y,sigma)) for x, y, sigma in zip(x_fit, y_fit, sigma_fit)])

    # subtracts the mean of each row (time course) from itself
    M = fit_timecourses - fit_timecourses.mean(1)[:,None] # (fit_timecourses.mean(1)).shape = 51
    M /= np.sqrt(np.sum(M**2, 1))[:,None] # scales each row (time course) of M by its Euclidean norm (L2 norm) to normalize the rows to unit length

    R = 1000
    noise_level = 1.25

    Y = np.random.randn(R, len(y)) * noise_level + y[None,:] #<---------------------IMPORTANT: Creating 1000 noisy signals from a single given model_signal

    rho = (M @ Y.T)**2 # rho.shape: (51, 1000)

    rho_max = np.argmax(rho, 0) # rho_max.shape: (1000, )

    counts = np.bincount(rho_max, minlength=rho.shape[0]) # numpy.bincount(arr, weights = None, min_len = 0)

    fig, ax = plt.subplots()
    ax.stem(x_fit, counts / R * 100) # creates a stem plot on the subplot,  normalize the counts by the total number of samples, and multiplying by 100 to express the result as a percentage.
    ax.set_title(noise_level)

    mask = x_fit > -0.7
    M2 = M[mask]
    rho2 = (M2 @ Y.T)**2

    rho2_max = np.argmax(rho2, 0)

    counts = np.bincount(rho2_max, minlength=rho[mask].shape[0])

    ax.stem(x_fit[mask], counts / R * 100, markerfmt = 'ro')


if __name__ == "__main__":
    testRun()

