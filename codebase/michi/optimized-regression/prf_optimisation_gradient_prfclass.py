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
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import sys
sys.path.append('/z/fmrilab/home/dlinhardt/pythonclass')
from PRFclass import PRF

study = 'stimsim22'
sub = '001' # 001..003
ses = '001' # 001..005
task = 'barStandard' # 'barRot', 'barDouble'
run = '01' # 01..02

a = PRF.from_docker(study, sub, ses, task, run, analysis='01')
a.maskROI('V1')
a.loadStim(buildTC=False)

# a.voxelTC TC of all voxels in mask
# a.stimImages stimulus convolved, 203x203, mask is a.window, pos are a.X0, a.Y0

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


stim = scp.io.loadmat('/z/fmri/data/stimsim22/BIDS/sourcedata/stimuli/eightbar_standard_tr1.5_images.mat', squeeze_me=True)

TR = float(stim['params']['tr'])

V = stim['stimulus']['images'].flatten()[0]

t = np.arange(V.shape[-1]) * TR

B = (V != 128).astype(float)

f = 101. / float(B.shape[0])

M = rescale(B, [f, f, 1], anti_aliasing=True) # scp.ndimage.zoom(B, [0.1, 0.1, 1])

stim_extend = 9.

I,J = np.indices(M.shape[:2])
rv = stim_extend
dv = 2 * stim_extend / (M.shape[0]-1)

X = -rv + dv * J
Y =  rv - dv * I

hrf = spm_hrf_compat(t[:int(32/TR)])

mask = M.sum(2) > 0
Im = I[mask]
Jm = J[mask]
# Xm = X[mask]
# Ym = Y[mask]
Mm = M[mask]
Mmh = np.apply_along_axis(lambda tc: np.convolve(tc, hrf)[:tc.shape[0]], 1, Mm)

R = np.ones((V.shape[-1], 1)) / np.sqrt(V.shape[-1])

R_regress = np.eye(V.shape[-1]) - R @ R.T

i, j = sym.symbols('i j', cls=sym.Idx)
r, d = sym.symbols('r d', real=True)
mx, my, sigma = sym.symbols(r'\mu_x \mu_y \sigma', real=True)

x = -r + d * j
y =  r - d * i

prf = sym.exp(-((x - mx)**2 + (y - my)**2) / (2 * sigma**2))

parameters = [mx, my, sigma, d, r, i, j]

prf_np = sym.lambdify(parameters, prf, modules='numpy')

dprf_dmx    = prf.diff(mx)
dprf_dmy    = prf.diff(my)
dprf_dsigma = prf.diff(sigma)

dprf_dmx_np    = sym.lambdify(parameters, dprf_dmx,    modules='numpy')
dprf_dmy_np    = sym.lambdify(parameters, dprf_dmy,    modules='numpy')
dprf_dsigma_np = sym.lambdify(parameters, dprf_dsigma, modules='numpy')

def signal(x):
    pRFm          = prf_np(        *x, dv, rv, Im, Jm)
    dpRF_dmx_m    = dprf_dmx_np(   *x, dv, rv, Im, Jm)
    dpRF_dmy_m    = dprf_dmy_np(   *x, dv, rv, Im, Jm)
    dpRF_dsigma_m = dprf_dsigma_np(*x, dv, rv, Im, Jm)

    s = pRFm @ Mmh
    s_star = R_regress @ s

    s_star_length = np.sqrt(np.sum(s_star**2))

    s_prime = s_star / s_star_length

    ds_star_dmx    = R_regress @ (dpRF_dmx_m    @ Mmh)
    ds_star_dmy    = R_regress @ (dpRF_dmy_m    @ Mmh)
    ds_star_dsigma = R_regress @ (dpRF_dsigma_m @ Mmh)

    ds_prime_dmx    = ds_star_dmx    / s_star_length - s_star / s_star_length ** 3  * (ds_star_dmx.T    @ s_star)
    ds_prime_dmy    = ds_star_dmy    / s_star_length - s_star / s_star_length ** 3  * (ds_star_dmy.T    @ s_star)
    ds_prime_dsigma = ds_star_dsigma / s_star_length - s_star / s_star_length ** 3  * (ds_star_dsigma.T @ s_star)

    return s_prime, ds_prime_dmx, ds_prime_dmy, ds_prime_dsigma

sigma = 2.

y, dx, dy, dsigma = signal([0, 0, sigma])

fig, axs = plt.subplots(1, 3, constrained_layout=True)

for ax, s, s_label in zip(axs, [dx, dy, dsigma], ['x', 'y', 'sigma']):
    ax.plot(y)
    ax.plot(s)
    ax.set_title(s_label)

Ng = 51

Xg, Yg = np.meshgrid(np.linspace(-rv, rv, Ng), np.linspace(-rv, rv, Ng))

Pg = np.vstack((Xg.flatten(), Yg.flatten(), sigma*np.ones(Xg.size))).T

Sg, Dxg, Dyg, Dsigmag = np.moveaxis(np.array([signal(x) for x in Pg]), 1, 0)

#%%

noise = np.random.randn(len(y))
cnr = 0.

fig = plt.figure()
ax_quiver = fig.add_subplot(1, 2, 1)
ax_err = fig.add_subplot(1, 2, 2, sharex=ax_quiver)
#ax_3d     = fig.add_subplot(1, 2, 2, projection='3d')

fig.subplots_adjust(bottom=0.3)

ax_x = fig.add_axes([0.1, 0.1, 0.5, 0.03])
x_slider = Slider(
    ax = ax_x,
    label='m_x',
    valmin=-stim_extend,
    valmax= stim_extend,
    valinit=0.)

ax_y = fig.add_axes([0.1, 0.066, 0.5, 0.03])
y_slider = Slider(
    ax = ax_y,
    label='m_y',
    valmin=-stim_extend,
    valmax= stim_extend,
    valinit=0.)

ax_cnr = fig.add_axes([0.1, 0.033, 0.5, 0.03])
cnr_slider = Slider(
    ax = ax_cnr,
    label='CNR',
    valmin=0.,
    valmax=2.,
    valinit=0.)

ax_new_noise = fig.add_axes([0.8, 0.033, 0.15, 0.1-0.033])
new_noise_button = Button(ax_new_noise, "New noise")

def update_tc():
    y, dx, dy, dsigma = signal([x_slider.val, y_slider.val, sigma])
    y /= y.max()
    return y

ln1 = ax_quiver.plot([x_slider.val], [y_slider.val], 'x')
ln2 = ax_err.plot([x_slider.val], [y_slider.val], 'x')

# def update_noise():
#     noise = np.random.randn(len(y))

def get_fits(y, cnr):
    yn = y + np.random.randn(len(y)) * cnr
    e = Sg @ yn
    ee = e ** 2

    dex     = 2 * e * (Dxg     @ yn)
    dey     = 2 * e * (Dyg     @ yn)
    desigma = 2 * e * (Dsigmag @ yn)

    return ee, dex, dey, desigma

y = update_tc()
# update_noise()
Eg, Exg, Eyg, Esigmag = get_fits(y, cnr)

def update_fits(y, cnr):
    return get_fits(y, cnr)

quiver = ax_quiver.quiver(Xg, Yg, Exg, Eyg, Esigmag, angles='xy', scale_units='xy', cmap='coolwarm', norm=plt.Normalize(-10, 10), scale=50)
ax_quiver.axis('equal')
ax_err.axis('equal')

def update_quiver(Eg, Exg, Eyg, Esigmag):
    quiver.set_UVC(Exg, Eyg, Esigmag)

def update_contour(Eg, Exg, Eyg, Esigmag):
    ax_err.clear()
    ax_err.contour(Xg, Yg, Eg.reshape(Xg.shape), levels=11)

# def plot_3d(Eg, Exg, Eyg, Esigmag):
#     return ax_3d.plot_surface(Xg, Yg, Eg.reshape(Xg.shape), edgecolor='royalblue', lw=0.5, alpha=0.3)

# surf = plot_3d(Eg, Exg, Eyg, Esigmag)

def update_plots(Eg, Exg, Eyg, Esigmag):
    update_quiver(Eg, Exg, Eyg, Esigmag)
    update_contour(Eg, Exg, Eyg, Esigmag)
    ln1[0].set_xdata([x_slider.val])
    ln1[0].set_ydata([y_slider.val])
    ln2 = ax_err.plot([x_slider.val], [y_slider.val], 'x')
    # surf = plot_3d(Eg, Exg, Eyg, Esigmag)

def update(event):
    # update_noise()
    y = update_tc()
    Eg, Exg, Eyg, Esigmag = update_fits(y, cnr_slider.val)
    update_plots(Eg, Exg, Eyg, Esigmag)

x_slider.on_changed(update)
y_slider.on_changed(update)
cnr_slider.on_changed(update)
new_noise_button.on_clicked(update)