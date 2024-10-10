import numpy as np
import scipy as scp
import nibabel as nib
import scipy.stats as sps
import sympy as sym
import matplotlib.pyplot as plt

import sys
if '/z/fmrilab/home/dlinhardt/pythonclass' not in sys.path:
    sys.path.append(r"Z:\home\dlinhardt\pythonclass")
    
from PRFclass import PRF
from PIL import Image
import os
from skimage.transform import resize
from skimage import img_as_bool
from glob import glob
from pathlib import Path
# import numba as nb
# from cvxopt import spmatrix


study = 'stimsim23'
base_dir = os.path.join(r"Y:\data", study)
prfanalyze_analysis = '01'
hemi = 'L'
subject = 'h001'
session = '001'
run     = '01'
task    = 'bar'

nDCT = 4

min_sigma = 0.5

# get the original results
ana = PRF.from_docker(study, subject, session, task, run, analysis=prfanalyze_analysis, hemi=hemi, baseP = "Y:\data")
ana.maskROI('V1')
# ana.maskVarExp(.1)
# ana.maskSigma(min_sigma)

prfprepare_options = ana._prfprepareOpts
prfprepare_analysis = ana._prfanalyzeOpts['prfprepareAnalysis']
analysis_space = ana._prfprepareOpts['analysisSpace']

stim_radius = 9.4

sigmas = np.linspace(0.5, 5, 8)
xy_space = np.linspace(-stim_radius, stim_radius, 51)

x,y,s = np.meshgrid(xy_space, xy_space, sigmas, indexing='ij')
search_space = np.vstack((x.flatten(), y.flatten(), s.flatten())).T

def makeTrends(nFrames = 300, nUniqueRep = 1, nDCT = 3):
    """Generate trends like mrVista's rmMakeTrends from the 'params' struct."""

    stims = [1]

    tf   = [int(nFrames / nUniqueRep)]
    ndct = [2 * nDCT + 1]
    t    = np.zeros((np.sum(tf), np.max([np.sum(ndct), 1])))
    start1 = np.hstack([0, np.cumsum(tf)])
    start2 = np.hstack([0, np.cumsum(ndct)])

    dcid = np.zeros(len(stims))

    for n in range(len(stims)):
        tc = np.linspace(0, 2.*np.pi, tf[n])[:,None]
        t[start1[n]:start1[n+1], start2[n]:start2[n+1]] = np.cos(tc.dot(np.arange(0, nDCT+0.5, 0.5)[None,:]))
        dcid[n] = start2[n]

    nt = t.shape[1]

    return t, nt, dcid

# SPMs HRF
def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio = 6,
                   normalize=True,
                  ):
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

# define the model
x, y = sym.symbols('x y', real=True)
mx, my, sigma = sym.symbols(r'\mu_x \mu_y \sigma', real=True)

prf = sym.exp(-((x - mx)**2 + (y - my)**2) / (2 * sigma**2))

parameters = [mx, my, sigma, x, y]

prf_np = sym.lambdify(parameters, prf, modules='numpy')

# @nb.njit
# def prf_np(mx, my, sigma, x, y):
#     return np.exp(-((x - mx)**2 + (y - my)**2) / (2 * sigma**2))

dprf_dmx    = prf.diff(mx)
dprf_dmy    = prf.diff(my)
dprf_dsigma = prf.diff(sigma)

dprf_dmx_dmx       = dprf_dmx.diff(mx)
dprf_dmx_dmy       = dprf_dmx.diff(my)
dprf_dmx_dsigma    = dprf_dmx.diff(sigma)
dprf_dmy_dmy       = dprf_dmy.diff(my)
dprf_dmy_dsigma    = dprf_dmy.diff(sigma)
dprf_dsigma_dsigma = dprf_dsigma.diff(sigma)

dprf_dmx_np    = sym.lambdify(parameters, dprf_dmx,    modules='numpy')
dprf_dmy_np    = sym.lambdify(parameters, dprf_dmy,    modules='numpy')
dprf_dsigma_np = sym.lambdify(parameters, dprf_dsigma, modules='numpy')

dprf_dmx_dmx_np       = sym.lambdify(parameters, dprf_dmx_dmx,       modules='numpy')
dprf_dmx_dmy_np       = sym.lambdify(parameters, dprf_dmx_dmy,       modules='numpy')
dprf_dmx_dsigma_np    = sym.lambdify(parameters, dprf_dmx_dsigma,    modules='numpy')
dprf_dmy_dmy_np       = sym.lambdify(parameters, dprf_dmy_dmy,       modules='numpy')
dprf_dmy_dsigma_np    = sym.lambdify(parameters, dprf_dmy_dsigma,    modules='numpy')
dprf_dsigma_dsigma_np = sym.lambdify(parameters, dprf_dsigma_dsigma, modules='numpy')

ml = np.array([(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)])

class Stimulus:
    def __init__(self, name, stim_res = 101):
        self.name = name

        stim_dir = os.path.join(base_dir, 'derivatives', 'prfprepare', f'analysis-{prfprepare_analysis}', f'sub-{subject}', 'stimuli')

        stim_path = list(glob(os.path.join(stim_dir, f"task-{name}*.nii.gz")))[0]

        #stim_radius = 9.4
        self.stim_radius = stim_radius

        stim_nifti = nib.load(stim_path)

        self.TR = stim_nifti.header['pixdim'][4]

        stim = stim_nifti.get_fdata()

        stim_low_res = np.array([np.array(Image.fromarray(stim[:,:,0,i]).resize((stim_res, stim_res), Image.Resampling.LANCZOS)) for i in range(stim.shape[3])])
        stim_low_res = np.moveaxis(stim_low_res, 0, -1)
        stim_low_res = np.clip(stim_low_res, 0, 1) # Lanczos can slightly over-/undershoot from 0-1

        X,Y = np.meshgrid(np.linspace(-stim_radius, stim_radius, stim_res), np.linspace(-stim_radius, stim_radius, stim_res))

        mask = stim_low_res.max(2) > 0

        self.X = X[mask]
        self.Y = Y[mask]
        self.V_unconv = stim_low_res[mask]

        self.N = self.V_unconv.shape[1]

        self.t = np.arange(self.N) * self.TR

        self.hrf = spm_hrf_compat(self.t[:int(32/self.TR)])

        self.V = np.apply_along_axis(lambda tc: np.convolve(tc, self.hrf)[:tc.shape[0]], 1, self.V_unconv)

        R, nt, dcid = makeTrends(nFrames = self.N, nDCT = nDCT)
        q, r = np.linalg.qr(R)

        R = q * np.sign(q[0,0])

        R_regress = np.eye(self.N) - R @ R.T

        self.R = R
        self.R_regress = R_regress

        print(name)

    def signal(self, x, return_derivatives = True):
        pRFm          = prf_np(        *x, self.X, self.Y)

        s = pRFm @ self.V
        s_star = self.R_regress @ s

        s_star_length = np.sqrt(np.sum(s_star**2))
        
        if s_star_length < 1e-16:
            s_star_length = 1e-16
            s_star = np.zeros_like(s_star)

        s_prime = s_star / s_star_length

        if not return_derivatives:
            return s_prime

        dpRF_dmx_m    = dprf_dmx_np(   *x, self.X, self.Y)
        dpRF_dmy_m    = dprf_dmy_np(   *x, self.X, self.Y)
        dpRF_dsigma_m = dprf_dsigma_np(*x, self.X, self.Y)
        
        dpRF_dmx_dmx_m       = dprf_dmx_dmx_np(      *x, self.X, self.Y)
        dpRF_dmx_dmy_m       = dprf_dmx_dmy_np(      *x, self.X, self.Y)
        dpRF_dmx_dsigma_m    = dprf_dmx_dsigma_np(   *x, self.X, self.Y)
        dpRF_dmy_dmy_m       = dprf_dmy_dmy_np(      *x, self.X, self.Y)
        dpRF_dmy_dsigma_m    = dprf_dmy_dsigma_np(   *x, self.X, self.Y)
        dpRF_dsigma_dsigma_m = dprf_dsigma_dsigma_np(*x, self.X, self.Y)

        ds_star_dmx    = self.R_regress @ (dpRF_dmx_m    @ self.V)
        ds_star_dmy    = self.R_regress @ (dpRF_dmy_m    @ self.V)
        ds_star_dsigma = self.R_regress @ (dpRF_dsigma_m @ self.V)
        
        ds_star_dmx_dmx       = self.R_regress @ (dpRF_dmx_dmx_m       @ self.V)
        ds_star_dmx_dmy       = self.R_regress @ (dpRF_dmx_dmy_m       @ self.V)
        ds_star_dmx_dsigma    = self.R_regress @ (dpRF_dmx_dsigma_m    @ self.V)
        ds_star_dmy_dmy       = self.R_regress @ (dpRF_dmy_dmy_m       @ self.V)
        ds_star_dmy_dsigma    = self.R_regress @ (dpRF_dmy_dsigma_m    @ self.V)
        ds_star_dsigma_dsigma = self.R_regress @ (dpRF_dsigma_dsigma_m @ self.V)

        ds_prime_dmx    = ds_star_dmx    / s_star_length - s_star / s_star_length ** 3  * (ds_star_dmx.T    @ s_star)
        ds_prime_dmy    = ds_star_dmy    / s_star_length - s_star / s_star_length ** 3  * (ds_star_dmy.T    @ s_star)
        ds_prime_dsigma = ds_star_dsigma / s_star_length - s_star / s_star_length ** 3  * (ds_star_dsigma.T @ s_star)
        
        def ds_prime_dm_dl(ds_star_dm, ds_star_dl, ds_star_dm_dl):
            return ds_star_dm_dl / s_star_length - ds_star_dm / s_star_length ** 3 * (s_star.T @ ds_star_dl) - \
                   (ds_star_dl / s_star_length ** 3 - 3 * s_star / s_star_length ** 5 * (s_star.T @ ds_star_dl)) * (s_star.T @ ds_star_dm) - \
                   s_star / s_star_length ** 3 * (ds_star_dl.T @ ds_star_dm + s_star.T @ ds_star_dm_dl)
                   
        ds_prime_dmx_dmx       = ds_prime_dm_dl(ds_star_dmx,    ds_star_dmx,    ds_star_dmx_dmx)
        ds_prime_dmx_dmy       = ds_prime_dm_dl(ds_star_dmx,    ds_star_dmy,    ds_star_dmx_dmy)
        ds_prime_dmx_dsigma    = ds_prime_dm_dl(ds_star_dmx,    ds_star_dsigma, ds_star_dmx_dsigma)
        ds_prime_dmy_dmy       = ds_prime_dm_dl(ds_star_dmy,    ds_star_dmy,    ds_star_dmy_dmy)
        ds_prime_dmy_dsigma    = ds_prime_dm_dl(ds_star_dmy,    ds_star_dsigma, ds_star_dmy_dsigma)
        ds_prime_dsigma_dsigma = ds_prime_dm_dl(ds_star_dsigma, ds_star_dsigma, ds_star_dsigma_dsigma)

        return s_prime, (ds_prime_dmx, ds_prime_dmy, ds_prime_dsigma), (ds_prime_dmx_dmx, ds_prime_dmx_dmy, ds_prime_dmx_dsigma, ds_prime_dmy_dmy, ds_prime_dmy_dsigma, ds_prime_dsigma_dsigma)
        
    def fit(self, x, y, return_derivatives = True):
        if return_derivatives:
            s_prime, ds_prime_dm, ds_prime_dm_dl = self.signal(x, return_derivatives)
                    
            e         = s_prime.dot(y)
            de_dm     = [        ds_dm    @ y for ds_dm    in ds_prime_dm   ]
            de_dm_dl  = [        ds_dk_dl @ y for ds_dk_dl in ds_prime_dm_dl]
            de2_dm    = [2 * e * de_k         for de_k     in de_dm         ]
            de2_dm_dl = [2 * (de_dm[l] * de_dm[m] + e * de_dml) for (m,l), de_dml in zip(ml, de_dm_dl)]
            
            return e, de_dm, de_dm_dl, de2_dm, de2_dm_dl
        else:
            s_prime = self.signal(x, return_derivatives)
            return s_prime.dot(y)
    

# do something
stim = Stimulus(task)
max_ecc = 2. * stim.stim_radius

S_and_deriv = [stim.signal(xys, return_derivatives=True) for xys in search_space]

#%%

deriv_weights = np.array([1./np.square(s_a_d[1]).sum(1) for s_a_d in S_and_deriv])

M = np.eye(4)
M[0,0] = xy_space[1] - xy_space[0]
M[1,1] = xy_space[1] - xy_space[0]
M[2,2] = sigmas[1] - sigmas[0]
M[0,3] = xy_space[0]
M[1,3] = xy_space[0]
M[2,3] = sigmas[0]

for i in range(3):
    weights_i = deriv_weights[:,i].reshape((len(xy_space), len(xy_space), len(sigmas)))
    nii = nib.Nifti1Image(weights_i, M)
    nib.save(nii, Path('prf_fine_fit_blue', f'weights_{i}.nii'))
                                           
