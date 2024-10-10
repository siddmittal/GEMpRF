#%%
import numpy as np
import scipy as scp
import nibabel as nib
import scipy.stats as sps
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons
from pathlib import Path
import re
from PIL import Image
import pickle

import sys
#sys.path.append('/z/home/dlinhardt/pythonclass')
sys.path.append('Z:/home/dlinhardt/pythonclass')
from PRFclass import PRF

study = 'stimsim22'
subjects = [1]
sessions = [1]
runs     = [1]
tasks    = ['barStandard']

nDCT = 4

stimuli_file = Path(f'{study}-stimuli.pkl')

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

###################------------Some stuff---START------------#######
x, y = sym.symbols('x y', real=True)
mx, my, sigma = sym.symbols(r'\mu_x \mu_y \sigma', real=True)

prf = sym.exp(-((x - mx)**2 + (y - my)**2) / (2 * sigma**2))

parameters = [mx, my, sigma, x, y]

prf_np = sym.lambdify(parameters, prf, modules='numpy')

dprf_dmx    = prf.diff(mx)
dprf_dmy    = prf.diff(my)
dprf_dsigma = prf.diff(sigma)

dprf_dmx_np    = sym.lambdify(parameters, dprf_dmx,    modules='numpy')
dprf_dmy_np    = sym.lambdify(parameters, dprf_dmy,    modules='numpy')
dprf_dsigma_np = sym.lambdify(parameters, dprf_dsigma, modules='numpy')

n_samples_sigma = 13
sigma_min = 0.2
sigma_max = 5

sigmas = np.linspace(sigma_min, sigma_max, n_samples_sigma)
###################------------Some stuff---END------------#######


class Stimulus:
    def __init__(self, name, n_samples_xy = 51, oversample = 0., stim_res = 101):
        self.name = name
        
        stim_dir = Path("Y:/", "data", study, "derivatives", "prfprepare", "analysis-01", f"sub-{subjects[0]:03}", "stimuli")
        
        stim_path = list(stim_dir.glob(f"task-{name}*"))[0]
        
        stim_radius = 9.4
        
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
        
        self.n_samples_xy = n_samples_xy
        
        self.oversample = oversample
        
        self.max_xy = max(np.abs(X).max(), np.abs(Y).max())      
        self.sampling_max = (1. + oversample) * self.max_xy
        
        self.Xg, self.Yg = np.meshgrid(np.linspace(-self.sampling_max, self.sampling_max, n_samples_xy), np.linspace(-self.sampling_max, self.sampling_max, n_samples_xy))
        
        self.sampling_tcs = []
        
        print(name)
        
        for sigma in sigmas:
            print(sigma)
            Pg = np.vstack((self.Xg.flatten(), self.Yg.flatten(), sigma*np.ones(self.Xg.size))).T

            self.sampling_tcs.append(np.moveaxis(np.array([self.signal(x) for x in Pg]), 1, 0))
        
    def signal(self, x):
        pRFm          = prf_np(        *x, self.X, self.Y)
        dpRF_dmx_m    = dprf_dmx_np(   *x, self.X, self.Y)
        dpRF_dmy_m    = dprf_dmy_np(   *x, self.X, self.Y)
        dpRF_dsigma_m = dprf_dsigma_np(*x, self.X, self.Y)
        
        s = pRFm @ self.V
        s_star = self.R_regress @ s
        
        s_star_length = np.sqrt(np.sum(s_star**2))
        
        s_prime = s_star / s_star_length
        
        ds_star_dmx    = self.R_regress @ (dpRF_dmx_m    @ self.V)
        ds_star_dmy    = self.R_regress @ (dpRF_dmy_m    @ self.V)
        ds_star_dsigma = self.R_regress @ (dpRF_dsigma_m @ self.V)
        
        ds_prime_dmx    = ds_star_dmx    / s_star_length - s_star / s_star_length ** 3  * (ds_star_dmx.T    @ s_star)
        ds_prime_dmy    = ds_star_dmy    / s_star_length - s_star / s_star_length ** 3  * (ds_star_dmy.T    @ s_star)
        ds_prime_dsigma = ds_star_dsigma / s_star_length - s_star / s_star_length ** 3  * (ds_star_dsigma.T @ s_star)
        
        return s_prime, ds_prime_dmx, ds_prime_dmy, ds_prime_dsigma

################----Stuff
if stimuli_file.exists():
    stimuli = pickle.load(stimuli_file.open('rb'))
else:
    stimuli = {}

####################    


class Measurement:
    def __init__(self, subject, session, run, task):
        self.subject = subject
        self.session = session
        self.run     = run
        self.task    = task
        
        self.data = PRF.from_docker(study, f"{self.subject:03}", f"{self.session:03}", task, f"{self.run:02}", baseP='Y:\data')
        self.data.maskROI('V1')
        
        global stimuli
        
        if task not in stimuli:
            stimuli[task] = Stimulus(task)
            
            if all([t in stimuli for t in tasks]):
                pickle.dump(stimuli, stimuli_file.open('wb'))

####################----Stuff
fig, ax = plt.subplot_mosaic(
    [
        ['quiver', 'err', 'subject'],
        ['quiver', 'err', 'session'],
        ['quiver', 'err', 'run'],
        ['quiver', 'err', 'task'],
        ['quiver', 'err', 'info'],
        ['sigma', 'sigma', 'sigma'],
        ['voxel', 'voxel', 'voxel']
    ],
    width_ratios=[5, 5, 1],
    height_ratios=[3, 5, 2, 3, 1, 0.25, 0.25],
    layout='constrained',
)
########################################

def toSubjectString(subject):
    return f"sub-{subject:03}"
def fromSubjectString(subject):
    return int(re.search('sub-(?P<id>\d{3})', subject).groupdict()['id'])
def toSessionString(session):
    return f"ses-{session:03}"
def fromSessionString(session):
    return int(re.search('ses-(?P<id>\d{3})', session).groupdict()['id'])
def toRunString(run):
    return f"run-{run:02}"
def fromRunString(run):
    return int(re.search('run-(?P<id>\d{2})', run).groupdict()['id'])

####################-------------Stuff
subject_rbtn = RadioButtons(ax['subject'], [toSubjectString(subject) for subject in subjects])
session_rbtn = RadioButtons(ax['session'], [toSessionString(session) for session in sessions])
run_rbtn     = RadioButtons(ax['run'],     [toRunString(    run)     for run     in runs])
task_rbtn    = RadioButtons(ax['task'], tasks)

sigma_slider = Slider(ax['sigma'], 'Sigma', sigma_min, sigma_max, valstep=sigmas)
voxel_slider = Slider(ax['voxel'], 'Voxel', 0, 1000, valstep=1)

info_txt = TextBox(ax['info'], 'info')

measurement = None
##################################################

def load_measurement():
    subject = fromSubjectString(subject_rbtn.value_selected)
    session = fromSessionString(session_rbtn.value_selected)
    run     = fromRunString(run_rbtn.value_selected)
    task    = task_rbtn.value_selected
    
    global measurement
    
    measurement = Measurement(subject, session, run, task)
        
    voxel_slider.valmax = measurement.data.voxelTC.shape[0]
    voxel_slider.ax.set_xlim(voxel_slider.valmin,voxel_slider.valmax)
        
##########-----------Stuff        
load_measurement()
########################
#%%
def get_fits():
    global measurement
    
    y = measurement.data.voxelTC[int(voxel_slider.val)]
    y = sps.zscore(y)
    
    stim = stimuli[measurement.task]
    
    sigma_index = np.argmin(np.abs(sigmas - sigma_slider.val))
    
    Sg, Dxg, Dyg, Dsigmag = stim.sampling_tcs[sigma_index]  ## How are the derivatives computed?
    
    e = Sg @ y
    ee = e ** 2
    
    dex     = 2 * e * (Dxg     @ y)
    dey     = 2 * e * (Dyg     @ y)
    desigma = 2 * e * (Dsigmag @ y)
    
    return ee, dex, dey, desigma

########################-------------Stuff
Eg, Exg, Eyg, Esigmag = get_fits() ##<<<<<<-----------------

quiver = ax['quiver'].quiver(stimuli[measurement.task].Xg, stimuli[measurement.task].Yg, Exg, Eyg, Esigmag, angles='xy', scale_units='xy', cmap='coolwarm', norm=plt.Normalize(-5, 5))
ax['err'].sharex(ax['quiver'])
vx = int(voxel_slider.val)
ln1, = ax['quiver'].plot([measurement.data.x[vx]], [-measurement.data.y[vx]], 'x')
CS = ax['err'].contour(stimuli[measurement.task].Xg, stimuli[measurement.task].Yg, Eg.reshape(stimuli[measurement.task].Xg.shape), levels=11)
ln2, = ax['err'].plot([measurement.data.x[vx]], [-measurement.data.y[vx]], 'x')
#ax['err'].clabel(CS, CS.levels, inline=True)
ax['quiver'].axis('equal')
ax['err'].axis('equal')

info_txt.set_val(f'Max error: {Eg.max():.5f}')
info_txt.set_val(f'Max error: {Eg.max():.5f}\nx: {measurement.data.x[vx]:.2f}/y: {measurement.data.y[vx]:.2f}/sigma: {measurement.data.s[vx]:.2f}\nvar exp: {measurement.data.varexp[vx]:.3f}')
#####################################################






























