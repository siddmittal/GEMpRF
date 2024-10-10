#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:52:10 2023

@author: dlinhardt
"""

import sys
# sys.path.append('/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/dlinhardt/pythonclass')
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")


from PRFclass import PRF
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='L', orientation='MP')
vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='L', baseP= "Y:\\data",  orientation='MP')
vista.maskVarExp(.1)

# oprf_file = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/stimsim23/derivatives/oprf/sub-sidtest/ses-001/sub-sidtest_ses-001_task-prf_run-01_hemi-L_results.txt'
oprf_file = "D:\\results\\fmri\\sid-fmri-refined_fit.txt"

with open(oprf_file, 'r') as f:
    lines = f.readlines()

oprf_x = []
oprf_y = []
oprf_s = []

for line in lines:
    split_line = [ a for a in line.split('[')[1].split(']')[0].split(' ') if a]
    
    oprf_x.append(float(split_line[0]))
    oprf_y.append(float(split_line[1]))
    oprf_s.append(float(split_line[2]))

oprf_x = np.array(oprf_x)
oprf_y = np.array(oprf_y)
oprf_s = np.array(oprf_s)

#%%
plt.close('all')

def plot_func(title, X, Y, xy_min, xy_max):
    plt.figure(constrained_layout=True)
    plt.title(title)
    
    plt.scatter(X, Y, c='blue')
    plt.plot((xy_min,xy_max), (xy_min,xy_max), 'r')
    
    plt.xlim(xy_min,xy_max)
    plt.ylim(xy_min,xy_max)
    plt.gca().set_aspect('equal', 'box')
    plt.grid()
    
    
plot_func('X', vista.x,  oprf_y[vista._varExpMsk], -10, 10)
plot_func('Y', vista.y,  oprf_x[vista._varExpMsk], -10, 10)
plot_func('S', vista.s,  oprf_s[vista._varExpMsk], 0, 5)


#%%
def plot_comp_curves(title, X, Y, xy_min, xy_max):
    plt.figure(constrained_layout=True)
    plt.title(title)
    
    plt.plot(X, label='vista')
    plt.plot(Y, label='oprf')
    
    plt.legend()
    plt.ylim(xy_min,xy_max)
    

# plot_comp_curves('X', vista.x,  oprf_y[vista._varExpMsk])

sq_su_vista = np.sqrt(vista.x0**2 + vista.y0**2)# + vista.s0**2)
sq_su_oprf  = np.sqrt(oprf_x**2   + oprf_y**2  )# + oprf_s**2)


plot_comp_curves('X', sq_su_vista[::10],  sq_su_oprf[::10], 0, 50)
# plot_comp_curves('X', sq_su_vista[::10],  oprf_y[::10], -10, 10)
