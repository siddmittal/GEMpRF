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

vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='', orientation='VF', baseP= "Y:\\data")
# vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='08', hemi='L', baseP= "Y:\\data", orientation='MP', method='oprf')#PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='L', baseP= "Y:\\data", orientation='MP')
vista.maskVarExp(.1)

#oprf = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='01', hemi='L', orientation='MP', method='oprf')
oprf = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='06', hemi='', baseP= "Y:\\data", orientation='VF', method='oprf')


#%%
plt.close('all')

def plot_func(title, X, Y, xy_min, xy_max):
    plt.figure(constrained_layout=True)
    plt.title(title)
    
    plt.scatter(X, Y, c='blue')
    plt.plot((xy_min,xy_max), (xy_min,xy_max), 'r')
    
    plt.xlim(xy_min,xy_max)
    plt.ylim(xy_min,xy_max)
    
    plt.xlabel('vista')
    plt.ylabel('oprf')
    
    plt.gca().set_aspect('equal', 'box')
    plt.grid()
    
    
plot_func('X', vista.x,  oprf.x0[vista._varExpMsk], -10, 10)
plot_func('Y', vista.y,  oprf.y0[vista._varExpMsk], -10, 10)
plot_func('S', vista.s,  oprf.s0[vista._varExpMsk], 0, 5)
plot_func('VE', vista.varexp0,  oprf.varexp0, -3, 1)

#%%
plt.figure(constrained_layout=True)
plt.scatter(vista.r, vista.s, label='vista')
plt.scatter(oprf.r0[vista._varExpMsk], oprf.s0[vista._varExpMsk], label='oprf')
plt.xlabel('ecc')
plt.ylabel('sigma')
plt.grid()
plt.legend()
plt.xlim(0,10)
plt.ylim(0,)


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
sq_su_oprf  = np.sqrt(oprf.x0**2   + oprf.y0**2  )# + oprf_s**2)


plot_comp_curves('X', sq_su_vista[::10],  sq_su_oprf[::10], 0, 50)
# plot_comp_curves('X', sq_su_vista[::10],  oprf_y[::10], -10, 10)

print()
