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
import seaborn as sns

#vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='L', orientation='MP')
vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='L', baseP= "Y:\\data", orientation='MP')
vista.maskVarExp(.1)

#oprf = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='01', hemi='L', orientation='MP', method='oprf')
oprf = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='01', hemi='L', baseP= "Y:\\data", orientation='MP', method='oprf')


#%%
plt.close('all')


## Sid Seaborn test
def make_seaborn_plots(oprf_data, vista_data):
    my_data = np.hstack((oprf_data, vista_data))
    # Plot sepal width as a function of sepal_length across days
    g = sns.lmplot(
        data=my_data,
        x="bill_length_mm", y="bill_depth_mm", hue="species",
        height=5
    )

    # Use more informative axis labels than are provided by default
    g.set_axis_labels("Snoot length (mm)", "Snoot depth (mm)")



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
    
make_seaborn_plots(oprf.y0[vista._varExpMsk], vista.x)
    
plot_func('X', vista.x,  oprf.y0[vista._varExpMsk], -10, 10)
plot_func('Y', vista.y,  oprf.x0[vista._varExpMsk], -10, 10)
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
