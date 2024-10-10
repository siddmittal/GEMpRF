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

sub = '002'
ses = '001'

#vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='L', orientation='MP')
fprf = PRF.from_docker('', sub, ses, 'prf', '01', analysis='02', hemi='BOTH', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='VF', method='fprf')

vista = PRF.from_docker('', sub, ses, 'prf', '01', analysis='01', hemi='BOTH', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='MP', method='vista')

samsrf = PRF.from_samsrf('', sub, ses, 'prf', '01', analysis='01',baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='MP')

oprf = PRF.from_docker('', sub, ses, 'prf', '01', analysis='02', hemi='BOTH', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='VF', method='oprf')
vista.maskVarExp(.05)


#%%
# plt.close('all')


###--Coverage Plots
vista.maskROI('V1')
vista.maskVarExp(.1)
vista.plot_covMap()
vista.plot_covMap(show=True)
vista.plot_covMap(save=True)


to_comp = (vista, oprf)

def plot_func(title, X, Y, xlab, ylab, xy_min, xy_max):
    f = plt.figure(constrained_layout=True)
    plt.title(title)
    
    plt.scatter(X, Y, c='blue', s=.3)
    #plt.scatter(X, Y, c='blue')
    plt.plot((xy_min,xy_max), (xy_min,xy_max), 'r')
    
    plt.xlim(xy_min,xy_max)
    plt.ylim(xy_min,xy_max)
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    plt.gca().set_aspect('equal', 'box')
    plt.grid()
    
    return f
    
    
f = plot_func('X', to_comp[0].x,  to_comp[1].x0[vista._varExpMsk], 
              to_comp[0]._prfanalyze_method, to_comp[1]._prfanalyze_method, -10, 10)
#f.savefig('l')
plot_func('Y',  to_comp[0].y,  to_comp[1].y0[vista._varExpMsk], 
          to_comp[0]._prfanalyze_method, to_comp[1]._prfanalyze_method, -10, 10)

plot_func('S',  to_comp[0].s,  to_comp[1].s0[vista._varExpMsk], 
          to_comp[0]._prfanalyze_method, to_comp[1]._prfanalyze_method, 0, 5)

plot_func('VE', to_comp[0].varexp0,  to_comp[1].varexp0, 
          to_comp[0]._prfanalyze_method, to_comp[1]._prfanalyze_method, 0, 1)


#%%
plt.figure(constrained_layout=True)
plt.scatter(vista.r, vista.s, label='vista')
plt.scatter(oprf.r0[vista._varExpMsk], oprf.s0[vista._varExpMsk], label='fprf')
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
    plt.plot(Y, label='fprf')
    
    plt.legend()
    plt.ylim(xy_min,xy_max)
    

# plot_comp_curves('X', vista.x,  oprf_y[vista._varExpMsk])

sq_su_vista = np.sqrt(vista.x0**2 + vista.y0**2)# + vista.s0**2)
sq_su_oprf  = np.sqrt(oprf.x0**2   + oprf.y0**2  )# + oprf_s**2)


plot_comp_curves('X', sq_su_vista[::10],  sq_su_oprf[::10], 0, 50)
# plot_comp_curves('X', sq_su_vista[::10],  oprf_y[::10], -10, 10)