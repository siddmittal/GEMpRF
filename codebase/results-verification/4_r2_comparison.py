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

r2_simple = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='07', hemi='R', baseP= "Y:\\data",  orientation='MP', method='oprf')
r2_simple.maskVarExp(.01)

r2_with_e = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='06', hemi='R', baseP= "Y:\\data",  orientation='MP', method='oprf')



#%%
# plt.close('all')

to_comp = (r2_simple, r2_with_e)

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
    
    

plot_func('VE', to_comp[0].varexp0,  to_comp[1].varexp0, 
          "r2_simple", "r2_with_e", 0, 1)


print
