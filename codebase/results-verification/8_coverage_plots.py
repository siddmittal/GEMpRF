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

#vista = PRF.from_docker('', sub, ses, 'prf', '01', analysis='01', hemi='BOTH', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='MP', method='vista')
vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='03', hemi='', orientation='VF', baseP= "Y:\\data")

##--Coverage Plots
vista.maskROI('V1')
vista.maskVarExp(.1)
vista.plot_covMap()
vista.plot_covMap(force = True, show=True)
vista.plot_covMap(force = True, save=True)


#oprf = PRF.from_docker('', sub, ses, 'prf', '01', analysis='02', hemi='BOTH', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='VF', method='oprf')
oprf = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '01', analysis='06', hemi='', baseP= "Y:\\data", orientation='VF', method='oprf')


#%%
# plt.close('all')


###--Coverage Plots
oprf.maskROI('V1')
oprf.maskVarExp(.1)
oprf.plot_covMap()
oprf.plot_covMap(force = True, show=True, maxEcc = 9)
oprf.plot_covMap(force = True, save=True, maxEcc = 9)

print("done")