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

subjects = ['001', '002']
sessions = ['001','002', '003']

sns.set_style('darkgrid')

def plot_func(title, X, Y, xlab, ylab, xy_min, xy_max, gt):
    # f = plt.figure(constrained_layout=True)
    # plt.title(title)
    
    # plt.scatter(X, Y, c='blue', s=.3)
    # #plt.scatter(X, Y, c='blue')
    # plt.plot((xy_min,xy_max), (xy_min,xy_max), 'r')
    
    # plt.xlim(xy_min,xy_max)
    # plt.ylim(xy_min,xy_max)
    
    # plt.xlabel(xlab)
    # plt.ylabel(ylab)
    
    # plt.gca().set_aspect('equal', 'box')
    # plt.grid()

    g = sns.JointGrid(x=X, y=Y)
    g.plot(sns.scatterplot, sns.histplot)
    g.set_axis_labels(xlab, ylab)
    g.refline(x=gt, y=gt, color='r')
    
    g.ax_marg_x.set_xlim(xy_min,xy_max)
    g.ax_marg_y.set_ylim(xy_min,xy_max)
    
    Result_image_path= f"D:/results/comparison-plots/samsrf_ana01vs02/{title}.svg"

    plt.savefig(Result_image_path, bbox_inches='tight')

    return g.fig

for sub in subjects:
    for ses in sessions:
        # vista = PRF.from_docker('', sub, ses, 'prf', '01', analysis='01', hemi='BOTH', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='MP', method='vista')
        # samsrf = PRF.from_samsrf('', sub, ses, 'prf', '01', analysis='02', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='MP')

        samsrf1 = PRF.from_samsrf('', sub, ses, 'prf', '01', analysis='01', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='MP')
        samsrf2 = PRF.from_samsrf('', sub, ses, 'prf', '01', analysis='02', baseP="Y:\\data\\tests\\oprf_test\\BIDS", orientation='MP')

        samsrf2.maskVarExp(.05)

        # to_comp = (vista, samsrf)   
        to_comp = (samsrf1, samsrf2)   

        gt_sigma = 1 if '001' in sub else 1.8         
            
        plot_func(f'Sigma-sub-{sub}-ses-{ses}',  to_comp[0].s0[samsrf2._varExpMsk],  to_comp[1].s, 
                to_comp[0]._prfanalyze_method+'01', to_comp[1]._prfanalyze_method+'01', 0, 2, gt_sigma)


        # #%%
        # plt.figure(constrained_layout=True)
        # plt.scatter(vista.r, vista.s, label='vista')
        # plt.scatter(samsrf.r0[vista._varExpMsk], samsrf.s0[vista._varExpMsk], label='samsrf')
        # plt.xlabel('ecc')
        # plt.ylabel('sigma')
        # plt.grid()
        # plt.legend()
        # plt.xlim(0,10)
        # plt.ylim(0,)

sns.set_style('white')