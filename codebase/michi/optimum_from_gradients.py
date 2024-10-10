#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:03:32 2024

@author: mwoletz
"""

import numpy as np
from scipy.spatial.transform import Rotation
from itertools import product

# we will simulate a 3D "Gaussian" function
# the mean and covariance matrix will be randomly generated
# the mean will be within the unit cube centred at (0.5, 0.5, 0.5)
# 8 samples will be drawn at the corners of the unit cube

def random_mean():
    return np.random.rand(3)

def random_covariance(sigma_min = 0.5, sigma_max=3):
    # draw random direction
    R = Rotation.random().as_matrix()
    l = np.random.rand(3)
    C = R @ np.diag(l) @ R.T
    
    return C

def fun(x, C, m):
    Ci = np.linalg.inv(C)
    x = np.atleast_2d(x)
    xp = x - m[None,:]
    
    f = np.array([np.exp(-0.5 * xi[None,:] @ Ci @ xi[:,None])[0,0] for xi in xp])
    g = -xp @ Ci.T * f[:,None]
    return f, g

X = np.array(list(product([0,1], repeat=3)))

def run_random_estimate():
    # draw random mean & covariance matrix
    m = random_mean()
    C = random_covariance()
    
    # estimte function and gradient
    F, G = fun(X, C, m)
    
    # normalise gradients
    G_norm = np.linalg.norm(G, axis=1)
    G_norm = np.maximum(G_norm, 1e-6)
    N = G / G_norm[:,None]
    
    # compute the R and q matrix for least squares solution
    R = np.sum([ np.eye(3) - np.outer(n, n)      for n    in     N    ], axis=0) 
    q = np.sum([(np.eye(3) - np.outer(n, n)) @ x for n, x in zip(N, X)], axis=0)
    
    m_est = np.linalg.solve(R, q)
    
    return C, m, m_est

N_iterations = 1000

Cs     = []
ms     = []
ms_est = []

for i in range(N_iterations):
    C, m, m_est = run_random_estimate()
    
    Cs.append(C)
    ms.append(m)
    ms_est.append(m_est)

ms     = np.array(ms)
ms_est = np.array(ms_est)

d = ms - ms_est

mean_d = d.mean(0)
cov_d  = np.cov(d, rowvar=False)

print(f"Mean difference:\n{mean_d}")
print(f"difference covariance:\n{cov_d}")
