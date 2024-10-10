#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:24:09 2023

@author: mwoletz

-> Generating trends, performing orthogonal projections and fitting data to a grid

"""

import numpy as np
import time
import matplotlib.pyplot as plt

# Define a function to generate trends
def makeTrends(nFrames=300, nUniqueRep=1, nDCT=3):
    # Generates trends similar to 'rmMakeTrends' from the 'params' struct in mrVista.
    # nFrames: Number of frames in the trend.
    # nUniqueRep: Number of unique repetitions.
    # nDCT: Number of discrete cosine transform components.

    stims = [1]
    tf = [int(nFrames / nUniqueRep)]
    ndct = [2 * nDCT + 1]
    t = np.zeros((np.sum(tf), np.max([np.sum(ndct), 1])))
    start1 = np.hstack([0, np.cumsum(tf)])
    start2 = np.hstack([0, np.cumsum(ndct)])
    dcid = np.zeros(len(stims))

    for n in range(len(stims)):
        tc = np.linspace(0, 2.*np.pi, tf[n])[:, None]
        t[start1[n]:start1[n+1], start2[n]:start2[n+1]] = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))
        dcid[n] = start2[n]
        
    nt = t.shape[1]
    return t, nt, dcid

# Set the number of frames
N = 300

# Generate trends using the makeTrends function
t, ntm, dcid = makeTrends(N)

# Perform QR decomposition on t
q, r = np.linalg.qr(t)
q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
R = q

# Set the number of targets
N_targets = 1000

# Create trends and perform QR decomposition
trends = np.vstack([np.linspace(0, 1, N)**i for i in range(4)]).T
q, r = np.linalg.qr(trends)
q *= np.sign(q[0, 0])
targets = np.random.randn(N, N_targets) * 20 + q @ np.random.randn(q.shape[1], N_targets) * 100 # np.random.randn(q.shape[1], N_targets) = (4 x 1000)

# Set grid size
grid_size = 100

# Generate a random grid
grid = np.random.randn(N, grid_size)

# Initialize an orthogonal grid
grid_ortho = np.zeros_like(grid)

# Measure starting time
t_start = time.time_ns()

# Calculate orthogonal projection using O = (I - R @ R^T)
O = (np.eye(N) - R @ R.T)

# Iterate through the grid columns
for i in range(grid_size):
    x = grid[:, i] #s
    x = O @ x #s_prime 
    x /= np.sqrt(x @ x)
    grid_ortho[:, i] = x

# Calculate squared projection of orthogonal grid onto targets
proj_squared = (grid_ortho.T @ targets)**2 #I guess, (y^T . s_prime)^2
best_fit_proj = np.argmax(proj_squared, axis=0)

# Measure ending time
t_end = time.time_ns()

# Calculate time taken for projection
t_proj = t_end - t_start

# Print projection time
print(f'Time for projection: {t_proj * 1e-9}')

# Initialize an array for errors
errors = np.zeros((grid_size, N_targets))

# Measure starting time for fitting
t_start = time.time_ns()

# Iterate through the grid columns
for i in range(grid_size):
    # Construct a matrix X
    X = np.hstack((grid[:, i][:, None], t))
    
    # Calculate the matrix M
    M = np.eye(N) - X @ np.linalg.inv(X.T @ X) @ X.T
    
    # Iterate through targets
    for j in range(N_targets):
        y = targets[:, j]
        ee = y @ M @ y
        errors[i, j] = ee
        
# Find indices of minimum errors for each target
best_fit_errors = np.argmin(errors, axis=0)

# Measure ending time for fitting
t_end = time.time_ns()

# Calculate time taken for fitting
t_fit = t_end - t_start

# Print fitting time and speedup
print(f'Time for fit: {t_fit * 1e-9}')
print(f'Speedup: {t_fit / t_proj}')

# Check if the best fits from both methods are equal
print('Best fits are equal:', np.all(best_fit_errors == best_fit_proj))
