#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)

Script to extract response properties of the target distribution of the synthetic data. 
"""

import os
import numpy as np
from scipy.stats import pearsonr

import stimulus as stim
import data_synthetic as ds

def run(root="/project/2420084.01", n_voxels=1024):

    # Load stimuli
    stimuli = stim.BensonStimulus(os.path.join(root, "resources"))
    stimuli.resample_stimulus(0.5)

    # Initialize synthetic data generators
    train_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (1, 30, 30), "train", "time")
    valid_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (1, 40, 40), "test", "time")

    # Pre-allocate memory for results
    t_train = np.zeros((n_voxels, 4), dtype="float32") # target parameters train
    t_valid = np.zeros((n_voxels, 4), dtype="float32") # target parameters valid
    y_train = np.zeros((n_voxels, 4), dtype="float32") # predicted parameters train (and used for valid)
    r_train = np.zeros((n_voxels,), dtype="float32") # squared Pearson's correlation between predicted train data and observed train data
    r_valid = np.zeros((n_voxels,), dtype="float32") # squared Pearson's correlation between predicted valid data and observed valid data
    ctime = np.zeros((n_voxels,), dtype="float32") # time it took to train the model

    # Loop individual voxels
    for i in range(n_voxels):

        # Generate train data
        x, t_train[i, :] = train_generator.generate_random()

        # Fit parameters
        y_train[i, :] = t_train[i, :]

        # Generate prediction of train data
        y = train_generator.generate_prediction(*y_train[i, :])

        # Compute explained variance of train data (might be overfitted)
        try:
            r_train[i] = pearsonr(x.flatten(), y.flatten())[0]**2
        except:
            print("warning: r could not be estimated with parameters: ", y_train[i, :])
            r_train[i] = 0

        # Generate valid data
        x, t_valid[i, :] = valid_generator.generate_random()

        # Generate prediction of the valid data using trained parameters
        y = valid_generator.generate_prediction(*y_train[i, :])

        # Compute explained variance of train data (might be overfitted)
        try:
            r_valid[i] = pearsonr(x.flatten(), y.flatten())[0]**2
        except:
            print("warning: r could not be estimated with parameters: ", y_train[i, :])
            r_valid[i] = 0

        # Print some statistics
        print("voxel {:5d}/{} train: {:.2f} ({:.2f}) valid: {:.2f} ({:.2f})".format(1+i, n_voxels, r_train[i], np.mean(r_train[:1+i]), r_valid[i], np.mean(r_valid[:1+i])))

    # Make output folder if it does not already exist
    if not os.path.exists(os.path.join(root, "derivatives", "target")):
        os.makedirs(os.path.join(root, "derivatives", "target"))

    # Save results
    np.savez(os.path.join(root, "derivatives", "target", "results_synthetic.npz"), 
        t_train=t_train, t_valid=t_valid, y_train=y_train, r_train=r_train, r_valid=r_valid, ctime=ctime)


if __name__ == "__main__":
    #root = "/project/2420084.01"
    root = "/Users/jordythielen/2420084.01"
    run(root=root, n_voxels=1024)
