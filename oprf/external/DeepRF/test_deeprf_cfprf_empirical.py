#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)

Script to test the hybrid approach (DeepRF and CFpRF) on empirical data.
"""

import os
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import fmin_powell
import time

import stimulus as stim
import data_synthetic as ds
import data_empirical as de

def error_function(theta, bounds, x, fn):
    for t, b in zip(theta, bounds):
        if (b[0] and t < b[0]) or (b[1] and t > b[1]):
            return np.finfo("float32").max
    yh = fn(*theta)
    if np.any(np.isnan(yh)) or np.any(np.isinf(yh)):
        return np.finfo("float32").max
    return np.sum((x - yh)**2)

def run(root="/project/2420084.01", subject=0, deeprf="resnet50_time"):

    # Load stimuli
    stimuli = stim.BensonStimulus(os.path.join(root, "resources"))
    stimuli.resample_stimulus(0.5)

    # Initialize data generators
    train_dataset = de.BensonDataset(stimuli, root, subject, "train", "time")
    valid_dataset = de.BensonDataset(stimuli, root, subject, "test", "time")
    n_voxels = len(train_dataset)
    train_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (1, 30, 30), "train", "time")
    valid_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (1, 40, 40), "test", "time")

    # Load deeprf predicted parameters
    y_deeprf = np.load(os.path.join(root, "derivatives", deeprf, "results_empirical_{}.npz".format(train_dataset.subject)))["y_train"]

    # Set fine fit bounds
    h_bound = (-2, 2)
    s_bound = (1/stimuli.pixperdeg, stimuli.radius_deg)
    x_bound = (-stimuli.radius_deg, stimuli.radius_deg)
    y_bound = (-stimuli.radius_deg, stimuli.radius_deg)
    bounds = (h_bound, s_bound, x_bound, y_bound)

    # Pre-allocate memory for results
    y_train = np.zeros((n_voxels, 4), dtype="float32") # predicted parameters train (and used for valid)
    r_train = np.zeros((n_voxels,), dtype="float32") # squared Pearson's correlation between predicted train data and observed train data
    r_valid = np.zeros((n_voxels,), dtype="float32") # squared Pearson's correlation between predicted valid data and observed valid data
    ctime = np.zeros((n_voxels,), dtype="float32") # time it took to train the model

    # Loop individual voxels
    for i in range(n_voxels):

        # Generate train data
        x = train_dataset.get_item(i)

        # Start
        start_time = time.time()

        # Coarse fit from deeprf
        coarse_fit = y_deeprf[i, :]

        # Fine fit
        y_train[i, :] = fmin_powell(error_function, coarse_fit, args=(bounds, x, train_generator.generate_prediction), disp=False)  

        # Finish
        ctime[i] = time.time() - start_time

        # Generate prediction of train data
        y = train_generator.generate_prediction(*y_train[i, :])

        # Compute explained variance of train data (might be overfitted)
        try:
            r_train[i] = pearsonr(x.flatten(), y.flatten())[0]**2
        except:
            print("warning: r could not be estimated with parameters: ", y_train[i, :])
            r_train[i] = 0

        # Generate valid data
        x = valid_dataset.get_item(i)

        # Generate prediction of the valid data using trained parameters
        y = valid_generator.generate_prediction(*y_train[i, :])

        # Compute explained variance of train data (might be overfitted)
        try:
            r_valid[i] = pearsonr(x.flatten(), y.flatten())[0]**2
        except:
            print("warning: r could not be estimated with parameters: ", y_train[i, :])
            r_valid[i] = 0

        # Print some statistics
        print("subject: {} ({:d})  voxel: {:5d}/{}  ctime: {:6.2f} ({:6.2f})  train: {:.2f} ({:.2f})  valid: {:.2f} ({:.2f})".format(
            train_dataset.subject, subject, 1+i, n_voxels, ctime[i], np.mean(ctime[:1+i]), r_train[i], np.mean(r_train[:1+i]), r_valid[i], np.mean(r_valid[:1+i])))

    # Make output folder if it does not already exist
    if not os.path.exists(os.path.join(root, "derivatives", "{}_cfprf".format(deeprf))):
        os.makedirs(os.path.join(root, "derivatives", "{}_cfprf".format(deeprf)))

    # Save results
    np.savez(os.path.join(root, "derivatives", "{}_cfprf".format(deeprf), "results_empirical_{}.npz".format(train_dataset.subject)), 
        y_train=y_train, r_train=r_train, r_valid=r_valid, ctime=ctime)

if __name__ == "__main__":
    root = "/project/2420084.01"
    #root = "/Users/jordythielen/2420084.01"
    n_subjects = 29
    for i_subject in range(n_subjects):
        run(root=root, subject=i_subject, deeprf="resnet50_chan")
