#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)

Script that trains DeepRF on CPU or GPU using synthetic data.
"""

import os
import numpy as np
import time

import deeprf
import data_synthetic as ds
import stimulus as stim

def run(root="/project/2420084.01", n_voxels=1024, cat_dim="time", n_layers=50):

    # Load stimuli
    stimuli = stim.BensonStimulus(os.path.join(root, "resources"))
    stimuli.resample_stimulus(0.5)

    # Initialize synthetic data generators
    train_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (0, 10, 10), "train", cat_dim)
    valid_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (0, 20, 20), "test", cat_dim)

    # Set up model
    if cat_dim == "time":
        num_in_channels = 1
    else:
        num_in_channels = 6
    if n_layers == 50:
        model = deeprf.get_resnet50(num_in_channels=num_in_channels)
    elif n_layers == 34:
        model = deeprf.get_resnet34(num_in_channels=num_in_channels)
    elif n_layers == 18:
        model = deeprf.get_resnet18(num_in_channels=num_in_channels)
    else:
        raise Exception("Unknown n_layers:", n_layers)

    # Set up trainer
    trainer = deeprf.Trainer(model, os.path.join(root, "derivatives"), train_generator, valid_generator, n_epochs=200, suffix=cat_dim)

    # Train model
    start_time = time.time()
    train_loss, valid_loss = trainer.train_model()
    ctime = time.time() - start_time
    print("Training took {:.2f} seconds ({:.2f} minutes or {:.2f} hours)".format(ctime, ctime/60, ctime/60/60))

    # Save model
    trainer.save_losses()
    trainer.plot_losses()

if __name__ == "__main__":
    root = "/project/2420084.01"
    #root = "/Users/jordythielen/2420084.01"
    run(root=root, n_voxels=1024, cat_dim="chan", n_layers=50)
