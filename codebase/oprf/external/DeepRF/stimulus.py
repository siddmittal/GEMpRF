#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os
import h5py
import numpy as np
from scipy.ndimage import zoom

class BensonStimulus():
    """
    Benson, N. C., Jamison, K. W., Arcaro, M. J., Vu, A. T., Glasser, M. F., Coalson, T. S., ... & Kay, K. (2018).
    The Human Connectome Project 7 Tesla retinotopy dataset: Description and population receptive field analysis.
    Journal of vision, 18(13), 23-23.

    AnalysePRF results including the stimulus bitmaps from OSF: https://osf.io/bw9ec/
        
    From page 3 Figure 1:
    Each run lasted 300 seconds in total, there are 6 runs in total.
    Each CCW/CW/EXP/CON run starts and ends with 22 seconds baseline.
    Each BAR1/BAR2 run starts and ends with 16 seconds baseline and has 12 seconds baseline half-way through.
    Each run performs eight stimulus sweeps of 32 seconds within a run.
    Sweeps are:
        - for CCW a counter-clockwise rotating wedge;
        - for CW a clock-wise rotating wedge;
        - for EXP an expanding ring;
        - for CON an contracting ring;
        - for BAR1/BAR2 eight bar directions in the order R, U, L, D, UR, UL, DL, DR (Right, Left, Up, Down).
    Note: The two BAR runs presented identical stimuli, i.e. the stimulus protocol in BAR1 and BAR2 is the same.
    """

    runs = ["RETCCW", "RETCW", "RETEXP", "RETCON", "RETBAR1", "RETBAR2"]
    n_volumes = 300
    width_pix = 200
    width_cm = 28.5
    distance_cm = 101.5
    radius_deg = 8
    pixperdeg = width_pix / 2. / (np.arctan(width_cm / 2. / distance_cm) / np.pi * 180.0)
    tr = 1.0
    scale = 1.0

    def __init__(self, path):
        """
        Sets up the stimulus and stimulus properties for the HCP retinotopy dataset. 

        args:
            path (str): path to the dataset

        notes:
            The stimulus will be represented as a numpy.ndarray of shape [runs, volumes, height, width]
        """
        self.path = path

        # Loop runs
        self.stimulus = np.zeros((len(self.runs), self.n_volumes, self.width_pix, self.width_pix), dtype="float32") 
        for i, run in enumerate(self.runs):

            # Remove number indicating bar run 1 or 2 (the stimuli of these are identical)
            if "RETBAR" in run and len(run) > 6:
                run = run[:6]  

            # Read stimulus
            with h5py.File(os.path.join(self.path, run + "small.mat"), "r") as fid:
                self.stimulus[i, :, :, :] = (np.array(fid["stim"]) > 127).astype("float32")

    def resample_stimulus(self, scale):
        """
        Resamples the stimulus along the spatial dimensions (width and height).

        args:
            scale (float): scale to resample the stimulus with
        """
        self.scale = scale

        self.stimulus = self.stimulus.reshape((-1, self.width_pix, self.width_pix))

        stimulus = np.zeros((self.stimulus.shape[0], int(np.round(self.width_pix * scale)), int(np.round(self.width_pix * scale))), dtype="float32")
        for i in range(self.stimulus.shape[0]):
            stimulus[i, :, :] = zoom(self.stimulus[i, :, :], scale, mode="nearest")
        self.stimulus = stimulus

        self.width_pix = self.stimulus.shape[2]
        self.pixperdeg = self.width_pix / 2. / (np.arctan(self.width_cm / 2. / self.distance_cm) / np.pi * 180.0)

        self.stimulus = self.stimulus.reshape((len(self.runs), -1, self.width_pix, self.width_pix))
