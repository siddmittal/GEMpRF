#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os
import h5py
import numpy as np

from scipy.stats import zscore
import nibabel as nib


class BensonDataset():
    """
    Benson, N. C., Jamison, K. W., Arcaro, M. J., Vu, A. T., Glasser, M. F., Coalson, T. S., ... & Kay, K. (2018).
    The Human Connectome Project 7 Tesla retinotopy dataset: Description and population receptive field analysis.
    Journal of vision, 18(13), 23-23.

    Data downloaded from ConnectomeDB: https://db.humanconnectome.org/
    AnalysePRF results including the Wang 2015 atlas from OSF: https://osf.io/bw9ec/

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

    subjects = ["126426", "130114", "130518", "134627", "135124", "146735", "165436", "167440", "177140", "180533",
                "186949", "193845", "239136", "360030", "385046", "401422", "463040", "550439", "552241", "644246", 
                "654552", "757764", "765864", "878877", "905147", "943862", "971160", "973770", "995174"]
    rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]

    def __init__(self, stimulus, root, subject, train_test_split=None, cat_dim="time"):
        """
        Sets up a specific empirical dataset of an individual participant of the HCP retinotopy dataset.

        args:
            stimulus: (stimulus.BensonStimulus) stimulus class containing the stimulus itself as a numpy.ndarray of shape [runs, voxels, height, width]
            root: (str) path to the dataset
            subject: (int or str) specific subject to test, either as string identifier, or integer index
            train_test_split: (str) load the training ("train") or testing ("test") split, or all (None) (default: None)
            cat_dim: (str) dimension along which to concatenate individual stimulus runs, "time" for time dimension or "chan" for channel/runs dimension (default: "time")

        notes:
            The data will be represented as a numpy.ndarray of shape [runs/channels, volumes, voxels]
        """
        self.stimulus = stimulus
        self.root = root
        self.train_test_split = train_test_split
        self.cat_dim = cat_dim
        if isinstance(subject, int):
            self.subject = self.subjects[subject]
        else:
            if subject in self.subjects:
                self.subject = subject
            else:
                raise Exception("Subject not found!")

        # Read voxel labels from atlas
        with h5py.File(os.path.join(root, "derivatives", "analyseprf", "atlas.mat"), "r") as fid:
            atlas = np.array(fid["wang2015"]).astype("int").flatten()
            labels = ["".join(chr(c[0]) for c in fid[ref[0]]) for ref in fid["wang2015labels"]]
        labels_idx = [i for i, lab in enumerate(labels) if lab in self.rois]

        # Make ROI mask
        self.mask = np.full(atlas.size, False)
        for lab in labels_idx:
            self.mask[atlas == lab] = True
        self.n_voxels = np.sum(self.mask)

        # Loop runs
        self.data = np.zeros((len(self.stimulus.runs), self.stimulus.n_volumes, self.n_voxels), dtype="float32")
        for i, run in enumerate(self.stimulus.runs):

            # Handle phase encoding direction
            if run in ["RETCCW", "RETEXP", "RETBAR1"]:
                run = "tfMRI_" + run + "_7T_AP"  
            else:
                run = "tfMRI_" + run + "_7T_PA"

            # Load data
            self.data[i, :, :] = nib.load(os.path.join(root, "sourcedata", self.subject, "MNINonLinear", "Results", run, "{}_Atlas_MSMAll_hp2000_clean.dtseries.nii".format(run))).get_fdata()[:, self.mask]

            # To percent signal change
            self.data[i, :, :] = 100.0 * self.data[i, :, :] / np.mean(self.data[i, :, :], axis=0, keepdims=True) - 100.0  

        # Split data
        if train_test_split is not None:
        	self.split_data()

        # Concatenate runs over time
        if self.cat_dim == "time":
            self.data = np.reshape(self.data, (1, -1, self.n_voxels))

        # Preprocess data
        self.data = zscore(self.data, axis=1)

    def __len__(self):
        """
        Returns the size of the datasets, i.e., the number of voxels.

        returns:
            length (int): the number of voxels in the dataset
        """
        return self.data.shape[2]

    def get_item(self, index):
        """
        Returns a specific voxel from the dataset. 

        args:
            index (int): the index of the voxel to return

        returns:
            data (numpt.ndarray): the empirical fMRI voxel of shape [1, volumes] if cat_dim="time" or [runs/channels, voxel] if cat_dim="chan"
        """
        return self.data[:, :, index]  # channels, samples, voxel

    def split_data(self):
        """
        Splits the data in a training or testing split (based on the train_test_split parameter).
        Specifically, the training split will include the first halfs of the odd runs and the second 
        halfs of the even runs. The testing split will contain the remaining halfs. Additionally, 
        blank intervals are removed to align the underlying stimulus protocol. Finally, for the 
        testing split, the last two bar runs are sitched in order to make the stimulus protocal
        again identical by means of the specific bar directions within those runs. 
        """

    	# Loop runs
        data = np.zeros((len(self.stimulus.runs), self.stimulus.n_volumes//2-22, self.n_voxels), dtype="float32")
        for i, run in enumerate(self.stimulus.runs):

            # Split train and test
            if self.train_test_split == "train" and i % 2 == 0 or self.train_test_split == "test" and i % 2 == 1:
                if "RETBAR" in run:
                    data[i, :, :] = self.data[i, 16:self.stimulus.n_volumes//2-6, :]  # baseline 12 sec pre and post run and 8 seconds half-way
                else:
                    data[i, :, :] = self.data[i, 22:self.stimulus.n_volumes//2, :]  # baseline 16 sec pre and post run
            else:
                if "RETBAR" in run:
                    data[i, :, :] = self.data[i, self.stimulus.n_volumes//2+6:-16, :]  # baseline 12 sec pre and post run and 8 seconds half-way
                else:
                    data[i, :, :] = self.data[i, self.stimulus.n_volumes//2:-22, :]  # baseline 16 sec pre and post run

        # Make the order of directions equal for train and test split in BAR runs
        if self.train_test_split == "test":
            data[[-2, -1], :, :] = data[[-1, -2], :, :]

        self.data = data
