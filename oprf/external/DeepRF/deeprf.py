#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import data_synthetic as ds

    
class MyDatasetGenerator(Dataset):
    
    def __init__(self, generator):
        """
        Custom dataset that calls the synthetic data generator. I.e., a dataset 
        for pytorch that accepts numpy input.

        args:
            generator (object): the synthetic data generator
        """
        self.generator = generator
        
    def __len__(self):
        """
        Length of the dataset.

        returns:
            length (int): number of voxels in the dataset, note, this is not
            really used, only to break an epoch. 
        """
        return len(self.generator)
    
    def __getitem__(self, idx):
        """
        Sample a new voxel.

        args:
            idx (int): index of the voxel, note, this is not at all used, as 
            the generator wil always sample new data

        returns:
            x (torch.FloatTensor): the voxel time series with size [channels, samples]
            t (torch.FloatTensor): the target labels 
        """
        x, t = self.generator.generate_random()
        x = torch.FloatTensor(x.astype("float32"))
        t = torch.FloatTensor(t.astype("float32"))
        return x, t


class MyDataset(Dataset):
    
    def __init__(self, data):
        """
        Custom dataset that calls the empircal dataset. I.e., a dataset for pytorch 
        that accepts numpy input.

        args:
            data (numpy.ndarray): the empircal dataset of shape [voxels, channels, samples]
        """
        self.data = data
        
    def __len__(self):
        """
        Length of the dataset.

        returns:
            length (int): number of voxels in the dataset
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a new voxel.

        args:
            idx (int): index of the voxel

        returns:
            x (torch.FloatTensor): the voxel time series with size [channels, samples]
        """
        x = self.data[idx, :, :]
        x = torch.FloatTensor(x.astype("float32"))
        return x


class Trainer():

    def __init__(self, model, path, train_generator, valid_generator, batch_size=64, learning_rate=0.001, betas=(0.9, 0.999), n_epochs=10, suffix=""):
        """
        Represents the pipelines to train a DeepRF model.

        args:
            model (torch.nn.Module): the neural net
            path (str): path to save the model to
            train_generator (object): synthetic data generator for training
            valid_generator (object): synthetic data generator for validation
            batch_size (int): batch size (default 64)
            learning_rate (float): learning rate for the optimizer (default: 0.001)
            betas (tuple): parameters for Adam optimizer running averages (default: (0.9, 0.999))
            n_epochs (int): number of training epochs to run (default: 10)
            suffix (str): suffix to add to saved model outputs (default: "")
        """
        self.model = model
        if len(suffix) > 0: suffix = "_" + suffix
        self.path = os.path.join(path, self.model.name + suffix)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.train_generator = train_generator
        self.valid_generator = valid_generator

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.n_epochs = n_epochs

        self.losses = None

    def train_model(self):
        """
        Trains the DeepRF model.

        returns:
            train_losses (numpy.ndarray) the training losses per epoch
            valid_losses (numpy.ndarray) the validation losses per epoch

        notes:
            saves the model as trained after n_epochs as "model.pt"
            saves the "best" model by means of least validation error as "best_model.pt"
            within these saves, one can find the model as "model_state_dict", the epoch 
            as "epoch" and the time it took to train in "ctime"
        """
        start_time = time.time()

        # Select GPU if avaiable, otherwise use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print("Training", self.model.name, "on", device)

        # Set up optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas)

        # Set up datasets
        data = dict(
            train=DataLoader(MyDatasetGenerator(self.train_generator), batch_size=self.batch_size), 
            valid=DataLoader(MyDatasetGenerator(self.valid_generator), batch_size=self.batch_size))

        # Initialize losses
        self.losses = dict(
            train=np.zeros(self.n_epochs), 
            valid=np.zeros(self.n_epochs))

        # Keep track of the best model by means of the validation loss
        best_loss = np.inf

        # Loop epochs
        for i_epoch in range(self.n_epochs):

            # Loop training and validation
            for phase in ["train", "valid"]:

                # Loop batches
                for x, t in data[phase]:
                    x, t = x.to(device), t.to(device)

                    # Training
                    if phase == "train":
                        self.model.train()
                        optimizer.zero_grad()
                        y = self.model(x)
                        loss = criterion(y, t)
                        loss.backward()
                        optimizer.step()

                    # Validation
                    else:
                        self.model.eval()
                        with torch.no_grad():
                            y = self.model(x)
                            loss = criterion(y, t)

                    # Save losses
                    self.losses[phase][i_epoch] += (loss.item() / len(data[phase]))
            
            # Print losses
            print("epoch: {:02d} train_loss: {:.04f} valid_loss: {:.04f}".format(
                1 + i_epoch, self.losses["train"][i_epoch], self.losses["valid"][i_epoch]))

            # Save a snapshot of the best model so far by means of validation loss
            if self.losses["valid"][i_epoch] < best_loss:
                ctime = time.time() - start_time
                best_loss = self.losses["valid"][i_epoch]
                torch.save({"epoch":i_epoch, "ctime":ctime, "model_state_dict":self.model.state_dict()}, os.path.join(self.path, "best_model.pt"))

        # Compute the total training time
        ctime = time.time() - start_time

        # Save model
        torch.save({"epoch":i_epoch, "ctime":ctime, "model_state_dict":self.model.state_dict()}, os.path.join(self.path, "model.pt"))

        # Return losses
        return self.losses["train"], self.losses["valid"]

    def save_losses(self):
        """
        Save the losses to disk as losses.npz.
        """
        np.savez(os.path.join(self.path, "losses.npz"), train=self.losses["train"], valid=self.losses["valid"])

    def plot_losses(self):
        """
        Plot the losses and save the figure as losses.pdf.
        """
        plt.figure(figsize=(15, 3))
        plt.plot(np.arange(self.n_epochs), self.losses["train"], label="train")
        plt.plot(np.arange(self.n_epochs), self.losses["valid"], label="valid")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, "losses.pdf"), dpi=300, transparent=True, bbox_inches="tight")


class Tester():

    def __init__(self, model, path, suffix=""):
        """
        Represents the pipelines to test/apply a DeepRF model.

        args:
            model (torch.nn.Module): the neural net
            path (str): path to save the model to
            suffix (str): suffix to add to saved model outputs (default: "")
        """
        self.model = model
        if len(suffix) > 0: suffix = "_" + suffix
        self.path = os.path.join(path, self.model.name + suffix)

    def test_model(self, x, test_time_dropout=False):
        """
        Apply DeepRF to an fMRI voxel represented as numpy.ndarray.

        args:
            x (numpy.ndarray): the fMRI time series with shape [channels, samples]
            test_time_dropout (bool): whether or not to sample parameters with 
            test-time dropout to compute a distribution over the parameters for
            uncertainty estimation (default: False)
        """

        # Select GPU if avaiable, otherwise use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Append dimensions if needed to represent [batch, channels, samples]
        if x.ndim == 1:
            x = torch.from_numpy(x[np.newaxis, np.newaxis, :])
        if x.ndim == 2:
            x = torch.from_numpy(x[np.newaxis, :, :])

        # Push data to device
        x = x.to(device)

        # Apply DeepRF
        self.model.eval()
        if test_time_dropout:
            self.model.apply(apply_dropout)
        with torch.no_grad():
            y = self.model(x).numpy()

        return y

    def test_model_batch(self, X, batch_size=64):
        """
        Apply DeepRF to a dataset of fMRI voxels represented as numpy.ndarray.

        args:
            X (numpy.ndarray): the fMRI time series with shape [voxels, channels, samples]
            batch_size (int): batch size (default: 64)
        """

        # Select GPU if avaiable, otherwise use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print("Testing", self.model.name, "on", device)

        # Set up dataset
        data = DataLoader(MyDataset(X), batch_size=batch_size)

        # Initialize predictions
        y = np.zeros((X.shape[0], 4), dtype="float32")

        # Apply DeepRF
        self.model.eval()
        with torch.no_grad():

            # Loop batches
            for i, x in enumerate(data):
                x = x.to(device)
                y[i * batch_size:(1 + i) * batch_size, :] = self.model(x).cpu()

        return y

    def load_model(self, model="best_model"):
        """
        Loads a (pre-trained) model.

        args:
            model (str): filename of the model (default: "best_model")
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(os.path.join(self.path, "{}.pt".format(model)), map_location=device)["model_state_dict"])


def apply_dropout(lay):
    """
    Enable dropout (i.e., set it in training modus).

    args:
        lay (torch.nn.Module): the layer to enable dropout for
    """
    if type(lay) == nn.Dropout:
        lay.train()

def get_alexnet(num_in_channels=1, num_outputs=4):
    """
    Get the AlexNet model architecture.
    has about 61M parameters.

    args:
        num_in_channels (int): number of input channels (default: 1)
        num_outputs: number of outputs (default: 4)
    """
    from net_alexnet_1d import AlexNet
    return AlexNet(num_in_channels=num_in_channels, num_outputs=num_outputs)

def get_resnet18(num_in_channels=1, num_outputs=4):
    """
    Get the 18-layer ResNet model architecture.
    has about 11M parameters.

    args:
        num_in_channels (int): number of input channels (default: 1)
        num_outputs: number of outputs (default: 4)
    """
    from net_resnet_1d import PreActResNet18
    return PreActResNet18(num_in_channels=num_in_channels, num_outputs=num_outputs)

def get_resnet34(num_in_channels=1, num_outputs=4):
    """
    Get the 34-layer ResNet model architecture.
    has about 21M parameters.

    args:
        num_in_channels (int): number of input channels (default: 1)
        num_outputs: number of outputs (default: 4)
    """
    from net_resnet_1d import PreActResNet34
    return PreActResNet34(num_in_channels=num_in_channels, num_outputs=num_outputs)

def get_resnet50(num_in_channels=1, num_outputs=4):
    """
    Get the 50-layer ResNet model architecture.
    has about 25M parameters.

    args:
        num_in_channels (int): number of input channels (default: 1)
        num_outputs: number of outputs (default: 4)
    """
    from net_resnet_1d import PreActResNet50
    return PreActResNet50(num_in_channels=num_in_channels, num_outputs=num_outputs)

def get_resnet101(num_in_channels=1, num_outputs=4):
    """
    Get the 101-layer ResNet model architecture.
    has about 43M parameters.

    args:
        num_in_channels (int): number of input channels (default: 1)
        num_outputs: number of outputs (default: 4)
    """
    from net_resnet_1d import PreActResNet101
    return PreActResNet101(num_in_channels=num_in_channels, num_outputs=num_outputs)

def get_resnet152(num_in_channels=1, num_outputs=4):
    """
    Get the 152-layer ResNet model architecture.
    has about 58M parameters.

    args:
        num_in_channels (int): number of input channels (default: 1)
        num_outputs: number of outputs (default: 4)
    """
    from net_resnet_1d import PreActResNet152
    return PreActResNet152(num_in_channels=num_in_channels, num_outputs=num_outputs)
