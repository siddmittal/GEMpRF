#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import numpy as np

from scipy.special import gamma
from scipy.integrate import trapz
from scipy.stats import zscore

class Voxel(object):

    def __init__(self, noise, signal):
        """
        Represents an fMRI voxel.

        args:
            noise (Noise): the noise component
            signal (Signal): the signal component
        """
        self.noise = noise
        self.signal = signal

    def __call__(self, x):
        """
        Combines the signal and noise component to generate a synthetic fMRI time series.

        args:
            x (numpy.ndarray): the stimulus of shape [volumes, height, width]
        """
        if self.noise is None:
            y = self.signal(x)
        else:
            y = self.noise(self.signal(x))
        y = 100.0 * y / np.nanmean(y) - 100.0
        return zscore(y)

class Noise(object):

    def __init__(self, w, *args):
        """
        Represents the noise component of an fMRI voxel.

        args:
            w (float): noise amplitude
            *args (object): noise components, e.g. LowFrequency, Physiological, System, Task, and/or Temporal.
        """
        self.noise = args
        self.w = w

    def __call__(self, x):
        """
        Combines the noise components to add noise to a synthetic fMRI time series.

        args:
            x (numpy.ndarray): the fMRI time-series of shape [volumes]
        """
        d = np.sum([w ** 2 for w in self.w]) ** 0.5
        y = d * x / np.sum(self.w)
        y = np.sum([w * noise(y) for w, noise in zip(self.w, self.noise)], axis=0) / d
        return y

class LowFrequency(object):

    def __init__(self, CNR, TR, T=128.0):
        """
        Represents low frequency noise (e.g., scanner drift).

        args:
            CNR (float): contrast to noise ratio
            TR (float): fMRI repetition time in seconds
            T (float): largest fluctuation to model in seconds
        """
        self.CNR = CNR
        self.T = T
        self.TR = TR

    def __call__(self, x):
        """
        Combines the noise components to add noise to a synthetic fMRI time series.

        args:
            x (numpy.ndarray): the fMRI time-series of shape [volumes]
        """
        n = x.shape[0]
        sigma = np.std(x) / self.CNR
        t = np.linspace(0.5 * np.pi / n, np.pi * (1 - 0.5 / n), n, dtype='float32')
        b = sigma * (self.T / (self.TR * n)) ** 0.5 * np.sum([np.cos(k * t) for k in range(1, int(2 * self.TR * n / self.T + 1))])
        return x + b

class Physiological(object):

    def __init__(self, CNR, TR, f=(1.17, 0.2)):
        """
        Represents physiological noise (i.e., heartrate and respiration).

        args:
            CNR (float): contrast to noise ratio
            TR (float): fMRI repetition time in seconds
            f (tuple): the modeled frequencies in Hertz for e.g. heartrate and respiration (default: (1.17, 0.2))
        """
        self.CNR = CNR
        self.TR = TR
        self.f = f

    def __call__(self, x):
        """
        Combines the noise components to add noise to a synthetic fMRI time series.

        args:
            x (numpy.ndarray): the fMRI time-series of shape [volumes]
        """
        n = x.shape[0]
        sigma = np.std(x) / self.CNR
        t = np.linspace(0, 2 * np.pi * self.TR * (n - 1), n, dtype='float32')
        b = sigma * (np.cos(self.f[0] * t) + np.sin(self.f[1] * t))
        return x + b
    
class System(object):

    def __init__(self, CNR, random_generator):
        """
        Represents system noise (i.e., measurement noise).

        args:
            CNR (float): contrast to noise ratio
            random_generator (numpy.random.RandomState): random number generator to draw random samples from
        """
        self.CNR = CNR
        self.random_generator = random_generator

    def __call__(self, x):
        """
        Combines the noise components to add noise to a synthetic fMRI time series.

        args:
            x (numpy.ndarray): the fMRI time-series of shape [volumes]
        """
        n = x.shape[0]
        sigma = np.std(x) / self.CNR
        b = self.random_generator.normal(scale=sigma, size=n).astype('float32')
        return x + b
    
class Task(object):

    def __init__(self, CNR, random_generator):
        """
        Represents task noise (e.g., participant motion). Specifically, noise is only added when the task is "on".

        args:
            CNR (float): contrast to noise ratio
            random_generator (numpy.random.RandomState): random number generator to draw random samples from
        """
        self.CNR = CNR
        self.random_generator = random_generator

    def __call__(self, x):
        """
        Combines the noise components to add noise to a synthetic fMRI time series.

        args:
            x (numpy.ndarray): the fMRI time-series of shape [volumes]
        """
        n = x.shape[0]
        sigma = np.std(x) / self.CNR
        nonzero = np.nonzero(zapsmall(x - np.mean(x)))[0]
        b = np.zeros(n, dtype='float32')
        if len(nonzero) > 0:
            b[nonzero] = self.random_generator.normal(scale=sigma * (n / len(nonzero)) ** 0.5, size=len(nonzero))
        return x + b
    
def zapsmall(x, digits=7):
    """
    args:
        digits (int): (default: 7)
    """
    if not isinstance(digits, int):
        sys.exit("invalid 'digits'")
    inan = np.isnan(x)
    if np.all(inan):
        return x
    mx = np.max(np.absolute(x[np.logical_not(inan)]))
    return np.round(x, int(np.maximum(0, digits - np.log10(mx))) if mx > 0 else digits)

class Temporal(object):

    def __init__(self, CNR, random_generator, phi=0.2):
        """
        Represents temporal noise (e.g., autocorrelatedness).

        args:
            CNR (float): contrast to noise ratio
            random_generator (numpy.random.RandomState): random number generator to draw random samples from
            phi (float): amplitude of first order component (default: 0.2)
        """
        self.CNR = CNR
        self.random_generator = random_generator
        self.phi = phi

    def __call__(self, x):
        """
        Combines the noise components to add noise to a synthetic fMRI time series.

        args:
            x (numpy.ndarray): the fMRI time-series of shape [volumes]
        """
        n = x.shape[0]
        sigma = np.std(x) / self.CNR
        b = self.random_generator.normal(scale=sigma, size=n).astype('float32')
        for t in range(1, n):
            b[t] = self.phi * b[t - 1] + b[t]
        return x + b
    
class Signal(object):

    def __init__(self, PSC, b, hemodynamic, population):
        """
        Represents the signal component of an fMRI voxel.

        args:
            PSC (float): percent signal change
            b (float): signal amplitude
            hemodynamic (object): the hemodynamic response function to convolve the population response with, e.g. DoubleGamme
            population (object): the population response, e.g. Gaussian
        """
        self.PSC = PSC
        self.b = b
        self.hemodynamic = hemodynamic
        self.population = population

    def __call__(self, x):
        """
        Combines the hemodynamic and population response to generate a noise-free synthetic fMRI time series.

        args:
            x (numpy.ndarray): the stimulus of shape [volumes, height, width]
        """
        y = self.hemodynamic(self.population(x))
        y -= np.mean(y)
        y = self.PSC * self.b * y / (100.0 * np.max(np.fabs(y)) + 1e-8) + self.b
        return y

class DoubleGamma(object):

    def __init__(self, TR, delay, T=32.0):
        """
        Represents a double gamma hemodaynamic response function.

        args:
            TR (float): fMRI repetition time in seconds
            delay (float): delay of the HRF in seconds
            T (float): length of the modeled HRF in seconds (default 32.0)
        """
        alpha_1 = 5 + delay / TR
        beta_1 = 1.0
        c = 0.1
        alpha_2 = 15 + delay / TR
        beta_2 = 1.0
        t = np.arange(0, T, TR)
        hrf = np.array( ( ( t**alpha_1 * beta_1**alpha_1 * np.exp(-beta_1 * t)) / gamma(alpha_1)) - c *
                        ( ( t**alpha_2 * beta_2**alpha_2 * np.exp(-beta_2 * t)) / gamma(alpha_2)), dtype='float32')
        hrf /= trapz(hrf)
        self.w = hrf

    def __call__(self, x):
        """
        Combines the hemodynamic and population response by means of convolution.

        args:
            x (numpy.ndarray): the signal of shape [volumes]
        """
        y = np.convolve(self.w, x)[:x.shape[0]]
        return y
        
class Gaussian(object):

    def __init__(self, FOV_x, FOV_y, sigma_x, sigma_y, x_0, y_0):
        """
        Represents a Gaussian population response.

        args:
            FOV_x (int): field of view horizontally
            FOV_y (int): field of view vertically
            sigma_x (float): size of the population response (pRF) horizontally
            sigma_y (float): size of the population response (pRF) vertically
            x_0 (float): x position of the (center of) population response (pRF)
            y_0 (float): y position of the (center of) population response (pRF)
        """
        y, x = np.mgrid[-FOV_y / 2 + 0.5 : FOV_y / 2 + 0.5 : 1,
                        -FOV_x / 2 + 0.5 : FOV_x / 2 + 0.5 : 1].astype('float32')
        self.w = np.flipud(1.0 * np.exp(-((x - x_0) ** 2 / (2 * sigma_x ** 2) + (y - y_0) ** 2 / (2 * sigma_y ** 2))))

    def __call__(self, x):
        """
        Combines the stimulus and the population to generate a a population response.

        args:
            x (numpy.ndarray): the stimulus of shape [volumes, height, width]
        """
        y = np.sum(self.w * x, axis=(1, 2))
        return y

class SyntheticDataGenerator(object):

    def __init__(self, stimulus, data_size=1024, seed=0, train_test_split=None, cat_dim="time"):
        """
        Represents a data generator for synthetic fMRI time series.

        args:
            stimulus: (object) stimulus class containing the stimulus itself as a numpy.ndarray of shape [runs, voxels, height, width], e.g. stimulus.BensonStimulus
            data_size: (int) number of voxels (default: 1024)
            seed: (int or tuple) seed for the random number generator (default: 0)
            train_test_split: (str) load the training ("train") or testing ("test") split, or all (None) (default: None)
            cat_dim: (str) dimension along which to concatenate individual stimulus runs, "time" for time dimension or "chan" for channel/runs dimension (default: "time")
        """
        if isinstance(seed, int) or len(seed) == 1:
            seed = (seed, seed, seed)
        self.random_generator_t = np.random.RandomState(seed[0]) # used to generate random parameters
        self.random_generator_x = np.random.RandomState(seed[1]) # used to generate random signal/noise
        self.random_generator_y = np.random.RandomState(seed[2]) # used to generate predictions
        self.stimulus = stimulus
        self.data_size = data_size
        self.train_test_split = train_test_split
        self.cat_dim = cat_dim

    def __len__(self):
        return self.data_size

    def generate_random(self, add_noise=True):
        """
        Generate a random fMRI time series with random parameters.

        args:
            add_noise (bool): whether or not to add noise to the signal (default: True)

        returns:
            data (numpt.ndarray): the simulated fMRI voxel of shape [1, volumes] if cat_dim="time" or [runs/channels, voxel] if cat_dim="chan"
            targets (numpt.ndarray): the underlying ground truth parameters of in vector format with [delay, sigma, x_0, y_0]
        """

        # Sample parameters
        delay = self.random_generator_t.uniform(-2, 2)
        sigma = self.random_generator_t.uniform(1/self.stimulus.pixperdeg, self.stimulus.radius_deg)
        x_pos = self.random_generator_t.uniform(-self.stimulus.radius_deg, self.stimulus.radius_deg)
        y_pos = self.random_generator_t.uniform(-self.stimulus.radius_deg, self.stimulus.radius_deg)

        # Make sure RFs are within the radius to prevent zero signal
        while np.sqrt(x_pos**2 + y_pos**2) > self.stimulus.radius_deg:
            x_pos = self.random_generator_t.uniform(-self.stimulus.radius_deg, self.stimulus.radius_deg)
            y_pos = self.random_generator_t.uniform(-self.stimulus.radius_deg, self.stimulus.radius_deg)

        # Generate signal
        self.data = self.generate_prediction(delay, sigma, x_pos, y_pos, add_noise)

        # Set target vector
        self.targets = np.array([delay, sigma, x_pos, y_pos]).astype("float32")

        return self.data, self.targets

    def generate_prediction(self, delay, sigma, x_pos, y_pos, add_noise=False):
        """
        Generate an fMRI time series with specific parameters.

        args:
            delay (float): delay of the HRF in seconds
            sigma (float): size of the pRF in visual degrees
            x_pos (float): x position of the pRF in visual degrees
            y_pos (float): y position of the pRF in visual degrees
            add_noise (bool): whether or not to add noise to the signal (default: True)

        returns:
            data (numpt.ndarray): the simulated fMRI voxel of shape [1, volumes] if cat_dim="time" or [runs/channels, voxel] if cat_dim="chan"
        """

        # Generate signal
        doublegamma = DoubleGamma(self.stimulus.tr, delay)
        gaussian = Gaussian(self.stimulus.width_pix, self.stimulus.width_pix, 
            sigma*self.stimulus.pixperdeg, sigma*self.stimulus.pixperdeg, 
            x_pos*self.stimulus.pixperdeg, y_pos*self.stimulus.pixperdeg)
        if add_noise:
            percentsignalchange = self.random_generator_y.normal(3.0, 0.25)
        else:
            percentsignalchange = 3.0
        bias = 800.0
        signal = Signal(percentsignalchange, bias, doublegamma, gaussian)

        # Generate noise
        if add_noise:
            contrasttonoiseratio = np.exp(self.random_generator_y.uniform(np.log(0.5), np.log(2.0)))
            lowfrequency = LowFrequency(contrasttonoiseratio, self.stimulus.tr)
            physiological = Physiological(contrasttonoiseratio, self.stimulus.tr)
            system = System(contrasttonoiseratio, self.random_generator_y)
            task = Task(contrasttonoiseratio, self.random_generator_y)
            temporal = Temporal(contrasttonoiseratio, self.random_generator_y)
            noise = Noise(self.random_generator_y.rand(5), lowfrequency, physiological, system, task, temporal)
        else:
            noise = None

        # Create voxel
        voxel = Voxel(noise, signal)

        # Generate observed activity
        self.data = np.zeros((len(self.stimulus.runs), self.stimulus.n_volumes), dtype="float32")
        for i in range(len(self.stimulus.runs)):
            self.data[i, :] = voxel(self.stimulus.stimulus[i, :, :, :]).astype("float32")

        # Split data
        if self.train_test_split is not None:
            self.split_data()

        # Concatenate over time
        if self.cat_dim == "time":
            self.data = np.reshape(self.data, (1, -1))

        return self.data

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
        data = np.zeros((len(self.stimulus.runs), self.stimulus.n_volumes//2-22), dtype="float32")
        for i, run in enumerate(self.stimulus.runs):

            # Split train and test
            if self.train_test_split == "train" and i % 2 == 0 or self.train_test_split == "test" and i % 2 == 1:
                if "RETBAR" in run:
                    data[i, :] = self.data[i, 16:self.stimulus.n_volumes//2-6]  # baseline 12 sec pre and post run and 8 seconds half-way
                else:
                    data[i, :] = self.data[i, 22:self.stimulus.n_volumes//2]  # baseline 16 sec pre and post run
            else:
                if "RETBAR" in run:
                    data[i, :] = self.data[i, self.stimulus.n_volumes//2+6:-16]  # baseline 12 sec pre and post run and 8 seconds half-way
                else:
                    data[i, :] = self.data[i, self.stimulus.n_volumes//2:-22]  # baseline 16 sec pre and post run

        # Make the order of directions equal for train and test split in BAR runs
        if self.train_test_split == "test":
            data[[-2, -1], :] = data[[-1, -2], :]

        self.data = data
