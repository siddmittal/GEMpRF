#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:58:19 2021

@author: mwoletz
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import loadmat


class PRFMeasurement(object):
    def __init__(self, x0: ArrayLike = None, y0: ArrayLike = None, scale0: ArrayLike = None, variance_explained0: ArrayLike = None, mask: ArrayLike = None, gain: ArrayLike = None):
        assert len(x0) == len(y0) == len(scale0) == len(
            variance_explained0), 'x0, y0, scale0 and variance_explained0 need to be of same length.'
        self.__x0 = np.array(x0)
        self.__y0 = np.array(y0)
        self.__scale0 = np.array(scale0)
        self.__variance_explained0 = np.array(variance_explained0)
        self.__gain0 = np.array(gain)
        self.__additional_mask0 = np.array(mask, dtype=bool) if mask is not None else None
        self.set_mask(additional_mask=mask)

    @property
    def x0(self):
        return self.__x0

    @property
    def y0(self):
        return self.__y0

    @property
    def X0(self):
        return np.vstack((self.x0.flatten(), self.y0.flatten())).T

    @property
    def scale0(self):
        return self.__scale0

    @property
    def variance_explained0(self):
        return self.__variance_explained0

    @property
    def eccentricity0(self):
        return np.sqrt(self.x0**2 + self.y0**2)

    @property
    def gain0(self):
        return self.__gain0

    @property
    def polar_angle0(self):
        return np.arctan2(self.y0, self.x0)

    @property
    def N0(self):
        return len(self.__x0)

    @property
    def N(self):
        return len(self.x)

    @property
    def mask(self):
        return self.__mask

    @property
    def additional_mask0(self):
        return self.__additional_mask0

    def set_mask(self, variance_explained=None, eccentricity=None, additional_mask: ArrayLike = None):

        # remove any voxels that are not finite in value (includes NaNs)
        self.__mask = np.isfinite(self.x0) & np.isfinite(self.y0) & np.isfinite(
            self.scale0) & np.isfinite(self.variance_explained0)

        if variance_explained is not None:
            self.__mask = self.__mask & (self.variance_explained0 >= variance_explained)

        if eccentricity is not None:
            self.__mask = self.__mask & (self.eccentricity0 <= eccentricity)

        if additional_mask is not None:
            if len(additional_mask) == len(self.x0):
                self.__mask = self.__mask & np.array(additional_mask, dtype=bool)

        if self.additional_mask0 is not None:
            if len(self.additional_mask0) == len(self.x0):
                self.__mask = self.__mask & self.additional_mask0

        return self.__mask

    @property
    def x(self):
        return self.__x0[self.__mask]

    @property
    def y(self):
        return self.__y0[self.__mask]

    @property
    def X(self):
        return np.vstack((self.x.flatten(), self.y.flatten())).T

    def normalised_x(self, max_radius=8.):
        return self.x / max_radius

    def normalised_y(self, max_radius=8.):
        return self.y / max_radius

    def normalised_X(self, max_radius=8.):
        return self.X / max_radius

    @property
    def scale(self):
        return self.__scale0[self.__mask]

    @property
    def variance_explained(self):
        return self.variance_explained0[self.__mask]

    @property
    def eccentricity(self):
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def polar_angle(self):
        return np.arctan2(self.y, self.x)

    @property
    def gain(self):
        return self.__gain0[self.__mask]

    def with_noise(self, noise: ArrayLike):
        '''
        Creates a new measurement based on this measurement with added noise in the pRF position.

        Parameters
        ----------
        noise : ArrayLike
            An n x N0 x 2 array. n is the number of measurements to be created.

        Returns
        -------
        PRFMeasurement or List of PRFMeasurement
            Returns new instances of this measurement with added noise. If n > 1 this will be a list, otherwise a single PDFMeasurement.

        '''
        if noise.ndim < 3:
            noise = np.expand_dims(noise, 0)

        n = noise.shape[0]

        noisy_measurements = [
            PRFMeasurement(self.x0 + noise[i, :, 0],
                           self.y0 + noise[i, :, 1],
                           self.scale0,
                           self.variance_explained0,
                           self.mask)
            for i in range(n)]

        return noisy_measurements if n > 1 else noisy_measurements[0]

    def with_gaussian_noise(self, sigma: ArrayLike = 1.5, mean: ArrayLike = 0., n: np.uint = 1):
        '''
        Adds Gaussian noise to the pRF positions.

        Parameters
        ----------
        sigma : ArrayLike, optional
            The standard deviation of the added noise. Can be a vector of length two
            for unequal variances in x and y or a matrix corresponding to the square
            root of the covariance matrix. The default is 1.5.
        mean : ArrayLike, optional
            The mean of the additive noise. Can be a scalar or a two dimensional vector.
            The default is 0..
        n : np.uint, optional
            The number of copies to create. If n > 1 this function will return a list
            of PRFMeasurements, otherwise a single PRFMeasurement is returned. The default is 1.

        Returns
        -------
        PRFMeasurement or List of PRFMeasurement
            Returns new instances of this measurement with added Gaussian noise.
            If n > 1 this will be a list, otherwise a single PDFMeasurement.

        '''
        sigma = np.array(sigma)

        if sigma.size == 1:
            Sigma = np.eye(2) * sigma**2
        if sigma.size == 2:
            Sigma = np.diag(sigma**2)
        if sigma.size == 4:
            Sigma = sigma**2  # assume that the covariances are supplied in matrix form

        mean = np.array(mean)
        if mean.size == 1:
            mean = np.ones(2) * mean

        if mean.size == 2:
            mean = mean.flatten()

        #assert mean.size == 2, 'The mean must be one or two dimensional'
        if mean.ndim == 1:
            noise = np.random.multivariate_normal(mean, Sigma, [n, self.N0])
        else:
            noise = np.zeros((n, self.N0, 2))
            for i, mean_i in enumerate(mean):
                noise[:, i, :] = np.random.multivariate_normal(mean_i, Sigma, n)

        return self.with_noise(noise)

    def with_offcentre_skew_normal_noise(self, sigma0=1.5, sigma1_min=0.5, alpha_scale=3.0, max_radius=8.0, n: np.uint = 1):
        '''
        Adds additive skewed normal noise to the position of the pRF centres.
        The shape of the noise distribution depends on the eccentricity and the
        polar angle of each measurement. The noise will be standard symmetric Gaussian noise
        in the centre and will radially get more skewed with the direction determined
        by the polar angle.

        Parameters
        ----------
        sigma0 : _ScalarLike, optional
            The standard deviation of the noise in the centre and at the periphery
            in the direction from the centre. The default is 1.5.
        sigma1_min : _ScalarLike, optional
            The orthogonal standard deviation in orthogonal to the direction from 
            the centre at the maximal radius. The default is 0.5.
        alpha_scale : _ScalarLike, optional
            The skewness factor at the maximal radius. The default is 3.0.
        max_radius : _ScalarLike, optional
            The maximal radius of the pRFs used for calculating the radial changes. The default is 8.0.
        n : np.uint, optional
            The number of copies to create. If n > 1 this function will return a list
            of PRFMeasurements, otherwise a single PRFMeasurement is returned. The default is 1.

        Returns
        -------
        PRFMeasurement or List of PRFMeasurement
            Returns new instances of this measurement with added offcentre skewed normal noise.
            If n > 1 this will be a list, otherwise a single PDFMeasurement.

        '''
        v0 = sigma0**2 * np.ones(self.N0)

        # compute the relative eccentricity from 1 in the centre to 0 at max_radius. Could lead to problems if the eccentricity extends the max_radius.
        relative_eccentricity0 = (max_radius - self.eccentricity0) / max_radius

        # compute the orthotonal component of Omega as a function of the eccentricity
        v1 = (sigma1_min + relative_eccentricity0 * (sigma0 - sigma1_min))**2

        # import the R package used for the multivariate skew normal distribution
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.vectors import FloatVector
        numpy2ri.activate()

        sn = importr('sn')

        # precompute the polar angle and eccentricity
        theta = self.polar_angle0
        eccentricity = self.eccentricity0

        noise = np.zeros((n, self.N0, 2))

        for j in range(self.N0):
            # create Omega in a rotated coordinate system
            S = np.diag([v0[j], v1[j]])
            S = np.array([[v0, 0.], [0., v1]])

            # get the rotation as a function of the polar angle
            R = np.array([[np.cos(theta[j]), -np.sin(theta[j])],
                          [np.sin(theta[j]),  np.cos(theta[j])]])

            # compute omega from the diagonal matrix and the rotation matrix
            Omega = R @ S @ R.T
            Omega = np.maximum(Omega, Omega.T)  # make sure the matrix is symmetric

            # the magnitude of the skewness parameter alpha depends on the eccentricity and the orientation on the polar angle
            a = eccentricity[j] / max_radius * alpha_scale * np.exp(1j * theta[j])

            alpha = np.array([a.real, a.imag])

            # compute the delta parameter in order to compute the mean of the distribution and correct for it afterwards
            delta = 1 / np.sqrt(1 + alpha.T @ Omega @ alpha) * Omega.dot(alpha)

            mu = np.sqrt(2 / np.pi) * delta
            xi = np.zeros(2)  # - mu

            # sample the skewed distribution and set its mean to 0
            noise[:, j, :] = np.array(sn.rmsn(n, xi=FloatVector(xi), Omega=Omega, alpha=alpha)) - mu[None, :]

        return self.with_noise(noise)

    @classmethod
    def from_covdata(cls, fname):
        d = loadmat(fname, squeeze_me=True)

        x0 = d['CovData']['x0'].item()
        y0 = d['CovData']['y0'].item()

        variance_explained0 = d['CovData']['varexp'].item()

        scale0 = d['CovData']['size1'].item()

        return cls(x0, y0, scale0, variance_explained0)

    @classmethod
    def from_analyze_prf(cls, fname, mask=None, atlas_fname=None, atlas_name='glasser2016', roi='V1'):

        d = loadmat(fname, squeeze_me=True)
        results = d['allresults']

        if mask is None:
            if atlas_fname:
                rois = roi if isinstance(roi, list) else [roi]
                # we suppose that the atlas is a new Matlab HDF5 file
                import h5py
                with h5py.File(atlas_fname, 'r') as f:
                    labels = [''.join([chr(a) for a in list(f['#refs#'][label[0]])])
                              for label in f[f'{atlas_name}labels']]  # convert the labels to strings
                    indices = np.array(f[atlas_name], dtype=int).flatten()

                    roi_indices = [labels.index(roi) for roi in rois]

                    mask = np.isin(indices, roi_indices)
            else:
                mask = np.ones(results.shape[0], dtype=bool)

        polar_angle = np.deg2rad(results[mask, 0])
        eccentricity = results[mask, 1]
        gain = results[mask, 2]
        meanvol = results[mask, 3]
        variance_explained = results[mask, 4] / 100.
        scale = results[mask, 5]

        z = eccentricity * np.exp(1j * polar_angle)
        x = z.real
        y = z.imag

        return cls(x, y, scale, variance_explained, gain=gain)

    @classmethod
    def from_mrvista(cls, fname, coords=None, mask=None):
        d = loadmat(fname, squeeze_me=True)

        x0 = d['model']['x0'][()]
        y0 = d['model']['y0'][()]

        scale0 = d['model']['sigma'][()]['major'][()]

        with np.errstate(divide='ignore', invalid='ignore'):
            variance_explained0 = (1. - d['model']['rss'] / d['model']['rawrss'])

        gain0 = d['model']['beta'][()][:, 0]

        if coords is not None:
            coords = np.asfortranarray(loadmat(coords, squeeze_me=True)['coords'])

            if mask is not None:
                if not isinstance(mask, list):
                    mask = [mask]

                roi_coords = np.zeros((3, 0), dtype=np.uint16)
                for mask_file in mask:
                    roi_coords = np.hstack((roi_coords, loadmat(mask_file, squeeze_me=True)['ROI']['coords'][()]))
                roi_coords = np.asfortranarray(roi_coords)

                dtype = np.dtype({'names': [f'f{i}' for i in range(3)], 'formats': 3 * [np.uint16]})

                roi_mask_indices = np.intersect1d(coords.T.view(
                    dtype).T, roi_coords.T.view(dtype).T, return_indices=True)[1]

                roi_mask = np.zeros_like(x0, bool)
                roi_mask[roi_mask_indices] = True

                x0 = x0[roi_mask]
                y0 = y0[roi_mask]
                scale0 = scale0[roi_mask]
                variance_explained0 = variance_explained0[roi_mask]
                gain0 = gain0[roi_mask]
        else:
            if mask is not None:
                raise ValueError('coords must be supplied for masking a MrVista file.')

        return cls(x0, y0, scale0, variance_explained0, gain=gain0)

    @classmethod
    def from_average(cls, prf_measurements):
        x = np.mean([prf.x0 for prf in prf_measurements], axis=0)
        y = np.mean([prf.y0 for prf in prf_measurements], axis=0)
        scale = np.mean([prf.scale0 for prf in prf_measurements], axis=0)
        variance_explained = np.mean([prf.variance_explained0 for prf in prf_measurements], axis=0)
        gain = np.mean([prf.gain0 for prf in prf_measurements], axis=0) if all(
            [prf.gain0 is not None for prf in prf_measurements]) else None

        return cls(x, y, scale, variance_explained, gain=gain)


class AnalyzePRFStimulus(object):
    def __init__(self, run_files, tr=1., max_eccentricity=8.):
        self._run_files = run_files
        self._tr = tr
        self._max_eccentricity = max_eccentricity

        self._load_run_files()

    def _load_run_files(self):
        from scipy.io import loadmat
        import h5py

        run_stimuli = []

        for run_file in self.run_files:
            if h5py.is_hdf5(run_file):
                file = h5py.File(run_file, 'r')
                # apparently, the array needs to be transposed after reading...
                run_stimuli.append(np.transpose(file['stim']))
                file.close()
            else:
                run = loadmat(run_file, squeeze_me=True)
                run_stimuli.append(run['stim'])

        self._run_stimuli = run_stimuli

    def get_run_indices(self, run_indices=None):
        if run_indices is None:
            return np.arange(len(self.run_files))
        else:
            return run_indices

    def convolve(self, run_indices=None, filter_function=None):
        run_indices = self.get_run_indices(run_indices)

        selected_runs = [self.run_stimuli[i] for i in run_indices]

        if filter_function is not None:
            filtered_runs = [filter_function(run) for run in selected_runs]
        else:
            filtered_runs = selected_runs

        from .utils import analyze_prf_hrf

        hrf = analyze_prf_hrf(self.tr, self.tr)  # same call as in analyzePRF.m

        return [np.apply_along_axis(lambda v: np.convolve(v, hrf, 'same'), 2, run) for run in filtered_runs]

    @property
    def max_eccentricity(self):
        return self._max_eccentricity

    @property
    def tr(self):
        return self._tr

    @property
    def run_files(self):
        return self._run_files

    @property
    def run_stimuli(self):
        return self._run_stimuli

    @property
    def stimulus_positions(self):
        positions = []

        for stimulus in self.run_stimuli:
            x, y = self._unitcoordinates(stimulus.shape[0])
            x *= 2. * self.max_eccentricity
            y *= 2. * self.max_eccentricity

            positions.append((x, y))

        return positions

    def _linspacepixels(self, x1, x2, n):
        dif = ((x2 - x1) / n) / 2.
        return np.linspace(x1 + dif, x2 + dif, n)

    def _unitcoordinates(self, res):
        return np.meshgrid(self._linspacepixels(-0.5, 0.5, res), self._linspacepixels(0.5, -0.5, res))

    def sample(self, prf_measurements, run_indices=None, filter_function=None):
        if not isinstance(prf_measurements, list):
            prf_measurements = [prf_measurements]

        convolved_stimuli = self.convolve(run_indices, filter_function)

        positions = self.stimulus_positions
        positions = [positions[i] for i in self.get_run_indices(run_indices)]

        samples = []
        for prf_measurement in prf_measurements:
            measurement_samples = []
            for position, stimulus in zip(positions, convolved_stimuli):
                xx, yy = position
                x = xx.flatten()
                y = yy.flatten()

                prf = np.exp(- ((x[None, :] - prf_measurement.x[:, None])**2 + (y[None, :] - prf_measurement.y[:, None])
                             ** 2) / (2 * prf_measurement.scale[:, None]**2)) / (2. * np.pi * prf_measurement.scale[:, None]**2)

                tc = prf.dot(stimulus.reshape((-1, stimulus.shape[2])))

                measurement_samples.append(tc)

            samples.append(measurement_samples)

        return samples


class PRFSubject(object):
    def __init__(self):
        pass
