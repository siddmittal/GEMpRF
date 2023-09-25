import numpy as np
from typing import List

from .prf_receptive_field_response import ReceptiveFieldResponse


# Ordinary Least Squares (OLS)
class OLS:
    def __init__(self
                 , modelled_signals
                 , measured_signals : List[ReceptiveFieldResponse]
                 , type = 'ordinary least squares - simplified'):
        self.modelled_signals = modelled_signals
        self.measured_signals = measured_signals
        self.N = len(modelled_signals[0]) # number of data points in each timecourse
    
    def _make_Trends(self, nDCT=3):
        # Generates trends similar to 'rmMakeTrends' from the 'params' struct in mrVista.    
        # nDCT: Number of discrete cosine transform components.
        
        tf = self.N # this is equal to 300
        ndct = 2 * nDCT + 1
        trends = np.zeros((np.sum(tf), np.max([np.sum(ndct), 1])))        
        
        tc = np.linspace(0, 2.*np.pi, tf)[:, None]        
        trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))

        nTrends = trends.shape[1]
        return trends, nTrends       
    
    def _get_orthogonal_trends(self, trends):
        q, r = np.linalg.qr(trends) # QR decomposition
        q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
        return q
    
    def _compute_orthonormal_model_signals(self, R):
        # Calculate orthogonal projection using O = (I - R @ R^T)
        O = (np.eye(self.N) - R @ R.T) # R: Orthonormal Regressors
        
        # compute orthonormal version of the modelled signals
        total_rows = (len(self.modelled_signals))
        total_cols = len(self.modelled_signals[0])
        orthonormalized_modelled_signals =  np.zeros((total_rows, total_cols)) # pre-allocate array
        for i in range(len(self.modelled_signals)):
            s = self.modelled_signals[i] #s
            s = O @ s #s_prime 
            s /= np.sqrt(s @ s) # orthonormalized_model_signal
            orthonormalized_modelled_signals[i] = s

        return orthonormalized_modelled_signals
    
    # Fitting: Calculate the square of the projection of orthonormal modelled singlas (s_prime) onto targets
    def compute_proj_squared(self):   
        trends, _ = self._make_Trends()
        R = self._get_orthogonal_trends(trends=trends)
        orthonormal_modelled_signals = self._compute_orthonormal_model_signals(R=R)

        # grid of orthonormal modelled signals
        grid_s_prime = np.vstack([timecourse for timecourse in orthonormal_modelled_signals]).T

        # grid of measured/simulated target signals
        grid_y  = np.vstack([timecourse for timecourse in self.measured_signals]).T
        
        # (y^T . s_prime)^2
        proj_squared = (grid_s_prime.T @ grid_y)**2 
        
        # find best matches along the rows (vertical axis) of the array
        best_fit_proj = np.argmax(proj_squared, axis=0)
        return best_fit_proj

