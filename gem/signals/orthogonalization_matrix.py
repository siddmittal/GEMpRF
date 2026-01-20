# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Medical University of Vienna",
"@Desc    :   None",
        
"""
import numpy as np
import cupy as cp
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

class OrthoMatrix:
    def __init__(self, nDCT, num_frame_stimulus):
        self.nDCT = nDCT
        self.O_gpu = None
        self.num_frame_stimulus = num_frame_stimulus

    def get_orthogonalization_matrix(self):
        if self.O_gpu is not None:
            return self.O_gpu

        nDCT = self.nDCT
        tc = np.linspace(0, 2.*np.pi, self.num_frame_stimulus)[:, None]

        # Generate (2 * nDCT + 1) cosine regressors:
        # frequencies = [0, 0.5, 1.0, ..., nDCT]
        # e.g., nDCT = 1  →  cos(0t), cos(0.5t), cos(1.0t)  →  3 DCTs        
        trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))

        q, r = np.linalg.qr(trends) # QR decomposition
        q *= np.sign(q[0, 0]) # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0

        with cp.cuda.Device(ggm.get_instance().default_gpu_id):
            R_gpu = cp.asarray(q)
            self.O_gpu = (cp.eye(self.num_frame_stimulus)  - cp.dot(R_gpu, R_gpu.T))

        return self.O_gpu

