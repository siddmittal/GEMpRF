#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:19:46 2021

@author: mwoletz
"""

import numpy as np

def getArrowPath(can, from_canvas_position, to_canvas_position, width = 1.5, head_length = 8.0, head_width = 6.0, notch = 0.0):
    '''
    Creates a reportlab path representing an arrow for a starting and end position
    in canvas coordinates. If the distance between arrow starting and end position is
    smaller than the length of the arrow head only part of the arrow head will be 
    drawn as a triangle.

    Parameters
    ----------
    can : reportlab.pdfgen.canvas.Canvas
        The canvas to draw the arrow on.
    from_canvas_position : tuple of floats
        2D position of the arrow origin in canvas coordinates
    to_canvas_position : tuple of floats
        2D position of the arrow head in canvas coordinates.
    width : float, optional
        The arrow width in points. The default is 1.5.
    head_length : float, optional
        The arrow head length in points. The default is 8.0.
    head_width : float, optional
        The arrow head width in points. The default is 6.0.
    notch : float, optional
        The notch in the back of the arrow head in points. The default is 0.0.

    Returns
    -------
    p : Path object or None
        The reportlab path object corresponding to the arrow or None if the start and end positions are the same.

    '''
    c1p = np.array(from_canvas_position)
    c2p = np.array(to_canvas_position)
    
    c12p = c2p - c1p
    c12l = np.sqrt(c12p[0]**2. + c12p[1]**2.)

    if c12l > 0:
        c12s = c12p / c12l
        c12n = np.array([c12s[1], -c12s[0]])
        wa = width * 0.5

        p = can.beginPath()

        if c12l > head_length - notch:
            wh = (head_width-width) * 0.5

            cn1 = c1p + wa*c12n
            cn2 = cn1 + (c12l - head_length + notch) * c12s
            cn3 = cn2 + wh*c12n - notch * c12s
            cn4 = c2p
            cn5 = cn4 - c12s * head_length - c12n * head_width * 0.5
            cn6 = cn5 + wh*c12n + notch * c12s
            cn7 = c1p - wa*c12n

            p.moveTo(cn1[0], cn1[1])
            p.lineTo(cn2[0], cn2[1])
            p.lineTo(cn3[0], cn3[1])
            p.lineTo(cn4[0], cn4[1])
            p.lineTo(cn5[0], cn5[1])
            p.lineTo(cn6[0], cn6[1])
            p.lineTo(cn7[0], cn7[1])
            p.close()
            return p
        else:
            wh = head_width * c12l / head_length * 0.5
            
            cn1 = c1p + wh*c12n
            cn2 = c2p
            cn3 = c1p - wh*c12n
            
            p.moveTo(cn1[0], cn1[1])
            p.lineTo(cn2[0], cn2[1])
            p.lineTo(cn3[0], cn3[1])
            p.close()
            
            return p
    else:
        return None
    
def getScatterPath(can, position, size=1):
    '''
    This method is just a wrapper for creating a circular path.

    Parameters
    ----------
    can : reportlab.canvas.Canvas
        The canvas to draw on.
    position : ArrayLike
        Two dimensional position of the centre.
    size : float, optional
        The width of the scatter icon. The default is 1.

    Returns
    -------
    p : Path
        Reportlab path object.

    '''
    p = can.beginPath()
    p.circle(*position, size/2)
    return p

def normalised_direction(from_point, to_point):
    direction = np.array(to_point) - np.array(from_point)
    length = np.linalg.norm(direction)
    if length > 0:
        direction /= length
    return direction
    
def hotelling_t_square_test(X, mu=0, weights=1):
    '''
    Performs a Hotelling's t²-test (multivariate t-test). See: https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution

    Parameters
    ----------
    X : array
        Array of shape N samples * D dimensions to test against.
        
    mu : array or float
        Expected value to test against.
    weights : array or float
        Weights for each row in X.

    Returns
    -------
    t² test statistic, p-value.

    '''
    from scipy.stats import f as f_distribution
    
    X = np.array(X)
    
    # the number of dimensions
    D = X.shape[1]
    
    mu = np.array(mu) if not np.isscalar(mu) else np.ones(D) * mu
    
    weights = np.array(weights) if not np.isscalar(weights) else np.ones(X.shape[0])
    
    # filter out all nan values (subject that did not include any values at this point)
    mask = (~np.any(np.isnan(X), axis=1)) & (~np.isnan(weights)) & (weights > 0.)
    N = mask.sum()
    
    if N < 2: # if there are either no samples left, or just one, which is not enough for the estimateion -> the result is defined as insignificant
        return 0., 1.
    
    X = X[mask]
    
    weights = weights[mask]
    
    # compute the mean (we want to test if this is != mu)
    X_m = np.average(X, axis=0, weights=weights)
    #X_p = X - X_m[None,:]
    
    # compute the sample covariance matrix of the mean
    S = np.cov(X, ddof=1, aweights=weights, rowvar=False) / N # (X_p.T @ X_p / (N-1.)) / N
    
    if np.linalg.det(S) > 1e-16:    
        # compute the test statistic (this statistic is Hotelling t-squared distributed)
        t_2 = (X_m - mu) @ np.linalg.inv(S) @ (X_m - mu).T
        
        # the Hotelling distribution is proportional to an F-distribution, compute the F distributed statistic
        f = (N - D) / (D * (N - 1)) * t_2
        p = f_distribution.sf(f, D, N-D)
    else:
        t_2 = 0.
        p   = 1.
    
    return t_2, p

def analyze_prf_hrf(tr, duration=0.1):
    '''
    Function ported from analyze_prf.
    generate a predicted HRF to a stimulus of duration <duration>, 
    with data sampled at a TR of <tr>.
    
    the resulting HRF is a row vector whose first point is 
    coincident with stimulus onset.  the HRF is normalized such 
    that the maximum value is one.
    
    the predicted HRF is generated based on experimentally observed
    HRFs from various datasets.

    Parameters
    ----------
    duration : float
        The duration of the stimulus in seconds.
        should be a multiple of 0.1 (if not, we round to the nearest 0.1)..
    tr : float
        The TR in seconds.

    Returns
    -------
    None.

    '''
    
    
    hrf = np.array([0,5.34e-06,3.55e-05,0.000104,0.00022,0.000388,0.00061,0.000886,0.00122,0.00159,0.00202,0.00249,0.003,0.00354,0.00411,0.00471,0.00533,0.00596,0.0066,0.00725,0.0079,0.00855,0.0092,0.00984,0.0105,0.0111,0.0117,0.0122,0.0128,0.0133,0.0138,0.0143,0.0148,0.0152,0.0156,0.016,0.0163,0.0166,0.0169,0.0171,0.0174,0.0176,0.0177,0.0179,0.018,0.0181,0.0181,0.0181,0.0182,0.0181,0.0181,0.018,0.018,0.0178,0.0177,0.0176,0.0174,0.0173,0.0171,0.0169,0.0167,0.0164,0.0162,0.0159,0.0157,0.0154,0.0151,0.0148,0.0146,0.0143,0.014,0.0136,0.0133,0.013,0.0127,0.0124,0.0121,0.0118,0.0114,0.0111,0.0108,0.0105,0.0102,0.00985,0.00954,0.00923,0.00893,0.00862,0.00832,0.00803,0.00773,0.00744,0.00716,0.00688,0.0066,0.00633,0.00607,0.00581,0.00555,0.0053,0.00505,0.00481,0.00458,0.00435,0.00412,0.0039,0.00369,0.00348,0.00328,0.00308,0.00289,0.0027,0.00252,0.00234,0.00217,0.002,0.00184,0.00168,0.00153,0.00139,0.00124,0.00111,0.000974,0.000846,0.000722,0.000603,0.000488,0.000377,0.000271,0.000168,6.96e-05,-2.51e-05,-0.000116,-0.000203,-0.000287,-0.000367,-0.000443,-0.000516,-0.000586,-0.000653,-0.000716,-0.000777,-0.000835,-0.00089,-0.000942,-0.000991,-0.00104,-0.00108,-0.00112,-0.00116,-0.0012,-0.00123,-0.00127,-0.0013,-0.00133,-0.00135,-0.00138,-0.0014,-0.00142,-0.00144,-0.00146,-0.00147,-0.00149,-0.0015,-0.00151,-0.00152,-0.00153,-0.00154,-0.00155,-0.00155,-0.00156,-0.00156,-0.00156,-0.00156,-0.00157,-0.00156,-0.00156,-0.00156,-0.00156,-0.00155,-0.00155,-0.00154,-0.00154,-0.00153,-0.00153,-0.00152,-0.00151,-0.0015,-0.00149,-0.00148,-0.00147,-0.00146,-0.00145,-0.00144,-0.00143,-0.00142,-0.00141,-0.0014,-0.00138,-0.00137,-0.00136,-0.00135,-0.00133,-0.00132,-0.00131,-0.00129,-0.00128,-0.00127,-0.00125,-0.00124,-0.00122,-0.00121,-0.0012,-0.00118,-0.00117,-0.00115,-0.00114,-0.00113,-0.00111,-0.0011,-0.00108,-0.00107,-0.00106,-0.00104,-0.00103,-0.00101,-0.001,-0.000987,-0.000973,-0.00096,-0.000947,-0.000933,-0.00092,-0.000907,-0.000894,-0.000881,-0.000868,-0.000855,-0.000843,-0.00083,-0.000818,-0.000805,-0.000793,-0.000781,-0.000769,-0.000758,-0.000746,-0.000734,-0.000723,-0.000711,-0.0007,-0.000689,-0.000678,-0.000667,-0.000657,-0.000646,-0.000636,-0.000625,-0.000615,-0.000605,-0.000595,-0.000585,-0.000576,-0.000566,-0.000557,-0.000547,-0.000538,-0.000529,-0.00052,-0.000511,-0.000503,-0.000494,-0.000486,-0.000477,-0.000469,-0.000461,-0.000453,-0.000445,-0.000438,-0.00043,-0.000422,-0.000415,-0.000408,-0.000401,-0.000393,-0.000387,-0.00038,-0.000373,-0.000366,-0.00036,-0.000353,-0.000347,-0.000341,-0.000335,-0.000329,-0.000323,-0.000317,-0.000311,-0.000305,-0.0003,-0.000294,-0.000289,-0.000284,-0.000278,-0.000273,-0.000268,-0.000263,-0.000258,-0.000254,-0.000249,-0.000244,-0.00024,-0.000235,-0.000231,-0.000226,-0.000222,-0.000218,-0.000214,-0.00021,-0.000206,-0.000202,-0.000198,-0.000194,-0.000191,-0.000187,-0.000184,-0.00018,-0.000177,-0.000173,-0.00017,-0.000167,-0.000163,-0.00016,-0.000157,-0.000154,-0.000151,-0.000148,-0.000145,-0.000143,-0.00014,-0.000137,-0.000134,-0.000132,-0.000129,-0.000127,-0.000124,-0.000122,-0.000119,-0.000117,-0.000115,-0.000113,-0.00011,-0.000108,-0.000106,-0.000104,-0.000102,-9.99e-05,-9.79e-05,-9.6e-05,-9.41e-05,-9.22e-05,-9.04e-05,-8.86e-05,-8.68e-05,-8.51e-05,-8.34e-05,-8.17e-05,-8.01e-05,-7.85e-05,-7.69e-05,-7.54e-05,-7.39e-05,-7.24e-05,-7.09e-05,-6.95e-05,-6.81e-05,-6.67e-05,-6.54e-05,-6.4e-05,-6.28e-05,-6.15e-05,-6.02e-05,-5.9e-05,-5.78e-05,-5.66e-05,-5.55e-05,-5.44e-05,-5.33e-05,-5.22e-05,-5.11e-05,-5.01e-05,-4.9e-05,-4.8e-05,-4.71e-05,-4.61e-05,-4.52e-05,-4.42e-05,-4.33e-05,-4.24e-05,-4.16e-05,-4.07e-05,-3.99e-05,-3.91e-05,-3.82e-05,-3.75e-05,-3.67e-05,-3.59e-05,-3.52e-05,-3.45e-05,-3.37e-05,-3.3e-05,-3.24e-05,-3.17e-05,-3.1e-05,-3.04e-05,-2.98e-05,-2.91e-05,-2.85e-05,-2.79e-05,-2.74e-05,-2.68e-05,-2.62e-05,-2.57e-05,-2.51e-05,-2.46e-05,-2.41e-05,-2.36e-05,-2.31e-05,-2.26e-05,-2.21e-05,-2.17e-05,-2.12e-05,-2.08e-05,-2.03e-05,-1.99e-05,-1.95e-05,-1.91e-05,-1.87e-05,-1.83e-05,-1.79e-05,-1.75e-05,-1.72e-05,-1.68e-05,-1.64e-05,-1.61e-05,-1.58e-05,-1.54e-05,-1.51e-05,-1.48e-05,-1.45e-05,-1.42e-05,-1.38e-05,-1.36e-05,-1.33e-05,-1.3e-05,-1.27e-05,-1.24e-05,-1.22e-05,-1.19e-05,-1.17e-05,-1.14e-05,-1.12e-05,-1.09e-05,-1.07e-05,-1.05e-05,-1.02e-05,-1e-05,-9.81e-06,-9.6e-06,-9.39e-06,-9.19e-06,-8.99e-06,-8.8e-06,-8.61e-06,-8.43e-06,-8.25e-06,-8.07e-06,-7.89e-06,-7.72e-06,-7.56e-06,-7.39e-06,-7.24e-06,-7.08e-06,-6.93e-06,-6.78e-06,-6.63e-06,-6.49e-06,-6.35e-06,-6.21e-06,-6.08e-06])
    
    # convolve to get the predicted response to the desired stimulus duration
    trold = 0.1
    hrf = np.convolve(hrf,np.ones(int(max(1,round(duration/trold)))));
    
    from scipy.interpolate import PchipInterpolator
    # resample to desired TR
    pchip = PchipInterpolator(np.arange(len(hrf))*trold, hrf)
    hrf = pchip(np.arange(0, len(hrf)*trold, tr))
    
    # make the peak equal to one
    return hrf / max(hrf)

def analyze_prf_atlas_mask(atlas_fname, atlas_name='glasser2016', roi='V1'):
    rois = roi if isinstance(roi, list) else [roi]
    
    # we suppose that the atlas is a new Matlab HDF5 file
    import h5py
    with h5py.File(atlas_fname, 'r') as f:
        labels =  [''.join([chr(a[0]) for a in list(f['#refs#'][label[0]])]) for label in f[f'{atlas_name}labels']] # convert the labels to strings
        indices = np.array(f[atlas_name], dtype=int).flatten()
        
        roi_indices = [labels.index(roi) for roi in rois]
        
        mask = np.isin(indices, roi_indices)
    
    return mask