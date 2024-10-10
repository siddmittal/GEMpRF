
# imports
import numpy as np

R = np.array([[ 0.5       ,  0.63245553, -0.5       ,  0.31622777],
       [ 0.5       ,  0.31622777,  0.5       , -0.63245553],
       [ 0.5       , -0.31622777,  0.5       ,  0.63245553],
       [ 0.5       , -0.63245553, -0.5       , -0.31622777]])


result = R@R.T

# Print the result
print(result)




# R = np.array([
#     0.5, 0.63245553, -0.5, 0.31622777,
#     0.5, 0.31622777, 0.5, -0.63245553,
#     0.5, -0.31622777, 0.5, 0.63245553,
#     0.5, -0.63245553, -0.5, -0.31622777
# ]).reshape(4, 4)


nStimulusFrames = 4
nTotal_signals = 6
nRows_R = nStimulusFrames
nCols_R = nStimulusFrames
nRows_S = nTotal_signals
nCols_S = nStimulusFrames

dummy_s = np.array([  # 4x6 matrix
    1, 2, 1, 1,
    1, 2, 1, 1,
    1, 3, 1, 1,
    1, 4, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1
])

dummy_r = np.array([  # 4x4 matrix
    1, 1, 1, 1,
    2, 2, 3, 4,
    1 , 1, 2, 2,
    0, 0, 1, 1
])

S = dummy_s.reshape((nTotal_signals, nCols_S))
R = dummy_r.reshape((nRows_R, nCols_S))

# Calculate orthogonal projection using O = (I - R @ R^T)
O = (np.eye(nRows_R) - R @ R.T) # R: Orthonormal Regressors

# compute orthonormal version of the modelled signals
orthonormalized_modelled_signals =  np.zeros((nTotal_signals, nCols_S)) # pre-allocate array
for i in range(nTotal_signals):
    s = S[i, :] #s
    s = O @ s #s_prime 
    # TODO: s /= np.sqrt(s @ s) # orthonormalized_model_signal
    orthonormalized_modelled_signals[i] = s


print("done!")

