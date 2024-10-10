import numpy as np
import matplotlib.pyplot as plt

X =  np.array([[-1.62, -1.08, -0.54],
       [-1.62, -1.08, -0.54],
       [-1.62, -1.08, -0.54]])

Y = np.array([[2.16, 2.16, 2.16],
       [2.7 , 2.7 , 2.7 ],
       [3.24, 3.24, 3.24]])

Z= np.array([[1846.76519619, 1912.83024484, 1907.85033005, 1884.44377117,
       1936.11219261, 1917.34261408, 1864.70501227, 1909.25344747,
       1886.32844313]]).reshape(3,3)
plt.figure()
plt.title('Sample Contour Plot')
plt.gca().contour(X , Y , Z)
plt.scatter(X,Y,c=Z,s=20)

print('done')

