import numpy as np
import matplotlib.pyplot as plt

x_coordinates = np.linspace(start=-1, stop=1, num=10)
y_coordinates = np.linspace(-3, 3, 5)

X, Y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")

Z = X**2 + Y**2

plt.imshow(Z,  extent=[-3, 3, -1, 1])

print('done')

