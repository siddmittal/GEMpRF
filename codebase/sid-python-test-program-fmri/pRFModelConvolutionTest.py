import numpy as np
import matplotlib.pyplot as plt

def generate_2d_gaussian(sigma, size, resolution, shift):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x - shift, y - shift)  # Shift the coordinates by 'shift'
    Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return X, Y, Z

def generate_triangle(size, resolution):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    mask = (X >= 0) & (Y >= 0) & (X + Y <= size)
    Z[mask] = 1
    return X, Y, Z

sigma = 1
size = 5
resolution = 100

X1, Y1, Z1 = generate_2d_gaussian(sigma, size, resolution, shift=2)
X2, Y2, Z2 = generate_triangle(size, resolution)

Z3 = Z1 * Z2

fig = plt.figure()

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X1, Y1, Z1, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Gaussian Function')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X2, Y2, Z2, cmap='Blues')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Triangle Function')

ax3 = fig.add_subplot(133)
ax3.imshow(Z3, extent=(-size, size, -size, size), cmap='hot', origin='lower')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Gaussian * Triangle')

plt.tight_layout()
plt.show()
