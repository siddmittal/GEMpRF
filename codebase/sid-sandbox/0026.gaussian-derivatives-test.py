

import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x_range = np.linspace(-8, 8, 100)

# Define the mean and standard deviation (sigma) for the Gaussian curve
mean = 0
sigma = 1

# Calculate the Gaussian values for the original Gaussian curve
gaussian_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / sigma) ** 2)

# Calculate the derivative of the Gaussian curve with respect to the mean
derivative_mean = -gaussian_values * (x_range - mean) / (sigma ** 2)

# Calculate the derivative of the Gaussian curve with respect to x
derivative_x = (-1 / sigma) * gaussian_values * (x_range - mean)

# Create a plot to visualize the Gaussian curve and its derivatives
plt.figure(figsize=(8, 6))

plt.plot(x_range, gaussian_values, label='Gaussian Curve', color='blue')
plt.plot(x_range, derivative_mean, label='Derivative w.r.t. Mean', color='red', linestyle='--')
plt.plot(x_range, derivative_x, label='Derivative w.r.t. X', color='green', linestyle=':')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('Gaussian Curve and Its Derivatives')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

print('done')
