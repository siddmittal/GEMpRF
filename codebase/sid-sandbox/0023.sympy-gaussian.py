# import numpy as np
# import sympy as sp
# from sympy.utilities.lambdify import lambdify
# import matplotlib.pyplot as plt

# # Define the symbols
# x, y, mean_x, mean_y, std_dev = sp.symbols('x y mean_x mean_y std_dev')

# # Define the Gaussian formula
# gaussian_formula = sp.exp(-((x - mean_x)**2 + (y - mean_y)**2) / (2 * std_dev**2))

# # Create a lambdified function
# gaussian_func = lambdify((x, y, mean_x, mean_y, std_dev), gaussian_formula, 'numpy')

# # Generate your mesh grids
# range_points_X, range_points_Y = np.meshgrid(np.linspace(-9, 9, 101), np.linspace(-9, 9, 101))

# # Set the standard deviation
# std_dev = 1.0

# # Create a list to store Gaussian curves
# gaussians = []

# # Loop through different mean values
# for mean_x_value in np.linspace(-9, 9, 11):
#     for mean_y_value in np.linspace(-9, 9, 11):
#         # Calculate the Gaussian values for each mean value
#         gaussian_values = gaussian_func(range_points_X, range_points_Y, mean_x_value, mean_y_value, std_dev)
#         gaussians.append(gaussian_values)

# # The 'gaussians' list now contains a grid of Gaussian curves for different mean values.

# print('done')


import sympy as sp

# Define the symbols
x, y, mean_x, mean_y, std_dev = sp.symbols('x y mean_x mean_y std_dev')

# Define the Gaussian formula
gaussian_formula = sp.exp(-((x - mean_x)**2 + (y - mean_y)**2) / (2 * std_dev**2))

# Calculate the marginal derivative with respect to mean_x
marginal_derivative = sp.diff(gaussian_formula, mean_x)

# Print the formula
sp.pprint(marginal_derivative)

