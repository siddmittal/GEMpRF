import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f_x_c with c=2
def f_x_c(x, c=2):
    return x + 2 * x * c

# Create an array of x-values
x = np.linspace(0, 10, 100)
c = np.linspace(0, 10, 100)

# Create a meshgrid from x and c
X, C = np.meshgrid(x, c)

# Compute the Z-values for the surface plot
Z = f_x_c(X, C)

# Compute the line using c=2
line = f_x_c(x, c=2)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
ax.plot_surface(X, C, Z, cmap='viridis')

# Plot the line
ax.plot(x, line, color='red', label='f(x,c=2)')

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('C')
ax.set_zlabel('f(x,c)')

# Add a legend
ax.legend()

# Show the plot
plt.show()
print('done')



###############################################################################################################
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the range of x values
# x = np.linspace(0, 10, 100)  # Adjust the range as needed

# # Define the constant 'c' to shift the line up or down
# c = 5  # Adjust 'c' as needed

# # Calculate the corresponding y values
# y = x + c

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the line in 3D
# ax.plot(x, y, x)

# # Set labels for the axes
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')

# # Show the plot
# plt.show()

###############################################################################################################
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define the values of y, x, and c
# fxc = [0, 3, 6, 9, 12]
# x = [0, 1, 2, 3, 4]
# c = [2, 2, 2, 2, 2]

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the data points
# ax.plot(x, c, fxc, c='r', marker='o')

# # Set labels for the axes
# ax.set_xlabel('X Axis')
# ax.set_ylabel('C Axis')
# ax.set_zlabel('f(x,c) Axis')

# # Show the plot
# plt.show()
