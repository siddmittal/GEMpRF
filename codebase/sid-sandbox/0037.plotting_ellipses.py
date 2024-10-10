import matplotlib.pyplot as plt
import numpy as np

# Step 1: Set the variables
groundtruth_position = (0, 0)
groundtruth_sigma_x = 1
groundtruth_sigma_y = 2

# Use the maximum value between groundtruth_sigma_x and groundtruth_sigma_y to decide the ticks
max_sigma = max(groundtruth_sigma_x, groundtruth_sigma_y)

# Calculate the tick range for both axes
tick_range = max_sigma * 1.5

# Step 2: Create a figure and plot the groundtruth_position as a single point in black color
fig, ax = plt.subplots()
ax.scatter(*groundtruth_position, color='blue')

# Step 3: Set custom ticks for both axes
x_ticks = np.arange(groundtruth_position[0] - tick_range, groundtruth_position[0] + tick_range + 1)
y_ticks = np.arange(groundtruth_position[1] - tick_range, groundtruth_position[1] + tick_range + 1)

# Set the ticks on the plot
ax.set_xlim(-tick_range, tick_range)
ax.set_ylim(-tick_range, tick_range)


# Ensure the aspect ratio is equal to make the plot square-shaped
ax.set_aspect('equal')

# Create an ellipse using groundtruth_sigma_x and groundtruth_sigma_y with a dashed blue line
theta = np.linspace(0, 2 * np.pi, 100)
x = groundtruth_sigma_x * np.cos(theta) + groundtruth_position[0]
y = groundtruth_sigma_y * np.sin(theta) + groundtruth_position[1]
ax.plot(x, y, linestyle='--', color='blue')

# Display the plot
plt.grid(True)
plt.show()
