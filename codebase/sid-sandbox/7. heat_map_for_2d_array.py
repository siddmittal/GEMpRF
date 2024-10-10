

import numpy as np
import matplotlib.pyplot as plt

nRows_grid = 4
nCols_grid = 6

# Analyze fitting results
grid_fitting_results = [[0 for _ in range(nCols_grid)] for _ in range(nRows_grid)]

# test
value = 0
for i in range(nRows_grid):
    for j in range(nCols_grid):
        grid_fitting_results[i][j] = value
        value += 1

 #Set the color mapping range
vmin = np.min(grid_fitting_results)  # Minimum value for color mapping
vmax = np.max(grid_fitting_results)  # Maximum value for color mapping

# Create a heatmap plot
plt.imshow(grid_fitting_results, cmap='YlGnBu', origin='lower', extent=[0, nCols_grid, 0, nRows_grid], vmin=vmin, vmax=vmax, interpolation='nearest')  # Added interpolation

# Annotate each box with its value
for i in range(nRows_grid):
    for j in range(nCols_grid):
        plt.text(j + 0.5, i + 0.5, f"{grid_fitting_results[i][j]:.2f}", color='black', ha='center', va='center')

# Customize grid lines at integer values
plt.xticks(range(nCols_grid))
plt.yticks(range(nRows_grid))
plt.grid(which='both', color='gray', linestyle='-', linewidth=1)

# Add labels    
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Heatmap')

# Show the plot
plt.colorbar(label='Color Coding')
plt.show()
