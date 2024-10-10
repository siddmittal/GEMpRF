import matplotlib.pyplot as plt
import numpy as np

def plot_dots_on_line(start, end, num_dots):
    line_points = np.linspace(start, end, num_dots)
    return line_points

# Function to plot dots at specific locations
def plot_specific_dots(locations):
    for location in locations:
        ax.plot(location[0], location[1], color='black', marker='o')
        ax.text(location[0], location[1], f'({location[0]}, {location[1]})', fontsize=8, ha='left', va='bottom')

# Create a figure and axis
fig, ax = plt.subplots()

# Create a circle
circle = plt.Circle((0, 0), 9, color='blue', fill=False)
ax.add_patch(circle)

# Create a hair cross without arrays
ax.plot([0, 0], [-9, 9], color='red')  # Vertical line
ax.plot([-9, 9], [0, 0], color='red')  # Horizontal line

# Create lines at 45-degree angles
angles = np.deg2rad([45, 135, 225, 315])
for angle in angles:
    x = np.linspace(0, 9 * np.cos(angle), 6)
    y = np.linspace(0, 9 * np.sin(angle), 6)
    ax.plot(x, y, color='green')

# Define points for dots on the hair cross lines
vertical_dots = plot_dots_on_line(0, 9, 4)
horizontal_dots = plot_dots_on_line(3, 9, 3)

# Define specific locations for dots on the hair cross lines
specific_locations = [
    [3, 3],
    [6, 6],
    [9, 9]
]

# Plot dots at specific locations
plot_specific_dots(specific_locations)

# Plot dots on vertical and horizontal lines
for point in vertical_dots:
    ax.plot(0, point, color='black', marker='o')
    ax.text(0.5, point, f'(0, {int(point)})', fontsize=8, ha='left', va='center')

for point in horizontal_dots:
    ax.plot(point, 0, color='black', marker='o')
    ax.text(point, 0.5, f'({int(point)}, 0)', fontsize=8, ha='center', va='bottom')

# Set x and y axis limits to display outermost limits
plt.xlim(-11, 11)
plt.ylim(-11, 11)

# Show the plot with dots on hair cross lines and visible outermost limits
plt.gca().set_aspect('equal', adjustable='box')
plt.axhline(y=0, color='black',linewidth=0.5)  # Show x-axis
plt.axvline(x=0, color='black',linewidth=0.5)  # Show y-axis
plt.xticks(np.arange(-11, 12, 2))  # Show ticks at every integer
plt.yticks(np.arange(-11, 12, 2))  # Show ticks at every integer
# plt.xticks(np.array([-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11]))  # Show ticks at every integer
# plt.yticks(np.array([-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11]))  # Show ticks at every integer
plt.grid(True)  # Show grid

plt.tight_layout()
plt.savefig('C:/Users/siddh/Desktop/MedUni/Papers/GEM-pRF/simulated_prf_locations.png', bbox_inches='tight')

plt.show()

print()
