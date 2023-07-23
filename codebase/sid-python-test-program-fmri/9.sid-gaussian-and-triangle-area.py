import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def generate_triangle(vertices):
    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    x.append(vertices[0][0])
    y.append(vertices[0][1])
    return x, y

def create_masked_gaussian_areas(initial_triangle_vertices, fig, ax):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    masked_gaussian_areas = []

    # Create an empty plot for the second subplot
    curve_plot, = ax2.plot([])

    for angle in range(-1, -1200, -1):
        # Apply rotational matrix to the initial triangle vertices
        rotated_triangle_vertices = apply_rotational_matrix(initial_triangle_vertices, angle)

        # Multiply the Gaussian curve with the rotated triangle
        masked_Z = multiply_gaussian_with_triangle(X, Y, Z, rotated_triangle_vertices)

        # Check if there is any overlapping area
        if np.any(masked_Z > 0):
            # Calculate the total probability density (area under the masked Gaussian)
            total_probability = calculate_total_probability(masked_Z, step_size)

            # Append the resulting area to the masked_gaussian_areas array
            masked_gaussian_areas.append(total_probability)

            # Clear the existing plot and redraw the necessary elements in the first subplot
            ax1.clear()
            ax1.set_xlim(-4, 4)
            ax1.set_ylim(-4, 4)
            ax1.set_title(f"Angle: {angle} degrees")
            ax1.contour(X, Y, masked_Z, cmap='viridis')
            ax1.axhline(0, color='gray', linestyle='--')  # Add quadrant lines
            ax1.axvline(0, color='gray', linestyle='--')
            rotated_triangle_x, rotated_triangle_y = generate_triangle(rotated_triangle_vertices)
            ax1.plot(rotated_triangle_x, rotated_triangle_y, 'r-', linewidth=2)

            # Update the plot data in the second subplot
            curve_plot.set_data(range(len(masked_gaussian_areas)), masked_gaussian_areas)
            ax2.set_xlim(0, 1199)  # Set the x-axis size to the maximum number of iterations

            plt.pause(0.01)

    return masked_gaussian_areas


def apply_rotational_matrix(vertices, angle):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_vertices = [rotation_matrix.dot(vertex) for vertex in vertices]
    return rotated_vertices

def generate_2d_gaussian():
    sigma = 1
    size = 2
    mean = [2, 2]
    x = np.linspace(mean[0] - size, mean[0] + size, 100)
    y = np.linspace(mean[1] - size, mean[1] + size, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X - mean[0])**2 / (2 * sigma**2)) * np.exp(-(Y - mean[1])**2 / (2 * sigma**2))
    return X, Y, Z

def multiply_gaussian_with_triangle(X, Y, Z, triangle_vertices):
    # Create a mask for the triangle shape
    triangle_path = Path(triangle_vertices)
    mask = triangle_path.contains_points(np.vstack((X.flatten(), Y.flatten())).T).reshape(X.shape)

    # Multiply the Gaussian values with the mask
    masked_Z = np.where(mask, Z, 0)

    return masked_Z

def calculate_total_probability(masked_Z, step_size=0.01):
    # Calculate the total probability density (area under the remaining Gaussian)
    total_probability = np.sum(masked_Z) * step_size**2

    return total_probability


def create_masked_gaussian_areas(initial_triangle_vertices, fig):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    masked_gaussian_areas = []

    for angle in range(-1, -1200, -1):
        # Apply rotational matrix to the initial triangle vertices
        rotated_triangle_vertices = apply_rotational_matrix(initial_triangle_vertices, angle)

        # Multiply the Gaussian curve with the rotated triangle
        masked_Z = multiply_gaussian_with_triangle(X, Y, Z, rotated_triangle_vertices)

        # Check if there is any overlapping area
        if np.any(masked_Z > 0):
            # Calculate the total probability density (area under the masked Gaussian)
            total_probability = calculate_total_probability(masked_Z, step_size)

            # Append the resulting area to the masked_gaussian_areas array
            masked_gaussian_areas.append(total_probability)

            # Clear the existing plot and redraw the necessary elements in the first subplot
            ax1.clear()
            ax1.set_xlim(-4, 4)
            ax1.set_ylim(-4, 4)
            ax1.set_title(f"Angle: {angle} degrees")
            ax1.contour(X, Y, masked_Z, cmap='viridis')
            ax1.axhline(0, color='gray', linestyle='--')  # Add quadrant lines
            ax1.axvline(0, color='gray', linestyle='--')
            rotated_triangle_x, rotated_triangle_y = generate_triangle(rotated_triangle_vertices)
            ax1.plot(rotated_triangle_x, rotated_triangle_y, 'r-', linewidth=2)

            # Plot the curve for masked_gaussian_areas in the second subplot
            ax2.clear()
            ax2.plot(masked_gaussian_areas)

            plt.pause(0.01)

    return masked_gaussian_areas


# Create the initial plot and keep it open
masked_gaussian_plot, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.axhline(0, color='gray', linestyle='--')  # Add quadrant lines
ax.axvline(0, color='gray', linestyle='--')

# Generate and plot the 2D Gaussian curve
X, Y, Z = generate_2d_gaussian()
ax.contour(X, Y, Z, cmap='viridis')

# Generate and plot a triangle
initial_triangle_vertices = [(0, 0), (3, 4), (4, 2)]
triangle_x, triangle_y = generate_triangle(initial_triangle_vertices)
ax.plot(triangle_x, triangle_y, 'r-', linewidth=2)

# Set the step size for calculating the total probability density
step_size = 0.01

# Create masked_gaussian_areas array using the for-loop
masked_gaussian_areas = create_masked_gaussian_areas(initial_triangle_vertices, masked_gaussian_plot)

# Show the final plot
# plt.show()

# Plot the masked_gaussian_areas
# plt.figure(figsize=(8, 6))
# plt.plot(masked_gaussian_areas)
# plt.xlabel("Angle")
# plt.ylabel("Area")
# plt.title("Masked Gaussian Areas")
# plt.show()
