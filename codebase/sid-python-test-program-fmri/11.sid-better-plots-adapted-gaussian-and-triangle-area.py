

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

#Generate Triangle
def generate_triangle(vertices):
    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    x.append(vertices[0][0])
    y.append(vertices[0][1])
    return x, y

#Generate Gaussian
def generate_2d_gaussian():
    sigma = 1
    size = 2
    mean = [2, 2]
    x = np.linspace(mean[0] - size, mean[0] + size, 100)
    y = np.linspace(mean[1] - size, mean[1] + size, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X - mean[0])**2 / (2 * sigma**2)) * np.exp(-(Y - mean[1])**2 / (2 * sigma**2))
    return X, Y, Z

# Apply rotation matrix
def apply_rotational_matrix(vertices, angle):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_vertices = [rotation_matrix.dot(vertex) for vertex in vertices]
    return rotated_vertices

#Apply Masking to the Gaussian Curver
def multiply_gaussian_with_triangle(X, Y, Z, triangle_vertices):
    # Create a mask for the triangle shape
    triangle_path = Path(triangle_vertices)
    mask = triangle_path.contains_points(np.vstack((X.flatten(), Y.flatten())).T).reshape(X.shape)

    # Multiply the Gaussian values with the mask
    masked_Z = np.where(mask, Z, 0)

    return masked_Z

# Compute area under the curve
def calculate_total_probability(masked_Z, step_size=0.01):
    # Calculate the total probability density (area under the remaining Gaussian)
    total_probability = np.sum(masked_Z) * step_size**2

    return total_probability

#def generate_time_course(initial_triangle_vertices, fig, ax):
def compute_area_under_curve(time, initial_triangle_vertices, rotating_triangle_plot):
    # simulate time and rotation of triangle 
    # Compute anti-clockwise rotation angle and apply rotational matrix to the initial triangle
    angle = -1*time
    rotated_triangle_vertices = apply_rotational_matrix(initial_triangle_vertices, angle)

    # Multiply the Gaussian curve with the rotated triangle
    masked_Z = multiply_gaussian_with_triangle(X, Y, Z, rotated_triangle_vertices)

    # Check if there is any overlapping area    
    # Calculate the total probability density (area under the masked Gaussian)
    total_probability = calculate_total_probability(masked_Z, step_size)

    # Update the subplot for rotating triangle (Clear the existing plot and redraw)
    rotating_triangle_plot.clear()
    rotating_triangle_plot.set_xlim(-4, 4)
    rotating_triangle_plot.set_ylim(-4, 4)
    rotating_triangle_plot.set_xlabel('X')
    rotating_triangle_plot.set_ylabel('Y')
    rotating_triangle_plot.set_title(f"Time: {time}, Angle: {angle} degrees")
    subplot_x3.contour(X, Y, masked_Z, cmap='viridis')
    subplot_x3.axhline(0, color='gray', linestyle='--')  # Add quadrant lines
    subplot_x3.axvline(0, color='gray', linestyle='--')
    rotated_triangle_x, rotated_triangle_y = generate_triangle(rotated_triangle_vertices)
    rotating_triangle_plot.plot(rotated_triangle_x, rotated_triangle_y, 'r-', linewidth=2)

    plt.pause(0.01)
    

    return total_probability


#################################################################
########################----Program-------#######################
#################################################################
# Generate and plot the 2D Gaussian curve
X, Y, Z = generate_2d_gaussian()

# Generate a triangle
initial_triangle_vertices = [(0, 0), (3, 4), (4, 2)]
triangle_x, triangle_y = generate_triangle(initial_triangle_vertices)


#Define a plot to display all information
plt.ion() #Turn on interactive mode
completeOverviewFigure = plt.figure()

# subplot to display 3D gaussian
subplot_x1 = completeOverviewFigure.add_subplot(241, projection='3d')
subplot_x1.set_xlim(-4, 4)
subplot_x1.set_ylim(-4, 4)
subplot_x1.set_zlim(0, 1)
subplot_x1.set_xlabel('X')
subplot_x1.set_ylabel('Y')
subplot_x1.set_zlabel('Z')
subplot_x1.plot_surface(X, Y, Z, cmap='viridis')
subplot_x1.set_title('Gaussian')
plt.show()


# subplot to display an overlayed initial-triangle over gaussian
subplot_x2 = completeOverviewFigure.add_subplot(242, projection='3d')
subplot_x2.set_xlim(-4, 4)
subplot_x2.set_ylim(-4, 4)
subplot_x2.set_zlim(0, 1)
subplot_x2.set_xlabel('X')
subplot_x2.set_ylabel('Y')
subplot_x2.set_zlabel('Z')
subplot_x2.plot_surface(X, Y, Z, cmap='viridis')
subplot_x2.set_title('Inital - Triangle & Gaussian')
subplot_x2.plot(triangle_x, triangle_y, 'r-', linewidth=2)


# subplot to display rotating triangle over gaussian
subplot_x3 = completeOverviewFigure.add_subplot(243)

# subplot to display time course
subplot_x4 = completeOverviewFigure.add_subplot(212)
subplot_x4.set_xlabel('time')
subplot_x4.set_ylabel('r(t)')
subplot_x4.set_title('r(t) curve - area')

subplot_x4.set_xlim(0, 1200)
subplot_x4.set_ylim(0, 0.5)
at_time = np.arange(1200)  # Generate an array from 0 to 1199 (1200 steps)
area_under_curve = np.zeros(1200)  # Initialize the area array with zeros
tc_curve, = subplot_x4.plot(at_time, area_under_curve, 'b-')

# Set the step size for calculating the total probability density
step_size = 0.01

for time in range(0, 1200, 1):
    area_under_curve[time] = compute_area_under_curve(time, initial_triangle_vertices, subplot_x3)
    tc_curve.set_ydata(area_under_curve)
    completeOverviewFigure.canvas.draw()
    completeOverviewFigure.canvas.flush_events()




