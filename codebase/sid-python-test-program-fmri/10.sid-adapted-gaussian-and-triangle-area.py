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


#def generate_time_course(initial_triangle_vertices, fig, ax):
def generate_time_course(initial_triangle_vertices, rotating_triangle_plot, timecourse_plot):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    

    # Create an empty plot for the second subplot
    # curve_plot, = ax2.plot([])

    # array to store computed areas under the masked gaussian curve
    masked_gaussian_areas = []

    # simulate time and rotation of triangle
    for time in range(1, 1200, 1):         
        # Compute anti-clockwise rotation angle and apply rotational matrix to the initial triangle
        angle = -1*time
        rotated_triangle_vertices = apply_rotational_matrix(initial_triangle_vertices, angle)

        # Multiply the Gaussian curve with the rotated triangle
        masked_Z = multiply_gaussian_with_triangle(X, Y, Z, rotated_triangle_vertices)

        # Check if there is any overlapping area
        if np.any(masked_Z > 0):
            # Calculate the total probability density (area under the masked Gaussian)
            total_probability = calculate_total_probability(masked_Z, step_size)

            # Append the resulting area to the masked_gaussian_areas array
            masked_gaussian_areas.append(total_probability)

            # Update the subplot for rotating triangle (Clear the existing plot and redraw)
            rotating_triangle_plot.clear()
            rotating_triangle_plot.set_xlim(-4, 4)
            rotating_triangle_plot.set_ylim(-4, 4)
            # rotating_triangle_plot.set_zlim(0, 1)
            rotating_triangle_plot.set_xlabel('X')
            rotating_triangle_plot.set_ylabel('Y')
            # rotating_triangle_plot.set_zlabel('Z')
            rotating_triangle_plot.set_title(f"Time: {time}, Angle: {angle} degrees")
            #subplot_x3.plot_surface(X, Y, masked_Z, cmap='viridis')
            subplot_x3.contour(X, Y, masked_Z, cmap='viridis')
            subplot_x3.axhline(0, color='gray', linestyle='--')  # Add quadrant lines
            subplot_x3.axvline(0, color='gray', linestyle='--')
            rotated_triangle_x, rotated_triangle_y = generate_triangle(rotated_triangle_vertices)
            rotating_triangle_plot.plot(rotated_triangle_x, rotated_triangle_y, 'r-', linewidth=2)

            # Update the subplot for time-course     
            timecourse_plot.clear()
            timecourse_plot.set_xlabel('time')
            timecourse_plot.set_ylabel('r(t)')
            timecourse_plot.set_title('r(t) curve - area')
            timecourse_plot.set_xlim(0, 1200)
            timecourse_plot.set_ylim(0, 0.5)
            timecourse_plot.plot(masked_gaussian_areas)
        
            plt.pause(0.01)

    return masked_gaussian_areas


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

#################################################################
########################----Program-------#######################
#################################################################
# Generate and plot the 2D Gaussian curve
X, Y, Z = generate_2d_gaussian()

# Generate a triangle
initial_triangle_vertices = [(0, 0), (3, 4), (4, 2)]
triangle_x, triangle_y = generate_triangle(initial_triangle_vertices)


#Define a plot to display all information
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
# subplot_x3 = completeOverviewFigure.add_subplot(243, projection='3d')
subplot_x3 = completeOverviewFigure.add_subplot(243)

# subplot to display time course
subplot_x4 = completeOverviewFigure.add_subplot(212)


# Set the step size for calculating the total probability density
step_size = 0.01

# Create masked_gaussian_areas array using the for-loop
masked_gaussian_areas = generate_time_course(initial_triangle_vertices, subplot_x3, subplot_x4)
