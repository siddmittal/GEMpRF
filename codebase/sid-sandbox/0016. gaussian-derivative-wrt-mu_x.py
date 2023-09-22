import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the parameters for the Gaussian distribution
sigma_x = 1.0
sigma_y = 1.0
mu_y = 0.0  # Keep mu_y fixed at 0 for this analysis

# Create a grid of x and y values using meshgrid
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)

# Initialize the figure with reduced plot size
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Initial value for mu_x
initial_mu_x = 0.0

# Calculate the Gaussian distribution and its derivative with respect to mu_x
gaussian_2d = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
    -0.5 * ((X - initial_mu_x)**2 / sigma_x**2 + (Y - mu_y)**2 / sigma_y**2)
)
derivative_2d = -gaussian_2d * (X - initial_mu_x) / sigma_x**2

# Create the contour plots for 2D Gaussian and its derivative
contour1 = ax1.contourf(X, Y, gaussian_2d, cmap='viridis', levels=20)
ax1.set_title('2D Gaussian Distribution')
contour2 = ax2.contourf(X, Y, derivative_2d, cmap='viridis', levels=20)
ax2.set_title('Derivative with respect to mu_x')

# Create a slider axis
slider_ax = plt.axes([0.1, 0.01, 0.65, 0.03])

# Create the mu_x slider
mu_x_slider = Slider(slider_ax, 'mu_x', -5.0, 5.0, valinit=initial_mu_x)

# Define an update function for the slider
def update_mu_x(val):
    mu_x = mu_x_slider.val
    
    # Clear the previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    # Calculate the Gaussian distribution and its derivative with the updated mu_x
    gaussian_2d = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
        -0.5 * ((X - mu_x)**2 / sigma_x**2 + (Y - mu_y)**2 / sigma_y**2)
    )
    derivative_2d = -gaussian_2d * (X - mu_x) / sigma_x**2
    
    # Create new contour plots for 2D Gaussian and its derivative
    contour1 = ax1.contourf(X, Y, gaussian_2d, cmap='viridis', levels=20)
    ax1.set_title('2D Gaussian Distribution')
    contour2 = ax2.contourf(X, Y, derivative_2d, cmap='viridis', levels=20)
    ax2.set_title('Derivative with respect to mu_x')
    
    # Calculate the 1D Gaussian distribution and its derivative
    gaussian_1d = (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(
        -0.5 * ((x - mu_x)**2 / sigma_x**2)
    )
    derivative_1d = -gaussian_1d * (x - mu_x) / sigma_x**2
    
    # Create plots for 1D Gaussian and its derivative
    ax3.plot(x, gaussian_1d, color='blue', label='1D Gaussian')
    ax3.set_title('1D Gaussian Distribution')
    ax4.plot(x, derivative_1d, color='red', label='Derivative')
    ax4.set_title('Derivative with respect to mu_x')
    
    # Redraw the canvas
    fig.canvas.draw_idle()

# Connect the slider to the update function
mu_x_slider.on_changed(update_mu_x)

# Set tight layout for better spacing
plt.tight_layout()

plt.show()
