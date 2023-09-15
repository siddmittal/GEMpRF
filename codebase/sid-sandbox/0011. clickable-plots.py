import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

class ClickablePlots:
    def __init__(self, nRows, nCols):
        self.nRows_search_space = nRows
        self.nCols_search_space = nCols

        # Create linspace for x and y legends
        self.x_legends = np.linspace(-9, 9, nCols)
        self.y_legends = np.linspace(-9, 9, nRows)

        # Create grid coordinates
        self.grid_x, self.grid_y = np.meshgrid(np.arange(nCols), np.arange(nRows))

        # Plotting related
        self.fig = plt.figure()
        gs = GridSpec(1, 2, width_ratios=[1, 1])  # Create a grid with 1 row and 2 columns

        # Create the 2D subplot on the right
        self.ax = self.fig.add_subplot(gs[1])
        self.ax.set_title("Click on this 2D subplot")

        # Create a white 2D array for the 10x10 square with grid lines
        self.square_data = np.ones((nRows, nCols), dtype=int) * 255  # White color (255 in grayscale)

        # Plot the 2D square in the 2D subplot with grid lines
        self.ax.imshow(self.square_data, cmap='gray', extent=[0, nCols, 0, nRows], vmin=0, vmax=255, origin='lower', interpolation='none')

        # Set the x and y axis ticks and labels using legends
        self.ax.set_xticks(np.arange(nCols), minor=True)
        self.ax.set_yticks(np.arange(nRows), minor=True)
        self.ax.set_xticklabels([f"{int(legend)}" for legend in self.x_legends])
        self.ax.set_yticklabels([f"{int(legend)}" for legend in self.y_legends])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.ax.grid(True, color='black', linestyle='--', linewidth=1)  # Display grid lines in black

        # Add hair-cross along the middle of x and y-axis
        self.ax.axhline(nRows // 2 + 0.5, color='red', linestyle='--', linewidth=1)  # Horizontal line
        self.ax.axvline(nCols // 2 + 0.5, color='red', linestyle='--', linewidth=1)  # Vertical line

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def on_click(self, event):
        if event.inaxes == self.ax:  # Check if the click event is on the 2D subplot
            x_grid = event.xdata
            y_grid = event.ydata
            print(f"Clicked on 2D subplot at (legend_x, legend_y) = ({x_grid:.1f}, {y_grid:.1f})")
            print(f"Grid coordinates: ({int(x_grid)}, {int(y_grid)})")

            self.fig.canvas.draw()  # Redraw the figure

# Example usage:
my_plot = ClickablePlots(9, 9)
