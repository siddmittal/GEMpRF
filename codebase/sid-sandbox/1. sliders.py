import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Create the main figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Set initial slider value
initial_value = 5

# Create a slider widget
slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(slider_ax, 'Value', 0, 10, valinit=initial_value)

# Create a textbox to display the current value
value_text = plt.text(0.02, 0.9, f'Current Value: {initial_value}', transform=ax.transAxes)

# Function to update the value displayed in the textbox
def update(val):
    value = slider.val
    value_text.set_text(f'Current Value: {value}')
    fig.canvas.draw_idle()

# Attach the update function to the slider's on_changed event
slider.on_changed(update)

plt.show()
