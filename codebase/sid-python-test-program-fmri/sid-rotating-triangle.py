import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

triangle = Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]], closed=True, fc='blue', alpha=0.5)
ax.add_patch(triangle)

lines = [ax.plot([], [], 'k--')[0] for _ in range(4)]
angles = np.linspace(0, 2*np.pi, 361)
circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linestyle='--')
ax.add_artist(circle)

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    theta = np.radians(-frame % 360)  # Negative sign for clockwise rotation

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    rotated_points = np.dot(triangle.get_xy(), rotation_matrix.T)
    triangle.set_xy(rotated_points)

    for i, line in enumerate(lines):
        line.set_data([0, np.cos(angles[i])], [0, np.sin(angles[i])])

    return lines + [triangle]

ani = FuncAnimation(fig, update, frames=360, init_func=init, blit=True)
plt.show()
