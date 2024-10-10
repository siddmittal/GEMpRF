# NOTE: Use "pip_fmri_py3.8" environment for this script.

import numpy as np
import matplotlib.pyplot as plt


import cairosvg
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def pol2cart(r,phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x,y

phi = np.linspace(-np.pi, np.pi)
# r = np.linspace(0,7,10) + np.cumsum(np.linspace(0,1,10) ) ## #NOTE: to incorporate cortical magnification
r = np.arange(0, 10, 1) ##np.linspace(0,10,10) ## #NOTE: without cortical magnification

phi_m, r_m = np.meshgrid(phi, r)
# plt.polar(phi_m.flatten(), r_m.flatten(), '.')
# plt.plot(*pol2cart(r_m,phi_m), '.')
plt.scatter(*pol2cart(r_m,phi_m), s=50) # NOTE: s=200 used for paper
min_value = -10
max_value = 10

plt.xticks(np.arange(min_value, max_value+1, 2.0))
plt.yticks(np.arange(min_value, max_value+1, 2.0))
plt.grid()
plt.gca().set_aspect('equal','box')

# selected pRF positions
plt.scatter(0, 0, color='red', s=100)
plt.scatter(3, 3, color='red', s=100)
plt.scatter(6, 6, color='red', s=100)

# Save the figure to the specified path with no white space
plt.tight_layout()
# plt.savefig("C:/Users/siddh/Downloads/grid-cortical-magnification-3.svg", bbox_inches='tight', transparent=True)
plt.savefig(r"D:\results\comparison-plots\gem-paper-simulated-data/chosen-pRF-posistions-simulated-data.svg", bbox_inches='tight', transparent=True)


plt.show()



print()


# def pol2cart(r,phi):
#     x = r * np.cos(phi)
#     y = r * np.sin(phi)
#     return x,y

# phi = np.linspace(-np.pi, np.pi)
# r = np.linspace(0,8)
# plt.polar(phi, r)
# phi_m, r_m = np.meshgrid(phi, r)
# plt.polar(phi_m, r_m, '.')
# plt.polar(phi_m.flatten(), r_m.flatten(), '.')
# r = np.linspace(0,8,10)
# phi_m, r_m = np.meshgrid(phi, r)
# plt.polar(phi_m.flatten(), r_m.flatten(), '.')
# r = np.logspace(0,8,10)
# phi_m, r_m = np.meshgrid(phi, r)
# plt.polar(phi_m.flatten(), r_m.flatten(), '.')
# # r = np.linspace(0,7,10) + np.linspace(0,1) 
# r = np.linspace(0,7,10) + np.linspace(0,1,10) 
# phi_m, r_m = np.meshgrid(phi, r)
# plt.polar(phi_m.flatten(), r_m.flatten(), '.')
# np.diff(r)
# np.linspace(0,1) 
# np.cumsum(np.linspace(0,1) )

# r = np.linspace(0,7,10) + np.cumsum(np.linspace(0,1,10) )
# plt.polar(phi_m.flatten(), r_m.flatten(), '.')
# phi_m, r_m = np.meshgrid(phi, r)
# plt.polar(phi_m.flatten(), r_m.flatten(), '.')
# np.diff(r)

# plt.plot(*pol2cart(r_m,phi_m), '.')
# plt.scatter(*pol2cart(r_m,phi_m))
# plt.scatter(*pol2cart(r_m,phi_m))
# plt.scatter(*pol2cart(r_m,phi_m))
# plt.grid()
# plt.gca().set_aspect('equal','box')	