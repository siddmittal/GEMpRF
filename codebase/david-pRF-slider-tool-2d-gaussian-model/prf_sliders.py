import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import scipy.stats as st
from scipy.io import loadmat
import os

maxEcc = 9

# load the stim images
script_directory = os.path.dirname(os.path.abspath(__file__))
fffFit_path = os.path.join(script_directory, 'fffFit.mat')
params = loadmat(fffFit_path)['params'][0]
window = params['stim'][0]['stimwindow'][0][0].flatten().reshape(101, 101).astype('bool')
stimImages = params['analysis'][0]['allstimimages'][0][0].T
stimImagesUnConv = params['analysis'][0]['allstimimages_unconvolved'][0][0].T


def _createmask(shape, otherRratio=None):
    x0, y0 = shape[0] // 2, shape[1] // 2
    n = shape[0]
    if otherRratio is not None:
        r = shape[0] / 2 * otherRratio
    else:
        r = shape[0] // 2

    y, x = np.ogrid[-x0:n - x0, -y0:n - y0]
    return x * x + y * y <= r * r


# The image to be plotted
def f(x, y, s):

    xx = np.linspace(-maxEcc, maxEcc, 101)

    covMap = np.zeros((len(xx), len(xx)))

    kern1dx = st.norm.pdf(xx, x, s)
    kern1dy = st.norm.pdf(xx, -y, s)
    kern2d  = np.outer(kern1dx, kern1dy)
    covMap  = kern2d / kern2d.max()

    msk = _createmask(covMap.shape)

    covMap[~msk] = 0
    covMap = covMap.T

    return xx, kern1dx, kern1dy, covMap

# Sid's test function to display
def displayInfo(valueToDisplay, title):    
    plt.figure()  # Open a new plot window
    plt.imshow(valueToDisplay, cmap='gray')
    plt.axis('off')    
    plt.title(title)
    plt.show()


# the TC
def g(x, y, s):
    xx = np.linspace(-maxEcc, maxEcc, 101)
    X0, Y0 = np.meshgrid(xx, xx)

    pRF = np.exp(((x - X0)**2 + (y - Y0)**2) / (-2 * s**2))
    displayInfo(pRF, 'pRF')

    tc = pRF[window].dot(stimImages)
    displayInfo(stimImages, 'stimImages')
    tc /= tc.max()

    return tc, np.linspace(0, len(stimImages.T) * 2, len(stimImages.T))


t = np.linspace(0, 1, 1000)

# Define initial parameters
initX = 0
initY = 0
initS = 1

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(2, 2)
xx, plx, ply, cov = f(initX, initY, initS)
im = ax[1, 0].imshow(cov, extent=[-maxEcc, maxEcc, -maxEcc, maxEcc])

xhist, = ax[0, 0].plot(xx, plx)
ax[0, 0].set_xlim((-maxEcc, maxEcc))
ax[0, 0].set_ylim((-.02, .5))
ax[0, 0].get_shared_x_axes().join(ax[0, 0], ax[1, 0])
ax[0, 0].set_aspect(np.diff(ax[0, 0].get_xlim())[0] / np.diff(ax[0, 0].get_ylim())[0])

yhist, = ax[1, 1].plot(ply, xx)
ax[1, 1].set_ylim((-maxEcc, maxEcc))
ax[1, 1].set_xlim((-.02, .5))
ax[1, 1].get_shared_y_axes().join(ax[1, 1], ax[1, 0])
ax[1, 1].set_aspect(np.diff(ax[1, 1].get_xlim())[0] / np.diff(ax[1, 1].get_ylim())[0])

ty, tx = g(initX, initY, initS)
tc, = ax[0, 1].plot(tx, ty)
# ax[0, 1].set_aspect(np.diff(ax[1, 1].get_xlim())[0] / np.diff(ax[1, 1].get_ylim())[0])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.15, bottom=0.25)

# Make a horizontal slider to control x.
axX = fig.add_axes([0.2, 0.1, 0.25, 0.03])
X_slider = Slider(
    ax=axX,
    label='X pos [°]',
    valmin=-maxEcc,
    valmax=maxEcc,
    valinit=initX,
)

# Make a vertically oriented slider to control y.
axY = fig.add_axes([0.1, 0.25, 0.0225, 0.3])
Y_slider = Slider(
    ax=axY,
    label='Y pos [°]',
    valmin=-maxEcc,
    valmax=maxEcc,
    valinit=initY,
    orientation="vertical"
)

# Make a vertically oriented slider to control sigma
axS = fig.add_axes([0.2, 0.15, 0.25, 0.03])
S_slider = Slider(
    ax=axS,
    label='pRF size [°]',
    valmin=.1,
    valmax=4,
    valinit=initS,
)


# The function to be called anytime a slider's value changes
def update(val):
    xx, plx, ply, cov = f(X_slider.val, Y_slider.val, S_slider.val)
    im.set_data(cov)
    tc.set_ydata(g(X_slider.val, Y_slider.val, S_slider.val)[0])
    xhist.set_ydata(plx)
    yhist.set_xdata(np.flip(ply))
    fig.canvas.draw_idle()


# register the update function with each slider
X_slider.on_changed(update)
Y_slider.on_changed(update)
S_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    X_slider.reset()
    Y_slider.reset()
    S_slider.reset()


button.on_clicked(reset)

plt.show()
