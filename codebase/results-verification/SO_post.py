import matplotlib.pyplot as plt
import numpy as np
import cairosvg
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# SVG file
svg_filepath = "Y:/data/stimsim23/derivatives/prfresult/analysis-06/covMap/sub-sidtest/ses-001/oprf_covmap.svg"

def convert_svg_to_png(svg_content):
    # Convert SVG to PNG
    png_output = cairosvg.svg2png(bytestring=svg_content)

    # Convert the PNG bytes to an image object
    img = plt.imread(BytesIO(png_output))

    return img


def plot_all(title, method_1_centerx0, method_2_centery0, method_1_centery0, method_2_centerx0, method_1_sigma, method_2_sigma, method_1_r2, method_2_r2, Result_image_path):    
    fig, axes = plt.subplots(3, 2, figsize=(12, 18), gridspec_kw={'hspace': 0.6, 'wspace': 0.4})

    # Increase label and title font size
    label_font_size = 26

    # Plot 1
    ax1 = axes[0, 0]
    ax1.scatter(method_2_centerx0, method_1_centery0)
    ax1.set_xlabel('x', fontsize=label_font_size)
    ax1.set_ylabel('y', fontsize=label_font_size)
    ax1.text(-0.3, 1.2, '(a)', transform=ax1.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')

    # Plot 2
    ax2 = axes[0, 1]
    ax2.scatter(method_2_centery0, method_1_centerx0)
    ax2.set_xlabel('x', fontsize=label_font_size)
    ax2.set_ylabel('y', fontsize=label_font_size)
    ax2.text(-0.3, 1.2, '(b)', transform=ax2.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')

    # Plot 3
    ax3 = axes[1, 0]
    ax3.scatter(method_2_sigma, method_1_sigma)
    ax3.set_xlabel('x', fontsize=label_font_size)
    ax3.set_ylabel('y', fontsize=label_font_size)
    ax3.text(-0.3, 1.2, '(c)', transform=ax3.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')

    # Plot 4
    ax4 = axes[1, 1]
    ax4.scatter(method_2_r2, method_1_r2)
    ax4.set_xlabel('x', fontsize=label_font_size)
    ax4.set_ylabel('y', fontsize=label_font_size)
    ax4.text(-0.3, 1.2, '(d)', transform=ax4.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')

    # Plot 5: read and plot SVG content
    ax5 = axes[2, 0]
    with open(svg_filepath, 'rb') as svg_file:
        svg_content = svg_file.read()
        img = convert_svg_to_png(svg_content)
        ax5.imshow(img)    
    ax5.set_xticks([]) # remove ticks for the SVG plot
    ax5.set_yticks([])

    # Plot 6: read and plot SVG content
    ax6 = axes[2, 1]
    with open(svg_filepath, 'rb') as svg_file:
        svg_content = svg_file.read()
        img = convert_svg_to_png(svg_content)
        ax6.imshow(img)    
    ax6.set_xticks([]) # remove ticks for the SVG plot
    ax6.set_yticks([])


    # Set the title for the entire figure
    plt.suptitle(title, fontsize=24)

    # Remove legends
    for ax in axes.flatten():
        ax.legend().set_visible(False)

    # Save the figure to the specified path with no white space
    plt.tight_layout()
    plt.savefig(Result_image_path)

    # Show the figure (optional)
    plt.show()
    


plot_all('', np.random.randint(100, size=(5)), np.random.randint(100, size=(5))
         , np.random.randint(100, size=(5)), np.random.randint(100, size=(5))
         , np.random.randint(100, size=(5)), np.random.randint(100, size=(5))
         , np.random.randint(100, size=(5)), np.random.randint(100, size=(5))
         , Result_image_path="D:/All.svg")