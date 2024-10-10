import cairosvg
import pandas as pd 
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.units import cm, inch
import cairosvg
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# from reportlab.pdfbase.ttfonts import TTFont
# from reportlab.pdfbase import pdfmetrics
# pdfmetrics.registerFont(TTFont("Arial", "C:/Users/siddh/Downloads/arial.ttf"))


def convert_svg_to_png(svg_content):
    # Convert SVG to PNG
    png_output = cairosvg.svg2png(bytestring=svg_content)

    # Convert the PNG bytes to an image object
    img = plt.imread(BytesIO(png_output))

    return img

def plot_all(title, xlabel, ylabel, vista_centerx0, oprf_centery0, vista_centery0, oprf_centerx0, vista_sigma, oprf_sigma, vista_r2, oprf_r2, Result_image_path):
    # Create a 3x2 grid of subplots with increased spacing and no white space on the sides
    fig, axes = plt.subplots(2, 2, figsize=(12, 18), gridspec_kw={'hspace': 0.1, 'wspace': 0.6}, constrained_layout=True)

    # Increase label and title font size
    label_font_size = 15
    title_font_size = 20
    numbering_text_x = -0.3
    numbering_text_y = 1.0

    # Plot 1
    ax1 = axes[0, 0]
    ax1.scatter(vista_centerx0, oprf_centery0, s=1)  # Swap axes
    ax1.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
    ax1.set_ylabel(ylabel, fontsize=label_font_size)  # Swap labels
    ax1.set_title(r'$\mu_x$', fontsize=title_font_size)
    ax1.text(numbering_text_x, numbering_text_y, '(a)', transform=ax1.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
    ax1.set_xlim(-9, 9)
    ax1.plot((-9, 9), (-9, 9), 'r')
    ax1.set_xlim(-9, 9)
    ax1.set_ylim(-9, 9)
    ax1.set_aspect('equal', 'box')

    # Plot 2
    ax2 = axes[0, 1]
    ax2.scatter(vista_centery0, oprf_centerx0, s=1)  # Swap axes
    ax2.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
    ax2.set_ylabel(ylabel, fontsize=label_font_size)  # Swap labels
    ax2.set_title(r'$\mu_y$', fontsize=title_font_size)
    ax2.text(numbering_text_x, numbering_text_y, '(b)', transform=ax2.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
    ax2.set_xlim(-9, 9)
    ax2.plot((-9, 9), (-9, 9), 'r')
    ax2.set_xlim(-9, 9)
    ax2.set_ylim(-9, 9)
    ax2.set_aspect('equal', 'box')

    # Plot 3
    ax3 = axes[1, 0]
    ax3.scatter(vista_sigma, oprf_sigma, s=1)  # Swap axes
    ax3.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
    ax3.set_ylabel(ylabel, fontsize=label_font_size)  # Swap labels
    ax3.set_title(r'$\sigma$', fontsize=title_font_size)
    ax3.text(numbering_text_x, numbering_text_y, '(c)', transform=ax3.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
    ax3.set_xlim(0, 5)
    ax3.plot((0, 5), (0, 5), 'r')
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 5)
    ax3.set_aspect('equal', 'box')

    # Plot 4
    ax4 = axes[1, 1]
    ax4.scatter(vista_r2, oprf_r2, s=1)  # Swap axes
    ax4.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
    ax4.set_ylabel(ylabel, fontsize=label_font_size)  # Swap labels
    ax4.set_title(r'$R^2$', fontsize=title_font_size)
    ax4.text(numbering_text_x, numbering_text_y, '(d)', transform=ax4.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
    ax4.set_xlim(0, 0.8)
    ax4.plot((0, 0.8), (0, 0.8), 'r')
    ax4.set_xlim(0, 0.8)
    ax4.set_ylim(0, 0.8)
    ax4.set_aspect('equal', 'box')

    # # Plot 5: read and plot SVG content
    # ax5 = axes[2, 0]
    # with open("Y:/data/stimsim23/derivatives/prfresult/analysis-03/covMap/sub-sidtest/ses-001/vista_covmap.svg", 'rb') as svg_file:
    #     svg_content = svg_file.read()
    #     img = convert_svg_to_png(svg_content)
    #     ax5.imshow(img)    
    # ax5.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
    # ax5.set_title('Coverage Map', fontsize=title_font_size)
    # ax5.text(numbering_text_x - 0.01, numbering_text_y, '(e)', transform=ax5.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
    # ax5.set_xticks([]) # remove ticks for the SVG plot
    # ax5.set_yticks([])

    # # Plot 6: read and plot SVG content
    # ax6 = axes[2, 1]
    # with open("Y:/data/stimsim23/derivatives/prfresult/analysis-06/covMap/sub-sidtest/ses-001/oprf_covmap.svg", 'rb') as svg_file:
    #     svg_content = svg_file.read()
    #     img = convert_svg_to_png(svg_content)
    #     ax6.imshow(img)    
    # ax6.set_xlabel(ylabel, fontsize=label_font_size)  # Swap labels
    # ax6.set_title('Coverage Map', fontsize=title_font_size)
    # ax6.text(numbering_text_x - 0.01, numbering_text_y, '(f)', transform=ax6.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
    # ax6.set_xticks([]) # remove ticks for the SVG plot
    # ax6.set_yticks([])    

    # Set the title for the entire figure
    plt.suptitle(title, fontsize=24)

    # Remove legends
    for ax in axes.flatten():
        ax.legend().set_visible(False)

    # Save the figure to the specified path with no white space
    plt.tight_layout()
    plt.savefig(Result_image_path, bbox_inches='tight')

    # Show the figure (optional)
    plt.show()
    print("test")
    



def merge_json_data(file_path_1, file_path_2):
    try:
        # Load data from the first JSON file (Method 1)
        with open(file_path_1, 'r') as method1_file:
            data_1 = json.load(method1_file)

        # Load data from the second JSON file (Method 2)
        with open(file_path_2, 'r') as method2_file:
            data_2 = json.load(method2_file)

        # Merge data from both methods
        consolidated_data = data_1 + data_2        

        return consolidated_data

    except FileNotFoundError:
        print("One or both of the JSON files not found.")
        return None, None


#############################################################---------------------------Main--------------------------------------------------################################################
file_path = "D:/code/sid-git/fmri/codebase/results-verification/comparison-dictionary.xlsx"  # Replace with your file path
df = pd.read_excel(file_path, sheet_name = 'bar-nojump_comparison-plots')

# Iterate through rows in the DataFrame
for index, row in df.iterrows():
    plot_title = row['Title']
    result_filename = row['ResultFilename']
    xlabel = row['xlabel']
    ylabel = row['ylabel']
    xDataSrc = row['XAxisDataSrc']
    yDataSrc = row['YAxisDataSrc']
    
    xLeftHemiDataSrc, xRightHemiDataSrc = xDataSrc.split('+')
    x_axis_data = merge_json_data(xLeftHemiDataSrc, xRightHemiDataSrc)


    yLeftHemiDataSrc, yRightHemiDataSrc = yDataSrc.split('+')
    y_axis_data = merge_json_data(yLeftHemiDataSrc, yRightHemiDataSrc)

    ## masking
    threshold  = 0.1
    r2_mask = [entry['R2'] is not None and not math.isnan(entry['R2']) and entry['R2'] > threshold for entry in x_axis_data]

    # Vista Data
    vista_centerx0 = [entry['Centerx0'] for i, entry in enumerate(x_axis_data) if r2_mask[i]]
    vista_centery0 = [entry['Centery0'] for i, entry in enumerate(x_axis_data) if r2_mask[i]]
    vista_sigma = [entry['sigmaMajor'] for i, entry in enumerate(x_axis_data) if r2_mask[i]]
    vista_r2 = [entry['R2'] for i, entry in enumerate(x_axis_data) if r2_mask[i]]

    # opRF Data
    oprf_centerx0 = [entry['Centerx0'] for i, entry in enumerate(y_axis_data) if r2_mask[i]]
    oprf_centery0 = [entry['Centery0'] for i, entry in enumerate(y_axis_data) if r2_mask[i]]
    oprf_sigma = [entry['sigmaMajor'] for i, entry in enumerate(y_axis_data) if r2_mask[i]]
    oprf_r2 = [entry['R2'] for i, entry in enumerate(y_axis_data) if r2_mask[i]]
    
    plot_all(title=plot_title
             , xlabel=xlabel
             , ylabel=ylabel
             , vista_centerx0=vista_centerx0
             , oprf_centery0=oprf_centery0
            , vista_centery0=vista_centery0
            , oprf_centerx0= oprf_centerx0
            , vista_sigma= vista_sigma
            , oprf_sigma= oprf_sigma
            , vista_r2= vista_r2
            , oprf_r2= oprf_r2
            , Result_image_path = os.path.join('D:/results/comparison-plots/compare-grid-configs/', result_filename))

print("Done...")
