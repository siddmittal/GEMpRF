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

# # from reportlab.pdfbase.ttfonts import TTFont
# # from reportlab.pdfbase import pdfmetrics
# pdfmetrics.registerFont(TTFont("Arial", "C:/Users/siddh/Downloads/arial.ttf"))


def convert_svg_to_png(svg_content):
    # Convert SVG to PNG
    png_output = cairosvg.svg2png(bytestring=svg_content)

    # Convert the PNG bytes to an image object
    img = plt.imread(BytesIO(png_output))

    return img

def plot_all(title, vista_centerx0, oprf_centery0, vista_centery0, oprf_centerx0, vista_sigma, oprf_sigma, vista_r2, oprf_r2, Result_image_path):
    # Create a 3x2 grid of subplots with increased spacing and no white space on the sides
    fig, axes = plt.subplots(2, 2, figsize=(12, 18), gridspec_kw={'hspace': 0.1, 'wspace': 0.6})

    our_method_name = 'GEM'

    # Increase label and title font size
    label_font_size = 15
    title_font_size = 20
    numbering_text_x = -0.3
    numbering_text_y = 1.0

    # Plot 1
    ax1 = axes[0, 0]
    ax1.scatter(vista_centerx0, oprf_centery0, s=1)  # Swap axes
    ax1.set_xlabel('vista', fontsize=label_font_size)  # Swap labels
    ax1.set_ylabel(our_method_name, fontsize=label_font_size)  # Swap labels
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
    ax2.set_xlabel('vista', fontsize=label_font_size)  # Swap labels
    ax2.set_ylabel(our_method_name, fontsize=label_font_size)  # Swap labels
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
    ax3.set_xlabel('vista', fontsize=label_font_size)  # Swap labels
    ax3.set_ylabel(our_method_name, fontsize=label_font_size)  # Swap labels
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
    ax4.set_xlabel('vista', fontsize=label_font_size)  # Swap labels
    ax4.set_ylabel(our_method_name, fontsize=label_font_size)  # Swap labels
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
    # ax5.set_xlabel('vista', fontsize=label_font_size)  # Swap labels
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
    # ax6.set_xlabel(our_method_name, fontsize=label_font_size)  # Swap labels
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
    # plt.savefig(Result_image_path, bbox_inches='tight')

    # Show the figure (optional)
    plt.show()
    print("test")
    




def plot_func(title, X, Y, xlab, ylab, xy_min, xy_max, Result_image_path):
    f = plt.figure(constrained_layout=True, figsize=(10, 10), dpi=100)
    plt.title(title)
    
    # plt.scatter(X, Y, c='blue', s=.3)
    # plt.scatter(X, Y, c='blue')
    plt.scatter(X, Y, alpha=0.5, label='Data Points')
    plt.plot((xy_min,xy_max), (xy_min,xy_max), 'r')
    
    plt.xlim(xy_min,xy_max)
    plt.ylim(xy_min,xy_max)
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    plt.gca().set_aspect('equal', 'box')


    plt.savefig(Result_image_path, bbox_inches='tight')

    plt.grid()
    plt.show()
    
    return f

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

##############################################----------------Main()----------------------------################################################
oprf_results_directory = 'Y:/data/stimsim23/derivatives/prfanalyze-oprf/analysis-11/sub-sidtest/ses-002'

for root, dirs, files in os.walk(oprf_results_directory):
    for directory in dirs:
        if directory.find('nojump') != -1: # i.e. don't process the results for bar-jump
            continue

        current_config_results_dir_path = os.path.join(root, directory)
        print(f'Processing -- {directory}')

        vista_data = merge_json_data("Y:/data/stimsim23/derivatives/prfanalyze-vista/analysis-11/sub-sidtest/ses-002/sub-sidtest_ses-002_task-bar_run-01_hemi-L_estimates.json",
                                    "Y:/data/stimsim23/derivatives/prfanalyze-vista/analysis-11/sub-sidtest/ses-002/sub-sidtest_ses-002_task-bar_run-01_hemi-R_estimates.json")



        oprf_data = merge_json_data(os.path.join(current_config_results_dir_path, "sub-sidtest_ses-002_task-bar_run-01_hemi-L_bold.json"),
                                    os.path.join(current_config_results_dir_path, "sub-sidtest_ses-002_task-bar_run-01_hemi-R_bold.json"))

        ### For bar-NOJUMP analysis        
        # vista_data = merge_json_data("Y:/data/stimsim23/derivatives/prfanalyze-vista/analysis-11/sub-sidtest/ses-002/sub-sidtest_ses-002_task-barnojump_run-01_hemi-L_estimates.json",
        #                     "Y:/data/stimsim23/derivatives/prfanalyze-vista/analysis-11/sub-sidtest/ses-002/sub-sidtest_ses-002_task-barnojump_run-01_hemi-R_estimates.json")

        # oprf_data = merge_json_data(os.path.join(current_config_results_dir_path, "sub-sidtest_ses-002_task-barnojump_run-01_hemi-L_bold.json"),
        #                     os.path.join(current_config_results_dir_path, "sub-sidtest_ses-002_task-barnojump_run-01_hemi-R_bold.json"))

        ## masking
        threshold  = 0.1
        r2_mask = [entry['R2'] is not None and not math.isnan(entry['R2']) and entry['R2'] > threshold for entry in vista_data]

        # Vista Data
        vista_centerx0 = [entry['Centerx0'] for i, entry in enumerate(vista_data) if r2_mask[i]]
        vista_centery0 = [entry['Centery0'] for i, entry in enumerate(vista_data) if r2_mask[i]]
        vista_sigma = [entry['sigmaMajor'] for i, entry in enumerate(vista_data) if r2_mask[i]]
        vista_r2 = [entry['R2'] for i, entry in enumerate(vista_data) if r2_mask[i]]

        # opRF Data
        oprf_centerx0 = [entry['Centerx0'] for i, entry in enumerate(oprf_data) if r2_mask[i]]
        oprf_centery0 = [entry['Centery0'] for i, entry in enumerate(oprf_data) if r2_mask[i]]
        oprf_sigma = [entry['sigmaMajor'] for i, entry in enumerate(oprf_data) if r2_mask[i]]
        oprf_r2 = [entry['R2'] for i, entry in enumerate(oprf_data) if r2_mask[i]]

        plot_title = directory
        plot_all(plot_title, vista_centerx0, oprf_centery0
                , vista_centery0, oprf_centerx0
                , vista_sigma, oprf_sigma
                , vista_r2, oprf_r2
                , Result_image_path = f"D:/results/comparison-plots/compare-grid-configs/vista-vs-gem_{directory}.svg")

print("Done...")