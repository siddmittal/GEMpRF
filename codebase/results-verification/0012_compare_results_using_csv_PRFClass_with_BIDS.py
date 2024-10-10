import sys
# sys.path.append('/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/dlinhardt/pythonclass')
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")
from PRFclass import PRF

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

def plot_all(title, xlabel, ylabel, xData, yData, mask, vista_centerx0, oprf_centery0, vista_centery0, oprf_centerx0, vista_sigma, oprf_sigma, vista_r2, oprf_r2, Result_image_path, plot_coverage_map, xCovMapPath, yCovMapPath):
    # Create a 3x2 grid of subplots with increased spacing and no white space on the sides
    fig, axes = plt.subplots(3, 2, figsize=(12, 18), gridspec_kw={'hspace': 0.1, 'wspace': 0.6}, constrained_layout=True)

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
    ax1.set_xlim(-13.5,13.5)
    ax1.plot((-13.5,13.5), (-13.5,13.5), 'r')
    ax1.set_xlim(-13.5,13.5)
    ax1.set_ylim(-13.5,13.5)
    ax1.grid()
    ax1.set_aspect('equal', 'box')

    # Plot 2
    ax2 = axes[0, 1]
    ax2.scatter(vista_centery0, oprf_centerx0, s=1)  # Swap axes
    ax2.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
    ax2.set_ylabel(ylabel, fontsize=label_font_size)  # Swap labels
    ax2.set_title(r'$\mu_y$', fontsize=title_font_size)
    ax2.text(numbering_text_x, numbering_text_y, '(b)', transform=ax2.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
    ax2.set_xlim(-13.5,13.5)
    ax2.plot((-13.5,13.5), (-13.5,13.5), 'r')
    ax2.set_xlim(-13.5,13.5)
    ax2.set_ylim(-13.5,13.5)
    ax2.grid()
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
    # ax3.set_xlim(0, 5)
    # ax3.set_ylim(0, 5)
    ax3.grid()
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
    ax4.grid()
    ax4.set_aspect('equal', 'box')

    if(not plot_coverage_map):
        # Plot 5: Ecc. plot
        ax5 = axes[2, 0]
        ax5.scatter(xData.ecc0[mask], yData.ecc0[mask], s=1)  # Swap axes
        ax5.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
        ax5.set_ylabel(ylabel, fontsize=label_font_size)  # Swap labels
        ax5.set_title('Ecc', fontsize=title_font_size)
        ax5.text(numbering_text_x - 0.01, numbering_text_y, '(e)', transform=ax5.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
        ax5.set_xlim(0, 13.5)
        ax5.plot((0, 13.5), (0, 13.5), 'r')
        ax5.set_xlim(0, 13.5)
        ax5.set_ylim(0, 13.5)
        ax5.grid()
        ax5.set_aspect('equal', 'box')

        # Plot 6: ploar angle plot
        ax6 = axes[2, 1]
        ax6.scatter(xData.pol0[mask], yData.pol0[mask], s=1)  # Swap axes
        ax6.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
        ax6.set_ylabel(ylabel, fontsize=label_font_size)  # Swap labels
        ax6.set_title('Polar Angle', fontsize=title_font_size)
        ax6.text(numbering_text_x - 0.01, numbering_text_y, '(f)', transform=ax6.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
        ax6.set_xlim(0, 6.28)
        ax6.plot((0, 6.28), (0, 6.28), 'r')
        ax6.set_xlim(0, 6.28)
        ax6.set_ylim(0, 6.28)
        ax6.grid()
        ax6.set_aspect('equal', 'box')

    else:
        # Plot 5: read and plot SVG content
        ax5 = axes[2, 0]
        with open(xCovMapPath, 'rb') as svg_file:
            svg_content = svg_file.read()
            img = convert_svg_to_png(svg_content)
            ax5.imshow(img)    
        ax5.set_xlabel(xlabel, fontsize=label_font_size)  # Swap labels
        ax5.set_title('Coverage Map', fontsize=title_font_size)
        ax5.text(numbering_text_x - 0.01, numbering_text_y, '(e)', transform=ax5.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
        ax5.set_xticks([]) # remove ticks for the SVG plot
        ax5.set_yticks([])

        # Plot 6: read and plot SVG content
        ax6 = axes[2, 1]
        with open(yCovMapPath, 'rb') as svg_file:
            svg_content = svg_file.read()
            img = convert_svg_to_png(svg_content)
            ax6.imshow(img)    
        ax6.set_xlabel(ylabel, fontsize=label_font_size)  # Swap labels
        ax6.set_title('Coverage Map', fontsize=title_font_size)
        ax6.text(numbering_text_x - 0.01, numbering_text_y, '(f)', transform=ax6.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
        ax6.set_xticks([]) # remove ticks for the SVG plot
        ax6.set_yticks([])    

    # Set the title for the entire figure
    plt.suptitle(title, fontsize=24)

    # Remove legends
    for ax in axes.flatten():
        ax.legend().set_visible(False)

    # Save the figure to the specified path with no white space
    plt.tight_layout()
    plt.savefig(Result_image_path, bbox_inches='tight')

    # Show the figure (optional)
    # plt.show()
    print("test")
    

#############################################################---------------------------Main--------------------------------------------------################################################
file_path = "D:/code/sid-git/fmri/codebase/results-verification/comparison-dictionary.xlsx"  # Replace with your file path
df = pd.read_excel(file_path, sheet_name = 'stimuli-compare', dtype=str)

# Iterate through rows in the DataFrame
for index, row in df.iterrows():
    plot_title = row['Title']
    result_filename = row['ResultFilename']
    plot_coverage_map = row['PlotCoverageMap']
    xlabel = row['xlabel']
    
    ## x-axis
    xStudy = row['xStudy']
    xSubject = row['xSubject']
    xSession = row['xSession']
    xTask = row['xTask']
    xRun = row['xRun']
    xMethod = row['xMethod']
    xAnalysis = row['xAnalysis']
    xHemi = row['xHemi'] if row['xHemi'] in ['L', 'R'] else ''
    xBaseP = row['xBaseP']
    xOrientation = row['xOrientation']

    ## y-axis
    ylabel = row['ylabel']
    yStudy = row['yStudy']
    ySubject = row['ySubject']
    ySession = row['ySession']
    yTask = row['yTask']
    yRun = row['yRun']
    yMethod = row['yMethod']
    yAnalysis = row['yAnalysis']
    yHemi = row['yHemi'] if row['yHemi'] in ['L', 'R'] else ''
    yBaseP = row['yBaseP']
    yOrientation = row['yOrientation']
    
    xAxisData = PRF.from_docker(xStudy, xSubject, xSession, xTask, xRun, method=xMethod, analysis=xAnalysis, hemi=xHemi, orientation=xOrientation, baseP= xBaseP)
    yAxisData = PRF.from_docker(yStudy, ySubject, ySession, yTask, yRun, method=yMethod, analysis=yAnalysis, hemi=yHemi, orientation=yOrientation, baseP= yBaseP)
    xAxisData.maskROI('V1')
    yAxisData.maskROI('V1')
    xAxisData.maskVarExp(0.1)
    yAxisData.maskVarExp(0.1)

    xAxisData.maskSigma(0.2, 10)
    yAxisData.maskSigma(0.2, 10)

    mask = xAxisData.mask & yAxisData.mask

    # coverage maps
    if(plot_coverage_map):
        xAxisData.plot_covMap(maxEcc = 9)
        xCovMapPath = xAxisData.plot_covMap(force = True, show=False, save=True, maxEcc = 9)
        yAxisData.plot_covMap(maxEcc = 9)
        yCovMapPath = yAxisData.plot_covMap(force = True, show=False, save=True, maxEcc = 9)

    plot_all(title=plot_title + f'\nX:{xAxisData.mask.sum()}, y:{yAxisData.mask.sum()}'
            , xlabel=xlabel
            , ylabel=ylabel    
            , xData=xAxisData
            , yData= yAxisData
            , mask=mask        
            , vista_centerx0=xAxisData.x0[mask]
            , oprf_centery0=yAxisData.x0[mask]
        , vista_centery0=xAxisData.y0[mask]
        , oprf_centerx0= yAxisData.y0[mask]
        , vista_sigma= xAxisData.s0[mask]
        , oprf_sigma= yAxisData.s0[mask]
        , vista_r2= xAxisData.varexp0[mask]
        , oprf_r2= yAxisData.varexp0[mask]
        , Result_image_path = os.path.join('D:/results/comparison-plots/compare-bar-nojump/', result_filename)
        , plot_coverage_map = plot_coverage_map
        , xCovMapPath=xCovMapPath
        , yCovMapPath = yCovMapPath)
    
    print('plotted...')

print("Done...")
