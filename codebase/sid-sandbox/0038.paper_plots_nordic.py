import sys
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")


#%% arrange them on a pdf
# from PdfImage import PdfImage
from placeImages import placeImage
from tqdm import tqdm
from os import path
import matplotlib.pyplot as plt
import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from svglib.svglib import svg2rlg

from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
pdfmetrics.registerFont(TTFont("Arial", "C:/Users/siddh/Downloads/arial.ttf"))

in_path = '' #'/z/fmri/data/stimsim23/plots/nordic_comp'

# set up arrangement
arrangement = np.array([['r','phi'],
                        ['s','varexp']])
numbering   = np.array([['A','B'],
                        ['C','D']])

# create a canvas
oFileName = f'sid_test.pdf'
oFilePath = "D:/results/comparison-plots"
c = canvas.Canvas(path.join(oFilePath,oFileName))

# get example file
drawing = svg2rlg(path.join(in_path, "D:/results/comparison-plots/test.svg"))

# Sid
pageWidth = 1000
num_fig_per_row = 2
single_plot_width = drawing.width
single_plot_height = drawing.height
margin_on_both_sides = 10
total_margin_width = margin_on_both_sides * (num_fig_per_row + 1)
available_width_per_plot = int((pageWidth - total_margin_width)/num_fig_per_row)
plot_height = available_width_per_plot / drawing.width * drawing.height

# set page size
plotH   = plot_height
plotW   = available_width_per_plot
marginH = margin_on_both_sides
marginW = margin_on_both_sides
pageW = 1000
pageH = 2000
c.setPageSize((pageW, pageH))

# # set page size
# plotH   = 10 * cm
# plotW   = plotH / drawing.height * drawing.width
# marginH = 0.15 * cm
# marginW = 0.15 * cm
# pageW = len(arrangement[0]) * (plotW + marginW) + marginW
# pageH = len(arrangement)    * (plotH + marginH) + marginH
# c.setPageSize((pageW, pageH))

# place images
c.setFont('Arial', 16)
c.drawCentredString(500 - 5, 16, "Low Noise")

for I,a in enumerate(arrangement.flatten()):
    I,J = np.unravel_index(I, (2,2))

    xpos = J     * (plotW) # (plotW + marginW) + marginW
    ypos = pageH - (I * (plotH + marginH) + marginH)

    placeImage(c, path.join(in_path, f"D:/results/comparison-plots/test.svg"), xpos, ypos, resize=(plotW,plotH), forceAR=True)

    # c.setFont('Arial', 16)
    # c.drawCentredString(xpos+20, ypos-16, numbering[I,J])

c.showPage()
c.save()
