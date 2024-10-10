import os
import json
import numpy as np
import pandas as pd

def load_json_data(sub, ses, run, hemi):
    if sub == '':
        sub = '001'
        ses = '001'
        run = '01'
        hemi = 'L' 
    filepath = f"Y:/data/stimsim24/BIDS/derivatives/prfanalyze-gem/analysis-03_AsusCorrect/sub-{sub:03}/ses-{ses:03}/sub-{sub:03}_ses-{ses:03}_task-bar_run-{run:02}_hemi-{hemi}_estimates.json"
    with open(filepath, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)




import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PREVIOUS_SUB = ''
PREVIOUS_SES = ''
PREVIOUS_RUN = ''

def update_data_info():
    global PREVIOUS_SUB, PREVIOUS_SES, PREVIOUS_RUN
    sub = sub_entry.get()
    ses = ses_entry.get()
    run = run_entry.get()
    if sub != PREVIOUS_SUB or ses != PREVIOUS_SES or run != PREVIOUS_RUN:
        PREVIOUS_SUB = sub
        PREVIOUS_SES = ses
        PREVIOUS_RUN = run
        data = load_json_data(sub, ses, run, 'L')
        voxel_slider.config(to=len(data)-1)
        update_plots(data, 0)

def update_plots(data, voxel_number):
    voxel_data = data.iloc[voxel_number]
    eccentricity = (voxel_data['Centerx0'] ** 2 + voxel_data['Centery0'] ** 2) ** 0.5

    # Scatter Plot
    scatter_figure.clear()
    scatter_plot = scatter_figure.add_subplot(111)
    scatter_plot.scatter(range(len(data)), (data['Centerx0'] ** 2 + data['Centery0'] ** 2) ** 0.5)
    scatter_plot.set_title('Eccentricity Scatter Plot')
    scatter_canvas.draw()

    # Histogram Plot
    hist_figure.clear()
    hist_plot = hist_figure.add_subplot(111)
    hist_plot.hist((data['Centerx0'] ** 2 + data['Centery0'] ** 2) ** 0.5, bins=20)
    hist_plot.set_title('Eccentricity Histogram')
    hist_canvas.draw()

def load_data():
    sub = sub_entry.get()
    ses = ses_entry.get()
    run = run_entry.get()
    hemi = hemi_entry.get()
    
    # data = load_json_data(sub, ses, run, hemi)

    # loading L and R data together
    data_L = load_json_data(sub, ses, run, 'L')
    data_R = load_json_data(sub, ses, run, 'R')
    data = pd.concat([data_L, data_R])
    
    # Update the label with Centerx0, Centery0 for the selected voxel
    voxel_number = voxel_slider.get()
    update_plots(data, voxel_number)
    
    param_label.config(text=f"Estimation Parameters Values: (Centerx0: {data.iloc[voxel_number]['Centerx0']}, Centery0: {data.iloc[voxel_number]['Centery0']})")

# Initialize the main window
root = tk.Tk()
root.title("pRF Estimation Viewer")

# Entry fields for sub, ses, run
tk.Label(root, text="Sub").grid(row=0, column=0)
sub_entry = tk.Entry(root)
sub_entry.grid(row=0, column=1)

tk.Label(root, text="Ses").grid(row=0, column=2)
ses_entry = tk.Entry(root)
ses_entry.grid(row=0, column=3)

tk.Label(root, text="Run").grid(row=0, column=4)
run_entry = tk.Entry(root)
run_entry.grid(row=0, column=5)

tk.Label(root, text="Hemi").grid(row=0, column=6)
hemi_entry = tk.Entry(root)
hemi_entry.grid(row=0, column=7)

# Label for displaying Centerx0, Centery0
param_label = tk.Label(root, text="Estimation Parameters Values: (Centerx0, Centery0)")
param_label.grid(row=1, column=0, columnspan=8)

# Slider for selecting the voxel number
voxel_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, label="Voxel Number Slider")
voxel_slider.grid(row=2, column=0, columnspan=8)

# Buttons for loading setup and lab images (placeholders)
setup_button = tk.Button(root, text="Load Setup Image", command=lambda: print("Setup Image Loaded"))
setup_button.grid(row=4, column=0, columnspan=4)

lab_button = tk.Button(root, text="Load Lab Image", command=lambda: print("Lab Image Loaded"))
lab_button.grid(row=4, column=4, columnspan=4)

# Button to load data based on inputs
load_button = tk.Button(root, text="Load Data", command=load_data)
load_button.grid(row=3, column=0, columnspan=8)

# Scatter plot area
scatter_figure = Figure(figsize=(5, 4), dpi=100)
scatter_canvas = FigureCanvasTkAgg(scatter_figure, master=root)
scatter_canvas.get_tk_widget().grid(row=5, column=0, columnspan=4)

# Histogram plot area
hist_figure = Figure(figsize=(5, 4), dpi=100)
hist_canvas = FigureCanvasTkAgg(hist_figure, master=root)
hist_canvas.get_tk_widget().grid(row=5, column=4, columnspan=4)

# Start the Tkinter main loop
root.mainloop()
