import os
import json
import pandas as pd
import numpy as np

# for plotting
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import cairosvg
from io import BytesIO

subs = ['001', '002']
sessions = ['001', '002', '003', '004', '005', '006']
runs = ['01', '02', '03', '04', '05', '0102030405avg']
hemis = ['L'] #['L', 'R']
pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/analysis-03_AsusCorrect"    
covmaps_basepath = f"D:/results/with-without-nordic-covmap/covMap"
# cephpath: f"Y:/data/stimsim24/BIDS/derivatives/prfresult/prfanalyze-gem/analysis-03/covMap/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-01_desc-V1-VarExp10-max_covmap.svg"

# # def load_all_data(isNordic : bool = False):
# #     pRF_estimations_json_data_list = []
# #     covmaps_list = []

# #     for sub in subs:
# #         for ses in sessions:
# #             for run in runs:
# #                 for hemi in hemis:
# #                     if isNordic:
# #                         pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_estimates.json"
# #                     else:
# #                         pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_estimates.json"
# #                     if os.path.exists(pRF_estimations_filepath):
# #                         with open(pRF_estimations_filepath, 'r') as file:
# #                             data = json.load(file)
# #                             for voxel_num, voxel_data in enumerate(data):
# #                                 voxel_data.update({
# #                                     "Sub": sub,
# #                                     "Ses": ses,
# #                                     "Run": run,
# #                                     "Hemi": hemi,
# #                                     "VoxelNum": voxel_num
# #                                 })
# #                                 pRF_estimations_json_data_list.append(voxel_data)

# #     return pd.DataFrame(pRF_estimations_json_data_list)

# # def load_all_covmap_data(isNordic : bool = False):
# #     covmaps_list = []

# #     for sub in subs:
# #         for ses in sessions:
# #             for run in runs:
# #                 if isNordic:
# #                     covmap_filepath = f"{covmaps_basepath}/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.png"
# #                 else:
# #                     covmap_filepath = f"{covmaps_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.png"

# #                 # Load and convert the SVG file to PNG
# #                 covmap_image = None
# #                 if os.path.exists(covmap_filepath):
# #                     covmap_image = Image.open(covmap_filepath)

# #                     # Store the data in a dictionary
# #                     covmap_data = {
# #                         "Sub": sub,
# #                         "Ses": ses,
# #                         "Run": run,
# #                         "CovMapPath": covmap_filepath,
# #                         "CovMapImage": covmap_image
# #                     }
# #                     covmaps_list.append(covmap_data)

# #     return pd.DataFrame(covmaps_list)


def load_all_data(isNordic : bool = False):
    pRF_estimations_json_data_list = []
    covmaps_list = []

    for sub in subs:
        for ses in sessions:
            for run in runs:
                if isNordic:
                    covmap_filepath = f"{covmaps_basepath}/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.png"
                else:
                    covmap_filepath = f"{covmaps_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.png"

                # Load and convert the SVG file to PNG
                covmap_image = None
                if os.path.exists(covmap_filepath):
                    # with open(covmap_filepath, 'rb') as svg_file:
                        # # png_image = cairosvg.svg2png(file_obj=svg_file)
                        # # covmap_image = Image.open(BytesIO(png_image))
                        covmap_image = Image.open(covmap_filepath)

                for hemi in hemis:
                    if isNordic:
                        pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                    else:
                        pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                    if os.path.exists(pRF_estimations_filepath):
                        with open(pRF_estimations_filepath, 'r') as file:
                            data = json.load(file)
                            for voxel_num, voxel_data in enumerate(data):
                                voxel_data.update({
                                    "Sub": sub,
                                    "Ses": ses,
                                    "Run": run,
                                    "Hemi": hemi,
                                    "VoxelNum": voxel_num,
                                    "CovMapPath": covmap_filepath,
                                    "CovMapImage": covmap_image
                                })
                                pRF_estimations_json_data_list.append(voxel_data)

    return pd.DataFrame(pRF_estimations_json_data_list)

# # covmap_data = load_all_covmap_data(isNordic=False)

# Load all data into a DataFrame
all_data_standard = load_all_data(isNordic=False)
all_data_standard['Denoising'] = 'Standard'
all_data_nordic = load_all_data(isNordic=True)
all_data_nordic['Denoising'] = 'Nordic'




def update_plots(filtered_data_standard, filtered_data_nordic, covmap_path_standard, covmap_path_nordic, voxel_number):
    # Calculate eccentricity for the filtered data
    filtered_data_standard['Eccentricity'] = (filtered_data_standard['Centerx0'] ** 2 + filtered_data_standard['Centery0'] ** 2) ** 0.5
    filtered_data_nordic['Eccentricity'] = (filtered_data_nordic['Centerx0'] ** 2 + filtered_data_nordic['Centery0'] ** 2) ** 0.5

    # Create an index for plotting on the x-axis (1, 2, 3, ...)
    filtered_data_standard['PlotIndex'] = np.arange(1, len(filtered_data_standard) + 1)
    filtered_data_nordic['PlotIndex'] = np.arange(1, len(filtered_data_nordic) + 1)

    # Scatter Plot
    combined_data = pd.concat([filtered_data_standard, filtered_data_nordic])
    scatter_figure.clear()
    scatter_plot = scatter_figure.add_subplot(111)
    sns.scatterplot(data=combined_data, x='PlotIndex', y='Eccentricity', hue='Denoising', style='Denoising', ax=scatter_plot)

    # scatter_plot.scatter(np.arange(filtered_data_standard.shape[0]) +1, filtered_data_standard['Eccentricity']) #NOTE: another method to plot, working
    # scatter_plot.scatter(np.arange(filtered_data_nordic.shape[0]) +1, filtered_data_nordic['Eccentricity'])
    scatter_plot.set_title('Eccentricity Scatter Plot')
    scatter_plot.set_xlabel('Seesion/Run')
    scatter_plot.set_ylabel('Eccentricity')
    scatter_canvas.draw()

    # Histogram Plot
    hist_figure.clear()
    hist_plot = hist_figure.add_subplot(111)    
    sns.histplot(data=combined_data, x='Eccentricity', hue='Denoising', ax=hist_plot, bins=10, kde=True) # Seaborn histogram for both datasets
    # # hist_plot.hist(filtered_data_standard['Eccentricity'], bins=10)
    hist_plot.set_title('Eccentricity Histogram')
    hist_plot.set_xlabel('Eccentricity')
    hist_plot.set_ylabel('Frequency')
    hist_canvas.draw()

    # coverage maps
    #NOTE: Things to do, load the svg file for the privovided paths of covmaps
    covmap_figure_standard.clear()
    covmap_plot_standard = covmap_figure_standard.add_subplot(111)    
    covmap_figure_nordic.clear()
    covmap_plot_nordic = covmap_figure_nordic.add_subplot(111)

    # Display the pre-loaded standard coverage map
    if filtered_data_standard['CovMapImage'].iloc[0] is not None:
        # covmap_plot_standard.imshow(filtered_data_standard['CovMapImage'].iloc[0])
        covmap_plot_standard.imshow(covmap_path_standard)
        covmap_plot_standard.axis('off')

    # Display the pre-loaded Nordic coverage map
    if filtered_data_nordic['CovMapImage'].iloc[0] is not None:
        # covmap_plot_nordic.imshow(filtered_data_nordic['CovMapImage'].iloc[0])
        covmap_plot_nordic.imshow(covmap_path_nordic)
        covmap_plot_nordic.axis('off')

    # Update the canvases
    covmap_canvas_standard.draw()
    covmap_canvas_nordic.draw()

    # Update the canvases
    covmap_canvas_standard.draw()
    covmap_canvas_nordic.draw()

def load_data():
    sub = sub_entry.get()
    ses = ses_entry.get()
    run = run_entry.get()
    voxel_number = voxel_slider.get()

    if sub == '':
        sub = '001'
        ses = '001'
        run = '01'
        voxel_number = 0

    # Filter the data based on selected sub and voxel number
    filtered_voxel_esitmations_data_standard = all_data_standard[(all_data_standard['Sub'] == sub) & (all_data_standard['VoxelNum'] == voxel_number)]
    filtered_voxel_esitmations_data_nordic = all_data_nordic[(all_data_nordic['Sub'] == sub) & (all_data_nordic['VoxelNum'] == voxel_number)]
    covmap_path_standard = all_data_standard[(all_data_standard['Sub'] == sub) & (all_data_standard['Ses'] == ses) & (all_data_standard['Run'] == run)]['CovMapImage'].values[0]
    covmap_path_nordic = all_data_nordic[(all_data_nordic['Sub'] == sub) & (all_data_nordic['Ses'] == ses) & (all_data_nordic['Run'] == run)]['CovMapImage'].values[0]

    # Update the label with Centerx0, Centery0 for the selected voxel
    update_plots(filtered_voxel_esitmations_data_standard, filtered_voxel_esitmations_data_nordic, covmap_path_standard, covmap_path_nordic, voxel_number)

    param_label.config(text=f"Estimation Parameters Values: (Centerx0: {filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']}, Centery0: {filtered_voxel_esitmations_data_standard.iloc[0]['Centery0']})")

# Initialize the main window
root = tk.Tk()
root.title("pRF Estimation Viewer")

# Entry fields for sub, ses, run
#...subject
tk.Label(root, text="Sub").grid(row=0, column=0)
sub_entry = tk.Entry(root)
sub_entry.grid(row=0, column=1)

#...session
tk.Label(root, text="Ses").grid(row=0, column=2)
ses_entry = tk.Entry(root)
ses_entry.grid(row=0, column=3)

#...run
tk.Label(root, text="Run").grid(row=0, column=4)
run_entry = tk.Entry(root)
run_entry.grid(row=0, column=5)


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

# covmap plot area - standard
covmap_figure_standard = Figure(figsize=(5, 4), dpi=100)
covmap_canvas_standard = FigureCanvasTkAgg(covmap_figure_standard, master=root)
covmap_canvas_standard.get_tk_widget().grid(row=11, column=0, columnspan=4)

# covmap plot area - standard
covmap_figure_nordic = Figure(figsize=(5, 4), dpi=100)
covmap_canvas_nordic = FigureCanvasTkAgg(covmap_figure_nordic, master=root)
covmap_canvas_nordic.get_tk_widget().grid(row=11, column=4, columnspan=4)



# Start the Tkinter main loop
root.mainloop()
