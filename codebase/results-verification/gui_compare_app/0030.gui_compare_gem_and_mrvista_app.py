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
import nibabel as nib

subs = ['001']
sessions = ['001']
runs = ['01', '02']

# subs = ['001', '002']
# sessions = ['001', '002', '003', '004', '005', '006']
# runs = ['01', '02', '03', '04', '05', '0102030405avg']

hemis = ['L'] #['L', 'R']
gem_pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/analysis-03_AsusCorrect"    
vista_pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/prfanalyze-vista/analysis-01"
gem_covmaps_basepath = f"D:/results/with-without-nordic-covmap/gem-covMap"
vista_covmaps_basepath = f"D:/results/with-without-nordic-covmap/vista-covMap"
fmri_measured_basepath = f"D:/results/with-without-nordic-covmap/prfprepare/analysis-01"
# cephpath: f"Y:/data/stimsim24/BIDS/derivatives/prfresult/prfanalyze-gem/analysis-03/covMap/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-01_desc-V1-VarExp10-max_covmap.svg"
fmri_measured_signal_length = None


def load_all_data(isVista : bool = False):
    pRF_estimations_json_data_list = []
    covmaps_list = []

    for sub in subs:
        for ses in sessions:
            for run in runs:
                if isVista:
                    covmap_filepath = f"{vista_covmaps_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.png"
                else:
                    covmap_filepath = f"{gem_covmaps_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.png"

                # Load and convert the SVG file to PNG
                covmap_image = None
                if os.path.exists(covmap_filepath):
                        covmap_image = Image.open(covmap_filepath)

                for hemi in hemis:
                    if isVista:
                        pRF_estimations_filepath = f"{vista_pRF_estimations_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                    else:
                        pRF_estimations_filepath = f"{gem_pRF_estimations_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_estimates.json"

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


###################################------------------------------------------------Data Loading-------------------------------------------------############################################

# Load all data into a DataFrame
all_data_GEM = load_all_data(isVista=False)
all_data_GEM['Method'] = 'GEM'
all_data_vista = load_all_data(isVista=True)
all_data_vista['Method'] = 'Vista'



###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
def update_plots(filtered_data_GEM, filtered_data_vista, covmap_GEM, covmap_vista):
    try:
        # Calculate eccentricity for the filtered data
        filtered_data_GEM['Eccentricity'] = (filtered_data_GEM['Centerx0'] ** 2 + filtered_data_GEM['Centery0'] ** 2) ** 0.5
        filtered_data_vista['Eccentricity'] = (filtered_data_vista['Centerx0'] ** 2 + filtered_data_vista['Centery0'] ** 2) ** 0.5

        # Create an index for plotting on the x-axis (1, 2, 3, ...)
        filtered_data_GEM['PlotIndex'] = np.arange(1, len(filtered_data_GEM) + 1)
        filtered_data_vista['PlotIndex'] = np.arange(1, len(filtered_data_vista) + 1)
        combined_data = pd.concat([filtered_data_GEM, filtered_data_vista])

        # Eccentricity Scatter Plot        
        ecc_scatter_figure.clear()
        ecc_scatter_plot = ecc_scatter_figure.add_subplot(111)
        sns.scatterplot(data=combined_data, x='PlotIndex', y='Eccentricity', hue='Method', style='Method', ax=ecc_scatter_plot)
        ecc_scatter_plot.set_title('Eccentricity')
        ecc_scatter_plot.set_xlabel('Session/Run')
        ecc_scatter_plot.set_ylabel('Eccentricity')
        ecc_scatter_canvas.draw()

        # Eccentricity Histogram Plot
        ecc_hist_figure.clear()
        ecc_hist_plot = ecc_hist_figure.add_subplot(111)    
        sns.histplot(data=combined_data, x='Eccentricity', hue='Method', ax=ecc_hist_plot, bins=10, kde=True) # Seaborn histogram for both datasets
        # # hist_plot.hist(filtered_data_GEM['Eccentricity'], bins=10)
        ecc_hist_plot.set_title('Eccentricity Histogram')
        ecc_hist_plot.set_xlabel('Eccentricity')
        ecc_hist_plot.set_ylabel('Frequency')
        ecc_hist_canvas.draw()        

        # Sigma Scatter Plot
        sigma_scatter_figure.clear()
        sigma_scatter_plot = sigma_scatter_figure.add_subplot(111)
        sns.scatterplot(data=combined_data, x='PlotIndex', y='sigmaMajor', hue='Method', style='Method', ax=sigma_scatter_plot)
        sigma_scatter_plot.set_title('pRF Size')
        sigma_scatter_plot.set_xlabel('Session/Run')
        sigma_scatter_plot.set_ylabel('pRF Size')
        sigma_scatter_canvas.draw()

        # Sigma Histogram Plot
        sigma_hist_figure.clear()
        sigma_hist_plot = sigma_hist_figure.add_subplot(111)    
        sns.histplot(data=combined_data, x='sigmaMajor', hue='Method', ax=sigma_hist_plot, bins=10, kde=True) # Seaborn histogram for both datasets
        sigma_hist_plot.set_title('pRF Size')
        sigma_hist_plot.set_xlabel('pRF Size')
        sigma_hist_plot.set_ylabel('Frequency')
        sigma_hist_canvas.draw()    

        # CovMap Plot
        covmap_figure_gem.clear()
        covmap_figure_vista.clear()
        if covmap_GEM is not None and covmap_vista is not None:                                    
            covmap_plot_GEM = covmap_figure_gem.add_subplot(111)                
            covmap_plot_vista = covmap_figure_vista.add_subplot(111)

            # Display the pre-loaded GEM coverage map
            if filtered_data_GEM['CovMapImage'].iloc[0] is not None:
                # covmap_plot_GEM.imshow(filtered_data_GEM['CovMapImage'].iloc[0])
                covmap_plot_GEM.imshow(covmap_GEM)
                covmap_plot_GEM.axis('off')
                covmap_plot_GEM.set_title('GEM')

            # Display the pre-loaded Vista coverage map
            if filtered_data_vista['CovMapImage'].iloc[0] is not None:
                covmap_plot_vista.imshow(covmap_vista)
                covmap_plot_vista.axis('off')
                covmap_plot_vista.set_title('Vista')                
        # Update the canvases
        covmap_canvas_gem.draw()
        covmap_canvas_vista.draw()
    
    except Exception as e:
        print(f"Error in update_plots: {e}")

def load_data():
    sub = subs[sub_listbox.curselection()[0]]
    ses = sessions[ses_listbox.curselection()[0]]
    run = runs[run_listbox.curselection()[0]]
    voxel_number = voxel_slider.get()
    show_covmap_value = show_covmap.get()
    # # r2_value = r2_slider.get()

    # Filter the data based on selected sub and voxel number
    # selected voxel's estimation data across all runs, all sessions (for a given suject)
    filtered_voxel_esitmations_data_GEM = all_data_GEM[(all_data_GEM['Sub'] == sub) & (all_data_GEM['VoxelNum'] == voxel_number)] # & (all_data_GEM['R2'] >= r2_value)
    filtered_voxel_esitmations_data_vista = all_data_vista[(all_data_vista['Sub'] == sub) & (all_data_vista['VoxelNum'] == voxel_number)]
    
    # coverage maps
    if show_covmap_value:
        covmap_GEM = all_data_GEM[(all_data_GEM['Sub'] == sub) & (all_data_GEM['Ses'] == ses) & (all_data_GEM['Run'] == run)]['CovMapImage'].values[0]
        covmap_vista = all_data_vista[(all_data_vista['Sub'] == sub) & (all_data_vista['Ses'] == ses) & (all_data_vista['Run'] == run)]['CovMapImage'].values[0]
    else:    
        covmap_GEM = None
        covmap_vista = None

    # Update the label with Centerx0, Centery0 for the selected voxel
    update_plots(filtered_voxel_esitmations_data_GEM, filtered_voxel_esitmations_data_vista, covmap_GEM, covmap_vista)

    # param_label.config(text=f"Estimation Parameters Values: (Centerx0: {filtered_voxel_esitmations_data_GEM.iloc[0]['Centerx0']}, Centery0: {filtered_voxel_esitmations_data_GEM.iloc[0]['Centery0']})")
    # param_label.config(text=f"R2: {filtered_voxel_esitmations_data_GEM.iloc[0]['R2']}")
    param_label.config(text=f"Single Voxel Estimation (GEM): (Ecc.: {round(np.sqrt(filtered_voxel_esitmations_data_GEM.iloc[0]['Centerx0']**2 + filtered_voxel_esitmations_data_GEM.iloc[0]['Centery0']**2), 2)}, size: {round(filtered_voxel_esitmations_data_GEM.iloc[0]['sigmaMajor'], 2)}, R2: {round(filtered_voxel_esitmations_data_GEM.iloc[0]['R2'], 2)})")

# Function to update slider from entry
def update_voxel_slider(event):    
    voxel_num = voxel_entry.get()
    try:
        voxel_slider.set(int(voxel_num))
        load_data()
    except ValueError:
        pass

# Function to update entry from slider
def update_voxel_entry(val):
    voxel_entry.delete(0, tk.END)
    voxel_entry.insert(0, str(val))
    load_data()


#############################################------------------------------used at first-working---------------------################################################
# Initialize the main window
root = tk.Tk()
root.title("pRF Estimation Viewer: GEM vs Vista")

# Listboxes for Sub, Ses, and Run with initial selection
# Labels for Sub, Ses, and Run
sub_label = tk.Label(root, text="Sub")
sub_label.grid(row=0, column=0)

ses_label = tk.Label(root, text="Ses")
ses_label.grid(row=0, column=1)

run_label = tk.Label(root, text="Run")
run_label.grid(row=0, column=2)

# Listboxes for Sub, Ses, and Run with initial selection
#....subjects
sub_listbox = tk.Listbox(root, height=3, exportselection=0)
for sub in subs:
    sub_listbox.insert(tk.END, sub)
sub_listbox.select_set(0)  # Default select first item
sub_listbox.grid(row=1, column=0)

#....sessions
ses_listbox = tk.Listbox(root, height=7, exportselection=0)
for ses in sessions:
    ses_listbox.insert(tk.END, ses)
ses_listbox.select_set(0)  # Default select first item
ses_listbox.grid(row=1, column=1)

#....runs
run_listbox = tk.Listbox(root, height=7, exportselection=0)
for run in runs:
    run_listbox.insert(tk.END, run)
run_listbox.select_set(0)  # Default select first item
run_listbox.grid(row=1, column=2)


# Label for displaying Centerx0, Centery0
param_label = tk.Label(root, text="R2")
param_label.grid(row=0, column=3, columnspan=8)

# Checkbox to show CovMap
show_covmap = tk.IntVar(value=1)
covmap_checkbox = tk.Checkbutton(root, text="Show CovMap", variable=show_covmap)
covmap_checkbox.grid(row=0, column=11)

# Slider and Entry for Voxel Number
voxel_slider = tk.Scale(root, from_=0, to=20000, orient=tk.HORIZONTAL, label="Voxel Number", length=600)
voxel_slider.bind("<ButtonRelease-1>", lambda event: update_voxel_entry(voxel_slider.get()))
voxel_slider.grid(row=2, column=0, columnspan=20)

voxel_entry = tk.Entry(root)
voxel_entry.grid(row=3, column=7)
voxel_entry.bind("<Return>", update_voxel_slider)
voxel_slider.set(15)  # Set initial value for the slider
update_voxel_entry(voxel_slider.get())  # Sync the entry with the slider

# # Slider for R2
# r2_slider = tk.Scale(root, from_=0.05, to=1, resolution=0.1, orient=tk.HORIZONTAL, label="R2 Filter", length=300)
# r2_slider.bind("<ButtonRelease-1>", lambda event: update_r2_slider(r2_slider.get()))
# r2_slider.grid(row=2, column=6, columnspan=20)


# Button to load data based on inputs
load_button = tk.Button(root, text="Load Data", command=load_data)
load_button.grid(row=3, column=0, columnspan=6)

# Eccentricity Scatter plot area
ecc_scatter_figure = Figure(figsize=(5, 4), dpi=100)
ecc_scatter_canvas = FigureCanvasTkAgg(ecc_scatter_figure, master=root)
ecc_scatter_canvas.get_tk_widget().grid(row=5, column=0, columnspan=4)

# Eccentricity Histogram plot area
ecc_hist_figure = Figure(figsize=(5, 4), dpi=100)
ecc_hist_canvas = FigureCanvasTkAgg(ecc_hist_figure, master=root)
ecc_hist_canvas.get_tk_widget().grid(row=5, column=4, columnspan=4)

# Sigma Scatter plot area
sigma_scatter_figure = Figure(figsize=(5, 4), dpi=100)
sigma_scatter_canvas = FigureCanvasTkAgg(sigma_scatter_figure, master=root)
sigma_scatter_canvas.get_tk_widget().grid(row=5, column=8, columnspan=4)

# Sigma Histogram plot area
sigma_hist_figure = Figure(figsize=(5, 4), dpi=100)
sigma_hist_canvas = FigureCanvasTkAgg(sigma_hist_figure, master=root)
sigma_hist_canvas.get_tk_widget().grid(row=5, column=12, columnspan=4)

# covmap plot area - GEM
covmap_figure_gem = Figure(figsize=(5, 4), dpi=100)
covmap_canvas_gem = FigureCanvasTkAgg(covmap_figure_gem, master=root)
covmap_canvas_gem.get_tk_widget().grid(row=11, column=0, columnspan=4)

# covmap plot area - vista
covmap_figure_vista = Figure(figsize=(5, 4), dpi=100)
covmap_canvas_vista = FigureCanvasTkAgg(covmap_figure_vista, master=root)
covmap_canvas_vista.get_tk_widget().grid(row=11, column=4, columnspan=4)



# Start the Tkinter main loop
root.mainloop()

##############################---------------------------------------------------------------------------################################################