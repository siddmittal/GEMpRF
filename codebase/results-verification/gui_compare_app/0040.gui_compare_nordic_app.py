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

# # ##### Verification of fmri data
# # import matplotlib.pyplot as plt
# # standard_fmri = nib.load(r"D:\results\with-without-nordic-covmap\prfprepare\analysis-01\sub-001\ses-001\func\sub-001_ses-001_task-bar_run-01_hemi-L_bold.nii.gz")
# # standard_Y_signals_cpu = standard_fmri.get_fdata()                        
# # standard_Y_signals_cpu = standard_Y_signals_cpu.reshape(-1, standard_Y_signals_cpu.shape[-1]) # reshape the BOLD response data to 2D

# # nordic_fmri = nib.load(r"D:\results\with-without-nordic-covmap\prfprepare\analysis-01\sub-001\ses-001nn\func\sub-001_ses-001nn_task-bar_run-01_hemi-L_bold.nii.gz")
# # nordic_Y_signals_cpu = nordic_fmri.get_fdata()
# # nordic_Y_signals_cpu = nordic_Y_signals_cpu.reshape(-1, nordic_Y_signals_cpu.shape[-1]) # reshape the BOLD response data to 2D

# # plt.plot(standard_Y_signals_cpu[1000, :])
# # plt.plot(nordic_Y_signals_cpu[1000, :])
# # plt.show()


# subs = ['001']
# sessions = ['001']
# runs = ['01', '02']

subs = ['001', '002']
sessions = ['001', '002', '003', '004', '005', '006']
runs = ['01', '02', '03', '04', '05', '0102030405avg']

hemis = ['L'] #['L', 'R']
pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/analysis-03_AsusCorrect"    
covmaps_basepath = f"D:/results/with-without-nordic-covmap/gem-covMap"
fmri_measured_basepath = f"D:/results/with-without-nordic-covmap/prfprepare/analysis-01"
# cephpath: f"Y:/data/stimsim24/BIDS/derivatives/prfresult/prfanalyze-gem/analysis-03/covMap/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-01_desc-V1-VarExp10-max_covmap.svg"
fmri_measured_signal_length = None


def load_all_data(isStandard : bool = False):
    pRF_estimations_json_data_list = []
    covmaps_list = []

    for sub in subs:
        for ses in sessions:
            for run in runs:
                if isStandard:
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
                    if isStandard:
                        pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                        fmri_measured_data_filepath = f"{fmri_measured_basepath}/sub-{sub}/ses-{ses}nn/func/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_bold.nii.gz"
                    else:
                        pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                        fmri_measured_data_filepath = f"{fmri_measured_basepath}/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_bold.nii.gz"

                    # Load fMRI measured data
                    if os.path.exists(fmri_measured_data_filepath):
                        bold_response_img = nib.load(fmri_measured_data_filepath)
                        Y_signals_cpu = bold_response_img.get_fdata()                        
                        Y_signals_cpu = Y_signals_cpu.reshape(-1, Y_signals_cpu.shape[-1]) # reshape the BOLD response data to 2D
                        fmri_measured_signal_length = Y_signals_cpu.shape[1]


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
                                    "CovMapImage": covmap_image,
                                    "fmri_data_path": fmri_measured_data_filepath,
                                    "fmri_data": Y_signals_cpu[voxel_num, :]
                                })
                                pRF_estimations_json_data_list.append(voxel_data)

    return pd.DataFrame(pRF_estimations_json_data_list)


# def load_timecourses_data(isNordic : bool = False):
#     fmri_measured_data_list = []

#     for sub in subs:
#         for ses in sessions:
#             for run in runs:
#                 for hemi in hemis:
#                     if isNordic:
#                         fmri_measured_data_filepath = f"{fmri_measured_basepath}/sub-{sub}/ses-{ses}nn/func/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_bold.nii.gz"
#                     else:
#                         fmri_measured_data_filepath = f"{fmri_measured_basepath}/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_bold.nii.gz"
#                     if os.path.exists(fmri_measured_data_filepath):
#                         bold_response_img = nib.load(fmri_measured_data_filepath)
#                         Y_signals_cpu = bold_response_img.get_fdata()                        
#                         Y_signals_cpu = Y_signals_cpu.reshape(-1, Y_signals_cpu.shape[-1]) # reshape the BOLD response data to 2D

#                         # Store the data in a dictionary
#                         fmri_data = {
#                             "Sub": sub,
#                             "Ses": ses,
#                             "Run": run,
#                             "fmri_data_path": fmri_measured_data_filepath,
#                             "fmri_data": Y_signals_cpu
#                         }
#                         fmri_measured_data_list.append(fmri_data)

#     return pd.DataFrame(fmri_measured_data_list)


###################################------------------------------------------------Data Loading-------------------------------------------------############################################
# # covmap_data = load_all_covmap_data(isNordic=False)

# all_fmri_data_standard = load_timecourses_data(isNordic=False)
# all_fmri_data_nordic = load_timecourses_data(isNordic=True)

# Load all data into a DataFrame
all_data_standard = load_all_data(isStandard=True)
all_data_standard['Denoising'] = 'Standard'
all_data_nordic = load_all_data(isStandard=False)
all_data_nordic['Denoising'] = 'Nordic'



###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
def update_plots(filtered_data_standard, filtered_data_nordic, covmap_standard, covmap_nordic, fmri_measured_timecourse_data_standard, fmri_measured_timecourse_data_nordic, voxel_number):
    try:
        # Calculate eccentricity for the filtered data
        filtered_data_standard['Eccentricity'] = (filtered_data_standard['Centerx0'] ** 2 + filtered_data_standard['Centery0'] ** 2) ** 0.5
        filtered_data_nordic['Eccentricity'] = (filtered_data_nordic['Centerx0'] ** 2 + filtered_data_nordic['Centery0'] ** 2) ** 0.5

        # Create an index for plotting on the x-axis (1, 2, 3, ...)
        filtered_data_standard['PlotIndex'] = np.arange(1, len(filtered_data_standard) + 1)
        filtered_data_nordic['PlotIndex'] = np.arange(1, len(filtered_data_nordic) + 1)
        combined_data = pd.concat([filtered_data_standard, filtered_data_nordic])

        # Eccentricity Scatter Plot        
        ecc_scatter_figure.clear()
        ecc_scatter_plot = ecc_scatter_figure.add_subplot(111)
        sns.scatterplot(data=combined_data, x='PlotIndex', y='Eccentricity', hue='Denoising', style='Denoising', ax=ecc_scatter_plot)
        ecc_scatter_plot.set_title('Eccentricity')
        ecc_scatter_plot.set_xlabel('Seesion/Run')
        ecc_scatter_plot.set_ylabel('Eccentricity')
        ecc_scatter_canvas.draw()

        # Eccentricity Histogram Plot
        ecc_hist_figure.clear()
        ecc_hist_plot = ecc_hist_figure.add_subplot(111)    
        sns.histplot(data=combined_data, x='Eccentricity', hue='Denoising', ax=ecc_hist_plot, bins=10, kde=True) # Seaborn histogram for both datasets
        # # hist_plot.hist(filtered_data_standard['Eccentricity'], bins=10)
        ecc_hist_plot.set_title('Eccentricity')
        ecc_hist_plot.set_xlabel('Eccentricity')
        ecc_hist_plot.set_ylabel('Frequency')
        ecc_hist_canvas.draw()        

        # Sigma Scatter Plot
        sigma_scatter_figure.clear()
        sigma_scatter_plot = sigma_scatter_figure.add_subplot(111)
        sns.scatterplot(data=combined_data, x='PlotIndex', y='sigmaMajor', hue='Denoising', style='Denoising', ax=sigma_scatter_plot)
        sigma_scatter_plot.set_title('pRF Size')
        sigma_scatter_plot.set_xlabel('Seesion/Run')
        sigma_scatter_plot.set_ylabel('pRF Size')
        sigma_scatter_canvas.draw()

        # Sigma Histogram Plot
        sigma_hist_figure.clear()
        sigma_hist_plot = sigma_hist_figure.add_subplot(111)    
        sns.histplot(data=combined_data, x='sigmaMajor', hue='Denoising', ax=sigma_hist_plot, bins=10, kde=True) # Seaborn histogram for both datasets
        sigma_hist_plot.set_title('pRF Size')
        sigma_hist_plot.set_xlabel('pRF Size')
        sigma_hist_plot.set_ylabel('Frequency')
        sigma_hist_canvas.draw()    

        # coverage maps
        covmap_figure_nordic.clear()        
        covmap_figure_standard.clear()        
        if covmap_standard is not None and covmap_nordic is not None:
            covmap_plot_standard = covmap_figure_standard.add_subplot(111)            
            covmap_plot_nordic = covmap_figure_nordic.add_subplot(111)

            # Display the pre-loaded standard coverage map
            if filtered_data_standard['CovMapImage'].iloc[0] is not None:
                # covmap_plot_standard.imshow(filtered_data_standard['CovMapImage'].iloc[0])
                covmap_plot_standard.imshow(covmap_standard)
                covmap_plot_standard.axis('off')
                covmap_plot_standard.set_title('Standard')

            # Display the pre-loaded Nordic coverage map
            if filtered_data_nordic['CovMapImage'].iloc[0] is not None:
                # covmap_plot_nordic.imshow(filtered_data_nordic['CovMapImage'].iloc[0])
                covmap_plot_nordic.imshow(covmap_nordic)
                covmap_plot_nordic.axis('off')
                covmap_plot_nordic.set_title('Nordic')                
        # Update the canvases
        covmap_canvas_standard.draw()
        covmap_canvas_nordic.draw()

        # Update the canvases
        covmap_canvas_standard.draw()
        covmap_canvas_nordic.draw()

        # Timecourses plot area
        combined_timecourses_data = pd.concat([fmri_measured_timecourse_data_standard, fmri_measured_timecourse_data_nordic])
        # # fmri_measured_timecourse_data_standard['PlotIndex'] = np.arange(0, fmri_measured_signal_length)
        # # fmri_measured_timecourse_data_nordic['PlotIndex'] = np.arange(0, fmri_measured_signal_length)
        # # timecourses_figure.clear()
        # # timecourses_plot = timecourses_figure.add_subplot(111)   
        # # sns.lineplot(data=combined_timecourses_data, x='PlotIndex', y='fmri_data', hue='Denoising', style='Denoising', ax=timecourses_plot) 
        # # timecourses_plot.set_title('Timecourses')
        # # timecourses_plot.set_xlabel('time')
        # # timecourses_plot.set_ylabel('Signal')
        # # timecourses_canvas.draw()
         
                 
        # Explode the fmri_data into separate rows for each time point
        combined_timecourses_data = combined_timecourses_data.explode('fmri_data').reset_index(drop=True)

        # Create a plot index that goes from 0 to fmri_measured_signal_length-1 for each unique denoising category
        combined_timecourses_data['PlotIndex'] = combined_timecourses_data.groupby('Denoising').cumcount()

        # Ensure fmri_data is in a numeric format suitable for plotting
        combined_timecourses_data['fmri_data'] = pd.to_numeric(combined_timecourses_data['fmri_data'])

        # Plotting
        timecourses_figure.clear()
        timecourses_plot = timecourses_figure.add_subplot(111)
        sns.lineplot(
            data=combined_timecourses_data,
            x='PlotIndex',
            y='fmri_data',
            hue='Denoising',
            style='Denoising',
            ax=timecourses_plot
        )
        timecourses_plot.set_title('Timecourses')
        timecourses_plot.set_xlabel('Time')
        timecourses_plot.set_ylabel('Signal')
        timecourses_canvas.draw()
    
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
    filtered_voxel_esitmations_data_standard = all_data_standard[(all_data_standard['Sub'] == sub) & (all_data_standard['VoxelNum'] == voxel_number)] # & (all_data_standard['R2'] >= r2_value)
    filtered_voxel_esitmations_data_nordic = all_data_nordic[(all_data_nordic['Sub'] == sub) & (all_data_nordic['VoxelNum'] == voxel_number)]
    
    # coverage maps
    if show_covmap_value:
        covmap_standard = all_data_standard[(all_data_standard['Sub'] == sub) & (all_data_standard['Ses'] == ses) & (all_data_standard['Run'] == run)]['CovMapImage'].values[0]
        covmap_nordic = all_data_nordic[(all_data_nordic['Sub'] == sub) & (all_data_nordic['Ses'] == ses) & (all_data_nordic['Run'] == run)]['CovMapImage'].values[0]
    else:
        covmap_standard = None
        covmap_nordic = None

    # fmri measured data
    fmri_measured_data_timecourse_standard = all_data_standard[(all_data_standard['Sub'] == sub) & (all_data_standard['Ses'] == ses) & (all_data_standard['Run'] == run) & (all_data_standard['VoxelNum'] == voxel_number)]
    fmri_measured_data_timecourse_nordic = all_data_nordic[(all_data_nordic['Sub'] == sub) & (all_data_nordic['Ses'] == ses) & (all_data_nordic['Run'] == run) & (all_data_nordic['VoxelNum'] == voxel_number)]

    # Update the label with Centerx0, Centery0 for the selected voxel
    update_plots(filtered_voxel_esitmations_data_standard, filtered_voxel_esitmations_data_nordic, covmap_standard, covmap_nordic, fmri_measured_data_timecourse_standard, fmri_measured_data_timecourse_nordic, voxel_number)

    # param_label.config(text=f"Estimation Parameters Values: (Centerx0: {filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']}, Centery0: {filtered_voxel_esitmations_data_standard.iloc[0]['Centery0']})")
    # param_label.config(text=f"R2: {filtered_voxel_esitmations_data_standard.iloc[0]['R2']}, x: {filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']}")
    # param_label.config(text=f"Estimation: (x: {filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']}, y: {filtered_voxel_esitmations_data_standard.iloc[0]['Centery0']}, sigma: {filtered_voxel_esitmations_data_standard.iloc[0]['sigmaMajor']}, R2: {filtered_voxel_esitmations_data_standard.iloc[0]['R2']}")
    # param_label.config(text=f"Estimation: (x: {round(filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0'], 2)}, y: {round(filtered_voxel_esitmations_data_standard.iloc[0]['Centery0'], 2)}, sigma: {round(filtered_voxel_esitmations_data_standard.iloc[0]['sigmaMajor'], 2)}, R2: {round(filtered_voxel_esitmations_data_standard.iloc[0]['R2'], 2)})")
    param_label.config(text=f"Single Voxel Estimation: (Ecc.: {round(np.sqrt(filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']**2 + filtered_voxel_esitmations_data_standard.iloc[0]['Centery0']**2), 2)}, size: {round(filtered_voxel_esitmations_data_standard.iloc[0]['sigmaMajor'], 2)}, R2: {round(filtered_voxel_esitmations_data_standard.iloc[0]['R2'], 2)})")


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
root.title("pRF Estimation Viewer: Standard vs Nordic")

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
param_label = tk.Label(root, text="Estimation:")
param_label.grid(row=0, column=3, columnspan=8)

# Checkbox to show CovMap
show_covmap = tk.IntVar(value=1)
covmap_checkbox = tk.Checkbutton(root, text="Show CovMap", variable=show_covmap)
covmap_checkbox.grid(row=0, column=11)

# Slider and Entry for Voxel Number
voxel_slider = tk.Scale(root, from_=0, to=20000, orient=tk.HORIZONTAL, label="Voxel Number (Hemi-L)", length=600)
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

# covmap plot area - standard
covmap_figure_standard = Figure(figsize=(5, 4), dpi=100)
covmap_canvas_standard = FigureCanvasTkAgg(covmap_figure_standard, master=root)
covmap_canvas_standard.get_tk_widget().grid(row=11, column=0, columnspan=4)

# covmap plot area - nordic
covmap_figure_nordic = Figure(figsize=(5, 4), dpi=100)
covmap_canvas_nordic = FigureCanvasTkAgg(covmap_figure_nordic, master=root)
covmap_canvas_nordic.get_tk_widget().grid(row=11, column=4, columnspan=4)

# Timecourses plot area
timecourses_figure = Figure(figsize=(10, 4), dpi=100)
timecourses_canvas = FigureCanvasTkAgg(timecourses_figure, master=root)
timecourses_canvas.get_tk_widget().grid(row=11, column=8, columnspan=8)

# Start the Tkinter main loop
root.mainloop()

##############################---------------------------------------------------------------------------################################################