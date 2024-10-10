## sub-001, ses-005, run-04, voxel-27
## sub-002, ses-006, run-03
## sub-001, ses-006, run-04, voxel-90

import sys
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")
from PRFclass import PRF

import os
import json
import pandas as pd
import numpy as np

# for plotting
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import cairosvg
from io import BytesIO
import nibabel as nib


subs = ['002']
# sessions = ['001', '002']
# runs = ['01', '02', '03']

# subs = ['001', '002']
sessions = ['001', '002', '003', '004', '005', '006']
runs = ['01', '02', '03', '04', '05', '0102030405avg']

hemis = ['L'] #['L', 'R']

covmaps_basepath = f"D:/results/with-without-nordic-covmap/gem-covMap"
fmri_measured_basepath = f"D:/results/with-without-nordic-covmap/prfprepare/analysis-01"
fmri_measured_signal_length = None

# To get the information about the voxel's region (i.e. V1 or V2 or V3)
# gem_sub_001 = PRF.from_docker('', '001', '001', 'bar', '01', analysis='analysis-03_AsusCorrect' , baseP='Y:/data/stimsim24/BIDS/', orientation='VF', method='gem')
gem_mask_info_dict = {}
for sub in subs:
    gem_mask_info = PRF.from_docker(study ='', subject=sub, session='001', task='bar', run='01', hemi='L', analysis='analysis-03_AsusCorrect', baseP='Y:/data/stimsim24/BIDS/', orientation='VF', method='gem')
    gem_mask_info.maskROI("all") # gem_mask_info.maskROI(['V1', 'V2', 'V3'])
    gem_mask_info_dict[sub] = gem_mask_info

# gem_sub_001 = PRF.from_docker(study ='', subject='001', session='001', task='bar', run='01', hemi='L', analysis='analysis-03_AsusCorrect', baseP='Y:/data/stimsim24/BIDS/', orientation='VF', method='gem')
# gem_sub_002 = PRF.from_docker(study ='', subject='002', session='001', task='bar', run='01', hemi='L', analysis='analysis-03_AsusCorrect', baseP='Y:/data/stimsim24/BIDS/', orientation='VF', method='gem')
# gem_sub_001.maskROI(['V1', 'V2', 'V3'])
# gem_sub_002.maskROI(['V1', 'V2', 'V3'])

def load_all_data(pRF_estimations_basepath, isStandard : bool = False):
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
                        covmap_image = Image.open(covmap_filepath)

                for hemi in hemis:
                    if isStandard:
                        pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}nn/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                        # fmri_measured_data_filepath = f"{fmri_measured_basepath}/sub-{sub}/ses-{ses}nn/func/sub-{sub}_ses-{ses}nn_task-bar_run-{run}_hemi-{hemi}_bold.nii.gz"
                    else:
                        pRF_estimations_filepath = f"{pRF_estimations_basepath}/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_estimates.json"
                        # fmri_measured_data_filepath = f"{fmri_measured_basepath}/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-bar_run-{run}_hemi-{hemi}_bold.nii.gz"

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
gem_pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/analysis-03_AsusCorrect"    
vista_pRF_estimations_basepath = f"D:/results/with-without-nordic-covmap/prfanalyze-vista/analysis-01"
#...gem data
all_data_standard_gem = load_all_data(pRF_estimations_basepath=gem_pRF_estimations_basepath, isStandard=True)
all_data_standard_gem['Method'] = 'Standard-Gem'

#...vista data
all_data_standard_vista = load_all_data(pRF_estimations_basepath=vista_pRF_estimations_basepath, isStandard=True)
all_data_standard_vista['Method'] = 'Standard-Vista'

#...gem-nordic data
all_data_nordic_gem = load_all_data(pRF_estimations_basepath=gem_pRF_estimations_basepath, isStandard=False)
all_data_nordic_gem['Method'] = 'Nordic-Gem'

###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
###################################------------------------------------------------GUI-------------------------------------------------############################################
def update_plots(filtered_data_standard_gem, filtered_data_standard_vista, filtered_data_nordic_gem, covmap_standard_gem, covmap_nordic_gem, fmri_measured_timecourse_data_standard_gem = None, fmri_measured_timecourse_data_nordic_gem = None):
    try:
        # Calculate eccentricity for the filtered data
        filtered_data_standard_gem['Eccentricity'] = (filtered_data_standard_gem['Centerx0'] ** 2 + filtered_data_standard_gem['Centery0'] ** 2) ** 0.5
        filtered_data_standard_vista['Eccentricity'] = (filtered_data_standard_vista['Centerx0'] ** 2 + filtered_data_standard_vista['Centery0'] ** 2) ** 0.5
        filtered_data_nordic_gem['Eccentricity'] = (filtered_data_nordic_gem['Centerx0'] ** 2 + filtered_data_nordic_gem['Centery0'] ** 2) ** 0.5

        # Create an index for plotting on the x-axis (1, 2, 3, ...)
        filtered_data_standard_gem['PlotIndex'] = np.arange(1, len(filtered_data_standard_gem) + 1)
        filtered_data_standard_vista['PlotIndex'] = np.arange(1, len(filtered_data_standard_vista) + 1)
        filtered_data_nordic_gem['PlotIndex'] = np.arange(1, len(filtered_data_nordic_gem) + 1)
        combined_data = pd.concat([filtered_data_standard_gem, filtered_data_standard_vista, filtered_data_nordic_gem])

        # Eccentricity Scatter Plot        
        ecc_scatter_figure.clear()
        ecc_scatter_plot = ecc_scatter_figure.add_subplot(111)
        sns.scatterplot(data=combined_data, x='PlotIndex', y='Eccentricity', hue='Method', style='Method', ax=ecc_scatter_plot)
        ecc_scatter_plot.set_title('Eccentricity')
        ecc_scatter_plot.set_xlabel('Seesion/Run')
        ecc_scatter_plot.set_ylabel('Eccentricity')
        ecc_scatter_canvas.draw()

        # Eccentricity Histogram Plot
        ecc_hist_figure.clear()
        ecc_hist_plot = ecc_hist_figure.add_subplot(111)    
        sns.histplot(data=combined_data, x='Eccentricity', hue='Method', ax=ecc_hist_plot, bins=10, kde=True) # Seaborn histogram for both datasets
        # # hist_plot.hist(filtered_data_standard['Eccentricity'], bins=10)
        ecc_hist_plot.set_title('Eccentricity')
        ecc_hist_plot.set_xlabel('Eccentricity')
        ecc_hist_plot.set_ylabel('Frequency')
        ecc_hist_canvas.draw()        

        # Sigma Scatter Plot
        sigma_scatter_figure.clear()
        sigma_scatter_plot = sigma_scatter_figure.add_subplot(111)
        # sns.scatterplot(data=combined_data, x='PlotIndex', y='sigmaMajor', hue='Method', style='Method', ax=sigma_scatter_plot)
        # sns.lmplot(data=combined_data, x='PlotIndex', y='sigmaMajor', hue='Method', scatter=True, ci=None, ax=sigma_scatter_plot)
        # Plot scatter points with regression lines
        for method in combined_data['Method'].unique():
            subset = combined_data[combined_data['Method'] == method]
            sns.regplot(data=subset, x='PlotIndex', y='sigmaMajor', ax=sigma_scatter_plot, label=method, scatter_kws={'s': 20}, line_kws={'label': method})        
        sigma_scatter_plot.legend() # Add the legend
        sigma_scatter_plot.set_title('pRF Size')
        sigma_scatter_plot.set_xlabel('Seesion/Run')
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

        # coverage maps
        covmap_figure_nordic_gem.clear()        
        covmap_figure_standard_gem.clear()        
        if covmap_standard_gem is not None and covmap_nordic_gem is not None:
            covmap_plot_standard_gem = covmap_figure_standard_gem.add_subplot(111)            
            covmap_plot_nordic_gem = covmap_figure_nordic_gem.add_subplot(111)

            # Display the pre-loaded standard coverage map
            if filtered_data_standard_gem['CovMapImage'].iloc[0] is not None:
                # covmap_plot_standard.imshow(filtered_data_standard['CovMapImage'].iloc[0])
                covmap_plot_standard_gem.imshow(covmap_standard_gem)
                covmap_plot_standard_gem.axis('off')
                covmap_plot_standard_gem.set_title('Standard')

            # Display the pre-loaded Nordic coverage map
            if filtered_data_nordic_gem['CovMapImage'].iloc[0] is not None:
                # covmap_plot_nordic.imshow(filtered_data_nordic['CovMapImage'].iloc[0])
                covmap_plot_nordic_gem.imshow(covmap_nordic_gem)
                covmap_plot_nordic_gem.axis('off')
                covmap_plot_nordic_gem.set_title('Nordic')                
        # Update the canvases
        covmap_canvas_standard_gem.draw()
        covmap_canvas_nordic_gem.draw()

        # Update the canvases
        covmap_canvas_standard_gem.draw()
        covmap_canvas_nordic_gem.draw()

        #########################################-----------------------------Timecourses-----------------------------############################################
        # # # Timecourses plot area
        # # combined_timecourses_data = pd.concat([fmri_measured_timecourse_data_standard, fmri_measured_timecourse_data_nordic])                         
        # # # Explode the fmri_data into separate rows for each time point
        # # combined_timecourses_data = combined_timecourses_data.explode('fmri_data').reset_index(drop=True)

        # # # Create a plot index that goes from 0 to fmri_measured_signal_length-1 for each unique denoising category
        # # combined_timecourses_data['PlotIndex'] = combined_timecourses_data.groupby('Method').cumcount()

        # # # Ensure fmri_data is in a numeric format suitable for plotting
        # # combined_timecourses_data['fmri_data'] = pd.to_numeric(combined_timecourses_data['fmri_data'])

        # # # Plotting
        # # timecourses_figure.clear()
        # # timecourses_plot = timecourses_figure.add_subplot(111)
        # # sns.lineplot(
        # #     data=combined_timecourses_data,
        # #     x='PlotIndex',
        # #     y='fmri_data',
        # #     hue='Method',
        # #     style='Method',
        # #     ax=timecourses_plot
        # # )
        # # timecourses_plot.set_title('Timecourses')
        # # timecourses_plot.set_xlabel('Time')
        # # timecourses_plot.set_ylabel('Signal')
        # # timecourses_canvas.draw()

        ##########################------------------R2------------------############################################
        # R2 Scatter Plot
        r2_scatter_figure.clear()
        r2_scatter_plot = r2_scatter_figure.add_subplot(111)
        sns.scatterplot(data=combined_data, x='PlotIndex', y='R2', hue='Method', style='Method', ax=r2_scatter_plot)
        r2_scatter_plot.set_title('R2')
        r2_scatter_plot.set_xlabel('Seesion/Run')
        r2_scatter_plot.set_ylabel('R2')
        r2_scatter_canvas.draw()

        # R2 Histogram Plot
        r2_bar_figure.clear()
        r2_bar_plot = r2_bar_figure.add_subplot(111)        
        sns.barplot(data=combined_data, x='PlotIndex', y='R2', hue='Method', ax=r2_bar_plot)
        r2_bar_plot.set_title('R2')
        r2_bar_plot.set_xlabel('Session/Run')
        r2_bar_plot.set_ylabel('R2')
        r2_bar_plot.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))
        r2_bar_canvas.draw()

    
    except Exception as e:
        print(f"Error in update_plots: {e}")

def load_data():
    sub = subs[sub_listbox.curselection()[0]]
    ses = sessions[ses_listbox.curselection()[0]]
    run = runs[run_listbox.curselection()[0]]
    
    # NOTE: We are fixing the sorting for Ses-001 and Run-01, otherwise, we will get a different distribution of Eccentricity and pRF Size plots as soon as we change the value of ses or run.
    # descending_r2_voxelnum_indices = all_data_standard_gem[(all_data_standard_gem['Sub'] == sub) & (all_data_standard_gem['Ses'] == ses) & (all_data_standard_gem['Run'] == run) & (all_data_standard_gem['R2'] > 0.001)].sort_values(by=['R2'], ascending=False)['VoxelNum'].values
    descending_r2_voxelnum_indices = all_data_standard_gem[(all_data_standard_gem['Sub'] == sub) & (all_data_standard_gem['Ses'] == '001') & (all_data_standard_gem['Run'] == '01') & (all_data_standard_gem['R2'] > 0.001)].sort_values(by=['R2'], ascending=False)['VoxelNum'].values
  
    
    voxel_number = descending_r2_voxelnum_indices[voxel_slider.get()] # the voxel slider presents the sorted voxel indices based on R2 values, so find the actual Voxel-Number based on the sorted index
    show_covmap_value = True #show_covmap.get()
    # # r2_value = r2_slider.get()

    # find voxel's region
    try:
        region = gem_mask_info_dict[sub]._roiWhichArea[voxel_number] # "_roiWhichArea()" is a function from PRFclass.py
    except:
        region = "Unknown"

    # Filter the data based on selected sub and voxel number
    # selected voxel's estimation data across all runs, all sessions (for a given suject)
    filtered_voxel_esitmations_data_standard_gem = all_data_standard_gem[(all_data_standard_gem['Sub'] == sub) & (all_data_standard_gem['VoxelNum'] == voxel_number)]
    filtered_voxel_esitmations_data_standard_vista = all_data_standard_vista[(all_data_standard_vista['Sub'] == sub) & (all_data_standard_vista['VoxelNum'] == voxel_number)]
    filtered_voxel_esitmations_data_nordic_gem = all_data_nordic_gem[(all_data_nordic_gem['Sub'] == sub) & (all_data_nordic_gem['VoxelNum'] == voxel_number)]    

    # coverage maps
    if show_covmap_value:
        covmap_standard_gem = all_data_standard_gem[(all_data_standard_gem['Sub'] == sub) & (all_data_standard_gem['Ses'] == ses) & (all_data_standard_gem['Run'] == run)]['CovMapImage'].values[0]
        covmap_nordic_gem = all_data_nordic_gem[(all_data_nordic_gem['Sub'] == sub) & (all_data_nordic_gem['Ses'] == ses) & (all_data_nordic_gem['Run'] == run)]['CovMapImage'].values[0]
    else:
        covmap_standard_gem = None
        covmap_nordic_gem = None

    # # fmri measured data
    # fmri_measured_data_timecourse_standard = all_data_standard[(all_data_standard['Sub'] == sub) & (all_data_standard['Ses'] == ses) & (all_data_standard['Run'] == run) & (all_data_standard['VoxelNum'] == voxel_number)]
    # fmri_measured_data_timecourse_nordic = all_data_nordic[(all_data_nordic['Sub'] == sub) & (all_data_nordic['Ses'] == ses) & (all_data_nordic['Run'] == run) & (all_data_nordic['VoxelNum'] == voxel_number)]

    # Update the label with Centerx0, Centery0 for the selected voxel
    # update_plots(filtered_voxel_esitmations_data_standard, filtered_voxel_esitmations_data_nordic, covmap_standard, covmap_nordic, fmri_measured_data_timecourse_standard, fmri_measured_data_timecourse_nordic)
    update_plots(filtered_voxel_esitmations_data_standard_gem, filtered_voxel_esitmations_data_standard_vista, filtered_voxel_esitmations_data_nordic_gem, covmap_standard_gem, covmap_nordic_gem)

    # param_label.config(text=f"Estimation Parameters Values: (Centerx0: {filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']}, Centery0: {filtered_voxel_esitmations_data_standard.iloc[0]['Centery0']})")
    # param_label.config(text=f"R2: {filtered_voxel_esitmations_data_standard.iloc[0]['R2']}, x: {filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']}")
    # param_label.config(text=f"Estimation: (x: {filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0']}, y: {filtered_voxel_esitmations_data_standard.iloc[0]['Centery0']}, sigma: {filtered_voxel_esitmations_data_standard.iloc[0]['sigmaMajor']}, R2: {filtered_voxel_esitmations_data_standard.iloc[0]['R2']}")
    # param_label.config(text=f"Estimation: (x: {round(filtered_voxel_esitmations_data_standard.iloc[0]['Centerx0'], 2)}, y: {round(filtered_voxel_esitmations_data_standard.iloc[0]['Centery0'], 2)}, sigma: {round(filtered_voxel_esitmations_data_standard.iloc[0]['sigmaMajor'], 2)}, R2: {round(filtered_voxel_esitmations_data_standard.iloc[0]['R2'], 2)})")
    param_label.config(text=f"Single Voxel Estimation: (Region: {region}, Ecc.: {round(np.sqrt(filtered_voxel_esitmations_data_standard_gem.iloc[0]['Centerx0']**2 + filtered_voxel_esitmations_data_standard_gem.iloc[0]['Centery0']**2), 2)}, size: {round(filtered_voxel_esitmations_data_standard_gem.iloc[0]['sigmaMajor'], 2)}, R2: {round(filtered_voxel_esitmations_data_standard_gem.iloc[0]['R2'], 2)})")


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
Projector = False
if Projector:
    dpi = 80
    columnspan = 2
    figsize = (3, 2.5)
    column_idx_multiplier = 1
else:
    dpi = 100
    columnspan = 4
    figsize = (5, 4)
    column_idx_multiplier = 2


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

# # Checkbox to show CovMap
# show_covmap = tk.IntVar(value=1)
# covmap_checkbox = tk.Checkbutton(root, text="Show CovMap", variable=show_covmap)
# covmap_checkbox.grid(row=1, column=4)

# Slider and Entry for Voxel Number
# voxel_slider = tk.Scale(root, from_=0, to=20000, orient=tk.HORIZONTAL, label="Voxel Number (Hemi-L)", length=600)
# voxel_slider = tk.Scale(root, from_=0, to=len(descending_r2_voxelnum_indices-1), orient=tk.HORIZONTAL, label="Sorted Voxel Index (Hemi-L)", length=600)
voxel_slider = tk.Scale(root, from_=0, to=20000, orient=tk.HORIZONTAL, label="Sorted Voxel Index (Hemi-L)", length=600) # NOTE:  show the possibility to choose from 20000 sorted indices
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
ecc_scatter_figure = Figure(figsize=figsize, dpi=dpi)
ecc_scatter_canvas = FigureCanvasTkAgg(ecc_scatter_figure, master=root)
ecc_scatter_canvas.get_tk_widget().grid(row=5, column=0*column_idx_multiplier, columnspan=columnspan)

# Eccentricity Histogram plot area
ecc_hist_figure = Figure(figsize=figsize, dpi=dpi)
ecc_hist_canvas = FigureCanvasTkAgg(ecc_hist_figure, master=root)
ecc_hist_canvas.get_tk_widget().grid(row=5, column=2*column_idx_multiplier, columnspan=columnspan)

# Sigma Scatter plot area
sigma_scatter_figure = Figure(figsize=figsize, dpi=dpi)
sigma_scatter_canvas = FigureCanvasTkAgg(sigma_scatter_figure, master=root)
sigma_scatter_canvas.get_tk_widget().grid(row=5, column=4*column_idx_multiplier, columnspan=columnspan)

# Sigma Histogram plot area
sigma_hist_figure = Figure(figsize=figsize, dpi=dpi)
sigma_hist_canvas = FigureCanvasTkAgg(sigma_hist_figure, master=root)
sigma_hist_canvas.get_tk_widget().grid(row=5, column=6*column_idx_multiplier, columnspan=columnspan)

# covmap plot area - standard
covmap_figure_standard_gem = Figure(figsize=figsize, dpi=dpi)
covmap_canvas_standard_gem = FigureCanvasTkAgg(covmap_figure_standard_gem, master=root)
covmap_canvas_standard_gem.get_tk_widget().grid(row=11, column=0*column_idx_multiplier, columnspan=columnspan)

# covmap plot area - nordic
covmap_figure_nordic_gem = Figure(figsize=figsize, dpi=dpi)
covmap_canvas_nordic_gem = FigureCanvasTkAgg(covmap_figure_nordic_gem, master=root)
covmap_canvas_nordic_gem.get_tk_widget().grid(row=11, column=2*column_idx_multiplier, columnspan=columnspan)

# # # Timecourses plot area
# # timecourses_figure = Figure(figsize=(6, 2.5), dpi=dpi)
# # timecourses_canvas = FigureCanvasTkAgg(timecourses_figure, master=root)
# # timecourses_canvas.get_tk_widget().grid(row=11, column=4, columnspan=4)


#############-------NEW
# covmap plot area - standard
r2_scatter_figure = Figure(figsize=figsize, dpi=dpi)
r2_scatter_canvas = FigureCanvasTkAgg(r2_scatter_figure, master=root)
r2_scatter_canvas.get_tk_widget().grid(row=11, column=4*column_idx_multiplier, columnspan=columnspan)

# covmap plot area - nordic
r2_bar_figure = Figure(figsize=figsize, dpi=dpi)
r2_bar_canvas = FigureCanvasTkAgg(r2_bar_figure, master=root)
r2_bar_canvas.get_tk_widget().grid(row=11, column=6*column_idx_multiplier, columnspan=columnspan)

# Start the Tkinter main loop
root.mainloop()

##############################---------------------------------------------------------------------------################################################