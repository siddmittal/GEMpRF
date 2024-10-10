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
from scipy import stats

R2_vs_LINESLOPE_PLOT_DRAWN = False

subs = ['001']
# sessions = ['001', '006']
# runs = ['01', '02', '05', '0102030405avg']

# # subs = ['001', '002']
sessions = ['001', '002', '003', '004', '005', '006']
runs = ['01', '02', '03', '04', '05', '0102030405avg']

hemis = ['L'] #['L', 'R']

fmri_measured_basepath = f"D:/results/with-without-nordic-covmap/prfprepare/analysis-01"
fmri_measured_signal_length = None

# To get the information about the voxel's region (i.e. V1 or V2 or V3)
gem_mask_info_dict = {}
for sub in subs:
    gem_mask_info = PRF.from_docker(study ='', subject=sub, session='001', task='bar', run='01', hemi='L', analysis='analysis-03_AsusCorrect', baseP='Y:/data/stimsim24/BIDS/', orientation='VF', method='gem')
    gem_mask_info.maskROI("all") # gem_mask_info.maskROI(['V1', 'V2', 'V3'])
    gem_mask_info_dict[sub] = gem_mask_info

def load_all_data(pRF_estimations_basepath, isStandard : bool = False):
    pRF_estimations_json_data_list = []
    for sub in subs:
        for ses in sessions:
            for run in runs:
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
                                    "VoxelNum": voxel_num
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

def draw_r2_vs_prfsize_lineslope_plot(all_data_standard_gem, all_data_standard_vista, all_data_nordic_gem):
    sub = '001'

    # Group data by voxel number
    all_data_standard_gem_grouped = all_data_standard_gem[(all_data_standard_gem['Sub'] == sub)].groupby('VoxelNum')
    all_data_standard_vista_grouped = all_data_standard_vista[all_data_standard_vista['Sub'] == sub].groupby('VoxelNum')
    all_data_nordic_gem_grouped = all_data_nordic_gem[all_data_nordic_gem['Sub'] == sub].groupby('VoxelNum')

    total_num_voxels = len(all_data_standard_gem_grouped)

    # Compute average R2 for each voxel
    r2avg_data_standard_gem = all_data_standard_gem_grouped['R2'].mean()
    r2avg_data_standard_vista = all_data_standard_vista_grouped['R2'].mean()
    r2avg_data_nordic_gem = all_data_nordic_gem_grouped['R2'].mean()


    r2avg_vs_prfsizeSlope_list = []
    for voxel_num in range(total_num_voxels): # get all the sigma values for each voxel number (gor a given subject)
        r2avg = r2avg_data_standard_gem[voxel_num] # all_data_standard_gem_grouped[voxel_num]['R2'].mean()
        sigma_vals = all_data_standard_gem_grouped.get_group(voxel_num)['sigmaMajor']
        x_vals = np.arange(len(sigma_vals))
        slope, _, _, _, _ = stats.linregress(x_vals, sigma_vals)
        r2avg_vs_prfsizeSlope_list.append({ "VoxelNum":voxel_num, "R2avg": r2avg, "pRFSizeLineSlope":slope, "Method":"Standard-Gem"})

    for voxel_num in range(total_num_voxels):
        r2avg = r2avg_data_standard_vista[voxel_num]
        sigma_vals = all_data_standard_vista_grouped.get_group(voxel_num)['sigmaMajor']
        x_vals = np.arange(len(sigma_vals))
        slope, _, _, _, _ = stats.linregress(x_vals, sigma_vals)
        r2avg_vs_prfsizeSlope_list.append({ "VoxelNum":voxel_num, "R2avg": r2avg, "pRFSizeLineSlope":slope, "Method":"Standard-Vista"})

    for voxel_num in range(total_num_voxels):
        r2avg = r2avg_data_nordic_gem[voxel_num]
        sigma_vals = all_data_nordic_gem_grouped.get_group(voxel_num)['sigmaMajor']
        x_vals = np.arange(len(sigma_vals))
        slope, _, _, _, _ = stats.linregress(x_vals, sigma_vals)
        r2avg_vs_prfsizeSlope_list.append({ "VoxelNum":voxel_num, "R2avg": r2avg, "pRFSizeLineSlope":slope, "Method":"Nordic-Gem"})

    combined_data = pd.DataFrame(r2avg_vs_prfsizeSlope_list)
    correct_r2avg_combined_data = combined_data[(combined_data['R2avg'] > 0) & (combined_data['pRFSizeLineSlope'] < 3 ) & (combined_data['pRFSizeLineSlope'] > -3 )]
    # correct_r2avg_combined_data = combined_data[(combined_data['R2avg'] > 0)]

    r2_vs_lineslope_regression_figure.clear()
    r2_vs_lineslope_scatter_plot = r2_vs_lineslope_regression_figure.add_subplot(111)
    for method in correct_r2avg_combined_data['Method'].unique():
        subset = correct_r2avg_combined_data[correct_r2avg_combined_data['Method'] == method]
        # sns.regplot(data=subset, x='R2avg', y='pRFSizeLineSlope', ax=r2_vs_lineslope_scatter_plot, label=method, scatter_kws={'s': 4}, line_kws={'label': method})        
        sns.regplot(data=subset, x='R2avg', y='pRFSizeLineSlope', ax=r2_vs_lineslope_scatter_plot, label=method, scatter=False, line_kws={'label': method})   

    # sns.lineplot(
    #     data=correct_r2avg_combined_data, 
    #     x='R2avg', 
    #     y='pRFSizeLineSlope', 
    #     hue='Method', 
    #     ax=r2_vs_lineslope_scatter_plot
    # )
    r2_vs_lineslope_scatter_plot.legend() # Add the legend
    r2_vs_lineslope_scatter_plot.set_title('R2avg vs. pRF Size Regression Line Slope')
    r2_vs_lineslope_scatter_plot.set_xlabel('R2avg')
    r2_vs_lineslope_scatter_plot.set_ylabel('pRF Size Regression Line Slope')
    r2_vs_lineslope_regression_canvas.draw()


    ############----Draw Histogram (R2avg vs. LineSlope)----############
    # ...R2avg vs. pRF Size Regression Line Slope Histogram Plot
    r2_vs_lineslope_hist_figure.clear()
    r2_vs_lineslope_hist_plot = r2_vs_lineslope_hist_figure.add_subplot(111)
    bins = np.linspace(0, 1, 11)  # Create 10 bins from 0 to 1
    correct_r2avg_combined_data['R2_bin'] = np.digitize(correct_r2avg_combined_data['R2avg'], bins) - 1

    # ...create bin labels
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    correct_r2avg_combined_data['R2_bin_label'] = correct_r2avg_combined_data['R2_bin'].map(dict(enumerate(bin_labels)))


    # sns.boxplot(data=correct_r2avg_combined_data, x='R2_bin_label', y='pRFSizeLineSlope', hue='Method', ax=r2_vs_lineslope_hist_plot)    
    sns.boxplot(data=correct_r2avg_combined_data, x='R2_bin_label', y='pRFSizeLineSlope', hue='Method', whis=(0, 100), ax=r2_vs_lineslope_hist_plot)  

    # ...set x-ticks to a subset to avoid overcrowding
    selected_ticks = np.arange(0, len(bin_labels), 2)  # Show every 2nd tick
    r2_vs_lineslope_hist_plot.set_xticks(selected_ticks)
    r2_vs_lineslope_hist_plot.set_xticklabels([bin_labels[i] for i in selected_ticks])

    r2_vs_lineslope_hist_plot.set_title('R2avg vs. pRF Size Regression Line Slope')
    r2_vs_lineslope_hist_plot.set_xlabel('R2avg Bins')
    r2_vs_lineslope_hist_plot.set_ylabel('pRF Size Regression Line Slope')
    r2_vs_lineslope_hist_canvas.draw()   


# NOTE: Test
# draw_r2_vs_prfsize_lineslope_plot(all_data_standard_gem, all_data_standard_vista, all_data_nordic_gem)


def update_plots(filtered_data_standard_gem, filtered_data_standard_vista, filtered_data_nordic_gem):
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

        # # coverage maps
        # covmap_figure_nordic_gem.clear()        
        # r2_vs_lineslope_figure_standard_gem.clear()        
        # if covmap_standard_gem is not None and covmap_nordic_gem is not None:
        #     covmap_plot_standard_gem = r2_vs_lineslope_figure_standard_gem.add_subplot(111)            
        #     covmap_plot_nordic_gem = covmap_figure_nordic_gem.add_subplot(111)

        #     # Display the pre-loaded standard coverage map
        #     if filtered_data_standard_gem['CovMapImage'].iloc[0] is not None:
        #         # covmap_plot_standard.imshow(filtered_data_standard['CovMapImage'].iloc[0])
        #         covmap_plot_standard_gem.imshow(covmap_standard_gem)
        #         covmap_plot_standard_gem.axis('off')
        #         covmap_plot_standard_gem.set_title('Standard')

        #     # Display the pre-loaded Nordic coverage map
        #     if filtered_data_nordic_gem['CovMapImage'].iloc[0] is not None:
        #         # covmap_plot_nordic.imshow(filtered_data_nordic['CovMapImage'].iloc[0])
        #         covmap_plot_nordic_gem.imshow(covmap_nordic_gem)
        #         covmap_plot_nordic_gem.axis('off')
        #         covmap_plot_nordic_gem.set_title('Nordic')                
        # # Update the canvases
        # r2_vs_lineslope_canvas_standard_gem.draw()
        # covmap_canvas_nordic_gem.draw()

        # # Update the canvases
        # r2_vs_lineslope_canvas_standard_gem.draw()
        # covmap_canvas_nordic_gem.draw()

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


        global R2_vs_LINESLOPE_PLOT_DRAWN
        if R2_vs_LINESLOPE_PLOT_DRAWN is not True:            
            draw_r2_vs_prfsize_lineslope_plot(all_data_standard_gem, all_data_standard_vista, all_data_nordic_gem)
            R2_vs_LINESLOPE_PLOT_DRAWN = True
    
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

    update_plots(filtered_voxel_esitmations_data_standard_gem, filtered_voxel_esitmations_data_standard_vista, filtered_voxel_esitmations_data_nordic_gem)
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

# Slider and Entry for Voxel Number
voxel_slider = tk.Scale(root, from_=0, to=20000, orient=tk.HORIZONTAL, label="Sorted Voxel Index (Hemi-L)", length=600) # NOTE:  show the possibility to choose from 20000 sorted indices
voxel_slider.bind("<ButtonRelease-1>", lambda event: update_voxel_entry(voxel_slider.get()))
voxel_slider.grid(row=2, column=0, columnspan=20)

voxel_entry = tk.Entry(root)
voxel_entry.grid(row=3, column=7)
voxel_entry.bind("<Return>", update_voxel_slider)
voxel_slider.set(15)  # Set initial value for the slider
update_voxel_entry(voxel_slider.get())  # Sync the entry with the slider

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

# r2_vs_lineSlope Regression plot area
r2_vs_lineslope_regression_figure = Figure(figsize=figsize, dpi=dpi)
r2_vs_lineslope_regression_canvas = FigureCanvasTkAgg(r2_vs_lineslope_regression_figure, master=root)
r2_vs_lineslope_regression_canvas.get_tk_widget().grid(row=11, column=0*column_idx_multiplier, columnspan=columnspan)

# r2_vs_lineSlope Histogram plot area
r2_vs_lineslope_hist_figure = Figure(figsize=figsize, dpi=dpi)
r2_vs_lineslope_hist_canvas = FigureCanvasTkAgg(r2_vs_lineslope_hist_figure, master=root)
r2_vs_lineslope_hist_canvas.get_tk_widget().grid(row=11, column=2*column_idx_multiplier, columnspan=columnspan)


# r2 scatter plot area
r2_scatter_figure = Figure(figsize=figsize, dpi=dpi)
r2_scatter_canvas = FigureCanvasTkAgg(r2_scatter_figure, master=root)
r2_scatter_canvas.get_tk_widget().grid(row=11, column=4*column_idx_multiplier, columnspan=columnspan)

# r2-bar chart plot area - nordic
r2_bar_figure = Figure(figsize=figsize, dpi=dpi)
r2_bar_canvas = FigureCanvasTkAgg(r2_bar_figure, master=root)
r2_bar_canvas.get_tk_widget().grid(row=11, column=6*column_idx_multiplier, columnspan=columnspan)

# Start the Tkinter main loop
root.mainloop()

##############################---------------------------------------------------------------------------################################################