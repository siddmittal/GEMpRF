# Add my "oprf" package path
import sys
sys.path.append("D:/code/sid-git/fmri/")

# imports
import numpy as np
import os
import matplotlib.pyplot as plt

# Local Imports
from oprf.standard.prf_stimulus import Stimulus
from oprf.external.hrf_generator_script import spm_hrf_compat # HRF Generator

def write_stimulus_data_to_file(stimulus_data, output_dir):
    # Write each frame to a separate file
    for frame_num in range(stimulus_data.shape[2]):
        frame_data = stimulus_data[:, :, frame_num]
        frame_filename = os.path.join(output_dir, f"frame-{frame_num}.txt")
        with open(frame_filename, "w") as file:
            for row in frame_data:
                for value in row:
                    file.write(f"{value:.6f}\n")

# Read back the data to verify
def read_frame_data(frame_num, dir):
    frame_filename = os.path.join(dir, f"frame-{frame_num}.txt")
    frame_data = []
    with open(frame_filename, "r") as file:
        for line in file:
            value = float(line.strip())
            frame_data.append(value)
    return np.array(frame_data).reshape(101, 101)

##############################---Program---#######################
def RunProgram():
    # HRF Curve
    hrf_t = np.linspace(0, 30, 31)
    hrf_curve = spm_hrf_compat(hrf_t)

    # Load stimulus
    stimulus = Stimulus("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\sid-prf-fmri-data\\task-bar_apertures.nii.gz", size_in_degrees=9)
    stimulus.compute_resample_stimulus_data((101, 101, stimulus.org_data.shape[2]))
    stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)

    file_dir= "D:\\code\\sid-git\\fmri\\local-extracted-datasets\\stimulus-data-test"

    # Write stimulus data to file
    # access stimulus frame: "stimulus.resampled_hrf_convolved_data[:,:,299].shape" <<----all rows, all cols of frame-299th 
    # write_stimulus_data_to_file(stimulus_data=stimulus.resampled_hrf_convolved_data, output_dir= "D:\\code\\sid-git\\fmri\\local-extracted-datasets\\stimulus-data-test")

    # Read back to verify
    test_frame = read_frame_data(frame_num=100, dir=file_dir)

    # VERIFICATION
    plt.figure()
    plt.imshow(stimulus.resampled_hrf_convolved_data[:,:,200], cmap='RdYlBu', vmin=-1.0, vmax=1.0) # data which was written to file
    plt.figure()
    plt.imshow(read_frame_data(frame_num=200, dir=file_dir), cmap='RdYlBu', vmin=-1.0, vmax=1.0) # data read back from file
    plt.show()

    print("done!")

if __name__ == "__main__":
    RunProgram()

