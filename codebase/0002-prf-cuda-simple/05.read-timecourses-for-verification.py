import matplotlib.pyplot as plt
import os
import numpy as np

def read_timecourse_from_file():
    data_folder = "D:/code/sid-git/fmri/local-extracted-datasets/gpu-tests/result-timecourses/"

    # Initialize an empty list to hold all the data
    timecourses_data = []

    # Iterate through all the files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_folder, filename)
            timecourse = np.loadtxt(filepath)
            # Reshape the data to a 1D shape (nStimulusRows * nStimulusCols)
            timecourse = timecourse.reshape((-1,))  # Flatten to 1D
            timecourses_data.append(timecourse)

    # Now, gaussian_curves_data is a list where each element represents data from a single file
    # To convert it to a numpy array, you can use:
    timecourses_data = np.array(timecourses_data)
    return timecourses_data

####---main()
def main():
    read_timecourses_data = read_timecourse_from_file()

##################-----------Verification Program-------------####################
main()

print("done")