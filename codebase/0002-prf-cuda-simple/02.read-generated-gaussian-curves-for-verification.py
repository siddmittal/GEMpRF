import matplotlib.pyplot as plt
import os
import numpy as np

def generate_and_save_gaussian_curves():
    # Test space points
    nTestspace_rows = 1
    nTestspace_cols = 1
    xx = np.linspace(-9, +9, nTestspace_cols)
    yy = np.linspace(-9, +9, nTestspace_rows)
    X, Y = np.meshgrid(xx, yy)

    # Stimulus points
    nStimulus_rows = 3
    nStimulus_cols = 3
    stim_xx = np.linspace(-9, +9, nStimulus_cols)
    stim_yy = np.linspace(-9, +9, nStimulus_rows)
    stim_X, stim_Y = np.meshgrid(stim_xx, stim_yy)

    # Sigma
    sigma = 2

    # Create the directory to save the files
    data_folder = r'D:\\code\\sid-git\\fmri\\local-extracted-datasets\\gpu-tests\\resulting-gaussian-curves\\1 x 1 gaussian curves-python'
    os.makedirs(data_folder, exist_ok=True)

    for row in range(nTestspace_rows):
        for col in range(nTestspace_cols):
            mean_x = X[row, col]
            mean_y = Y[row, col]
            pRF = (
                np.exp(-((stim_X - mean_x) ** 2) / (2 * sigma ** 2))
                * np.exp(-((stim_Y - mean_y) ** 2) / (2 * sigma ** 2)
                )
            )

            # Reshape the Gaussian curve for saving
            pRF = pRF.flatten()

            # Save to a text file
            filename = os.path.join(data_folder, f"gc-({mean_x}, {mean_y}).txt")
            with open(filename, "w") as file:
                for value in pRF:
                    file.write(f"{value}\n")




def read_generated_gaussianCurves_from_file():
    data_folder = "D:/code/sid-git/fmri/local-extracted-datasets/gpu-tests/resulting-gaussian-curves/101 x 101 gaussian curves/"

    # Initialize an empty list to hold all the data
    gaussian_curves_data = []

    # Iterate through all the files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_folder, filename)
            curve_data = np.loadtxt(filepath)
            # Reshape the data to a 1D shape (nStimulusRows * nStimulusCols)
            curve_data = curve_data.reshape((-1,))  # Flatten to 1D
            gaussian_curves_data.append(curve_data)

    # Now, gaussian_curves_data is a list where each element represents data from a single file
    # To convert it to a numpy array, you can use:
    gaussian_curves_data = np.array(gaussian_curves_data)

####---main()
def main():
    generate_and_save_gaussian_curves()

##################-----------Verification Program-------------####################
main()

print("done")