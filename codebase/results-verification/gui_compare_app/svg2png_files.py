import os
from io import BytesIO
import cairosvg
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

subs = ['001', '002']
sessions = ['001', '002', '003', '004', '005', '006']
runs = ['01', '02', '03', '04', '05', '0102030405avg']
hemis = ['L'] #['L', 'R']
pRF_estimations_basepath = "D:/results/with-without-nordic-covmap/analysis-03_AsusCorrect"    
covmaps_basepath = f"D:/results/with-without-nordic-covmap/vista-covMap" #"D:/results/with-without-nordic-covmap/covMap"

def convert_svg_to_png(svg_filepath):
    png_filepath = svg_filepath.replace('.svg', '.png')
    try:
        if not os.path.exists(os.path.dirname(png_filepath)):
            os.makedirs(os.path.dirname(png_filepath))
        with open(svg_filepath, 'rb') as svg_file:
            png_image = cairosvg.svg2png(file_obj=svg_file)
            with open(png_filepath, 'wb') as png_file:
                png_file.write(png_image)
        print(f"Converted {svg_filepath} to {png_filepath}")
    except Exception as e:
        print(f"Failed to convert {svg_filepath}: {e}")

def covert_covmap_data_2_png(isNordic: bool = False):
    covmaps_svgfiles_list = []

    for sub in subs:
        for ses in sessions:
            for run in runs:
                if isNordic:
                    covmap_filepath = os.path.join(
                        covmaps_basepath, 
                        f"sub-{sub}", 
                        f"ses-{ses}nn", 
                        f"sub-{sub}_ses-{ses}nn_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.svg"
                    )
                else:
                    covmap_filepath = os.path.join(
                        covmaps_basepath, 
                        f"sub-{sub}", 
                        f"ses-{ses}", 
                        f"sub-{sub}_ses-{ses}_task-bar_run-{run}_desc-V1-VarExp10-max_covmap.svg"
                    )

                if os.path.exists(covmap_filepath):
                    covmaps_svgfiles_list.append(covmap_filepath)
                else:
                    print(f"File does not exist: {covmap_filepath}")

    with ProcessPoolExecutor() as executor:
        executor.map(convert_svg_to_png, covmaps_svgfiles_list)

###############-----------------Main()-----------------################
if __name__ == "__main__":
    covert_covmap_data_2_png(isNordic=False)
    covert_covmap_data_2_png(isNordic=True)