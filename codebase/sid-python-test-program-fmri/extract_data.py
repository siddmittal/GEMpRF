import os
import zipfile

def create_extracted_datasets_folder():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(script_directory, ".."))
    grandparent_directory = os.path.abspath(os.path.join(parent_directory, ".."))
    extracted_datasets_folder = os.path.join(grandparent_directory, "local-extracted-datasets")

    if not os.path.exists(extracted_datasets_folder):
        os.makedirs(extracted_datasets_folder)

    return extracted_datasets_folder

def extract_zip_file(zip_file_path):
    extracted_datasets_folder = create_extracted_datasets_folder()

    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    absolute_zip_file_path = os.path.join(current_script_directory, zip_file_path)

    with zipfile.ZipFile(absolute_zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_datasets_folder)

    print(f"Successfully extracted '{zip_file_path}' to '{extracted_datasets_folder}'.")



def create_local_folder_and_extract_data(zip_file_path):
    extract_zip_file(zip_file_path)
