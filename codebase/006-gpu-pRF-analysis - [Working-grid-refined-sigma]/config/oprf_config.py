import json
import os

class ConfigurationWrapper:
    config_data = None  # Class-level attribute to store the configuration data

    @classmethod
    def _read_json_file(cls):
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
        config_file_path = os.path.join(script_dir, "config.json")  # Construct the path to config.json

        if os.path.isfile(config_file_path):
            with open(config_file_path, "r") as config_file:
                cls.config_data = json.load(config_file)
        else:
            print("Config file not found. Make sure it's located in the same directory as the script.")
        
    @classmethod
    def load_configuration(cls):
        if cls.config_data is None:
            cls._read_json_file()

        # path_to_append
        cls.path_to_append = cls.config_data.get("path_to_append")

        # result_file_path
        cls.result_file_path = cls.config_data.get("result_file_path")        

        # results
        cls.results = cls.config_data.get("results")

        # Stimulus
        cls.stimulus = cls.config_data.get("stimulus")
        
        # Measure Data
        cls.measured_data = cls.config_data.get("measured_data")
        
        # Search Space
        cls.search_space = cls.config_data.get("search_space")  

        # Concatenated runs
        cls.concatenated_runs = cls.config_data.get("concatenated_runs")        

        # Test Space
        cls.test_space = cls.config_data.get("test_space")

        # Noise
        cls.noise = cls.config_data.get("noise")           

        # GPU
        cls.gpu = cls.config_data.get("gpu")           

# Usage:
if __name__ == "__main__":
    ConfigurationWrapper.load_configuration()

    # Access the configuration parameters using the class itself
    print(f"Visual Field Size: {ConfigurationWrapper.path_to_appended}")
    print(f"Stimulus: {ConfigurationWrapper.stimulus}")
