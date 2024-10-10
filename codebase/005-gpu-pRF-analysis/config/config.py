
import json
import os

class Configuration:
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
        self.config_file_path = os.path.join(script_dir, "config.json")  # Construct the path to config.json
        self.load_configuration()

    def load_configuration(self):
        if os.path.isfile(self.config_file_path):
            with open(self.config_file_path, "r") as config_file:
                config_data = json.load(config_file)

            # Visual field
            self.visual_field_size = self.asint(config_data["visual_field_size"])

            # Stimulus
            self.stimulus = config_data["stimulus"]
            self.stimulus_file_path = self.stimulus["filePath"]
            self.stimulus_width = self.asint(self.stimulus["width"])
            self.stimulus_height = self.asint(self.stimulus["height"])
            self.stimulus_num_frames = self.asint(self.stimulus["num_frames"])

            # Search Space
            self.search_space = config_data["search_space"]
            self.search_space_rows = self.asint(self.search_space["nRows"])
            self.search_space_cols = self.asint(self.search_space["nCols"])

            # Test Space
            self.test_space = config_data["test_space"]
            self.test_space_rows = self.asint(self.test_space["nRows"])
            self.test_space_cols = self.asint(self.test_space["nCols"])

            # Noise
            self.noise = config_data["noise"]
            self.noise_level = self.asfloat(self.noise["level"])
            self.noise_model = self.noise["model"]
            self.synthesis_ratio = self.asint(self.noise["synthesis_ratio"])
        else:
            print("Config file not found. Make sure it's located in the same directory as the script.")

    def asint(self, value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None  # Handle conversion errors as needed

    def asfloat(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None  # Handle conversion errors as needed

# Usage:
if __name__ == "__main__":
    config = Configuration()

    # Access the configuration parameters using the class instance
    print(f"Visual Field Size: {config.visual_field_size}")
    print(f"Stimulus File Path: {config.stimulus_file_path}")
    # Access and use other parameters as needed
