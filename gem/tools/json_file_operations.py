import json
import os
import datetime

class JsonMgr:
    @classmethod
    def write_to_file(cls, filepath, data):

        # create folders     
        # Extract the directory path from the file path
        directory_path = os.path.dirname(filepath)

        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Convert the list of dictionaries to a JSON string
        json_string = json.dumps(data, indent=4)

        # Write the JSON data to the file
        with open(filepath, 'w') as json_file:
            json_file.write(json_string)


    @classmethod
    def args2jsonEntry(cls, muX, muY, sigma, r2, signal):        
        json_entry = {
                    "Centerx0": round(float(muX), 4),
                    "Centery0": round(float(muY), 4),
                    "Theta": 0,
                    "sigmaMajor": round(float(sigma), 4),
                    "sigmaMinor": 0,
                    "R2": round(float(r2), 4),
                    "modelpred": signal.tolist()
                }
        return json_entry
    
    @classmethod
    def log_string(cls, message, filepath):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filepath, "a") as logfile:
            logfile.write("[{}] {}\n".format(timestamp, message))
                    
        return    