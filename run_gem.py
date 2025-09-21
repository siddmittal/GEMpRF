# -*- coding: utf-8 -*-
"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Medical University of Vienna",
"@Desc    :   None",

"""
import time
import datetime
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__))))
from gem_program import run
import argparse

def main():
    default_config = os.path.join(
        os.path.dirname(__file__), 'gem', 'configs', 'analysis_configs', 
        'analysis_config.xml'
    )

    # Check if config_filepath is provided as a command-line argument
    if len(sys.argv) == 1:
        # No argument provided
        print("\033[91m\n\n⚠️ You have not provided a configuration filepath as an argument.\033[0m")
        if os.path.exists(default_config):
            print(f"\033[38;5;208mThe program will try to use the default config filepath:\n\033[0m  {default_config}")
            choice = input("\033[38;5;208m\nDo you want to continue with this default config? [y/N]: \033[0m").strip().lower()
            if choice != 'y':
                print("Aborting program.")
                sys.exit(0)
            sys.argv.append(default_config)
        else:
            print(f"\033[91mError: The default configuration file does not exist:\n  {default_config}\033[0m")
            print("Aborting program.")
            sys.exit(1)

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the GEM pRF Analysis.")
    parser.add_argument('config_filepath', type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Start the analysis
    start_time = datetime.datetime.now()
    run()  # can access args.config_filepath if needed inside run()
    print(f"\nComplete Time taken: {datetime.datetime.now() - start_time}")

if __name__ == "__main__":
    main()