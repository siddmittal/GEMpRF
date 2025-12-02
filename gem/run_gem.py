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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

def run(config_filepath=None):
    # Check if cupy is available for GPU support
    try:
        import cupy
    except ImportError:
        print("\033[93m\nCuPy python package isn’t installed. Install it and ensure you have an NVIDIA GPU with a compatible CUDA toolkit.\033[0m")
        print("\033[91m\nAborting program\033[0m")
        sys.exit(1)        

    # Check if config_filepath is provided as a command-line argument or as a function argument
    if config_filepath is None:
        print("\033[91m\n\n⚠️ You have not provided a configuration filepath as an argument.\033[0m")
        print("\033[91m\nAborting program\033[0m")
        sys.exit(1)

    if not os.path.exists(config_filepath):
        print(f"\033[91mError: The configuration file does not exist:\n  {config_filepath}\033[0m")
        print("\033[91m\nAborting program\033[0m")
        sys.exit(1)

    # Start the analysis
    start_time = datetime.datetime.now()

    try:
        # When run as a package: python -m gem.run_gem
        from .init_setup import init_setup
    except ImportError:
        # When run directly: python run_gem.py
        from init_setup import init_setup

    init_setup(config_filepath=config_filepath)
    print(f"\nComplete Time taken: {datetime.datetime.now() - start_time}")
    sys.exit(0)

if __name__ == "__main__":
    __import__("utils.assert_no_gemprf").assert_no_gemprf.check_gemprf_not_installed()
    parser = argparse.ArgumentParser(description="Run the GEM pRF Analysis.")
    parser.add_argument("config_filepath", type=str, help="Path to the XML configuration file")
    args = parser.parse_args()
    run(args.config_filepath)