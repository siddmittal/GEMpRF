# -*- coding: utf-8 -*-
"""
"@Author  :   Siddharth Mittal",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2023-2025, Medical University of Vienna",
"@Desc    :   None",
        
"""

import subprocess
import sys


def check_gemprf_not_installed():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to run pip list: {e}")

    output = result.stdout.lower()

    if "gemprf" in output:
        print(f"\033[91mERROR: A pip-installed 'gemprf' package was found in the environment.\n"
              f"Please uninstall it to avoid conflicts with the local GEMpRF code.\033[0m")
        sys.exit(1)