"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   None",

"""

# gemprf/__init__.py

import importlib

# Load real gem package lazily through importlib
_gem = importlib.import_module("gem")

# Expose selected API from gemprf (not gem)
__all__ = ["run", "__version__"]

def run(*args, **kwargs):
    return _gem.run(*args, **kwargs)

__version__ = "0.1.11"
