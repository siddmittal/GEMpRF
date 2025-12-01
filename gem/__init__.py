"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   None",

"""

# Lazy loader to avoid circular imports and behave identical to source structure.

import importlib
import os

# Dynamically detect all submodules in the gem directory
_package_dir = os.path.dirname(__file__)
_submodules = {
    name
    for name in os.listdir(_package_dir)
    if os.path.isdir(os.path.join(_package_dir, name))
    and not name.startswith("_")
}

# expose run_gem.run at top-level:
def run(*args, **kwargs):
    from .run_gem import run as _run
    return _run(*args, **kwargs)

def __getattr__(name):
    if name in _submodules:
        return importlib.import_module(f"gem.{name}")
    raise AttributeError(f"module 'gem' has no attribute '{name}'")