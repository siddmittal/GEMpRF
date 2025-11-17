import sys
import importlib

_real_gem = importlib.import_module("gem")
_real_gem.__gemprf_version__ = "0.1.3" 

sys.modules[__name__] = _real_gem
