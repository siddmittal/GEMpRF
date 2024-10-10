#!/home/mrphys/jorthi/.conda/envs/deeprf/bin/python
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os

# Subject number in empirical dataset (note: negative for synthetic dataset)
subject = int(os.getenv("subject", "0"))

# Method to apply
method = os.getenv("method", "cfprf")

# Coarse-to-fine method
if method == "cfprf":
    if subject < 0:
        import test_cfprf_synthetic
        test_cfprf_synthetic.run()
    else:
        import test_cfprf_empirical
        test_cfprf_empirical.run(subject=subject)

# DeepRF
elif method == "deeprf":
    cat_dim = os.getenv("cat_dim", "time")
    n_layers = int(os.getenv("n_layers", "50"))
    n_dropout = int(os.getenv("n_dropout", "1000"))
    if subject < 0:
        import test_deeprf_synthetic
        test_deeprf_synthetic.run(cat_dim=cat_dim, n_layers=n_layers, n_dropout=n_dropout)
    else:
        import test_deeprf_empirical
        test_deeprf_empirical.run(subject=subject, cat_dim=cat_dim, n_layers=n_layers, n_dropout=n_dropout)

# DeepRF as coarse for the coarse-to-fine method
elif method == "deeprf_cfprf":
    deeprf = os.getenv("deeprf", "resnet50_time")
    if subject < 0:
        import test_deeprf_cfprf_synthetic
        test_deeprf_cfprf_synthetic.run(deeprf=deeprf)
    else:
        import test_deeprf_cfprf_empirical
        test_deeprf_cfprf_empirical.run(subject=subject, deeprf=deeprf)

# Unknown
else:
    raise Exception("Unknown method:", method)
