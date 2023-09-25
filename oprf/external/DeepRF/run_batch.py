#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os

# Wrapper script to be able to submit specific jobs
script = "/project/2420084.01/deeprf/new/run_job.py" 

# The subjects in the empirical dataset to run parallel jobs for
subjects = range(29)

# Method to jobmit jobs for
method = "cfprf"

# Toggle to submit empirical jobs
do_empirical = True

# Toggle to submit synthetic job
do_synthetic = True

# Specific settings for coarse-to-fine
if method == "cfprf":
    timreq = "18:00:00"
    memreq = "1gb"

# Specific settings for DeepRF
elif method == "deeprf":
    timreq = "00:10:00"
    memreq = "1gb"
    os.environ["cat_dim"] = "chan"
    os.environ["n_layers"] = "50"
    os.environ["n_dropout"] = "0"

# Specific settings for DeepRF plus coarse-to-fine
elif method == "deeprf_cfprf":
    timreq = "12:00:00"
    memreq = "1gb"
    os.environ["deeprf"] = "resnet50_chan"

# Unknown
else:
    raise Exception("Unknown method:", method)

# Submit empirical jobs
if do_empirical:
    for subject in subjects:
        job_name = "{}_{:02d}".format(method, subject)
        os.environ["subject"] = str(subject)
        os.environ["method"] = method
        command = "echo '{}' | qsub -V -N {} -l mem={},walltime={}".format(script, job_name, memreq, timreq)
        os.system(command)

# Submit synthetic job
if do_synthetic:
    job_name = "{}_sy".format(method)
    os.environ["subject"] = "-1"
    os.environ["method"] = method
    command = "echo '{}' | qsub -V -N {} -l mem={},walltime={}".format(script, job_name, memreq, timreq)
    os.system(command)
