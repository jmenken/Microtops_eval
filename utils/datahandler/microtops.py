#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:07:46 2019

@author: julia
"""

import numpy as np
import xarray as xr
import glob


path = "/Users/julia/Documents/MPI_Sonne/microtops/data/All_MAN_Data"
aod_sda = "AOD"
sampled = "daily" #"all_points", "daily", "series"
level = 20 #10, 15, 20

filepath = 1
"_daily.lev20"