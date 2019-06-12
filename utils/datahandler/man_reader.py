#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:07:46 2019

@author: julia
"""

import numpy as np
import xarray as xr
import glob
import os
import sys
import pandas as pd

class man_reader():
    
    def __init__(self, path="", pathout="", aod_sda="AOD", sampled="daily",
                 level=20):
        if sampled not in ["all_points", "daily", "series"]:
            print("Sampled must be either 'all_points', 'daily' or 'series'.")
        if level not in [10, 15, 20]:
            print("Level must be either 10, 15 or 20.")
        
        self.path = path
        self.pathout = pathout
        self.aod_sda = aod_sda
        self.sampled = sampled
        self.level = level
        self.file_nc = os.path.join(self.pathout, "","{}_{}_lev{}.nc".format(
                                    self.aod_sda, self.sampled, self.level))
    
    def read_man(self):
        # files containing data
        if self.aod_sda == "AOD":
            str_files = "*_{}.lev{}".format(self.sampled, self.level)
        filepath = os.path.join(self.path, "", self.aod_sda, "")
        filelist = sorted(glob.glob(filepath + str_files))
        
        # set dateparser and parsedates to read in dates
        dateparser = lambda x: pd.Timestamp.strptime(x, '%d:%m:%Y %H:%M:%S')
        parsedates = {"Datetime": [0,1]}
        # check if there are any files to read
        if not filelist:
            print("No files to read. Please make other specifications.")
            sys.exit()
        # iterate over files and create pd.Dataframe
        for file in filelist:
            temp = pd.read_csv(file, sep=",", header=4,
                               parse_dates=parsedates,
                               date_parser=dateparser,
                               na_values=-999.000000)
            try:
                df = pd.concat([df, temp], axis=0, sort=True)
            except NameError:
                df = temp
        
        # check in column names for spaces
        df.rename(lambda x: x.replace(" ", "_"), axis="columns", inplace=True)
        
        if self.sampled == "daily":
            df.Datetime = df.Datetime.apply(lambda x: x.replace(hour=0,
                                                                minute=0,
                                                                second=0))
        if self.level == 20:
            df = df.drop(columns=["Date(dd:mm:yyyy)", "Time(hh:mm:ss)",
                                  "Last_Processing_Date(dd:mm:yyyy)"])
        
        # create MultiIndex and sort
        df.set_index(["Datetime", "Latitude", "Longitude"], inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        return df
    
    def df_to_ds(self, df):
        # create xr.Dataset from pd.Dataframe
        ds = xr.Dataset(df)
        
        return ds
    
    def write_ds_to_nc(self, ds):
        # writes xr.Dataset with multiindex to netcdf
        ds = ds.reset_index("dim_0")
        ds.to_netcdf(self.file_nc, mode="w")
    
    def read_ds_from_nc(self):
        # reads xr.Dataset from netcdf and constructs multiindex
        ds = xr.open_dataset(self.file_nc)
        ds = ds.set_index(dim_0=["Datetime", "Latitude", "Longitude"])
        
        return ds
        
    def main(self):
        df = man.read_man()
        ds = man.df_to_ds(df)
        man.write_ds_to_nc(ds)
#        ds_new = man.read_ds_from_nc()
        
        return ds
    
if __name__ == "__main__":
    path = "/Users/julia/Documents/MPI_Sonne/microtops/data/All_MAN_Data/"
    man = man_reader(path, path, "AOD", "daily", 20)
    ds = man.main()
    
    