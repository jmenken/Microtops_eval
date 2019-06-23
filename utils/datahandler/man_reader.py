#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import pandas as pd
import numpy as np
import xarray as xr
from MainReader import MainReader
import dask.dataframe as ddf
from tqdm import tqdm

class MANReader(MainReader):
    
    def __init__(self, sampled="daily", level=20):
        super().__init__()
        self.sampled = sampled
        self.level = level
    
    def __check_init(self):
        if self.sampled not in ["all_points", "daily", "series"]:
            raise ValueError("Sampled must be either 'all_points', 'daily' or 'series'.")
        if self.level not in [10, 15, 20]:
            raise ValueError("Level must be either 10, 15 or 20.")
    
    def __assign_aod550(self, df):
        return df.assign(AOD_550nm=(- df["440_870nm_Angstrom_Exponent"]
                                    * np.log(.55 / .44)
                                    + (df["AOD_440nm"]).apply(np.log)
                                    ).apply(np.exp))
    
    def __dateparser(self):
        return lambda x: pd.Timestamp.strptime(x, '%d:%m:%Y %H:%M:%S')
        
    def read_data(self):    
        self.__check_init()
        
        filelist = sorted(glob.glob(os.path.join(self.config["MANDATA"]["PATH"])
                                    + "*_{}.lev{}".format(self.sampled, self.level)))
        if not filelist:
            raise ValueError("No files to read. Please make other selections.")
        
        df = pd.concat((pd.read_csv(file, sep=",", header=4,
                                    parse_dates={"Datetime": [0,1]},
                                    date_parser=self.__dateparser(),
                                    na_values=-999.000000)
                        for file in filelist), axis=0, sort=True)
        
        df.rename(lambda x: x.replace(" ", "_"), axis="columns", inplace=True)
        df.rename(lambda x: x.replace("-", "_"), axis="columns", inplace=True)
        df = self.__assign_aod550(df)
        
        if self.sampled == "daily":
            df.Datetime = df.Datetime.apply(lambda x: x.replace(hour=0,
                                                                minute=0,
                                                                second=0))
        if self.level == 20:
            df = df.drop(columns=["Date(dd:mm:yyyy)", "Time(hh:mm:ss)",
                                  "Last_Processing_Date(dd:mm:yyyy)"])
        
        df.set_index(["Datetime", "Latitude", "Longitude"], inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        return df
    
    def df_to_ds(self, df):
        # create xr.Dataset from pd.Dataframe
        return xr.Dataset(df)
    
    def write_ds_to_nc(self, ds):
        # writes xr.Dataset with multiindex to netcdf
        file_nc = os.path.join(self.config["MANDATA"]["PATHNC"],
                               "{}_lev{}.nc".format(self.sampled, self.level))
        ds = ds.reset_index("dim_0")
        ds.to_netcdf(file_nc, mode="w")
    
    def read_ds_from_nc(self):
        # reads xr.Dataset from netcdf and constructs multiindex
        file_nc = os.path.join(self.config["MANDATA"]["PATHNC"],
                               "{}_lev{}.nc".format(self.sampled, self.level))
        ds = xr.open_dataset(file_nc)
        return ds.set_index(dim_0=["Datetime", "Latitude", "Longitude"])
        
        
def main():
    man = MANReader()
    df = man.read_data()
    ds = man.df_to_ds(df)
    man.write_ds_to_nc(ds)
    
    return ds
    
if __name__ == "__main__":
    # man_ds = main()
    man = MANReader()
    df = man.read_data()
    ds = man.df_to_ds(df)

    import matplotlib.pyplot as plt
    import matplotlib
    from datetime import datetime as dt
    import cartopy.crs as ccrs
    import cartopy

    ds_AOD = ds.AOD_550nm.sortby(ds.AOD_550nm)

    # ds = ds.isel(dim_0=slice(5000, None))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)

    # ax.stock_img()

    # cmap = matplotlib.cm.get_cmap('viridis')
    # normalize = matplotlib.colors.Normalize(vmin=ds.AOD_550nm.values.min(), vmax=ds.AOD_550nm.values.min())
    # colors = [cmap(normalize(value)) for value in ds.AOD_550nm.values]

    lons = ds_AOD.Longitude.values
    lats = ds_AOD.Latitude.values
    im = ax.scatter(lons, lats, c=ds_AOD.values, s=0.1, cmap="inferno_r", vmax=2)
    plt.colorbar(im, extend="max")
    plt.tight_layout()
    plt.savefig("man_test.pdf")
    plt.show()
    