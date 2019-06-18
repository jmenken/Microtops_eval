import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import abc
from datetime import datetime as dt
from MainReader import MainReader

ALLOWED_SATELLITES = ["MODIS", "VIIRS", "MISR"]

class SatelliteReader(MainReader):
    def __init__(self, start_date=None, end_date=None):
        super().__init__()
        self.start_date = start_date # e.g. "2018-01-01"
        self.end_date = end_date # e.g. "2018-01-02"

    @staticmethod
    def __check_satellite_name(satellite):
        if not satellite in ALLOWED_SATELLITES:
            raise ValueError(f"Unknown satellite {satellite}. Please choose one of {' | '.join(ALLOWED_SATELLITES)}")
        else:
            return satellite

    def read_data(self, satellite):
        satellite = self.__check_satellite_name(satellite)
        ds = None

        if satellite == "MODIS":
            ds = self.__read_MODIS()

        elif satellite == "VIIRS":
            ds = self.__read_VIIRS()

        elif satellite == "MISR":
            ds = self.__read_MISR()

        assert ds is not None, "Something went wrong. Try to provide another satellite name."
        
        return ds

    @abc.abstractmethod
    def __read_MODIS(self):
        modis_nc = os.path.join((self.config["SATELLITES"]["MODISNC"]), "modis.nc")
        if os.path.exists(modis_nc):
            print("Reading modis.nc ...")
            ds = xr.open_dataset(modis_nc, chunks=100)
#            ds = ds.set_index(dim_0=["time", "lat", "lon"])
        else:
            print("Constructing modis.nc ...")
            files = self.__get_MODIS_files()
            
            cols = ["time", "lat", "lon", "AOD", "AOD_std", "AOD_L2_std",
                    "AOD_obs_err", "num"]
            df = pd.concat((pd.read_csv(file, delim_whitespace=True, skiprows=1,
                                        header=0, skipfooter=6, engine="python",
                                        parse_dates=[0], names=cols,
                                        date_parser=self.__get_MODIS_time())
                            for file in files), axis=0)
            
            # get daily data for each lat, lon
            groupers = [pd.Grouper(key="time", freq="D"), "lat", "lon"]
            agg_dict = {"AOD": np.mean, "AOD_std": np.mean, "AOD_L2_std": np.mean,
                        "AOD_obs_err": np.mean, "num": np.sum}
            df = df.groupby(groupers).agg(agg_dict)
            ds = xr.Dataset(df)
            
            ds = ds.reset_index("dim_0")
            ds.to_netcdf(modis_nc, mode="w")
        
        return ds
    
    def __get_MODIS_files(self):
        allfiles = sorted(glob.iglob(self.config["SATELLITES"]["MODIS"] + "*/*/*",
                                     recursive=True))
        files =[]
        for file in allfiles:
            start, end = (True, True)
            f_time = pd.Timestamp.strptime(file.split("/")[-1],
                                           "%Y%m%d%H_obsnew.txt")
            if self.start_date and f_time<pd.Timestamp(self.start_date):
                start = False
            if self.end_date and f_time>pd.Timestamp(self.end_date)\
                                        .replace(hour=23, minute=59):
                end = False
            if start and end:
                files += [file]
        if not files:
            raise ValueError ("No files to read.")
        return files
    
    @staticmethod
    def __get_MODIS_time():
        return lambda x: pd.Timestamp.strptime(x,"%Y%m%d%H%M")
    
    @abc.abstractmethod
    def __read_VIIRS(self):
        files = sorted(glob.glob(self.config["SATELLITES"]["VIIRS"] + "*"))
        ds = xr.open_mfdataset(files,
                               concat_dim="time",
                               preprocess=self.__get_VIIRS_time)
        ds = ds.sel(time=slice(self.start_date, self.end_date))
        return ds
    
    @staticmethod
    def __get_VIIRS_time(dataset):
        file_name = os.path.split(dataset.encoding['source'])[-1]
        file_str = "NOAA_EPS_AOD_%Y%m%d.nc"
        return dataset.assign(time=dt.strptime(file_name, file_str))

    @abc.abstractmethod
    def __read_MISR(self):
        files = sorted(glob.glob(self.config["SATELLITES"]["MISR"] + "*"))
        ds = xr.open_mfdataset(files,
                               group="Aerosol_Parameter_Average",
                               concat_dim="time",
                               preprocess=self.__get_MISR_time)
        ds = ds.sel(time=slice(self.start_date, self.end_date))
        return ds
    
    @staticmethod
    def __get_MISR_time(dataset):
        file_name = os.path.split(dataset.encoding['source'])[-1]
        file_str = "MISR_AM1_CGAS_%b_%d_%Y_F15_0032.nc"
        return dataset.assign(time=dt.strptime(file_name, file_str))

if __name__ == "__main__":
    Sr = SatelliteReader()
#    viirs = Sr.read_data("VIIRS")
#    misr = Sr.read_data("MISR")
    modis = Sr.read_data("MODIS")