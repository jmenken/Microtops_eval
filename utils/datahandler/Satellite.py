import os
import glob
import xarray as xr
import pandas as pd
import dask.dataframe as ddf
import dask.array as da
from dask.distributed import Client
from dask.delayed import delayed
import numpy as np
from datetime import datetime as dt
from .MainReader import MainReader
import logging
from tqdm import tqdm
import gc
import dask

ALLOWED_SATELLITES = ["MODIS", "VIIRS", "MISR"]

class SatelliteReader(MainReader):
    def __init__(self, start_date=None, end_date=None):
        super().__init__()
        self.start_date = start_date # e.g. "2018-01-01"
        self.end_date = end_date # e.g. "2018-01-02"
        self.client = Client()

    @staticmethod
    def __check_satellite_name(satellite):
        if not satellite in ALLOWED_SATELLITES:
            raise ValueError(f"Unknown satellite {satellite}. Please choose one of {' | '.join(ALLOWED_SATELLITES)}")
        else:
            return satellite

    def read_data(self, satellite):
        satellite = self.__check_satellite_name(satellite)
        ds = None
        logging.debug(f"Reading data for {satellite}")
        if satellite == "MODIS":
            ds = self.__read_MODIS()

        elif satellite == "VIIRS":
            ds = self.__read_VIIRS()

        elif satellite == "MISR":
            ds = self.__read_MISR()

        assert ds is not None, "Something went wrong. Try to provide another satellite name."
        
        return ds

    @staticmethod
    def __read_MODIS_NC(modis_nc):
        logging.info("Reading modis.nc ...")
        ds_loaded = xr.open_dataset(modis_nc, chunks=100)
        # return ds_loaded.set_index(dim_0=("time","lat","lon"))
        return ds_loaded

    @staticmethod
    def __chunk_list(list, size):
        for i in range(0, len(list), size):
            yield list[i:i + size]

    def __read_MODIS_text(self, modis_nc):
        logging.info("Constructing modis.nc ...")
        all_files = self.__get_MODIS_files()
        size = 1000

        cols = ["time", "lon", "lat", "AOD", "AOD_std", "AOD_L2_std",
                "AOD_obs_err", "num"]

        counter = 0
        chunks = [x for x in self._chunk_list(all_files, size)]
        for files in tqdm(chunks):
            save_name = modis_nc.replace(".nc", f"_{counter}.nc")
            dfs = ddf.read_csv(files, delim_whitespace=True, skiprows=1,
                               header=0, skipfooter=6, engine="python", names=cols)

            dfs["time"] = ddf.to_datetime(dfs["time"], format="%Y%m%d%H%M")
            df = dfs.compute()

            groupers = [pd.Grouper(key="time", freq="D"), "lat", "lon"]
            agg_dict = {"AOD": np.mean, "AOD_std": np.mean, "AOD_L2_std": np.mean,
                        "AOD_obs_err": np.mean, "num": np.sum}
            df = df.groupby(groupers).agg(agg_dict)

            self.df = df
            logging.debug("Now constructing xarray dataset")
            ds = xr.Dataset(df)
            logging.debug(f"constructed Dataset: {ds}")
            ds = ds.unstack()

            delayed_write = ds.to_netcdf("./tmp.nc", mode="w", compute=False)
            with self.client:
                delayed_write.compute()
            ds.close()
            del ds
            gc.collect()

            ds = xr.open_dataset("./tmp.nc", chunks=100)

            logging.debug("Unstacked the dataset")

            ds = self.__create_MODIS_grid(ds)

            delayed_write = ds.to_netcdf(save_name, mode="w", compute=False)
            with self.client:
                delayed_write.compute()
            if os.path.isfile(save_name):
                logging.debug(f"saved file to {save_name}")
            else:
                logging.critical(f"Modis nc file was not saved {save_name}")
            ds.close()
            counter += 1

        final_files = []
        for i in range(counter):
            final_files.append(modis_nc.replace(".nc", f"_{i}.nc"))

        ds = xr.open_mfdataset(final_files, concat_dim="time")

        delayed_write = ds.to_netcdf(modis_nc,compute=False)
        with self.client:
            delayed_write.compute()


    def __read_MODIS(self):
        modis_nc = os.path.join((self.config["SATELLITES"]["MODISNC"]), "modis.nc")
        if not os.path.exists(modis_nc):
            self.__read_MODIS_text(modis_nc)

        ds_loaded = self.__read_MODIS_NC(modis_nc)

        return ds_loaded
    
    def __get_MODIS_files(self):
        allfiles = sorted(glob.iglob(self.config["SATELLITES"]["MODIS"] + "*/*/*", recursive=True))

        files = []
        for file in tqdm(allfiles, desc="read"):
            start, end = (True, True)
            f_time = pd.Timestamp.strptime(os.path.split(file)[-1], "%Y%m%d%H_obsnew.txt")
            if self.start_date and (f_time < pd.Timestamp(self.start_date)):
                start = False
            if self.end_date and (f_time > pd.Timestamp(self.end_date).replace(hour=23, minute=59)):
                end = False
            if start and end:
                files += [file]
        if not files:
            raise FileNotFoundError("No files to read.")
        return files

    @staticmethod
    def __get_MODIS_time():
        return lambda x: pd.Timestamp.strptime(x, "%Y%m%d%H%M")


    def __create_MODIS_grid(self, ds):

        lats = np.arange(-89.5, 90, 1)
        lons = np.arange(-179.5, 180, 1)
        time = ds.time.values
        data = np.zeros((len(time), len(lats), len(lons)))
        data[:] = np.nan

        coord_dict = {
            "time": time,
            "lat": lats,
            "lon": lons
        }

        empty_data_array = xr.DataArray(data, dims=("time", "lat", "lon"), coords=coord_dict).chunk(100)

        ds_dict = {}
        for key in ds.keys():
            ds_dict[key] = empty_data_array.copy()

        ds_empty = xr.Dataset(ds_dict)
        logging.info("Created empty dataset")

        with self.client:
            ds_regridded = ds.combine_first(ds_empty).compute()

        return ds_regridded



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
    logging.basicConfig(level=logging.WARNING)
    Sr = SatelliteReader()
    viirs = Sr.read_data("VIIRS")
    misr = Sr.read_data("MISR")
    modis = Sr.read_data("MODIS")