import xarray as xr
import abc
from MainReader import MainReader
import glob
from datetime import datetime as dt

ALLOWED_SATELLITES = ["MODIS", "VIIRS", "MISR"]

class SatelliteReader(MainReader):
    def __init__(self, start_date=None, end_date=None):
        super().__init__()
        self.start_date = None
        self.end_date = None

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

    @abc.abstractmethod
    def __read_MODIS(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __read_VIIRS(self):
        files = sorted(glob.glob(self.config["SATELLITES"]["VIIRS"] + "*"))
        ds = xr.open_mfdataset(files, concat_dim="time",preprocess=self.__get_VIIRS_time)

    @staticmethod
    def __get_VIIRS_time(dataset):
        file_name = dataset.encoding['source']
        return dataset.assign(time=dt.strptime(file_name,"NOAA_EPS_AOD_%Y%m%d.nc"))



    @abc.abstractmethod
    def __read_MISR(self):
        files = sorted(glob.glob(self.config["SATELLITES"]["MISR"] + "*"))
        raise NotImplementedError


if __name__ == "__main__":
    Sr = SatelliteReader()
    ds = Sr.read_data("VIIRS")
