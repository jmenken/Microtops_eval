import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
import cartopy.crs as ccrs
import cartopy
from datahandler.man_reader import main as man_reader
from datahandler.Satellite import SatelliteReader
import numpy as np
import logging

def plot_man_data():
    ds_man = man_reader()
    ds_AOD = ds_man.AOD_550nm.sortby(ds_man.AOD_550nm)

    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)

    lons = ds_AOD.Longitude.values
    lats = ds_AOD.Latitude.values
    im = ax.scatter(lons, lats, c=ds_AOD.values, s=0.1, cmap="inferno_r", vmax=2)
    plt.colorbar(im, extend="max")
    plt.tight_layout()
    plt.savefig("man_test.pdf")
    plt.show()


def compare_man_satellite(man_data, satellite_data, compare_func):
    pass


def check_nan(data_array):
    if np.any(np.isnan(data_array.values)):
        logging.debug("There are still nans")
        return True
    else:
        return False

if __name__ == "__main__":
    man_data = man_reader()
    Sr = SatelliteReader()
    modis_data = Sr.read_data("MODIS")

    geo_window = 1.5 # in degrees
    time_window = 2 # in days

    for i, entry in enumerate(man_data.dim_0[:50]):
        time = man_data.Datetime.isel(dim_0=i).values
        lat = man_data.Latitude.isel(dim_0=i).values
        lon = man_data.Longitude.isel(dim_0=i).values

        lon_max = lon + geo_window
        lon_min = lon - geo_window
        lat_max = lat + geo_window
        lat_min = lat - geo_window
        time_max = time + np.timedelta64(time_window, "D")
        time_min = time - np.timedelta64(time_window, "D")

        satellite_AOD = modis_data.AOD.sel(time=time, method="nearest").sel(dict(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)))
        still_nans = check_nan(satellite_AOD)
        nan_counter = 0
        satellite_AOD_no_nan = satellite_AOD.dropna("lat", how="all").dropna("lon", how="all")
        while still_nans:
            nan_counter += 1
            satellite_AOD_no_nan = satellite_AOD_no_nan.dropna("lat", how="all").dropna("lon", how="all")
            still_nans = check_nan(satellite_AOD_no_nan)
            if nan_counter > 10:
                raise RuntimeError("Nans were not removed after 10 loops.")




