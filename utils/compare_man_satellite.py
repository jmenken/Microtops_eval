import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
import cartopy.crs as ccrs
import cartopy
from datahandler.man_reader import main as man_reader
from datahandler.Satellite import SatelliteReader
import numpy as np
import logging
import xarray as xr
from tqdm import tqdm

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

def distance(man, satellite):
    return man - satellite

def relative_error(man, satellite):
    return (man - satellite) / man * 100


def compare_man_modis(man_data, modis_data, compare_func):

    geo_window = 1.5  # in degrees
    time_window = 1  # in days
    man_data_index = np.where(man_data.Datetime.values == np.datetime64("2015-01-01"))
    man_data = man_data.isel(dim_0=slice(man_data_index[0][0], None))

    dists = []
    lats = []
    lons = []
    times = []

    for i, entry in enumerate(tqdm(man_data.dim_0)):
        time = man_data.Datetime.isel(dim_0=i).values
        lat = man_data.Latitude.isel(dim_0=i).values
        lon = man_data.Longitude.isel(dim_0=i).values

        lon_max = lon + geo_window
        lon_min = lon - geo_window
        lat_max = lat + geo_window
        lat_min = lat - geo_window
        time_max = time + np.timedelta64(time_window, "D")
        time_min = time - np.timedelta64(time_window, "D")

        try:
            satellite_AOD = modis_data.AOD.sel(
                dict(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max), time=slice(time_min, time_max)))
            satellite_AOD_stacked = satellite_AOD.stack(dim_0=("time", "lat", "lon"))
            satellite_AOD_stacked = satellite_AOD_stacked[satellite_AOD_stacked.notnull()]
            if len(satellite_AOD_stacked.values) == 0:
                continue

            satellite_AOD_point = satellite_AOD_stacked.unstack().sel(dict(time=time, lat=lat, lon=lon), method="nearest")
            man_AOD_point = man_data.isel(dim_0=i).AOD_550nm.unstack()
            dist = compare_func(man_AOD_point.values, satellite_AOD_point.values)
        except ValueError:
            logging.warning(f"Skipped value {i}")
            continue

        dists.append(dist)
        lats.append(lat)
        lons.append(lon)
        times.append(time)
    dists = np.asarray(dists)
    times = np.asarray(times)
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    return lats, lons, dists, times


def check_nan(data_array):
    if np.any(np.isnan(data_array.values)):
        logging.debug("There are still nans")
        return True
    else:
        return False

def plot_on_map(lats, lons, values, plot_name, sizes=None, extend="neither", cmap="inferno_r" ,vmin=None, vmax=None,
                value_name=""):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    if sizes is None:
        sizes = 1

    # ds = ds.isel(dim_0=slice(5000, None))
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cartopy.feature.LAND, facecolor="lightgrey")
    ax.add_feature(cartopy.feature.OCEAN, facecolor="grey")

    gl = ax.gridlines(draw_labels=True, linewidth=2, color='white', alpha=0.5, linestyle='--')
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    im = ax.scatter(lons, lats, c=values, s=sizes, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, extend=extend, label=value_name)
    # plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()


if __name__ == "__main__":
    Sr = SatelliteReader()
    modis_data = Sr.read_data("MODIS")

    man_data = man_reader()


    # relative distance:
    # lats, lons, dists, times = compare_man_modis(man_data, modis_data, relative_error)
    # plot_on_map(lats, lons, dists, "dists_relative.pdf", sizes=np.power(np.abs(dists/100)+1, 2), cmap="PiYG", vmin=-200, vmax=200,
    #             extend="both", value_name="Relative distance [%]")

    # absolute distance:
    lats, lons, dists, times = compare_man_modis(man_data, modis_data, distance)
    plot_on_map(lats, lons, dists, "dists_absolute.pdf", sizes=np.power(np.abs(dists)+1.5, 3), cmap="PiYG", vmin=-0.3, vmax=0.3,
                extend="both", value_name="Absolute distance")
