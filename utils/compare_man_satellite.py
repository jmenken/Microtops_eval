import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from utils.datahandler.man_reader import main as man_reader
from utils.datahandler.Satellite import SatelliteReader


def plot_man_data():
    ds_man = man_reader()
    ds_aod = ds_man.AOD_550nm.sortby(ds_man.AOD_550nm)

    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)

    lons = ds_aod.Longitude.values
    lats = ds_aod.Latitude.values
    im = ax.scatter(lons, lats, c=ds_aod.values, s=0.1, cmap="inferno_r", vmax=2)
    plt.colorbar(im, extend="max")
    plt.tight_layout()
    plt.savefig("man_test.pdf")
    plt.show()


def distance(man, sat):
    return man - sat


def relative_error(man, sat):
    return (man - sat) / man * 100


def compare_man_sat(man_sat, sat_name, compare_func):
    dist = compare_func(man_sat["MAN"], man_sat[sat_name])
    return dist


def join_man_sat(man_data, sat_data, sat_name):
    aod_sat = {"MODIS": "AOD", "VIIRS": "AOD550", "MISR": "Aerosol_Optical_Depth"}
    geo_window = 1.5  # in degrees
    time_window = 1  # in days
    sat_data = sat_data.sortby(["time", "lat", "lon"])
    starttime = sat_data.time.values[0] - np.timedelta64(time_window, "D")
    endtime = sat_data.time.values[-1] + np.timedelta64(time_window, "D")
    man_data = man_data.sel(dim_0=slice(starttime, endtime))

    data = {}
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

        sel_dict = dict(lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max),
                        time=slice(time_min, time_max))
        if sat_name == "MISR":
            sel_dict["Optical_Depth_Range"] = "all"

        try:
            sat_aod = sat_data[aod_sat[sat_name]].sel(sel_dict)
        except ValueError:
            logging.warning(f"Skipped value {i}")
            continue

        sat_aod_stacked = sat_aod.stack(dim_0=("time", "lat", "lon"))
        sat_aod_stacked = sat_aod_stacked[sat_aod_stacked.notnull()]

        if len(sat_aod_stacked.values) == 0:
            logging.warning(f"Only NaN values found. Skipped value {i}")
            continue

        sat_aod_unstacked = sat_aod_stacked.unstack().sortby(["time", "lat", "lon"])
        sat_aod_point = sat_aod_unstacked.sel(dict(time=time, lat=lat, lon=lon), method="nearest")
        man_aod_point = man_data.isel(dim_0=i).AOD_550nm.unstack()
        data[(time, float(lat), float(lon))] = [float(man_aod_point.values),
                                                float(sat_aod_point.values)]

    man_sat = pd.DataFrame.from_dict(data, orient="index",
                                     columns=["MAN", sat_name])
    index = pd.MultiIndex.from_tuples(man_sat.index,
                                      names=['time', 'lat', 'lon'])
    man_sat = man_sat.set_index(index)

    return man_sat


# def check_nan(data_array):
#     if np.any(np.isnan(data_array.values)):
#         logging.debug("There are still nans")
#         return True
#     else:
#         return False

def plot_on_map(data, plot_name, sizes=1, extend="neither",
                cmap="inferno_r", vmin=None, vmax=None, value_name=""):
    lats = data.reset_index()["lat"].values
    lons = data.reset_index()["lon"].values
    values = data.values
    logging.info("{} datapoints plotted".format(data.dropna().values.shape))

    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cartopy.feature.LAND, facecolor="lightgrey")
    ax.add_feature(cartopy.feature.OCEAN, facecolor="grey")

    gl = ax.gridlines(draw_labels=True, linewidth=2, color='white', alpha=0.5,
                      linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    im = ax.scatter(lons, lats, c=values, s=sizes, cmap=cmap, vmin=vmin,
                    vmax=vmax)
    plt.colorbar(im, extend=extend, label=value_name)
    # plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    s = "MISR"

    Sr = SatelliteReader()
    man_data = man_reader()

    logging.info("Reading satellite data of {}.".format(s))
    sat_data = Sr.read_data(s)

    logging.info("Joining MAN and satellite data.")
    man_sat = join_man_sat(man_data, sat_data, s)

    # relative distance:
    man_sat_rel = compare_man_sat(man_sat, s, relative_error)
    plot_on_map(man_sat_rel, "{}_dists_relative.pdf".format(s),
                sizes=np.power(np.abs(man_sat_rel.values/100)+1, 2),
                cmap="PiYG", vmin=-200, vmax=200, extend="both",
                value_name="Relative distance [%]")

    # absolute distance:
    man_sat_dist = compare_man_sat(man_sat, s, distance)
    plot_on_map(man_sat_dist, "{}_dists_absolute.pdf".format(s),
                sizes=np.power(np.abs(man_sat_dist.values)+1.5, 3),
                cmap="PiYG", vmin=-0.3, vmax=0.3, extend="both",
                value_name="Absolute distance")
