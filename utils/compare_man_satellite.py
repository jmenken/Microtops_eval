import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from utils.datahandler.man_reader import main as man_reader
from utils.datahandler.Satellite import SatelliteReader
import utils.plotting as plot

log = logging.getLogger(__name__)


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

    sat_data = sat_data[aod_sat[sat_name]].sortby(["time", "lat", "lon"])
    if sat_name == "MISR":
        sat_data = sat_data.sel(dict(Optical_Depth_Range="all"))

    starttime = sat_data.time.values[0] - np.timedelta64(time_window, "D")
    endtime = sat_data.time.values[-1] + np.timedelta64(time_window, "D")
    man_data = man_data.sel(dim_0=slice(starttime, endtime))
    sat_data = sat_data.sel(time=slice(starttime, endtime))

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

        try:
            sat_aod = sat_data.sel(
                dict(lat=slice(lat_min, lat_max),
                     lon=slice(lon_min, lon_max),
                     time=slice(time_min, time_max)))
        except ValueError:
            log.warning(f"Skipped value {i}")
            continue

        sat_aod_stacked = sat_aod.stack(dim_0=("time", "lat", "lon"))
        sat_aod_stacked = sat_aod_stacked[sat_aod_stacked.notnull()]

        if len(sat_aod_stacked.values) == 0:
            log.warning(f"Only NaN values found. Skipped value {i}")
            continue

        sat_aod_unstacked = sat_aod_stacked.unstack()
        sat_aod_sorted = sat_aod_unstacked.sortby(["time", "lat", "lon"])
        sat_aod_point = sat_aod_sorted.sel(dict(time=time, lat=lat, lon=lon),
                                              method="nearest")
        man_aod_point = man_data.isel(dim_0=i).AOD_550nm.unstack()
        data[(time, float(lat), float(lon))] = [float(man_aod_point.values),
                                                float(sat_aod_point.values)]

    df = pd.DataFrame.from_dict(data, orient="index", columns=["MAN", sat_name])
    index = pd.MultiIndex.from_tuples(df.index, names=['time', 'lat', 'lon'])
    df = df.set_index(index)

    return df


# def check_nan(data_array):
#     if np.any(np.isnan(data_array.values)):
#         log.debug("There are still nans")
#         return True
#     else:
#         return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pickle = True
    s = "MISR"
    if pickle:
        man_sat = pd.read_pickle("./MAN_{}.pd".format(s))
    else:
        Sr = SatelliteReader()
        man_data = man_reader()

        print("Reading satellite data of {}.".format(s))
        sat_data = Sr.read_data(s)

        print("Joining MAN and satellite data.")
        man_sat = join_man_sat(man_data, sat_data, s)
        man_sat.to_pickle("./MAN_{}.pd".format(s))

    # relative distance:
    man_sat_rel = compare_man_sat(man_sat, s, relative_error).dropna()
    lats = man_sat_rel.reset_index()["lat"].values
    lons = man_sat_rel.reset_index()["lon"].values
    plot.plot_on_map(man_sat_rel.values, lats, lons,
                     "{}_dists_relative.pdf".format(s),
                     sizefunc=plot.size_func_1, resizefunc=plot.resize_func_1,
                     cmap="PiYG", vmin=-200, vmax=200, extend="both",
                     value_name="Relative distance [%]")

    # absolute distance:
    man_sat_dist = compare_man_sat(man_sat, s, distance).dropna()
    lats = man_sat_dist.reset_index()["lat"].values
    lons = man_sat_dist.reset_index()["lon"].values
    plot.plot_on_map(man_sat_dist.values, lats, lons,
                     "{}_dists_absolute.pdf".format(s),
                     sizefunc=plot.size_func_2, resizefunc=plot.resize_func_2,
                     cmap="PiYG", vmin=-0.3, vmax=0.3, extend="both",
                     value_name="Absolute distance")
