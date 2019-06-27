import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from utils.datahandler.man_reader import main as man_reader

log = logging.getLogger(__name__)


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


def size_func_1(x):
    return np.power(np.abs(x / 100) + 1, 2)


def resize_func_1(x):
    return (np.sqrt(x) - 1) * 100


def size_func_2(x):
    return np.power(np.abs(x) + 1.5, 3)


def resize_func_2(x):
    return np.cbrt(x) - 1.5


def plot_on_map(values, lats, lons, plot_name, sizefunc=None, resizefunc=None,
                extend="neither", cmap="inferno_r", vmin=None, vmax=None,
                value_name=""):
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

    im = ax.scatter(lons, lats, c=values, s=sizefunc(values), cmap=cmap,
                    vmin=vmin, vmax=vmax)

    plt.colorbar(im, extend=extend, label=value_name)
    handles, labels = im.legend_elements(prop="sizes", alpha=0.6, num=6,
                                         func=resizefunc)
    max_point = handles[-1].get_markersize()
    ax.legend(handles, labels, frameon=False,
              labelspacing=0.069*max_point, loc='center left',
              bbox_to_anchor=(1.16 + (max_point*0.0019), 0.5))

    plt.savefig(plot_name)
    plt.show()
    log.info("{} datapoints plotted".format(values.shape))
