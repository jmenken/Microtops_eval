from configparser import ExtendedInterpolation
import configparser
import os


class MainReader(object):
    def __init__(self):
        self.config = self.__read_config()

    def __read_config(self, file="../../PATH.ini"):
        config_reader = configparser.ConfigParser(interpolation=ExtendedInterpolation())
        config = config_reader.read(os.path.abspath(file))
        return config
