from configparser import ExtendedInterpolation
import configparser
import os


class MainReader(object):
    def __init__(self):
        self.config = self.__read_config()

    def __read_config(self, file=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'PATH.ini'))):
        config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
        config.read(os.path.abspath(file))
        return config
