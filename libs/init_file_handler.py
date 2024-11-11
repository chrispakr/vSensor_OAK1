import os
from loguru import logger
import configparser as cp

class VsInitConfigObject:
    def __init__(self):
        self.serial:str = ""
        self.capture_width:int = 1012
        self.capture_height:int = 760
        self.lens_position:int = 130
        self.center_position:int = 420
        self.raw_image_width: int = 700
        self.raw_image_height: int = 300
        self.raw_image_height_offset: int = 0
        self.raw_image_width_offset: int = 0

class GeneralConfigObject:
    def __init__(self):
        self.fps = 60
        self.exposure_time_positive = 5000
        self.exposure_time_negative = 1200


class InitFileHandler:
    def __init__(self, init_config_file=None):
        self.init_file = init_config_file
        self._config = cp.ConfigParser()
        self.vs_front = VsInitConfigObject()
        self.vs_rear = VsInitConfigObject()
        self.general = GeneralConfigObject()
        if not os.path.exists(self.init_file):
            logger.info("no init-config found... - create new.")
            self._config = cp.ConfigParser()
            self._config.add_section('vs_front')
            self._config.add_section('vs_rear')
            self._config.add_section('general')
            self._config['vs_front'] = self.vs_front.__dict__
            self._config['vs_rear'] = self.vs_rear.__dict__
            self._config['general'] = self.general.__dict__
            self.save_config()
        else:
            self._config.read(self.init_file)
            self.vs_front.serial = self._config.get(section='vs_front', option='serial')
            self.vs_front.capture_width = self._config.getint(section='vs_front', option='capture_width')
            self.vs_front.capture_height = self._config.getint(section='vs_front', option='capture_height')
            self.vs_front.lens_position = self._config.getint(section='vs_front', option='lens_position')
            self.vs_front.center_position = self._config.getint(section='vs_front', option='center_position')
            self.vs_front.raw_image_width = self._config.getint(section='vs_front', option='raw_image_width')
            self.vs_front.raw_image_height = self._config.getint(section='vs_front', option='raw_image_height')
            self.vs_front.raw_image_height_offset = self._config.getint(section='vs_front', option='raw_image_height_offset')
            self.vs_front.raw_image_width_offset = self._config.getint(section='vs_front', option='raw_image_width_offset')
            self.vs_rear.serial = self._config.get(section='vs_rear', option='serial')
            self.vs_rear.capture_width = self._config.getint(section='vs_rear', option='capture_width')
            self.vs_rear.capture_height = self._config.getint(section='vs_rear', option='capture_height')
            self.vs_rear.lens_position = self._config.getint(section='vs_rear', option='lens_position')
            self.vs_rear.center_position = self._config.getint(section='vs_rear', option='center_position')
            self.vs_rear.raw_image_width = self._config.getint(section='vs_rear', option='raw_image_width')
            self.vs_rear.raw_image_height = self._config.getint(section='vs_rear', option='raw_image_height')
            self.vs_rear.raw_image_height_offset = self._config.getint(section='vs_rear', option='raw_image_height_offset')
            self.vs_rear.raw_image_width_offset = self._config.getint(section='vs_rear', option='raw_image_width_offset')
            self.general.fps = self._config.getint(section='general', option='fps')
            self.general.exposure_time_positive = self._config.getint(section='general', option='exposure_time_positive')
            self.general.exposure_time_negative = self._config.getint(section='general', option='exposure_time_negative')

    def save_config(self):
        self._config['vs_front'] = self.vs_front.__dict__
        self._config['vs_rear'] = self.vs_rear.__dict__
        self._config['general'] = self.general.__dict__
        logger.info(f"write init-config file to {self.init_file}")
        with open(self.init_file, 'w') as configfile:
            self._config.write(configfile)

    def swap_cameras(self):
        vs_front_config = self.vs_front.__dict__
        vs_rear_config = self.vs_rear.__dict__
        self.vs_front.__dict__ = vs_rear_config
        self.vs_rear.__dict__ = vs_front_config
        self.save_config()



