import os
import json
from dataclasses import dataclass, asdict, field
from typing import Optional
import traceback
import logging

@dataclass
class VsConfigParameter:
    """Vision system camera configuration parameters."""
    serial: str = ""
    capture_width: int = 1012
    capture_height: int = 760
    lens_position: int = 130
    center_offset: int = 0
    raw_image_crop_top: int = 0
    raw_image_width_offset: int = 0
    warp_factor: int = 55
    _raw_image_width: int = field(default=700, repr=False)
    _raw_image_height: int = field(default=400, repr=False)

    MIN_HEIGHT: int = 150
    MAX_HEIGHT: int = 400
    DEFAULT_HEIGHT: int = 250
    MIN_WIDTH: int = 300
    MAX_WIDTH: int = 800
    DEFAULT_WIDTH: int = 700

    @property
    def raw_image_height(self) -> int:
        return self._raw_image_height

    @raw_image_height.setter
    def raw_image_height(self, value: int) -> None:
        if not self.MIN_HEIGHT <= value <= self.MAX_HEIGHT:
            value = self.DEFAULT_HEIGHT
        self._raw_image_height = value

    @property
    def raw_image_width(self) -> int:
        return self._raw_image_width

    @raw_image_width.setter
    def raw_image_width(self, value: int) -> None:
        if not self.MIN_WIDTH <= value <= self.MAX_WIDTH:
            value = self.DEFAULT_WIDTH
        self._raw_image_width = value

@dataclass
class GeneralConfigObject:
    """General configuration parameters."""
    fps: int = 60
    exposure_time_positive: int = 5000
    exposure_time_negative: int = 1200

@dataclass
class ConfigData:
    """Complete configuration data structure."""
    vs_front: VsConfigParameter = field(default_factory=VsConfigParameter)
    vs_rear: VsConfigParameter = field(default_factory=VsConfigParameter)
    general: GeneralConfigObject = field(default_factory=GeneralConfigObject)

class ConfigFileHandler:
    """Handles reading and writing JSON configuration files for the vision system."""

    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger("base." + self.__class__.__name__)
        self.config_filename = config_file
        self.config = ConfigData()

        if not os.path.exists(self.config_filename):
            self.logger.info("No init-config found... - create new.")
            self.save_config()
        else:
            self._load_config()

    def _load_config(self) -> None:
        """Loads configuration from JSON file."""
        try:
            with open(self.config_filename, 'r') as f:
                data = json.load(f)

            # Load front camera config
            front_data = data.get('vs_front', {})
            self.config.vs_front = VsConfigParameter(**front_data)

            # Load rear camera config
            rear_data = data.get('vs_rear', {})
            self.config.vs_rear = VsConfigParameter(**rear_data)

            # Load general config
            general_data = data.get('general', {})
            self.config.general = GeneralConfigObject(**general_data)

            self._log_all_settings()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            # Create new config with defaults if loading fails
            traceback.print_exc()
            self.config = ConfigData()
            self.save_config()

    def _log_all_settings(self) -> None:
        """Logs all configuration settings."""
        self.logger.info("Vs-Rear Settings:")
        self._log_attributes(self.config.vs_rear)
        self.logger.info("Vs-Front Settings:")
        self._log_attributes(self.config.vs_front)
        self.logger.info("General Settings:")
        self._log_attributes(self.config.general)

    def _log_attributes(self, obj: object) -> None:
        """Logs all attributes of an object."""
        for name, value in asdict(obj).items():
            if not name.startswith('_') and not name.isupper():
                self.logger.info("{0:30}: {1}".format(name, value))

    def save_config(self) -> None:
        """Saves current configuration to JSON file."""
        config_dict = {
            'vs_front': asdict(self.config.vs_front),
            'vs_rear': asdict(self.config.vs_rear),
            'general': asdict(self.config.general)
        }

        self.logger.info(f"Writing config file to {self.config_filename}")
        try:
            with open(self.config_filename, 'w') as f:
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            traceback.print_exc()

    def swap_cameras(self) -> None:
        """Swaps configuration between front and rear cameras."""
        self.config.vs_front, self.config.vs_rear = self.config.vs_rear, self.config.vs_front
        self.save_config()

    @property
    def vs_front(self) -> VsConfigParameter:
        return self.config.vs_front

    @property
    def vs_rear(self) -> VsConfigParameter:
        return self.config.vs_rear

    @property
    def general(self) -> GeneralConfigObject:
        return self.config.general