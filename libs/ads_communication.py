import threading
import time
from enum import Enum
import pyads
import logging
import sys
import platform


class AdsHandler(threading.Thread):
    _externalLogger = None
    connected = False

    def setLogger(self, logger):
        self._externalLogger = logger

    @property
    def stop_film(self):
        return self.plc.read_by_name("", pyads.PLCTYPE_BOOL, handle=self._ads_handler_stop_film)

    @stop_film.setter
    def stop_film(self, value):
        self.plc.write_by_name("", value, pyads.PLCTYPE_BOOL, handle=self._ads_handler_stop_film)

    @property
    def picture_in_position(self):
        return self.plc.read_by_name("", pyads.PLCTYPE_BOOL, handle=self._ads_handler_picture_in_position)

    @picture_in_position.setter
    def picture_in_position(self, value):
        self.plc.write_by_name("", value, pyads.PLCTYPE_BOOL, handle=self._ads_handler_picture_in_position)

    @property
    def vs_front_edge_position(self):
        return self.plc.read_by_name("", pyads.PLCTYPE_INT, handle=self._ads_handler_vs_front_edge_position)

    @vs_front_edge_position.setter
    def vs_front_edge_position(self, value):
        self.plc.write_by_name("", value, pyads.PLCTYPE_INT, handle=self._ads_handler_vs_front_edge_position)

    @property
    def vs_rear_edge_position(self):
        return self.plc.read_by_name("", pyads.PLCTYPE_INT, handle=self._ads_handler_vs_rear_edge_position)

    @vs_rear_edge_position.setter
    def vs_rear_edge_position(self, value):
        self.plc.write_by_name("", value, pyads.PLCTYPE_INT, handle=self._ads_handler_vs_rear_edge_position)

    @property
    def vs_front_edge_state(self):
        return self.plc.read_by_name("", pyads.PLCTYPE_INT, handle=self._ads_handler_vs_front_edge_state)

    @vs_front_edge_state.setter
    def vs_front_edge_state(self, value):
        self.plc.write_by_name("", value, pyads.PLCTYPE_INT, handle=self._ads_handler_vs_front_edge_state)

    @property
    def vs_rear_edge_state(self):
        return self.plc.read_by_name("", pyads.PLCTYPE_INT, handle=self._ads_handler_vs_rear_edge_state)

    @vs_rear_edge_state.setter
    def vs_rear_edge_state(self, value):
        self.plc.write_by_name("", value, pyads.PLCTYPE_INT, handle=self._ads_handler_vs_rear_edge_state)

    def enableLogger(self, enable):
        self.loggerEnabled = enable

    def _str2bool(self, value):
        value = str(value)
        return value.lower() in ("yes", "true", "t", "1")

    def _logInfoAds(self, message):
        if self.loggerEnabled:
            log_message = "[ADS-CL ]" + " - " + message
            if self._externalLogger is None:
                logging.info(log_message)
            else:
                self._externalLogger.info(log_message)

    def __init__(self, external_logger=None, logger_enabled=True):
        threading.Thread.__init__(self)

        if external_logger is None:
            logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                                handlers=[logging.FileHandler("mqttLog.log"), logging.StreamHandler(sys.stdout)])
        else:
            self._externalLogger = external_logger

        self.loggerEnabled = logger_enabled
        if platform.system() == "Linux":
            pyads.open_port()
            pyads.set_local_address("192.168.0.30.1.1")
            pyads.close_port()

            pyads.ads.add_route('5.71.85.220.1.1', "192.168.0.10")

            pyads.add_route_to_plc("192.168.0.30.1.1", "192.168.0.30", "192.168.0.10", "admin", "geodyn2905", route_name="pi")

        self.plc = pyads.Connection('5.71.85.220.1.1', pyads.PORT_TC3PLC1, "192.168.0.10")
        self.plc.open()
        self._logInfoAds("connected to PLC...")

        self._ads_handler_stop_film = self.plc.get_handle("AdsVars.vs_stopFilm")
        self._ads_handler_picture_in_position = self.plc.get_handle("AdsVars.vs_filmIsInPosition")
        self._ads_handler_vs_front_edge_position = self.plc.get_handle("AdsVars.vs_front_edge_position")
        self._ads_handler_vs_rear_edge_position = self.plc.get_handle("AdsVars.vs_rear_edge_position")
        self._ads_handler_vs_front_edge_state = self.plc.get_handle("AdsVars.vs_front_edge_state")
        self._ads_handler_vs_rear_edge_state = self.plc.get_handle("AdsVars.vs_rear_edge_state")