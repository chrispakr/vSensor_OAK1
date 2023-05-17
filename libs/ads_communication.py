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
        return self.plc.read_by_name("",pyads.PLCTYPE_BOOL, handle=self._stop_film_handler)

    @stop_film.setter
    def stop_film(self, value):
        self.plc.write_by_name("", value, pyads.PLCTYPE_BOOL, handle=self._stop_film_handler)

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

    def __init__(self, mqtt_broker_ip="192.168.0.5", external_logger=None, logger_enabled=True, client_type=None):
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

        self._stop_film_handler = self.plc.get_handle("MAIN.vs_stopFilm")