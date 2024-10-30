#!/usr/bin/env python3
import threading

import cv2
import platform
import depthai as dai
import time
import contextlib
import sys
from timeloop import Timeloop
import logging
from datetime import datetime, timedelta
import mqtt_handler.mqtt_communication_handler_v2 as mqtt_communication_handler
import libs.functions as ef
import visionSensor
from libs.functions import ValueHandler
from configparser import ConfigParser
from collections import deque
from ads_handler.ads_handler import AdsHandler
from visionSensor import VisionSensor
from socket_handler import SocketHandler
from loguru import logger
import numpy as np

enable_chart = False
showOutput = False

logger.add(sys.stderr, format="{time} {level} | {message}", filter="my_module", level="INFO")
logger.add("visionSensor.log")

logger.info("Start visionSensorM4")

init_config_file = "../config/init.ini"
settings_config_file = "../config/settings.ini"

if platform.system() == "Windows":
    init_config_file = "config/init.ini"
    settings_config_file = "config/settings.ini"

debug_vs = True

write_init_config = False
write_settings_config = False

vs_front_config_name = "vs_front"
vs_rear_config_name = "vs_rear"

camera_fps = 45

# vs_op_modes = VisionSensorOperationMode
# vs_operation_mode = 0

live_view_fps_divider = 2
vs_front_live_view_frame_nr = 0
vs_rear_live_view_frame_nr = 0

exposure_value_positive = 5000
exposure_value_negative = 1200

vs_front_send_mqtt_image =      False
vs_rear_send_mqtt_image =       False
vs_front_send_mqtt_values =     False
vs_rear_send_mqtt_values =      False

vs_front_last_edge_position = 0
vs_rear_last_edge_position = 0

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    # handlers=[logging.FileHandler("vSensor.log"), logging.StreamHandler(sys.stdout)]
    handlers=[logging.StreamHandler(sys.stdout)]
)

# def log_info_general(message):
#     log_message = "[general ]" + " - " + message
#     logging.info(log_message)

tl = Timeloop()


# Read Init-Configuration
logger.info(f"read init-config file..")
config_init = ConfigParser()
config_settings = ConfigParser()
config_init.read(init_config_file)
config_settings.read(settings_config_file)

def write_init_config_to_file():
    logger.info(f"write init-config file..")
    with open(init_config_file, 'w') as configfile:
        config_init.write(configfile)

def write_settings_config_to_file():
    logger.info("write sensor-config file..")
    with open(settings_config_file, 'w') as configfile:
        config_settings.write(configfile)

logger.info(f"DephtAi-Version : {dai.__version__}")

devices_found = dai.Device.getAllAvailableDevices()

if len(devices_found) != 2:
    logger.info(f"found only one sensor... - exit program")
    exit()

for device in devices_found:
    logger.info(f"found sensor ({device.getMxId()}) on state: {device.state}")

found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_front_config_name, "serial"))
found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_rear_config_name, "serial"))

if not found_front_sensor or not found_rear_sensor:
    if len(devices_found) == 2:
        config_init.set(vs_front_config_name, "serial", devices_found[0].getMxId())
        config_init.set(vs_rear_config_name, "serial", devices_found[1].getMxId())
        write_init_config_to_file()

    found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_front_config_name, "serial"))
    found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_rear_config_name, "serial"))

if not found_front_sensor:
    raise RuntimeError("FrontSensor not found!")

if not found_rear_sensor:
    raise RuntimeError("RearSensor not found!")


if config_init.has_option("settings", "fps"):
    camera_fps = config_init.getint("settings", "fps")
    logger.info(f"set camera FPS to: {camera_fps}")
else:
    logger.info("No config param_image found for 'FPS' - set standard value of 45")
    camera_fps = 45


plc_handler = AdsHandler(local_host_ip="192.168.0.30", route_name="vSensor")
plc_handler.connect_to_plc()

time.sleep(2)

plc_handler.stop_film = False

cam_vs_front = VisionSensor(
    device_info_front_sensor,
    is_front_sensor=True,
    vs_name=vs_front_config_name,
    fps=camera_fps,
)

cam_vs_rear = VisionSensor(
    device_info=device_info_rear_sensor,
    is_front_sensor=False,
    vs_name=vs_rear_config_name,
    fps=camera_fps,
)

if config_settings.has_option(vs_front_config_name, "tile_width"):
    cam_vs_front.proc_image_width = config_settings.getint(vs_front_config_name, "tile_width")
if config_settings.has_option(vs_rear_config_name, "tile_width"):
    cam_vs_rear.proc_image_width = config_settings.getint(vs_rear_config_name, "tile_width")

if config_settings.has_option(vs_front_config_name, "stop_position"):
    cam_vs_front.stop_position = config_settings.getint(vs_front_config_name, "stop_position")
if config_settings.has_option(vs_rear_config_name, "stop_position"):
    cam_vs_rear.stop_position = config_settings.getint(vs_rear_config_name, "stop_position")

if config_settings.has_option(vs_front_config_name, "stop_offset_compensation"):
    cam_vs_front.stop_offset_compensation = config_settings.getint(vs_front_config_name, "stop_offset_compensation")
if config_settings.has_option(vs_rear_config_name, "stop_offset_compensation"):
    cam_vs_rear.stop_offset_compensation = config_settings.getint(vs_rear_config_name, "stop_offset_compensation")

if config_settings.has_option(vs_front_config_name, "edge_detection_range"):
    cam_vs_front.edge_detection_range = config_settings.getint(vs_front_config_name, "edge_detection_range")
if config_settings.has_option(vs_rear_config_name, "edge_detection_range"):
    cam_vs_rear.edge_detection_range = config_settings.getint(vs_rear_config_name, "edge_detection_range")

if config_settings.has_option(vs_front_config_name, "center_position"):
    cam_vs_front.image_center_position = config_settings.getint(vs_front_config_name, "center_position")
if config_settings.has_option(vs_rear_config_name, "center_position"):
    cam_vs_rear.image_center_position = config_settings.getint(vs_rear_config_name, "center_position")

if config_init.has_option(vs_front_config_name, "lens_position"):
    cam_vs_front._lens_position = config_init.getint(vs_front_config_name, "lens_position")

if config_init.has_option(vs_rear_config_name, "lens_position"):
    cam_vs_rear._lens_position = config_init.getint(vs_rear_config_name, "lens_position")


logger.info(f"Connect to MQTT-Broker...")
mqtt = mqtt_communication_handler.MqttHandler(logger_enabled=True, client_type="vsController", client_id="vsController", external_logger=logging)

@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsController_heartbeat, 3)

tl.start()

socket_handler = SocketHandler(host="192.168.0.30", port=4001, logger=logging)


cam_vs_front._enabled_lcm = False
cam_vs_rear._enabled_lcm = False

main_loop_count = 0
@tl.job(interval=timedelta(seconds=1))
def main_loops():
    global main_loop_count
    while True:
        logger.info(f"MainLoopCount: {main_loop_count}")
        main_loop_count = 0
        time.sleep(1.0)


with (contextlib.ExitStack() as stack):
    while True:
        main_loop_count += 1
        startTime = time.time()

        #######################################################################################################
        # Checking ADS-Values for any changes...
        #######################################################################################################

        # if plc_handler.connected:

        # auto-exposure camera front
        if plc_handler.vs_ctrl.auto_exposure_cameras.value:
            logger.info(f"ADS: Start AutoExposure....")
            plc_handler.vs_ctrl.is_ready.value = False
            cam_vs_front.auto_exposure_camera()
            cam_vs_rear.auto_exposure_camera()
            plc_handler.vs_ctrl.auto_exposure_cameras.value = False

        # auto-exposure camera front finished
        if cam_vs_front.autoExposureFinished and cam_vs_rear.autoExposureFinished:
            mean_exposure = (cam_vs_front.exposure_time + cam_vs_rear.exposure_time) // 2
            logger.info(f"AutoExposure Finished")
            logger.info(f"Exposure cam_front: {cam_vs_front.exposure_time}us")
            logger.info(f"Exposure cam_rear : {cam_vs_rear.exposure_time}us")
            logger.info(f"Set mean-exposure-time to: {mean_exposure}us")
            plc_handler.vs_ctrl.exposure_time.value = mean_exposure
            cam_vs_front.autoExposureFinished = False
            cam_vs_rear.autoExposureFinished = False
            plc_handler.vs_ctrl.is_ready.value = True

        # autofocus camera front
        if plc_handler.vs_ctrl.auto_focus_cameras.value:
            logger.info("ADS: Start AutoFocus Cameras....")
            plc_handler.vs_ctrl.is_ready.value = False
            cam_vs_front.auto_focus_camera()
            cam_vs_rear.auto_focus_camera()
            plc_handler.vs_ctrl.auto_focus_cameras.value = False

        # autofocus camera front finished
        if cam_vs_front.autoFocusFinished and cam_vs_rear.autoFocusFinished:
            logger.info("AutoFocus Finished")
            logger.info(f"Lens-Position cam_front: {cam_vs_front.focus_position}")
            logger.info(f"Lens-Position cam_rear : {cam_vs_rear.focus_position}")
            plc_handler.vs_ctrl.vs_front_focus_position.value = cam_vs_front.focus_position
            plc_handler.vs_ctrl.vs_rear_focus_position.value = cam_vs_rear.focus_position
            cam_vs_front.autoFocusFinished = False
            cam_vs_rear.autoFocusFinished = False
            plc_handler.vs_ctrl.is_ready.value = True

        # set film-type positive/negative
        if plc_handler.vs_ctrl.film_type_is_negative.new_value_available():
            cam_vs_front.film_type_is_negative = plc_handler.vs_ctrl.film_type_is_negative.value
            cam_vs_rear.film_type_is_negative = plc_handler.vs_ctrl.film_type_is_negative.value
            if plc_handler.vs_ctrl.film_type_is_negative.value:
                cam_vs_front.set_exposure_value(exposure_value_negative)
                cam_vs_rear.set_exposure_value(exposure_value_negative)
            else:
                cam_vs_front.set_exposure_value(exposure_value_positive)
                cam_vs_rear.set_exposure_value(exposure_value_positive)

        # swap sensors
        if plc_handler.vs_ctrl.swap_cameras.value:
            vs_front_serial = config_init.get(vs_front_config_name, "serial")
            vs_rear_serial = config_init.get(vs_rear_config_name, "serial")
            config_init.set(vs_front_config_name, "serial", vs_rear_serial)
            config_init.set(vs_rear_config_name, "serial", vs_front_serial)
            write_init_config = True
            plc_handler.vs_ctrl.swap_cameras.value = False

        # Check enableLowContrastMode
        if plc_handler.vs_ctrl.enable_lcm_mode.new_value_available():
            cam_vs_front.enable_low_contrast_mode = plc_handler.vs_ctrl.enable_lcm_mode.value
            cam_vs_rear.enable_low_contrast_mode = plc_handler.vs_ctrl.enable_lcm_mode.value

        # lcm set lcm_slope
        if plc_handler.vs_ctrl.lcm_slope.new_value_available():
            cam_vs_front.lcm_slope = plc_handler.vs_ctrl.lcm_slope.value
            cam_vs_rear.lcm_slope = plc_handler.vs_ctrl.lcm_slope.value

        # lcm set lcm_contrast
        if plc_handler.vs_ctrl.lcm_contrast_offset.new_value_available():
            cam_vs_front.lcm_contrast_offset = plc_handler.vs_ctrl.lcm_contrast_offset.value
            cam_vs_rear.lcm_contrast_offset = plc_handler.vs_ctrl.lcm_contrast_offset.value


        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setOperationMode):
            vs_operation_mode = mqtt.getMqttValue(mqtt.sTopics_vsController.get_setOperationMode)

        # set procImageWidth on vsFront
        if plc_handler.vs_ctrl.vs_front_proc_image_width.new_value_available():
            cam_vs_front.proc_image_width = int(plc_handler.vs_ctrl.vs_front_proc_image_width.value)
            config_settings.set(
                section=vs_front_config_name,
                option="tile_width",
                value=str(cam_vs_front.proc_image_width)
            )
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set procImageWidth on vsRear
        if plc_handler.vs_ctrl.vs_rear_proc_image_width.new_value_available():
            cam_vs_rear.proc_image_width = int(plc_handler.vs_ctrl.vs_rear_proc_image_width.value)
            config_settings.set(
                section=vs_rear_config_name,
                option="tile_width",
                value=str(cam_vs_rear.proc_image_width)
            )
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        # set centerPosition on vsFront
        if plc_handler.vs_ctrl.vs_front_image_center_position.new_value_available():
            cam_vs_front.image_center_position = int(plc_handler.vs_ctrl.vs_front_image_center_position.value)
            config_settings.set(
                section=vs_front_config_name,
                option="center_position",
                value=str(cam_vs_front.image_center_position)
            )
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set centerPosition on vsRear
        if plc_handler.vs_ctrl.vs_rear_image_center_position.new_value_available():
            cam_vs_rear.image_center_position = int(plc_handler.vs_ctrl.vs_rear_image_center_position.value)
            config_settings.set(
                section=vs_rear_config_name,
                option="center_position",
                value=str(cam_vs_rear.image_center_position)
            )
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        ##################################################################################################################
        # IMAGE PROCESSING PART
        ##################################################################################################################

        if cam_vs_front.new_image_available:
            if plc_handler.vs_ctrl.enable_live_view.value:
                if vs_front_live_view_frame_nr == live_view_fps_divider:
                    socket_handler.send_image(
                        image_rear=cam_vs_rear.numpy_image_array,
                        image_front=cam_vs_front.numpy_image_array
                    )
                    vs_front_live_view_frame_nr = 0
                vs_front_live_view_frame_nr += 1
            if abs(cam_vs_front.result.edge_position - vs_front_last_edge_position) > 1:
                plc_handler.vs_ctrl.vs_front_edge_position.value = cam_vs_front.result.edge_position
                vs_front_last_edge_position = cam_vs_front.result.edge_position
            plc_handler.vs_ctrl.vs_front_fps.value = cam_vs_front.fps

        if cam_vs_rear.new_image_available:
            if abs(cam_vs_rear.result.edge_position - vs_rear_last_edge_position) > 1:
                plc_handler.vs_ctrl.vs_front_edge_position.value = cam_vs_front.result.edge_position
                vs_rear_last_edge_position = cam_vs_rear.result.edge_position
            plc_handler.vs_ctrl.vs_rear_edge_position.value = cam_vs_rear.result.edge_position
            plc_handler.vs_ctrl.vs_rear_fps.value = cam_vs_rear.fps

        if plc_handler.vs_ctrl.capture_image.value:
            logger.debug(f"Send image_data data....")
            socket_handler.send_image(
                image_rear=cam_vs_rear.numpy_image_array,
                image_front=cam_vs_front.numpy_image_array
            )
            vs_front_send_mqtt_image = True
            vs_rear_send_mqtt_image = True
            plc_handler.vs_ctrl.capture_image.value = False

        if vs_front_send_mqtt_image:
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageData, cam_vs_front.get_base64_image())
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageWidth, cam_vs_front.result.image_width)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageHeight, cam_vs_front.result.image_height)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_image_statistics, str(cam_vs_front.calc_statistics()))
            vs_front_send_mqtt_image = False

        if vs_rear_send_mqtt_image:
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageData, cam_vs_rear.get_base64_image())
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageWidth, cam_vs_rear.result.image_width)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageHeight, cam_vs_rear.result.image_height)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_image_statistics, str(cam_vs_rear.calc_statistics()))
            vs_rear_send_mqtt_image = False

        if write_init_config:
            write_init_config_to_file()
            write_init_config = False

        if write_settings_config:
            write_settings_config_to_file()
            write_settings_config = False

        time.sleep(0.0001)


