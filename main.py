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
from libs.functions import ValueHandler
from configparser import ConfigParser
from collections import deque
from ads_handler.ads_handler import AdsHandler
from visionSensor import VisionSensor, VisionSensorOperationMode
from socket_handler import SocketHandler
import numpy as np

enable_chart = False
showOutput = False

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

vs_op_modes = VisionSensorOperationMode
vs_operation_mode = 0

live_view_fps_divider = 2
vs_front_live_view_frame_nr = 0
vs_rear_live_view_frame_nr = 0

exposure_value_positive = 5000
exposure_value_negative = 1200

vs_front_send_mqtt_image =      False
vs_rear_send_mqtt_image =       False
vs_front_send_mqtt_values =     False
vs_rear_send_mqtt_values =      False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    # handlers=[logging.FileHandler("vSensor.log"), logging.StreamHandler(sys.stdout)]
    handlers=[logging.StreamHandler(sys.stdout)]
)

def log_info_general(message):
    log_message = "[general ]" + " - " + message
    logging.info(log_message)

tl = Timeloop()

def compute_stats(image):
    if image is not None:
        vs_maximum_dn = 256  # for image depth of byte
        clipping_percent = 0.05  # in percent for clipping the histogram with 0.025% from left and 0.025% from right

        # computing histogram
        hist = cv2.calcHist([image], [0], None, [vs_maximum_dn], [0, vs_maximum_dn])
        hist = hist.flatten()

        # Clipping the histogram by CLIPPING_PERCENT/2 % from bottom and top
        cutoff = image.shape[0] * image.shape[1] * clipping_percent / 2

        image_min, image_max, _, _ = cv2.minMaxLoc(image)
        clip_min = image_min  # starting value for clipMin
        clip_max = image_max  # starting value for clipMax

        accumulate_right = 0
        accumulate_left = 0
        clip_left_found = False
        clip_right_found = False
        for i in range(hist.size):
            if not clip_right_found:
                accumulate_right += hist[hist.size - 1 - i]
                if accumulate_right < cutoff:
                    clip_max = hist.size - 2 - i
                else:
                    clip_right_found = True

            if not clip_left_found:
                accumulate_left += hist[i]
                if accumulate_left < cutoff:
                    clip_min = i
                else:
                    clip_left_found = True

            if clip_left_found and clip_right_found:
                break

        # computing the mean and standard deviation. Note that the returned values are two-dimensional
        mean, std_dev = cv2.meanStdDev(image)

        # flatten mean and std to obtain a vector and obtain the single value in it.
        return (image_min, image_max, clip_min, clip_max, round(mean.flatten()[0], 3), round(std_dev.flatten()[0], 3))
    else:
        return None

# Read Init-Configuration
log_info_general("read init-config file..")
config_init = ConfigParser()
config_settings = ConfigParser()
config_init.read(init_config_file)
config_settings.read(settings_config_file)

def write_init_config_to_file():
    log_info_general("write init-config file..")
    with open(init_config_file, 'w') as configfile:
        config_init.write(configfile)

def write_settings_config_to_file():
    log_info_general("write sensor-config file..")
    with open(settings_config_file, 'w') as configfile:
        config_settings.write(configfile)

log_info_general("DephtAi-Version : {}".format(dai.__version__))

devices_found = dai.Device.getAllAvailableDevices()

if len(devices_found) != 2:
    log_info_general("found only one sensor... - exit program")
    exit()

for device in devices_found:
    log_info_general("found sensor ({}) on state: {}".format(device.getMxId(), device.state))

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

for section in config_init.sections():
    print('Section:', section)
    # iterate over each option in the section
    for option in config_init.options(section):
        print('Option:', option)
        print('Value:', config_init.get(section, option))
    print()


if config_init.has_option("general", "fps"):
    camera_fps = config_init.getint("general", "fps")
else:
    log_info_general("No config parameter found for 'FPS' - set standard value of 45")
    camera_fps = 45

plc_handler = AdsHandler(local_host_ip="192.168.0.30", route_name="vSensor")

plc_handler.stop_film = False

cam_vs_front = VisionSensor(
    device_info_front_sensor,
    is_front_sensor=True,
    vs_name=vs_front_config_name,
    fps=camera_fps,
    logger=logging,
)

cam_vs_rear = VisionSensor(
    device_info=device_info_rear_sensor,
    is_front_sensor=False,
    vs_name=vs_rear_config_name,
    fps=camera_fps,
    logger=logging,
)

if config_settings.has_option(vs_front_config_name, "proc_image_width"):
    cam_vs_front.proc_image_width = config_settings.getint(vs_front_config_name, "proc_image_width")
if config_settings.has_option(vs_rear_config_name, "proc_image_width"):
    cam_vs_rear.proc_image_width = config_settings.getint(vs_rear_config_name, "proc_image_width")

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


log_info_general("Connect to MQTT-Broker...")
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
        log_info_general(f"MainLoopCount: {main_loop_count}")
        main_loop_count = 0
        time.sleep(1.0)


with (contextlib.ExitStack() as stack):
    while True:
        main_loop_count += 1
        startTime = time.time()

        #######################################################################################################
        # Checking ADS-Values for any changes...
        #######################################################################################################

        # auto-exposure camera front
        if plc_handler.vs_ctrl.auto_exposure_cameras.value:
            log_info_general("ADS: Start AutoExposure....")
            plc_handler.vs_ctrl.is_ready.value = False
            cam_vs_front.auto_exposure_camera()
            cam_vs_rear.auto_exposure_camera()
            plc_handler.vs_ctrl.auto_exposure_cameras.value = False

        # auto-exposure camera front finished
        if cam_vs_front.autoExposureFinished and cam_vs_rear.autoExposureFinished:
            mean_exposure = (cam_vs_front.exposure_time + cam_vs_rear.exposure_time) // 2
            log_info_general("AutoExposure Finished")
            log_info_general(f"Exposure cam_front: {cam_vs_front.exposure_time}us")
            log_info_general(f"Exposure cam_rear : {cam_vs_rear.exposure_time}us")
            log_info_general(f"Set mean-exposure-time to: {mean_exposure}us")
            plc_handler.vs_ctrl.exposure_time.value = mean_exposure
            cam_vs_front.autoExposureFinished = False
            cam_vs_rear.autoExposureFinished = False
            plc_handler.vs_ctrl.is_ready.value = True

        # autofocus camera front
        if plc_handler.vs_ctrl.auto_focus_cameras.value:
            log_info_general("ADS: Start AutoFocus Cameras....")
            plc_handler.vs_ctrl.is_ready.value = False
            cam_vs_front.auto_focus_camera()
            cam_vs_rear.auto_focus_camera()
            plc_handler.vs_ctrl.auto_focus_cameras.value = False

        # autofocus camera front finished
        if cam_vs_front.autoFocusFinished and cam_vs_rear.autoFocusFinished:
            log_info_general("AutoFocus Finished")
            log_info_general(f"Lens-Position cam_front: {cam_vs_front.focus_position}")
            log_info_general(f"Lens-Position cam_rear : {cam_vs_rear.focus_position}")
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
            log_info_general(f"ADS: set enableLowContrastMode to: {plc_handler.vs_ctrl.enable_lcm_mode.value}")
            cam_vs_front.enable_low_contrast_mode = plc_handler.vs_ctrl.enable_lcm_mode.value
            cam_vs_rear.enable_low_contrast_mode = plc_handler.vs_ctrl.enable_lcm_mode.value

        # lcm set lcm_slope
        if plc_handler.vs_ctrl.lcm_slope.new_value_available():
            log_info_general(f"ADS: set lcm_slope to: {plc_handler.vs_ctrl.lcm_slope.value}")
            cam_vs_front.lcm_slope = plc_handler.vs_ctrl.lcm_slope.value
            cam_vs_rear.lcm_slope = plc_handler.vs_ctrl.lcm_slope.value

        # lcm set lcm_contrast
        if plc_handler.vs_ctrl.lcm_contrast_offset.new_value_available():
            log_info_general(f"ADS: set lcm_contrast_offset to: {plc_handler.vs_ctrl.lcm_contrast_offset.value}")
            cam_vs_front.lcm_contrast_offset = plc_handler.vs_ctrl.lcm_contrast_offset.value
            cam_vs_rear.lcm_contrast_offset = plc_handler.vs_ctrl.lcm_contrast_offset.value


        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setOperationMode):
            vs_operation_mode = mqtt.getMqttValue(mqtt.sTopics_vsController.get_setOperationMode)
            log_info_general("ADS: change OperationMode to: {}".format(str(vs_operation_mode)))
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_getOperationMode, vs_operation_mode)

        # Capture Image
        if plc_handler.vs_ctrl.capture_image.value:
            vs_front_send_mqtt_image = True
            vs_front_send_mqtt_values = True
            plc_handler.vs_ctrl.capture_image.value = False

        # set procImageWidth on vsFront
        if plc_handler.vs_ctrl.vs_front_proc_image_width.new_value_available():
            cam_vs_front.proc_image_width = int(plc_handler.vs_ctrl.vs_front_proc_image_width.value)
            config_settings.set(vs_front_config_name, "proc_image_width", str(cam_vs_front.proc_image_width))
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set procImageWidth on vsRear
        if plc_handler.vs_ctrl.vs_rear_proc_image_width.new_value_available():
            cam_vs_rear.proc_image_width = int(plc_handler.vs_ctrl.vs_rear_proc_image_width.value)
            config_settings.set(vs_rear_config_name, "proc_image_width", str(cam_vs_rear.proc_image_width))
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        # set centerPosition on vsFront
        if plc_handler.vs_ctrl.vs_front_image_center_position.new_value_available():
            cam_vs_front.image_center_position = int(plc_handler.vs_ctrl.vs_front_image_center_position.value)
            config_settings.set(vs_front_config_name, "center_position", str(cam_vs_front.image_center_position))
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set centerPosition on vsRear
        if plc_handler.vs_ctrl.vs_rear_image_center_position.new_value_available():
            cam_vs_rear.image_center_position = int(plc_handler.vs_ctrl.vs_rear_image_center_position.value)
            config_settings.set(vs_rear_config_name, "center_position", str(cam_vs_rear.image_center_position))
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
            plc_handler.vs_ctrl.vs_front_edge_position.value = cam_vs_front.edge_position
            plc_handler.vs_ctrl.vs_front_fps.value = cam_vs_front.fps

        if cam_vs_rear.new_image_available:
            plc_handler.vs_ctrl.vs_rear_edge_position.value = cam_vs_rear.edge_position
            plc_handler.vs_ctrl.vs_rear_fps.value = cam_vs_rear.fps


        # if vs_rear_edge_state.new_value_available:
        #     plc_handler.vs_rear_edge_state = vs_rear_edge_state.value
        #     if vs_rear_edge_state.value == 0:
        #         mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgeDetected, value=0)
        #         mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_pictureIsInPosition, value=0)
        #         log_info_general("set ads_stop_film to false")
        #         plc_handler.stop_film = False
        #         vs_rear_last_mqtt_position_value = 0
        #     if vs_rear_edge_state.value == 1:
        #         mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgeDetected, value=1)
        #         mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_pictureIsInPosition, value=0)
        #     if vs_rear_edge_state.value == 2:
        #         mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgeDetected, value=1)
        #         mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_pictureIsInPosition, value=1)
        #         # log_info_general("set ads_stop_film to false")
        #         # plc_handler.stop_film = True


        # if vs_front_send_mqtt_image:
        #     if not vs_front_enable_live_view.value:
        #         log_info_general("send vsFront image over MQTT")
        #     if not showOutput:
        #         cam_vs_front.create_image_info()
        #     cam_vs_front.create_image_info_jpg()
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageData, cam_vs_front.image_info_base64)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageWidth, cam_vs_front.img_width)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageHeight, cam_vs_front.img_height)
        #     vs_front_send_mqtt_image = False
        #
        # if vs_rear_send_mqtt_image:
        #     if not vs_rear_enable_live_view.value:
        #         log_info_general("send vsRear image over MQTT")
        #     if not showOutput:
        #         cam_vs_rear.create_image_info()
        #     cam_vs_rear.create_image_info_jpg()
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageData, cam_vs_rear.image_info_base64)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageWidth, cam_vs_rear.img_width)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageHeight, cam_vs_rear.img_height)
        #     vs_rear_send_mqtt_image = False

        # if vs_front_send_mqtt_values:
        #     vs_front_stop_offset = vs_front_edge_position.value - cam_vs_front.stop_position
        #     log_info_general("#######################################################################################################")
        #     log_info_general("stop-delay-time: " + str((time.time() - send_stop_motor_time) * 1000))
        #     log_info_general("vsFront stop-Offset: " + str(vs_front_stop_offset))
        #     cam_vs_front.calc_statistics()
        #     vs_front_statistics = compute_stats(cam_vs_front.stat_image_full)
        #     if showOutput and cam_vs_front.stat_image_full is not None:
        #         cv2.imshow("Sensor-Front - Stat-Image", cam_vs_front.stat_image_full)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_rear_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive, cam_vs_front.exposure_time)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_image_statistics, str(vs_front_statistics))
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_pip, vs_front_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_pip, vs_rear_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getLcmStatistics, str(cam_vs_front.lcm_statistics))
        #     vs_front_send_mqtt_values = False
        #
        # if vs_rear_send_mqtt_values:
        #     vs_rear_stop_offset = vs_rear_edge_position.value - cam_vs_rear.stop_position
        #     log_info_general("vsRear stop-Offset: " + str(vs_rear_stop_offset))
        #     log_info_general("#######################################################################################################")
        #     cam_vs_rear.calc_statistics()
        #     vs_rear_statistics = compute_stats(cam_vs_rear.stat_image_full)
        #     if showOutput and cam_vs_rear.stat_image_full is not None:
        #         cv2.imshow("Sensor-Rear - Stat-Image", cam_vs_rear.stat_image_full)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_rear_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive, cam_vs_rear.exposure_time)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_image_statistics, str(vs_rear_statistics))
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_pip, vs_front_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_pip, vs_rear_edge_position.value)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getLcmStatistics, str(cam_vs_rear.lcm_statistics))
        #     vs_rear_send_mqtt_values = False

        # if film_move_direction.value == 1:
        #     if vs_operation_mode == vs_op_modes.REAR_SENSOR.value or vs_operation_mode == vs_op_modes.BOTH_SENSORS:
        #         if vs_rear_edge_state.previous_value == 1 and vs_rear_edge_state.value == 2:
        #             send_stop_motor_time = time.time()
        #             log_info_general("#######################################################################################################")
        #             log_info_general("REAR EDGE IN STOP-POSITION - send stop-command to plc")
        #             log_info_general("edge-position_front: " + str(vs_front_edge_position.value))
        #             log_info_general("edge-position_rear: " + str(vs_rear_edge_position.value))
        #             plc_handler.stop_film = True
        #             mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
        #             log_info_general("#######################################################################################################")
        #
        #     if vs_operation_mode == vs_op_modes.AUTO.value \
        #             or vs_operation_mode == vs_op_modes.FRONT_SENSOR.value \
        #             or vs_operation_mode == vs_op_modes.BOTH_SENSORS.value:
        #         if vs_front_edge_state.previous_value == 1 and vs_front_edge_state.value == 2:
        #             send_stop_motor_time = time.time()
        #             log_info_general("#######################################################################################################")
        #             log_info_general("FRONT EDGE IN STOP-POSITION - send stop-command to plc")
        #             log_info_general("edge-position_front: " + str(vs_front_edge_position.value))
        #             log_info_general("edge-position_rear: " + str(vs_rear_edge_position.value))
        #             plc_handler.stop_film = True
        #             mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
        #             log_info_general("#######################################################################################################")
        #
        # if film_move_direction.value == 2:
        #     if vs_operation_mode == vs_op_modes.FRONT_SENSOR.value or vs_operation_mode == vs_op_modes.BOTH_SENSORS.value:
        #         if vs_front_edge_state.previous_value == 1 and vs_front_edge_state.value == 2:
        #             send_stop_motor_time = time.time()
        #             log_info_general("#######################################################################################################")
        #             log_info_general("FRONT EDGE IN STOP-POSITION - send stop-command to plc")
        #             log_info_general("edge-position_front: " + str(vs_front_edge_position.value))
        #             log_info_general("edge-position_rear: " + str(vs_rear_edge_position.value))
        #             plc_handler.stop_film = True
        #             mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
        #             log_info_general("#######################################################################################################")
        #
        #     if vs_operation_mode == vs_op_modes.AUTO.value \
        #             or vs_operation_mode == vs_op_modes.REAR_SENSOR.value \
        #             or vs_operation_mode == vs_op_modes.BOTH_SENSORS.value:
        #         if vs_rear_edge_state.previous_value == 1 and vs_rear_edge_state.value == 2:
        #             send_stop_motor_time = time.time()
        #             log_info_general("#######################################################################################################")
        #             log_info_general("REAR EDGE IN STOP-POSITION - send stop-command to plc")
        #             log_info_general("edge-position_front: " + str(vs_front_edge_position.value))
        #             log_info_general("edge-position_rear: " + str(vs_rear_edge_position.value))
        #             plc_handler.stop_film = True
        #             mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
        #             log_info_general("#######################################################################################################")

        if write_init_config:
            write_init_config_to_file()
            write_init_config = False

        if write_settings_config:
            write_settings_config_to_file()
            write_settings_config = False

        # print("looptime: " + str(time.time() - startTime))

        time.sleep(0.0001)


