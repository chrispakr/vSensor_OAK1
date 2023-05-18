#!/usr/bin/env python3
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
import libs.ads_communication as ads_handler
from visionSensor import VisionSensor, VisionSensorOperationMode

enable_chart = False
showOutput = False

init_config_file = "../config/init.ini"
settings_config_file = "../config/settings.ini"

if platform.system() == "Windows":
    import matplotlib.pyplot as plt
    enable_chart = False
    init_config_file = "config/init.ini"
    settings_config_file = "config/settings.ini"

debug_vs = True

write_init_config = False
write_settings_config = False

vs_front_config_name = "vs_front"
vs_rear_config_name = "vs_rear"

camera_fps = 45

vs_op_modes = VisionSensorOperationMode()
vs_operation_mode = vs_op_modes.auto


mqtt_report_position_steps = 2

live_view_fps_divider = 12
vs_front_live_view_frame_nr = 0
vs_rear_live_view_frame_nr = 0

exposure_value_positive = 5000
exposure_value_negative = 1200

vs_front_stop_offset = 0
vs_rear_stop_offset = 0

send_stop_motor_time = time.time()
motor_stopped_time = time.time()

vs_front_last_mqtt_position_value = 0
vs_rear_last_mqtt_position_value = 0

chart_length = 2000
vs_front_chart_slope = deque(maxlen=chart_length)
vs_front_chart_edge_diff = deque(maxlen=chart_length)
vs_front_chart_x_image_nr = deque(maxlen=chart_length)
vs_front_chart_edge_position = deque(maxlen=chart_length)
vs_front_chart_edge_detected = deque(maxlen=chart_length)
vs_front_chart_in_contr = deque(maxlen=chart_length)
vs_front_chart_out_contr = deque(maxlen=chart_length)
vs_front_chart_total_slope_mean = deque(maxlen=chart_length)
vs_front_chart_slope_diff = deque(maxlen=chart_length)

if enable_chart:
    vs_front_chart_slope.append(0)
    vs_front_chart_edge_diff.append(0)
    vs_front_chart_x_image_nr.append(0)
    vs_front_chart_edge_position.append(0)
    vs_front_chart_edge_detected.append(0)
    vs_front_chart_in_contr.append(0)
    vs_front_chart_out_contr.append(0)
    vs_front_chart_total_slope_mean.append(0)
    vs_front_chart_slope_diff.append(0)


def clamp(num, v0, v1):
    return max(v0, min(num, v1))

image_is_centered =             ValueHandler(False)
film_move_direction =           ValueHandler(0)

vs_front_looking_for_picture_position = ValueHandler(False)
vs_rear_looking_for_picture_position = ValueHandler(False)

move_command = ValueHandler(0)

vs_front_send_mqtt_image =      False
vs_rear_send_mqtt_image =       False
vs_front_send_mqtt_values =     False
vs_rear_send_mqtt_values =      False


vs_front_sent_stop_position =   0
vs_rear_sent_stop_position =    0

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
    #devices = dai.Device.getAllAvailableDevices()
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

if config_init.has_option("general", "fps"):
    camera_fps = config_init.getint("general", "fps")
else:
    log_info_general("No config parameter found for 'FPS' - set standard value of 45")
    camera_fps = 45

plc_handler = ads_handler.AdsHandler()

plc_handler.stop_film = False

vs_front = VisionSensor(device_info_front_sensor, True, vs_front_config_name, fps=camera_fps, logger=logging)
vs_rear = VisionSensor(device_info_rear_sensor, False, vs_rear_config_name, fps=camera_fps, logger=logging)

if config_settings.has_option(vs_front_config_name, "proc_image_width"):
    vs_front.proc_image_width = config_settings.getint(vs_front_config_name, "proc_image_width")
if config_settings.has_option(vs_rear_config_name, "proc_image_width"):
    vs_rear.proc_image_width = config_settings.getint(vs_rear_config_name, "proc_image_width")

if config_settings.has_option(vs_front_config_name, "stop_position"):
    vs_front.stop_position = config_settings.getint(vs_front_config_name, "stop_position")
if config_settings.has_option(vs_rear_config_name, "stop_position"):
    vs_rear.stop_position = config_settings.getint(vs_rear_config_name, "stop_position")

if config_settings.has_option(vs_front_config_name, "stop_offset_compensation"):
    vs_front.stop_offset_compensation = config_settings.getint(vs_front_config_name, "stop_offset_compensation")
if config_settings.has_option(vs_rear_config_name, "stop_offset_compensation"):
    vs_rear.stop_offset_compensation = config_settings.getint(vs_rear_config_name, "stop_offset_compensation")

if config_settings.has_option(vs_front_config_name, "edge_detection_range"):
    vs_front.edge_detection_range = config_settings.getint(vs_front_config_name, "edge_detection_range")
if config_settings.has_option(vs_rear_config_name, "edge_detection_range"):
    vs_rear.edge_detection_range = config_settings.getint(vs_rear_config_name, "edge_detection_range")

if config_settings.has_option(vs_front_config_name, "center_position"):
    vs_front.image_center_position = config_settings.getint(vs_front_config_name, "center_position")
if config_settings.has_option(vs_rear_config_name, "center_position"):
    vs_rear.image_center_position = config_settings.getint(vs_rear_config_name, "center_position")

if config_init.has_option(vs_front_config_name, "lens_position"):
    vs_front._lens_position = config_init.getint(vs_front_config_name, "lens_position")

if config_init.has_option(vs_rear_config_name, "lens_position"):
    vs_rear._lens_position = config_init.getint(vs_rear_config_name, "lens_position")


log_info_general("Connect to MQTT-Broker...")
mqtt = mqtt_communication_handler.MqttHandler(logger_enabled=True, client_type="vsController", client_id="vsController", external_logger=logging)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_vsController_sensorVersion, "3")
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_getLcmModeEnabled, ef.bool2Int(False))
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_filmTypeIsNegative, vs_front.film_type_is_negative)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getStopPosition, vs_front.stop_position)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getStopPosition, vs_rear.stop_position)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getStopOffset, vs_front.stop_offset_compensation)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getStopOffset, vs_rear.stop_offset_compensation)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getEdgeDetectionRange, vs_front.edge_detection_range)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getEdgeDetectionRange, vs_rear.edge_detection_range)

mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getCenterPosition, vs_front.image_center_position)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getCenterPosition, vs_rear.image_center_position)

mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_getLcmModeEnabled, vs_front.enable_low_contrast_mode)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_getLcmSlope, vs_front.lcm_slope)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_getLcmContrastOffset, vs_front.lcm_contrast_offset)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setFilmTypeIsNegative)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_fmCtrl_filmMoveDirection)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_fmCtrl_moveCommand)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_swapSensors)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_autoExposureCamera)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_focusCamera)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_focusCamera)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_captureImage)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_captureImage)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setProcImageWidth)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setProcImageWidth)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setStopPosition)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setStopPosition)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setStopOffset)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setStopOffset)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setEdgeDetectionRange)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setEdgeDetectionRange)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setCenterPosition)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setCenterPosition)

mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setExposureTime)



# mqtt.sTopics_vsController.get_fmCtrl_moveCommand.log_output = True
# mqtt.sTopics_vsController.get_fmCtrl_filmMoveDirection.log_output = True
#
mqtt.pTopics_vsController.set_vsFront_edgePosition.log_output = True
# mqtt.pTopics_vsController.set_vsRear_edgePosition.log_output = True
# mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive.log_output = True
# mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive.log_output = True
#
# mqtt.pTopics_vsController.set_vsFront_image_statistics.log_output = True
# mqtt.pTopics_vsController.set_vsRear_image_statistics.log_output = True
#
# mqtt.pTopics_vsController.set_vsFront_imageData.log_output = True
# mqtt.pTopics_vsController.set_vsRear_imageData.log_output = True

mqtt.pTopics_vsController.set_vsFront_edgePosition.log_output = True
mqtt.sTopics_vsController.get_fmCtrl_filmMoveDirection.log_output = True
mqtt.sTopics_vsController.get_fmCtrl_moveCommand.log_output = True

@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsController_heartbeat, 3)

tl.start()

vs_front._enabled_lcm = False
vs_rear._enabled_lcm = False

if enable_chart:
    ed_value = 0
    plt.ion()
    fig, ax = plt.subplots()
    ax.grid()
    line_slope, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_slope, lw=1, label="slope", color="red")
    # line_edge_diff, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_edge_diff, lw=1, label="edgeDiff", color="blue")
    line_edge_detected, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_edge_detected, lw=2, label="edgeDetected", color="blue")
    # line_in_contrast, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_in_contr, lw=1, label="inContr", color="blue")
    # line_out_contrast, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_out_contr, lw=1, label="outContr", color="green")
    line_total_slope_mean, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_total_slope_mean, lw=1, label="total_slope_mean", color="green")
    # line_edge_position, = ax.plot(vs_front.chart_x_image_nr, vs_front.chart_edge_position)
    line_slope_diff, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_slope_diff, lw=1, label="slope_diff_rise", color="cyan")
    # line_slope_tile2, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_slope_tile[1], lw=1)

with contextlib.ExitStack() as stack:
    while True:
        startTime = time.time()

        #######################################################################################################
        # Checking incoming mqtt-messages / values
        #######################################################################################################

        # Check enableLiveView on vsFront
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView):
            vs_front.enable_live_view = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView))
            log_info_general("(vsFront) set enableLiveView to: " + str(vs_front.enable_live_view))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsFront_liveViewIsEnabled, ef.bool2Int(vs_front.enable_live_view))

        # Check enableLiveView on vsRear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView):
            vs_rear.enable_live_view = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView))
            log_info_general("(vsRear) set enableLiveView to: " + str(vs_rear.enable_live_view))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsRear_liveViewIsEnabled, ef.bool2Int(vs_rear.enable_live_view))

        # Check moveCommand
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_fmCtrl_moveCommand):
            move_command.value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_fmCtrl_moveCommand)

        if move_command.new_value_available:
            log_info_general("moveCommand(mqtt) changed to: " + str(move_command.value))
            if move_command.value == 0 and move_command.previous_value > 0:
                log_info_general("command-time-offset: " + str((time.time() - send_stop_motor_time)*1000) + " ms")

        # Check filmMoveDirection
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_fmCtrl_filmMoveDirection):
            film_move_direction.value = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_fmCtrl_filmMoveDirection))

        # focus camera front
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_focusCamera):
            log_info_general("trigger AutoFocus on vsRear")
            vs_front.auto_focus_camera()

        # focus camera rear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_focusCamera):
            log_info_general("trigger AutoFocus on vsRear")
            vs_rear.auto_focus_camera()

        # focus camera front finished
        if vs_front.autoFocusFinished:
            log_info_general("autoFocus on vs_front finished... ")
            vs_front.autoFocusFinished = False
            vs_front_send_mqtt_image = True

        # focus camera rear finished
        if vs_rear.autoFocusFinished:
            log_info_general("autoFocus on vs_rear finished... ")
            vs_rear.autoFocusFinished = False
            vs_rear_send_mqtt_image = True

        # auto exposure cameras
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_autoExposureCamera):
            vs_front.auto_exposure_camera()
            vs_rear.auto_exposure_camera()

        # auto exposure cameras finished
        if vs_front.autoExposureFinished and vs_rear.autoExposureFinished:
            exposure_mean = round((vs_front.exposure_time.value + vs_rear.exposure_time.value) // 2, -2)
            vs_front.set_exposure_value(exposure_mean)
            vs_rear.set_exposure_value(exposure_mean)
            vs_front.autoExposureFinished = False
            vs_rear.autoExposureFinished = False
            vs_front_send_mqtt_image = True
            vs_rear_send_mqtt_image = True
            log_info_general("set exposure to: {}".format(exposure_mean))

        # set exposure-time to cameras
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setExposureTime):
            t_exposure_time = mqtt.getMqttValue(mqtt.sTopics_vsController.get_setExposureTime)
            vs_front.set_exposure_value(t_exposure_time)
            vs_rear.set_exposure_value(t_exposure_time)
            vs_front_send_mqtt_image = True
            vs_rear_send_mqtt_image = True
            log_info_general("set exposure to: {}".format(t_exposure_time))

        # set film-type positive/negative
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setFilmTypeIsNegative):
            t_value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_setFilmTypeIsNegative)
            vs_front.film_type_is_negative = t_value
            vs_rear.film_type_is_negative = t_value
            if t_value:
                vs_front.set_exposure_value(exposure_value_negative)
                vs_rear.set_exposure_value(exposure_value_negative)
            else:
                vs_front.set_exposure_value(exposure_value_positive)
                vs_rear.set_exposure_value(exposure_value_positive)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_filmTypeIsNegative, ef.bool2Int(t_value))

        # swap sensors
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_swapSensors):
            vs_front_serial = config_init.get(vs_front_config_name, "serial")
            vs_rear_serial = config_init.get(vs_rear_config_name, "serial")
            config_init.set(vs_front_config_name, "serial", vs_rear_serial)
            config_init.set(vs_rear_config_name, "serial", vs_front_serial)
            write_init_config = True

        # Check enableLowContrastMode
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_setEnableLcmMode):
            t_value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_setEnableLcmMode)
            log_info_general("set enableLowContrastMode to: " + str(t_value))
            if t_value:
                vs_front.enable_low_contrast_mode = True
                vs_rear.enable_low_contrast_mode = True
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_getLcmModeEnabled, ef.bool2Int(vs_front.enable_low_contrast_mode))
            else:
                vs_front.enable_low_contrast_mode = False
                vs_rear.enable_low_contrast_mode = False
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_getLcmModeEnabled, ef.bool2Int(vs_front.enable_low_contrast_mode))

        # lcm set lcm_slope
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setLcmSlope):
            t_value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_setLcmSlope)
            # log_info_general("set lcmSlope to: " + str(t_value))
            vs_front.lcm_slope = t_value
            vs_rear.lcm_slope = t_value
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_getLcmSlope, vs_front.lcm_slope)

        # lcm set lcm_contrast
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setLcmContrastOffset):
            t_value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_setLcmContrastOffset)
            # log_info_general("set lcmContrast to: " + str(t_value))
            vs_front.lcm_contrast_offset = t_value
            vs_rear.lcm_contrast_offset = t_value
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_getLcmContrastOffset, vs_front.lcm_contrast_offset)

        # Capture Image vsFront
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_captureImage):
            vs_front_send_mqtt_image = True
            vs_front_send_mqtt_values = True

        # Capture Image vsRear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_captureImage):
            vs_rear_send_mqtt_image = True
            vs_rear_send_mqtt_values = True

        # set procImageWidth on vsFront
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setProcImageWidth):
            vs_front.proc_image_width = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setProcImageWidth))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsFront_getProcImageWidth, vs_front.proc_image_width)
            config_settings.set(vs_front_config_name, "proc_image_width", str(vs_front.proc_image_width))
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set procImageWidth on vsRear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setProcImageWidth):
            vs_rear.proc_image_width = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setProcImageWidth))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsRear_getProcImageWidth, vs_rear.proc_image_width)
            config_settings.set(vs_rear_config_name, "proc_image_width", str(vs_rear.proc_image_width))
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        # set stopPosition on vsFront
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setStopPosition):
            vs_front.stop_position = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setStopPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getStopPosition, vs_front.stop_position)
            config_settings.set(vs_front_config_name, "stop_position", str(vs_front.stop_position))
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set stopPosition on vsRear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setStopPosition):
            vs_rear.stop_position = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setStopPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getStopPosition, vs_rear.stop_position)
            config_settings.set(vs_rear_config_name, "stop_position", str(vs_rear.stop_position))
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        # set stopOffset on vsFront
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setStopOffset):
            vs_front.stop_offset_compensation = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setStopOffset)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getStopOffset, vs_front.stop_offset_compensation)
            config_settings.set(vs_front_config_name, "stop_offset_compensation", str(vs_front.stop_offset_compensation))
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set stopOffset on vsRear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setStopOffset):
            vs_rear._stop_offset_compensation = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setStopOffset)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getStopOffset, vs_rear.stop_offset_compensation)
            config_settings.set(vs_rear_config_name, "stop_offset_compensation", str(vs_rear.stop_offset_compensation))
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        # set edgeDetectionRange on vsFront
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setEdgeDetectionRange):
            vs_front.edge_detection_range = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setEdgeDetectionRange)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getEdgeDetectionRange, vs_front.edge_detection_range)
            config_settings.set(vs_front_config_name, "edge_detection_range", str(vs_front.edge_detection_range))
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set edgeDetectionRange on vsRear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setEdgeDetectionRange):
            vs_rear.edge_detection_range = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setEdgeDetectionRange)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getEdgeDetectionRange, vs_rear.edge_detection_range)
            config_settings.set(vs_rear_config_name, "edge_detection_range", str(vs_rear.edge_detection_range))
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        # set centerPosition on vsFront
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setCenterPosition):
            vs_front.image_center_position = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setCenterPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getCenterPosition, vs_front.image_center_position)
            config_settings.set(vs_front_config_name, "center_position", str(vs_front.image_center_position))
            write_settings_config = True
            vs_front_send_mqtt_image = True

        # set centerPosition on vsRear
        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setCenterPosition):
            vs_rear.image_center_position = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setCenterPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getCenterPosition, vs_rear.image_center_position)
            config_settings.set(vs_rear_config_name, "center_position", str(vs_rear.image_center_position))
            write_settings_config = True
            vs_rear_send_mqtt_image = True

        ##################################################################################################################
        # IMAGE PROCESSING PART
        ##################################################################################################################

        if vs_front.process_image() or vs_rear.process_image():
            if vs_front.enable_live_view:
                if vs_front_live_view_frame_nr == live_view_fps_divider:
                    # log_info_general("image")
                    vs_front_send_mqtt_image = True
                    vs_front_live_view_frame_nr = 0
                vs_front_live_view_frame_nr += 1

            if vs_rear.enable_live_view:
                if vs_rear_live_view_frame_nr == live_view_fps_divider:
                    vs_rear_send_mqtt_image = True
                    vs_rear_live_view_frame_nr = 0
                vs_rear_live_view_frame_nr += 1

            if showOutput:
                if vs_front.create_image_info():
                    cv2.imshow("Sensor-Front - Processing", vs_front.np_image_info_edge_line)
                    cv2.imshow("Sensor-Front - Tile Left", vs_front.np_image_tile_left)
                    cv2.imshow("Sensor-Front - Tile Right", vs_front.np_image_tile_right)

                if vs_rear.create_image_info():
                    cv2.imshow("Sensor-Rear - Processing", vs_rear.np_image_info_edge_line)
                    cv2.imshow("Sensor-Rear - Tile Left", vs_rear.np_image_tile_left)
                    cv2.imshow("Sensor-Rear - Tile Right", vs_rear.np_image_tile_right)

                if enable_chart:
                    vs_front_chart_slope.append(vs_front._total_edge_slope)
                    vs_front_chart_edge_diff.append(vs_front.edge_position_tile_diff * 10)
                    vs_front_chart_x_image_nr.append(vs_front.captured_images)
                    vs_front_chart_edge_position.append(vs_front.edge_position.value)
                    vs_front_chart_in_contr.append(vs_front._in_pic_median_total // 2)
                    vs_front_chart_out_contr.append(vs_front._out_pic_median_total // 2)
                    vs_front_chart_total_slope_mean.append(vs_front._slope_total_mean)
                    vs_front_chart_edge_detected.append(vs_front._new_edge_detected)
                    vs_front_chart_slope_diff.append(vs_front._slope_diff_falling)

                    # print(self.chart_slope)
                    # if vs_front.edge_detected.value:
                    #     vs_front_chart_edge_detected.append(150)
                    # else:
                    #     vs_front_chart_edge_detected.append(0)
                    if vs_front.captured_images % 10 == 0:
                        ax.set_xlim(min(vs_front_chart_x_image_nr), max(vs_front_chart_x_image_nr))
                        ax.set_ylim(0, 300)
                        line_slope.set_data(vs_front_chart_x_image_nr, vs_front_chart_slope)
                        # line_edge_diff.set_data(vs_front_chart_x_image_nr, vs_front_chart_edge_diff)
                        line_edge_detected.set_data(vs_front_chart_x_image_nr, vs_front_chart_edge_detected)
                        # line_edge_position.set_data(self.chart_x_image_nr, self.chart_edge_position)
                        # line_slope_tile2.set_data(self.chart_x_image_nr, self.chart_slope_tile[1])
                        # line_slope_tile1.set_data(self.chart_x_image_nr, self.chart_slope_tile[0])
                        # line_in_contrast.set_data(vs_front_chart_x_image_nr, vs_front_chart_in_contr)
                        # line_out_contrast.set_data(vs_front_chart_x_image_nr, vs_front_chart_out_contr)
                        line_total_slope_mean.set_data(vs_front_chart_x_image_nr, vs_front_chart_total_slope_mean)
                        line_slope_diff.set_data(vs_front_chart_x_image_nr, vs_front_chart_slope_diff)

            # check if exposureTime has changed on vsFront
            if vs_front.exposure_time.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive, vs_front.exposure_time.value)

            # check if exposureTime has changed on vsRear
            if vs_rear.exposure_time.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive, vs_rear.exposure_time.value)

            # vsFront - edge-detected
            if vs_front.edge_detected.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgeDetected, ef.bool2Int(vs_front.edge_detected.value))

            # vsRear - edge-detected
            if vs_rear.edge_detected.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgeDetected, ef.bool2Int(vs_rear.edge_detected.value))

            # if vs_front.edge_position.new_value_available:
            #     log_info_general("edge_front_sensor: {}".format(str(vs_front.edge_position.value)))
            #
            # if vs_rear.edge_position.new_value_available:
            #     log_info_general("edge_rear_sensor: {}".format(str(vs_rear.edge_position.value)))

            if vs_front.edge_status.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getEdgeStatus, vs_front.edge_status.value)

            if vs_rear.edge_status.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getEdgeStatus, vs_rear.edge_status.value)

            if vs_front.fps.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_fps, vs_front.fps.value)
                vs_front.fps.reset()

            if vs_rear.fps.new_value_available:
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_fps, vs_rear.fps.value)
                vs_rear.fps.reset()

            if vs_front_send_mqtt_image:
                if not vs_front.enable_live_view:
                    log_info_general("send vsFront image over MQTT")
                if not showOutput:
                    vs_front.create_image_info()
                vs_front.create_image_info_jpg()
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageData, vs_front.image_info_base64)
                if vs_front.img_width.new_value_available:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageWidth, vs_front.img_width.value)
                    vs_front.img_width.reset()
                if vs_front.img_height.new_value_available:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageHeight, vs_front.img_height.value)
                    vs_front.img_height.reset()
                vs_front_send_mqtt_image = False

            if vs_rear_send_mqtt_image:
                if not vs_rear.enable_live_view:
                    log_info_general("send vsRear image over MQTT")
                if not showOutput:
                    vs_rear.create_image_info()
                vs_rear.create_image_info_jpg()
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageData, vs_rear.image_info_base64)
                if vs_rear.img_width.new_value_available:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageWidth, vs_rear.img_width.value)
                    vs_rear.img_width.reset()
                if vs_rear.img_height.new_value_available:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageHeight, vs_rear.img_height.value)
                    vs_rear.img_height.reset()
                vs_rear_send_mqtt_image = False

            if vs_front_send_mqtt_values:
                vs_front_stop_offset = vs_front.edge_position.value - vs_front.stop_position
                log_info_general("#######################################################################################################")
                log_info_general("stop-delay-time: " + str((time.time() - send_stop_motor_time) * 1000))
                log_info_general("vsFront stop-Offset: " + str(vs_front_stop_offset))
                vs_front.calc_statistics()
                vs_front_statistics = compute_stats(vs_front.stat_image_full)
                if showOutput and vs_front.stat_image_full is not None:
                    cv2.imshow("Sensor-Front - Stat-Image", vs_front.stat_image_full)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front.edge_position.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_rear.edge_position.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive, vs_front.exposure_time.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_image_statistics, str(vs_front_statistics))
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_pip, vs_front.edge_position.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_pip, vs_rear.edge_position.value)
                vs_front_send_mqtt_values = False

            if vs_rear_send_mqtt_values:
                vs_rear_stop_offset = vs_rear.edge_position.value - vs_rear.stop_position
                log_info_general("vsRear stop-Offset: " + str(vs_rear_stop_offset))
                log_info_general("#######################################################################################################")
                vs_rear.calc_statistics()
                vs_rear_statistics = compute_stats(vs_rear.stat_image_full)
                if showOutput and vs_rear.stat_image_full is not None:
                    cv2.imshow("Sensor-Rear - Stat-Image", vs_rear.stat_image_full)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front.edge_position.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_rear.edge_position.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive, vs_rear.exposure_time.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_image_statistics, str(vs_rear_statistics))
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_pip, vs_front.edge_position.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_pip, vs_rear.edge_position.value)
                vs_rear_send_mqtt_values = False

            if debug_vs:
                pass

        if film_move_direction.value == 0 and film_move_direction.previous_value > 0:
            log_info_general("Motor Stopped...")
            vs_front_send_mqtt_image = True
            vs_rear_send_mqtt_image = True
            vs_front_send_mqtt_values = True
            vs_rear_send_mqtt_values = True

        # if move_command.value == 0 and move_command.previous_value > 0:
        #     log_info_general("receive moveCommand={}".format(str(move_command.value)))
        #     vs_front_send_mqtt_image = True
        #     vs_rear_send_mqtt_image = True
        #     vs_front_send_mqtt_values = True
        #     vs_rear_send_mqtt_values = True


        if vs_front.edge_position.value > vs_front_last_mqtt_position_value + mqtt_report_position_steps:
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front.edge_position.value)
            vs_front_last_mqtt_position_value = vs_front.edge_position.value

        if vs_rear.edge_position.value < 0 and not vs_rear_looking_for_picture_position.value:
            vs_rear_looking_for_picture_position.value = True
            plc_handler.stop_film = False
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 0)

        if vs_front.edge_position.value < 0 and not vs_front_looking_for_picture_position.value:
            vs_front_looking_for_picture_position.value = True
            plc_handler.stop_film = False
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 0)
            vs_front_last_mqtt_position_value = vs_front.edge_position.value

        if film_move_direction.value == 1:
            if vs_operation_mode == vs_op_modes.auto or vs_operation_mode == vs_op_modes.front_sensor or vs_operation_mode == vs_op_modes.both_sensors:
                if vs_rear.edge_position.new_value_available:
                    if 0 < vs_rear.edge_position.value <= (vs_rear.stop_position + vs_rear.stop_offset_compensation) and vs_rear_looking_for_picture_position.value:
                        send_stop_motor_time = time.time()
                        log_info_general("#######################################################################################################")
                        log_info_general("REAR EDGE IN STOP-POSITION - send stop-command to plc")
                        log_info_general("edge-position_front: " + str(vs_front.edge_position.value))
                        log_info_general("edge-position_rear: " + str(vs_rear.edge_position.value))
                        plc_handler.stop_film = True
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
                        log_info_general("#######################################################################################################")
                        vs_rear_looking_for_picture_position.value = False

            if vs_operation_mode == vs_op_modes.rear_sensor or vs_operation_mode == vs_op_modes.both_sensors:
                if vs_front.edge_position.new_value_available:
                    if vs_front.edge_position.value >= (vs_front.stop_position - vs_front.stop_offset_compensation) and vs_front_looking_for_picture_position.value:
                        send_stop_motor_time = time.time()
                        log_info_general("#######################################################################################################")
                        log_info_general("FRONT EDGE IN STOP-POSITION - send stop-command to plc")
                        log_info_general("edge-position_front: " + str(vs_front.edge_position.value))
                        log_info_general("edge-position_rear: " + str(vs_rear.edge_position.value))
                        plc_handler.stop_film = True
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
                        log_info_general("#######################################################################################################")
                        vs_front_looking_for_picture_position.value = False

        if film_move_direction.value == 2:
            if vs_operation_mode == vs_op_modes.auto \
                    or vs_operation_mode == vs_op_modes.rear_sensor \
                    or vs_operation_mode == vs_op_modes.both_sensors:
                if vs_rear.edge_position.new_value_available:
                    if vs_rear.edge_position.value >= (vs_rear.stop_position - vs_rear.stop_offset_compensation) \
                            and vs_rear_looking_for_picture_position.value:
                        send_stop_motor_time = time.time()
                        log_info_general("#######################################################################################################")
                        log_info_general("REAR EDGE IN STOP-POSITION - send stop-command to plc")
                        log_info_general("edge-position_front: " + str(vs_front.edge_position.value))
                        log_info_general("edge-position_rear: " + str(vs_rear.edge_position.value))
                        plc_handler.stop_film = True
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
                        log_info_general("#######################################################################################################")
                        vs_rear_looking_for_picture_position.value = False

            if vs_operation_mode == vs_operation_mode == vs_op_modes.front_sensor \
                    or vs_operation_mode == vs_op_modes.both_sensors:
                if vs_front.edge_position.new_value_available:
                    if 0 < vs_front.edge_position.value <= (vs_front.stop_position + vs_front.stop_offset_compensation) \
                            and vs_front_looking_for_picture_position.value:
                        send_stop_motor_time = time.time()
                        log_info_general("#######################################################################################################")
                        log_info_general("FRONT EDGE IN STOP-POSITION - send stop-command to plc")
                        log_info_general("edge-position_front: " + str(vs_front.edge_position.value))
                        log_info_general("edge-position_rear: " + str(vs_rear.edge_position.value))
                        plc_handler.stop_film = True
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, 1)
                        log_info_general("#######################################################################################################")
                        vs_front_looking_for_picture_position.value = False

        if vs_rear_looking_for_picture_position.new_value_available:
            log_info_general("LookingForEdge: " + str(vs_rear_looking_for_picture_position.value))

        if vs_front_looking_for_picture_position.new_value_available:
            log_info_general("LookingForEdge: " + str(vs_front_looking_for_picture_position.value))

        if write_init_config:
            write_init_config_to_file()
            write_init_config = False

        if write_settings_config:
            write_settings_config_to_file()
            write_settings_config = False

        vs_front.edge_position.reset()
        vs_rear.edge_position.reset()

        vs_front.edge_detected.reset()
        vs_rear.edge_detected.reset()

        vs_front.edge_is_in_position.reset()
        vs_rear.edge_is_in_position.reset()

        # vs_front.autoFocusEnabled.reset()
        # vs_rear.autoFocusEnabled.reset()

        vs_front.exposure_time.reset()
        vs_rear.exposure_time.reset()
        image_is_centered.reset()

        vs_front.edge_status.reset()
        vs_rear.edge_status.reset()

        film_move_direction.reset()
        move_command.reset()

        vs_front_looking_for_picture_position.reset()
        vs_rear_looking_for_picture_position.reset()

        if platform.system() == "Windows":
            if cv2.waitKey(1) == ord('q'):
                break

        time.sleep(0.0001)

        # print("looptime: " + str(time.time() - startTime))
