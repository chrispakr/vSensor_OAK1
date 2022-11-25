#!/usr/bin/env python3
import cv2
import io
import platform
import depthai as dai
import time
from threading import Event, Thread
import numpy as np
from pathlib import Path
import contextlib
import sys
import base64
from timeloop import Timeloop
import logging
from datetime import datetime, timedelta
import mqtt_communication_handler_v2
import libs.functions as ef
from configparser import ConfigParser
from turbojpeg import TurboJPEG
from collections import deque

enable_chart = False

if platform.system() == "Windows":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    enable_chart = True

# import calc_mean as CalcMean

debug_vs = True

edge_detection_slope_lcm = 30
edge_detection_slope_std = 50
contrast_offset_lcm = 40
contrast_offset_std = 100
max_edge_position_diff = 15

init_config_file = "config/init.ini"
settings_config_file = "config/settings.ini"
write_init_config = False
write_settings_config = False

vs_front_config_name = "vs_front"
vs_rear_config_name = "vs_rear"

line_color_red = (0, 0, 255)
line_color_green = (0, 255, 0)
line_color_orange = (0, 165, 255)
line_color_blue = (255, 0, 0)

capture_width = 860
capture_height = 600

preview_width = 800

update_setup_jpg = False

# preview_height_crop_factor = 0.693333333
# preview_height_crop = int(preview_height * preview_height_crop_factor)
# if preview_height_crop % 32 != 0:
#     print("preview_height_crop must be a multiple of 32")
#     preview_height_crop = ((preview_height_crop // 32) + 1) * 32
#     print("set preview_height_crop to {}".format(preview_height_crop))
#
# thumbnail_width = 576
# thumbnail_height = (thumbnail_width / 4) * 3 * preview_height_crop_factor

# if thumbnail_width % 32 != 0:
#     print("thumbnail_width must be a multiple of 32")
#
# if thumbnail_height % 32 != 0:
#     result = thumbnail_height / 32
#     thumbnail_height = (int(result) + 1) * 32
#     print(thumbnail_height)

camera_fps = 60

enable_live_view = False

reset_program = False

last_slope_value = 0

mqtt_report_position_steps = 2

chart_length = 2000
vs_front_chart_slope = deque(maxlen=chart_length)
vs_front_chart_edge_diff = deque(maxlen=chart_length)
vs_front_chart_x_image_nr = deque(maxlen=chart_length)
vs_front_chart_edge_position = deque(maxlen=chart_length)
vs_front_chart_edge_detected = deque(maxlen=chart_length)
vs_front_chart_edge_diff = deque(maxlen=chart_length)
vs_front_chart_in_contr = deque(maxlen=chart_length)
vs_front_chart_out_contr = deque(maxlen=chart_length)
vs_front_chart_total_slope_mean = deque(maxlen=chart_length)

if enable_chart:
    vs_front_chart_slope.append(0)
    vs_front_chart_edge_diff.append(0)
    vs_front_chart_x_image_nr.append(0)
    vs_front_chart_edge_position.append(0)
    vs_front_chart_edge_detected.append(0)
    vs_front_chart_in_contr.append(0)
    vs_front_chart_out_contr.append(0)
    vs_front_chart_total_slope_mean.append(0)

def clamp(num, v0, v1):
    return max(v0, min(num, v1))


class ValueHandler:
    def __init__(self, value=None):
        self.value = value
        self.last_value = value

    def is_different(self):
        if self.value != self.last_value:
            return True
        else:
            return False

    def reset(self):
        self.last_value = self.value


vs_front_enable_live_view = ValueHandler(False)
vs_rear_enable_live_view = ValueHandler(False)

image_is_centered = ValueHandler(False)

film_type_is_negative = ValueHandler(True)

vs_enable_low_contrast_mode = ValueHandler(True)

film_move_direction = ValueHandler(0)

send_thumbnail = ValueHandler(False)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    # handlers=[logging.FileHandler("vSensor.log"), logging.StreamHandler(sys.stdout)]
    handlers=[logging.StreamHandler(sys.stdout)]
)


def logInfoGeneral(message):
    log_message = "[general ]" + " - " + message
    logging.info(log_message)


if platform.system() == "Windows":
    logInfoGeneral("set parameters for windows-system")
    jpeg = TurboJPEG("libs/libturbojpeg.dll")
    showOutput = True

if platform.system() == "Linux":
    logInfoGeneral("set parameters for linux-system")
    jpeg = TurboJPEG()
    showOutput = False

tl = Timeloop()

class VisionSensor(ValueHandler):
    edge_detection_range = 20
    stop_position = 350
    lens_position = 160
    number_of_tiles = 2
    tn_scale_percent = 80
    fps_report_time = 2
    contrast_pic_height = 20
    contrast_pic_edge_offset = 10

    def __init__(self, device_info, is_front_sensor, vs_name):
        # general Variables
        self.name = vs_name
        self.device_info = device_info
        self.is_front_sensor = is_front_sensor
        self.is_main_sensor = is_front_sensor
        self.film_type_is_negative = True
        self.edge_position = ValueHandler(-1)
        self.last_edge_position = 0
        self.total_edge_slope = 0
        self.lcm_enabled = False
        self.edge_position_tile_diff = 0
        self.edge_detected = ValueHandler(False)
        self.edge_is_in_position = ValueHandler(False)
        self.image_center_position = 450
        self.proc_image_width = 400

        # processing image variables
        self.input_image_data = None
        self.raw_input_image = None
        self.np_image_tile_left = None
        self.np_image_tile_right = None
        self.np_image_info_edge_line = None
        self.np_image_info_setup_lines = None
        self.proc_image_centered = None
        self.crop_image_right = 50
        self.crop_image_left = 50

        self.left_edge_position = 0
        self.right_edge_position = 0

        self.left_edge_slope = 0
        self.right_edge_slope = 0

        self.left_edge_results = None
        self.right_edge_results = None

        self.arr_slope_total_mean = deque(maxlen=20)
        self.slope_total_mean = 0
        self.new_edge_detected = 0

        self.jpg_setup = None
        self.jpg_setup_base64 = None
        # self.edge_position_tile = []
        # self.edge_slope_tile = []
        # self.edge_result_tile = []
        # self.image_tile = []
        # self.chart_slope_tile = []
        # for i in range(self.number_of_tiles):
        #     self.edge_position_tile.append(ValueHandler(-1))
        #     self.edge_slope_tile.append(0)
        #     self.edge_result_tile.append(None)
        #     self.image_tile.append(None)
        #     self.chart_slope_tile.append(deque(maxlen=self.chart_length))
        #     self.chart_slope_tile[i].append(0)
        self.captured_images = 0

        # preview image variables
        self.jpg_image_data = None
        self.jpg_image = None
        self.image_jpg_info = None
        self.image_info_full = None
        self.jpg_info_full = None
        self.jpg_info_full_base64 = None
        self.image_info_tn = None
        self.jpg_info_tn = None
        self.jpg_info_tn_base64 = None
        self.jpg_video_base64 = None

        # fps variables
        self.fps_elapsed_time = datetime.now()
        self.fps_jpg_image_time = datetime.now()
        self.fps = 0
        self.fps_jpg_image = 0
        self._fps_counter = 0
        self._fps_counter_jpg_image = 0

        # statistics image variables
        self.stat_image_full = None
        self.in_pic_median_left = 0
        self.in_pic_median_right = 0
        self.out_pic_median_left = 0
        self.out_pic_median_right = 0
        self.in_pic_median_total = 0
        self.out_pic_median_total = 0

        # camera control variables
        self.camCtrl = None
        self.exposure_time = ValueHandler(0)
        self.af_start_time = datetime.now()
        self.autoFocusEnabled = False

        self.stop_position_min = self.stop_position - (self.edge_detection_range // 2)
        self.stop_position_max = self.stop_position + (self.edge_detection_range // 2)

        self.pipeline = None
        self.getPipeline()
        self.device = dai.Device(self.pipeline, self.device_info)
        self.image_edge_queue = self.device.getOutputQueue(name="image_edge_detection", maxSize=4, blocking=True)
        self.camera_control_queue = self.device.getInputQueue('control')

    def getPipeline(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.manip_edge_detetction = self.pipeline.create(dai.node.ImageManip)
        self.x_out_edge_detection = self.pipeline.create(dai.node.XLinkOut)

        self.controlIn = self.pipeline.create(dai.node.XLinkIn)

        self.x_out_edge_detection.setStreamName('image_edge_detection')
        self.controlIn.setStreamName('control')

        # Properties
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setPreviewSize(capture_width, capture_height)
        self.camRgb.setFps(camera_fps)
        self.camRgb.initialControl.setAutoFocusLensRange(120, 180)
        self.camRgb.initialControl.setManualFocus(int(config_init.get(self.name, "lens_position")))
        self.camRgb.initialControl.setManualExposure(1200, 100)
        self.camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        self.camRgb.setInterleaved(False)
        maxFrameSize = self.camRgb.getPreviewWidth() * self.camRgb.getPreviewHeight() * 3

        # self.manip_edge_detetction.initialConfig.setCropRect(0, 0, 1, (self.stop_position_max + 20) / preview_height)
        self.manip_edge_detetction.initialConfig.setCropRect(0, 0, 1, 0.75)
        self.manip_edge_detetction.setMaxOutputFrameSize(maxFrameSize)
        self.manip_edge_detetction.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        # Links
        self.camRgb.preview.link(self.manip_edge_detetction.inputImage)

        self.manip_edge_detetction.out.link(self.x_out_edge_detection.input)

        self.controlIn.out.link(self.camRgb.inputControl)

    def process_image(self):
        self.input_image_data = self.image_edge_queue.tryGet()
        if self.input_image_data is not None:
            self.edge_position.value = -1
            self.raw_input_image = self.input_image_data.getCvFrame()
            (full_image_height, full_image_width) = self.raw_input_image.shape[:2]
            if self.image_center_position < preview_width // 2:
                self.image_center_position = capture_width // 2
            self.proc_image_centered = self.raw_input_image[0:full_image_height, self.image_center_position - (preview_width // 2):self.image_center_position + (preview_width // 2)]
            self.np_image_tile_left = self.raw_input_image[0:self.stop_position + 50, self.image_center_position - self.proc_image_width:self.image_center_position]
            self.np_image_tile_right = self.raw_input_image[0:self.stop_position + 50, self.image_center_position:self.image_center_position + self.proc_image_width]

            self.left_edge_results = self.calc_edge_parameter(self.np_image_tile_left)
            self.right_edge_results = self.calc_edge_parameter(self.np_image_tile_right)

            if self.film_type_is_negative:
                self.left_edge_position = self.left_edge_results[3]
                self.left_edge_slope = int(self.left_edge_results[1])
                self.right_edge_position = self.right_edge_results[3]
                self.right_edge_slope = int(self.right_edge_results[1])
            else:
                self.left_edge_position = self.left_edge_results[2]
                self.left_edge_slope = int(self.left_edge_results[0])
                self.right_edge_position = self.right_edge_results[2]
                self.right_edge_slope = int(self.right_edge_results[0])

            self.edge_position_tile_diff = abs(self.left_edge_position - self.right_edge_position)
            self.total_edge_slope = self.left_edge_slope + self.right_edge_slope

            self.arr_slope_total_mean.append(self.total_edge_slope)
            self.slope_total_mean = sum(self.arr_slope_total_mean) // len(self.arr_slope_total_mean)

            if (self.total_edge_slope - self.slope_total_mean) > 50:
                self.new_edge_detected = 200
            if (self.slope_total_mean - self.total_edge_slope) > 50:
                self.new_edge_detected = 0
            # if self.edge_position_tile_diff < max_edge_position_diff:
            #     if self.lcm_enabled:
            #         if self.total_edge_slope > edge_detection_slope_lcm:
            #             self.edge_position.value = (self.left_edge_position + self.right_edge_position) // 2
            #     else:
            #         if self.total_edge_slope > edge_detection_slope_std:
            #             self.edge_position.value = (self.left_edge_position + self.right_edge_position) // 2

            if self.edge_position_tile_diff < max_edge_position_diff:
                if self.lcm_enabled:
                    if self.total_edge_slope > edge_detection_slope_lcm:
                        self.edge_position.value = (self.left_edge_position + self.right_edge_position) // 2
                else:
                    if self.total_edge_slope > edge_detection_slope_std:
                        self.edge_position.value = (self.left_edge_position + self.right_edge_position) // 2

            # self.calc_edge_position()

            if self.edge_position.value > (self.contrast_pic_height + self.contrast_pic_edge_offset):
                in_pic_contrast_max_pos = self.edge_position.value - self.contrast_pic_edge_offset
                in_pic_contrast_min_pos = in_pic_contrast_max_pos - self.contrast_pic_height
                in_pic_contrast_image_left = self.np_image_tile_left[in_pic_contrast_min_pos:in_pic_contrast_max_pos, 0:self.proc_image_width]
                in_pic_contrast_image_right = self.np_image_tile_right[in_pic_contrast_min_pos:in_pic_contrast_max_pos, 0:self.proc_image_width]
                self.in_pic_median_left = np.median(in_pic_contrast_image_left)
                self.in_pic_median_right = np.median(in_pic_contrast_image_right)
                self.in_pic_median_total = self.in_pic_median_left + self.in_pic_median_right
                out_pic_contrast_min_pos = self.edge_position.value + self.contrast_pic_edge_offset
                out_pic_contrast_max_pos = out_pic_contrast_min_pos + self.contrast_pic_height
                out_pic_contrast_image_left = self.np_image_tile_left[out_pic_contrast_min_pos:out_pic_contrast_max_pos, 0:self.proc_image_width]
                out_pic_contrast_image_right = self.np_image_tile_right[out_pic_contrast_min_pos:out_pic_contrast_max_pos, 0:self.proc_image_width]
                self.out_pic_median_left = np.median(out_pic_contrast_image_left)
                self.out_pic_median_right = np.median(out_pic_contrast_image_right)
                self.out_pic_median_total = self.out_pic_median_left + self.out_pic_median_right
                # if self.is_front_sensor:
                #     print(str(np.median(self.in_pic_median_left)) + " // " + str(np.median(self.out_pic_median_left)))

            if self.lcm_enabled:
                if self.in_pic_median_total + contrast_offset_lcm < self.out_pic_median_total:
                    self.edge_detected.value = True
                else:
                    self.edge_detected.value = False
                    self.edge_position.value = -1
            else:
                if self.in_pic_median_total + contrast_offset_std < self.out_pic_median_total:
                    self.edge_detected.value = True
                else:
                    self.edge_detected.value = False
                    self.edge_position.value = -1

            if self.edge_position.value > self.stop_position_min:
                self.edge_is_in_position.value = True
            else:
                self.edge_is_in_position.value = False

            self.captured_images += 1


            self._fps_counter += 1
            if (datetime.now() - self.fps_elapsed_time).seconds >= self.fps_report_time:
                self.fps = self._fps_counter // self.fps_report_time
                self.fps_elapsed_time = datetime.now()
                self._fps_counter = 0
            if int(self.input_image_data.getExposureTime().total_seconds() * 1000000) != self.exposure_time:
                self.exposure_time.value = int(self.input_image_data.getExposureTime().total_seconds() * 1000000)
            if self.input_image_data.getLensPosition() != self.lens_position:
                self.lens_position = self.input_image_data.getLensPosition()
                self.log_info_vSensor("lens-position changed to: {}".format(self.lens_position))
            if self.autoFocusEnabled:
                if (datetime.now() - self.af_start_time).seconds > 2:
                    self.autoFocusEnabled = False
                    self.setFocusValue(self.lens_position)
                    self.log_info_vSensor("disable AutoFocus")
                    config_init.set(self.name, "lens_position", str(self.lens_position))
                    writeInitConfigToFile()
            return True
        else:
            return False

    def calc_edge_position(self):








        # if self.edge_position_tile_diff < max_edge_position_diff:
        #     if not self.lcm_enabled:
        #         if self.total_edge_slope > edge_detection_slope_std:
        #             self.edge_position.value = (self.left_edge_position + self.right_edge_position) // 2
        #     else:
        #         if self.total_edge_slope > edge_detection_slope_lcm:
        #             self.edge_position.value = (self.left_edge_position + self.right_edge_position) // 2

        # if self.in_pic_contrast_image_total < 0:
        #     self.edge_detected.value = True
        # else:
        #     self.edge_detected.value = False


        self.last_edge_position = self.edge_position
        # if self.is_front_sensor:
        #     self.log_info_vSensor(self.edge_position.value)

    def create_image_info(self):
        if self.proc_image_centered is not None:
            self.np_image_info_edge_line = cv2.cvtColor(self.proc_image_centered, cv2.COLOR_GRAY2RGB)
            img_height, img_width, dim = self.np_image_info_edge_line.shape
            if self.edge_is_in_position.value:
                line_color = line_color_green
            else:
                line_color = line_color_orange
            cv2.line(self.np_image_info_edge_line, (0, self.edge_position.value), (img_width, self.edge_position.value), line_color, 2)
            self.np_image_info_setup_lines = self.np_image_info_edge_line
            cv2.line(self.np_image_info_setup_lines, (img_width // 2 - self.proc_image_width, 0),
                     (img_width // 2 - self.proc_image_width, img_height), line_color_blue, 1)
            cv2.line(self.np_image_info_setup_lines, (img_width // 2 + self.proc_image_width, 0),
                     (img_width // 2 + self.proc_image_width, img_height), line_color_blue, 1)
            center_image_line = ((img_width - self.crop_image_right - self.crop_image_left) // 2) + self.crop_image_left
            cv2.line(self.np_image_info_setup_lines, (img_width // 2, 0), (img_width // 2, img_height), line_color_red, 1)

    def create_setup_jpg(self):
        if self.np_image_info_setup_lines is not None:
            self.jpg_setup = jpeg.encode(self.np_image_info_setup_lines, quality=80)
            self.jpg_setup_base64 = base64.b64encode(self.jpg_setup)
            return True
        else:
            return False

    def create_thumbnail_jpg(self):
        np_image_info_tn = cv2.cvtColor(self.proc_image_centered, cv2.COLOR_GRAY2RGB)

        # rotate images for front/rear sensor
        if self.is_front_sensor:
            np_image_info_tn_rot = cv2.rotate(np_image_info_tn, 2)
        else:
            np_image_info_tn_rot = cv2.rotate(np_image_info_tn, 0)

        if self.edge_is_in_position.value:
            line_color = line_color_green
        else:
            line_color = line_color_orange

        tn_width = int(np_image_info_tn_rot.shape[1] * self.tn_scale_percent / 100)
        tn_height = int(np_image_info_tn_rot.shape[0] * self.tn_scale_percent / 100)
        self.image_info_tn = cv2.resize(np_image_info_tn_rot, (tn_width, tn_height))
        if self.is_front_sensor:
            edge_position_tn = self.edge_position.value * self.tn_scale_percent // 100
        else:
            edge_position_tn = tn_width - (self.edge_position.value * self.tn_scale_percent // 100)
        cv2.line(self.image_info_tn, (edge_position_tn, 0), (edge_position_tn, tn_height), line_color, 2)
        self.jpg_info_tn = jpeg.encode(self.image_info_tn, quality=80)
        self.jpg_info_tn_base64 = base64.b64encode(self.jpg_info_tn)

    def calc_statistics(self):
        if self.np_image_tile_left is not None and self.np_image_tile_right is not None:
            tile_height, tile_width = self.np_image_tile_left.shape
            stat_image_tile_left = self.np_image_tile_left[0:self.edge_position.value - 20, 0:tile_width]
            stat_image_tile_right = self.np_image_tile_right[0:self.edge_position.value - 20, 0:tile_width]
            self.stat_image_full = np.concatenate((stat_image_tile_left, stat_image_tile_right), axis=1)


    def autoFocusCamera(self):
        self.log_info_vSensor("Focus Camera...")
        self.autoFocusEnabled = True
        self.af_start_time = datetime.now()
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
        self.camCtrl.setAutoFocusTrigger()
        self.camera_control_queue.send(self.camCtrl)

    def setFocusValue(self, lens_position):
        self.log_info_vSensor("Set lens-position to: {}".format(lens_position))
        self.camCtrl = dai.CameraControl()
        # self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        self.camCtrl.setManualFocus(lens_position)
        self.camera_control_queue.send(self.camCtrl)

    @staticmethod
    def calc_edge_parameter(image):
        start_time = time.time()
        reduced = np.mean(image, axis=1)
        # compute the slope values at +/-3
        slope = [(reduced[i + 3] - reduced[i - 3]) for i in range(3, len(reduced) - 3, 1)]
        # compute the minVal, minPos, maxVal, maxPos of the slopes

        min_slope = min(slope)
        max_slope = max(slope)
        min_pos = slope.index(min_slope)
        max_pos = slope.index(max_slope)

        # prepare slopes and positions for output
        slope_and_pos_output = ([min_slope * -1, max_slope, min_pos, max_pos])
        return slope_and_pos_output

    def log_info_vSensor(self, message):
        message = str(message)
        if self.is_front_sensor:
            log_message = "[vs-front]" + " - " + message
        else:
            log_message = "[vs-rear ]" + " - " + message
        logging.info(log_message)


# Read Init-Configuration
logInfoGeneral("read init-config file..")
config_init = ConfigParser()
config_settings = ConfigParser()
config_init.read(init_config_file)
config_settings.read(settings_config_file)


def writeInitConfigToFile():
    logInfoGeneral("write init-config file..")
    with open(init_config_file, 'w') as configfile:
        config_init.write(configfile)


def writeSettingsConfigToFile():
    logInfoGeneral("write sensor-config file..")
    with open(settings_config_file, 'w') as configfile:
        config_settings.write(configfile)


if config_init.get("vs_front", "serial") == "":
    pass

devices_found = dai.DeviceBase.getAllAvailableDevices()
if len(devices_found) != 2:
    logInfoGeneral("found only one sensor... - exit program")
    exit()

for device in devices_found:
    logInfoGeneral("found sensor ({}) on state: {}".format(device.getMxId(), device.state))

found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_front_config_name, "serial"))
found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_rear_config_name, "serial"))

if not found_front_sensor or not found_rear_sensor:
    device = dai.Device.getAllAvailableDevices()
    if len(device) == 2:
        config_init.set(vs_front_config_name, "serial", device[0].getMxId())
        config_init.set(vs_rear_config_name, "serial", device[1].getMxId())
        writeInitConfigToFile()

    found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_front_config_name, "serial"))
    found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_init.get(vs_rear_config_name, "serial"))

if not found_rear_sensor:
    raise RuntimeError("RearSensor not found!")

if not found_rear_sensor:
    raise RuntimeError("RearSensor not found!")

logInfoGeneral("Connect to MQTT-Broker...")
mqtt = mqtt_communication_handler_v2.MqttHandler(logger_enabled=False, client_type="vsController")
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_swapSensors)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_focusCamera)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_focusCamera)
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_vsController_sensorVersion, "3")
mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(False))


@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsController_heartbeat, 1)


@tl.job(interval=timedelta(seconds=2))
def get_fps():
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_fps, vs_front.fps)
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_fps, vs_rear.fps)


vs_front = VisionSensor(device_info_front_sensor, True, vs_front_config_name)
vs_rear = VisionSensor(device_info_rear_sensor, False, vs_rear_config_name)

tl.start()

if config_settings.has_option(vs_front_config_name, "center_position"):
    vs_front.image_center_position = config_settings.getint(vs_front_config_name, "center_position")
if config_settings.has_option(vs_rear_config_name, "center_position"):
    vs_rear.image_center_position = config_settings.getint(vs_rear_config_name, "center_position")

if config_settings.has_option(vs_front_config_name, "proc_image_width"):
    vs_front.proc_image_width = config_settings.getint(vs_front_config_name, "proc_image_width")
if config_settings.has_option(vs_rear_config_name, "proc_image_width"):
    vs_rear.proc_image_width = config_settings.getint(vs_rear_config_name, "proc_image_width")

vs_front.lcm_enabled = vs_enable_low_contrast_mode.value
vs_rear.lcm_enabled = vs_enable_low_contrast_mode.value

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
    # line_slope_tile1, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_slope_tile[0], lw=1)
    # line_slope_tile2, = ax.plot(vs_front_chart_x_image_nr, vs_front_chart_slope_tile[1], lw=1)

with contextlib.ExitStack() as stack:
    while True:

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView):
            vs_front_enable_live_view.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsFront_liveViewIsEnabled, ef.bool2Str(vs_front_enable_live_view.value))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView):
            vs_rear_enable_live_view.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsRear_liveViewIsEnabled, ef.bool2Str(vs_rear_enable_live_view.value))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_fmCtrl_filmMoveDirection):
            film_move_direction.value = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_fmCtrl_filmMoveDirection))
            if film_move_direction.value == 1:
                vs_front.is_main_sensor = True
                vs_rear.is_main_sensor = False
            if film_move_direction.value == 2:
                vs_front.is_main_sensor = False
                vs_rear.is_main_sensor = True
            if film_move_direction.value == 0 and film_move_direction.last_value > 0:
                send_thumbnail.value = True

        vs_enable_low_contrast_mode.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_enable_low_contrast_mode))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setStopPosition):
            vs_front_stop_position.value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setStopPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getStopPosition, vs_front_stop_position.value)

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setStopPosition):
            vs_rear_stop_position.value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setStopPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getStopPosition, vs_rear_stop_position.value)

        if vs_enable_low_contrast_mode.is_different():
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled,
                              ef.bool2Str(vs_enable_low_contrast_mode.value))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_focusCamera):
            vs_front.autoFocusCamera()

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_focusCamera):
            vs_rear.autoFocusCamera()

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setFilmTypeIsNegative):
            film_type_is_negative.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_setFilmTypeIsNegative))
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_filmTypeIsNegative, ef.bool2Str(film_type_is_negative.value))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_swapSensors):
            config_init.set(vs_front_config_name, "serial", config_init.get(vs_rear_config_name, "serial"))
            config_init.set(vs_rear_config_name, "serial", config_init.get(vs_front_config_name, "serial"))
            write_init_config = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_enable_low_contrast_mode):
            if ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_enable_low_contrast_mode)):
                vs_front.lcm_enabled = True
                vs_rear.lcm_enabled = True
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(True))
            else:
                vs_front.lcm_enabled = False
                vs_rear.lcm_enabled = False
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(False))
            logInfoGeneral("set lcm_enabled to: {}".format(vs_front.lcm_enabled))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_captureImage) or mqtt.isNewMqttValueAvailable(
                mqtt.sTopics_vsController.get_vsRear_captureImage):
            send_thumbnail.value = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setProcImageWidth) or \
                mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setProcImageWidth):
            vs_front.proc_image_width = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setProcImageWidth))
            vs_rear.proc_image_width = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setProcImageWidth))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsFront_getProcImageWidth, vs_front.proc_image_width)
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsRear_getProcImageWidth, vs_rear.proc_image_width)
            config_settings.set(vs_front_config_name, "proc_image_width", str(vs_front.proc_image_width))
            config_settings.set(vs_rear_config_name, "proc_image_width", str(vs_rear.proc_image_width))
            update_setup_jpg = True
            write_settings_config = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setCenterPosition):
            vs_front.image_center_position = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setCenterPosition))
            logInfoGeneral("set centerPosition to: {}".format(vs_front.image_center_position))
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getCenterPosition, vs_front.image_center_position)
            config_settings.set(vs_front_config_name, "center_position", str(vs_front.image_center_position))
            update_setup_jpg = True
            write_settings_config = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setCenterPosition):
            vs_rear.image_center_position = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setCenterPosition))
            logInfoGeneral("set centerPosition to: {}".format(vs_front.image_center_position))
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getCenterPosition, vs_rear.image_center_position)
            config_settings.set(vs_rear_config_name, "center_position", str(vs_rear.image_center_position))
            update_setup_jpg = True
            write_settings_config = True



        # Process Front-Sensor
        if vs_front.process_image():
            if showOutput:
                vs_front.create_image_info()
                cv2.imshow("Sensor-Front - Processing", vs_front.np_image_info_edge_line)
                cv2.imshow("Sensor-Front - Tile Left", vs_front.np_image_tile_left)
                cv2.imshow("Sensor-Front - Tile Right", vs_front.np_image_tile_right)

                if enable_chart:
                    vs_front_chart_slope.append(vs_front.total_edge_slope)
                    vs_front_chart_edge_diff.append(vs_front.edge_position_tile_diff * 10)
                    vs_front_chart_x_image_nr.append(vs_front.captured_images)
                    vs_front_chart_edge_position.append(vs_front.edge_position.value)
                    vs_front_chart_in_contr.append(vs_front.in_pic_median_total // 2)
                    vs_front_chart_out_contr.append(vs_front.out_pic_median_total // 2)
                    vs_front_chart_total_slope_mean.append(vs_front.slope_total_mean)
                    vs_front_chart_edge_detected.append(vs_front.new_edge_detected)

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



        # Process Rear-Sensor
        if vs_rear.process_image():
            if showOutput:
                vs_rear.create_image_info()
                cv2.imshow("Sensor-Rear - Processing", vs_rear.np_image_info_edge_line)
                cv2.imshow("Sensor-Rear - Tile Left", vs_rear.np_image_tile_left)
                cv2.imshow("Sensor-Rear - Tile Right", vs_rear.np_image_tile_right)



        if vs_front.edge_position.is_different() or vs_rear.edge_position.is_different():

            # VS-Front - edge-position
            if vs_front.edge_position.is_different():
                # logInfoGeneral("new edge position: {}".format(vs_front.edge_position.value))
                if (vs_front.edge_position.value % mqtt_report_position_steps) == 0 or vs_front.edge_position.value == -1:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front.edge_position.value)

            # VS-Front - edge-detected
            if vs_front.edge_detected.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgeDetected, ef.bool2Str(vs_front.edge_detected.value))

            # VS-Front - edge-is-in-position
            if vs_front.edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_pictureIsInPosition,
                                  ef.bool2Str(vs_front.edge_is_in_position.value))

            # VS-Rear - edge-position
            if vs_rear.edge_position.is_different():
                # logInfoGeneral("{} // {}".format(vs_rear.edge_position.value, vs_front.edge_position.value))
                if (vs_rear.edge_position.value % mqtt_report_position_steps) == 0 or vs_rear.edge_position.value == -1:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_rear.edge_position.value)

            # VS-Rear - edge-detected
            if vs_rear.edge_detected.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgeDetected, ef.bool2Str(vs_rear.edge_detected.value))

            # VS-Rear - edge-is-in-position
            if vs_rear.edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_pictureIsInPosition, ef.bool2Str(vs_rear.edge_is_in_position.value))

            # Check if Image is Centered
            if vs_front.edge_position.value > -1 and vs_rear.edge_position.value > -1:

                image_is_centered.value = abs(vs_front.edge_position.value - vs_rear.edge_position.value) < 16 \
                                          and vs_rear.edge_is_in_position.value and vs_front.edge_is_in_position.value

                if image_is_centered.is_different():
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, ef.bool2Str(image_is_centered.value))
                    if image_is_centered.value:
                        logInfoGeneral("IMAGE CENTERED!! #################################")
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front.edge_position.value)
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_front.edge_position.value)
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive, vs_front.exposure_time.value)
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive, vs_rear.exposure_time.value)
                        # send_thumbnail = True

        print(str(vs_front.exposure_time.value) + " // " + str(vs_rear.exposure_time.value))

        if send_thumbnail.value:
            vs_front.calc_statistics()
            vs_rear.calc_statistics()
            if showOutput:
                cv2.imshow("Sensor-Front - Stat-Image", vs_front.stat_image_full)
                cv2.imshow("Sensor-Rear - Stat-Image", vs_rear.stat_image_full)
            vs_front.create_thumbnail_jpg()
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageDataTn, vs_front.jpg_info_tn_base64)
            # mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageData, vs_front.jp)
            vs_rear.create_thumbnail_jpg()
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageDataTn, vs_rear.jpg_info_tn_base64)
            if vs_front.exposure_time.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive, vs_front.exposure_time.value)
            if vs_rear.exposure_time.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive, vs_rear.exposure_time.value)
            send_thumbnail.value = False
            update_setup_jpg = True

        if update_setup_jpg:
            vs_front.create_image_info()
            vs_rear.create_image_info()
            vs_front.create_setup_jpg()
            vs_rear.create_setup_jpg()
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageData, vs_front.jpg_setup_base64)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageData, vs_rear.jpg_setup_base64)
            update_setup_jpg = False

        if debug_vs:
            pass

            # if vs_front.edge_is_in_position.is_different():
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_slope_tile1, vs_front.edge_slope_tile[0].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_slope_tile2, vs_front.edge_slope_tile[1].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_tile1, vs_front.edge_position_tile[0].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_tile2, vs_front.edge_position_tile[1].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_pip_debug, ef.bool2Str(vs_front.edge_detected.value))
            #
            # if vs_rear.edge_is_in_position.is_different():
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_slope_tile1, vs_rear.edge_slope_tile[0].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_slope_tile2, vs_rear.edge_slope_tile[1].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_tile1, vs_rear.edge_position_tile[0].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_tile2, vs_rear.edge_position_tile[1].value)
            #     mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_pip_debug, ef.bool2Str(vs_rear.edge_detected.value))

        if write_init_config:
            writeInitConfigToFile()
            write_init_config = False

        if write_settings_config:
            writeSettingsConfigToFile()
            write_settings_config = False

        vs_front_enable_live_view.reset()
        vs_rear_enable_live_view.reset()
        vs_enable_low_contrast_mode.reset()

        vs_front.edge_position.reset()
        vs_rear.edge_position.reset()

        vs_front.edge_detected.reset()
        vs_rear.edge_detected.reset()

        vs_front.edge_is_in_position.reset()
        vs_rear.edge_is_in_position.reset()

        vs_front.exposure_time.reset()
        vs_rear.exposure_time.reset()
        image_is_centered.reset()

        film_move_direction.reset()

        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.001)
