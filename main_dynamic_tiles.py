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

if platform.system() == "Windows":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

# import calc_mean as CalcMean

debug_vs = True
enable_chart = True

edge_detection_slope_lcm = 15
edge_detection_slope_std = 25
max_edge_position_diff = 10

config_file_init = "config/init.ini"
write_config = False

vs_front_config_name = "vs_front"
vs_rear_config_name = "vs_rear"

line_color_red = (0, 0, 255)
line_color_green = (0, 255, 0)
line_color_orange = (0, 165, 255)
line_color_blue = (255, 0, 0)

preview_width = 800
preview_height = 600

preview_height_crop_factor = 0.693333333
preview_height_crop = int(preview_height * preview_height_crop_factor)
if preview_height_crop % 32 != 0:
    print("preview_height_crop must be a multiple of 32")
    preview_height_crop = ((preview_height_crop // 32) + 1) * 32
    print("set preview_height_crop to {}".format(preview_height_crop))

thumbnail_width = 576
thumbnail_height = (thumbnail_width / 4) * 3 * preview_height_crop_factor

if thumbnail_width % 32 != 0:
    print("thumbnail_width must be a multiple of 32")

if thumbnail_height % 32 != 0:
    result = thumbnail_height / 32
    thumbnail_height = (int(result) + 1) * 32
    print(thumbnail_height)

camera_fps = 60

enable_live_view = False

reset_program = False

last_slope_value = 0



mqtt_report_position_steps = 5


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

vs_enable_low_contrast_mode = ValueHandler(False)

film_move_direction = ValueHandler(0)

send_thumbnail = ValueHandler(False)

s_half_frame_rate = """
    divide = 5
    curFrame = 0
    while True:
        frame = node.io['input'].get()
        if curFrame == 0:
            node.io['out'].send(frame)
        else:
            if curFrame == (divide - 1):
                curFrame = -1    
        curFrame += 1
    """

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


def update_chart(i):
    ax1.clear()
    ax1.plot(vs_front.chart_slope, vs_front.chart_x_image_nr)


class VisionSensor(ValueHandler):
    edge_detection_range = 20
    stop_position = 350
    lens_position = 160
    number_of_tiles = 2
    tn_scale_percent = 70
    chart_length = 2000

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
        self.enable_lcm = False
        self.edge_position_tile_diff = 0
        self.edge_detected = ValueHandler(False)
        self.edge_is_in_position = ValueHandler(False)


        # processing image variables
        self.proc_image_edge_data = None
        self.proc_image_edge = None
        self.proc_image_edge_cropped = None
        self.crop_image_right = 50
        self.crop_image_left = 50

        self.edge_position_tile = []
        self.edge_slope_tile = []
        self.edge_result_tile = []
        self.image_tile = []
        self.image_info_tile = []
        self.chart_slope_tile = []
        for i in range(self.number_of_tiles):
            self.edge_position_tile.append(ValueHandler(-1))
            self.edge_slope_tile.append(0)
            self.edge_result_tile.append(None)
            self.image_tile.append(None)
            self.image_info_tile.append(None)
            self.chart_slope_tile.append(deque(maxlen=self.chart_length))
            self.chart_slope_tile[i].append(0)

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
        self.fps_proc_image_time = datetime.now()
        self.fps_jpg_image_time = datetime.now()
        self.fps_proc_image = 0
        self.fps_jpg_image = 0
        self._fps_counter_proc_image = 0
        self._fps_counter_jpg_image = 0

        # camera control variables
        self.camCtrl = None
        self.exposure_time = ValueHandler(0)
        self.af_start_time = 0
        self.autoFocusEnabled = False

        self.stop_position_min = self.stop_position - (self.edge_detection_range // 2)
        self.stop_position_max = self.stop_position + (self.edge_detection_range // 2)

        self.pipeline = None
        self.getPipeline()
        self.device = dai.Device(self.pipeline, self.device_info)
        self.image_edge_queue = self.device.getOutputQueue(name="image_edge_detection", maxSize=4, blocking=True)
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=30, blocking=True)
        self.camera_control_queue = self.device.getInputQueue('control')

        if enable_chart:
            self.chart_slope = deque(maxlen=self.chart_length)
            self.chart_edge_diff = deque(maxlen=self.chart_length)
            self.chart_edge_detected = deque(maxlen=self.chart_length)
            self.chart_edge_position = deque(maxlen=self.chart_length)
            self.chart_x_image_nr = deque(maxlen=self.chart_length)
            self.chart_slope_mean = deque(maxlen=self.chart_length)
            self.cur_image = 0
            self.chart_slope.append(0)
            self.chart_edge_diff.append(0)
            self.chart_x_image_nr.append(0)
            self.chart_edge_detected.append(0)
            self.chart_edge_position.append(0)

    def getPipeline(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.jpgEncoder = self.pipeline.create(dai.node.VideoEncoder)
        self.manip_edge_detetction = self.pipeline.create(dai.node.ImageManip)
        self.full_crop = self.pipeline.create(dai.node.ImageManip)
        self.reduce_frame = self.pipeline.create(dai.node.Script)
        self.reduce_frame.setScript(s_half_frame_rate)
        self.reduce_frame.inputs['image'].setBlocking(True)
        self.reduce_frame.inputs['image'].setQueueSize(1)
        self.full_rotate = self.pipeline.create(dai.node.ImageManip)
        self.x_out_edge_detection = self.pipeline.create(dai.node.XLinkOut)
        # self.x_out_image2 = self.pipeline.create(dai.node.XLinkOut)
        self.x_out_video = self.pipeline.create(dai.node.XLinkOut)

        self.controlIn = self.pipeline.create(dai.node.XLinkIn)

        self.x_out_edge_detection.setStreamName('image_edge_detection')
        self.x_out_video.setStreamName('video')
        self.controlIn.setStreamName('control')

        # Properties
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setPreviewSize(preview_width, preview_height)
        self.camRgb.setFps(camera_fps)
        self.camRgb.initialControl.setAutoFocusLensRange(120, 180)
        self.camRgb.initialControl.setManualFocus(int(config_init.get(self.name, "lens_position")))
        self.camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        self.camRgb.setInterleaved(False)
        maxFrameSize = self.camRgb.getPreviewWidth() * self.camRgb.getPreviewHeight() * 3

        self.manip_edge_detetction.initialConfig.setCropRect(0, 0, 1, (self.stop_position_max + 20) / preview_height)
        self.manip_edge_detetction.setMaxOutputFrameSize(maxFrameSize)
        self.manip_edge_detetction.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        self.rr = dai.RotatedRect()
        self.rr.center.x, self.rr.center.y = preview_width // 2, preview_height_crop // 2
        self.rr.size.width, self.rr.size.height = preview_height_crop, preview_width
        if self.is_front_sensor:
            self.rr.angle = 90
        else:
            self.rr.angle = 270
        self.full_rotate.initialConfig.setCropRotatedRect(self.rr, False)
        self.full_rotate.initialConfig.setFrameType(dai.RawImgFrame.Type.NV12)

        self.jpgEncoder.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)

        # Links
        self.camRgb.preview.link(self.manip_edge_detetction.inputImage)
        self.camRgb.preview.link(self.reduce_frame.inputs['input'])

        self.manip_edge_detetction.out.link(self.x_out_edge_detection.input)

        self.reduce_frame.outputs['out'].link(self.full_rotate.inputImage)
        self.full_rotate.out.link(self.jpgEncoder.input)

        self.jpgEncoder.bitstream.link(self.x_out_video.input)

        self.controlIn.out.link(self.camRgb.inputControl)

    def calculate_edge_parameter(self, tile_id):
        self.image_tile.clear()
        self.proc_image_edge_data = self.image_edge_queue.tryGet()
        if self.proc_image_edge_data is not None:
            self.proc_image_edge = self.proc_image_edge_data.getCvFrame()
            (full_image_height, full_image_width) = self.proc_image_edge.shape[:2]
            self.proc_image_edge_cropped = self.proc_image_edge[0:full_image_height, self.crop_image_right:full_image_width - self.crop_image_left]
            (crop_image_height, crop_image_width) = self.proc_image_edge_cropped.shape[:2]
            tile_width = int(crop_image_width // self.number_of_tiles)
            for i in range(self.number_of_tiles):
                self.image_tile.append(self.proc_image_edge_cropped[0:full_image_height, (i * tile_width):(i * tile_width) + tile_width])

                self.edge_result_tile[i] = self.calc_edge_parameter(self.image_tile[i])
                # print(str(int(self.edge_result_tile[i][1])) + " // " + str(int(self.edge_result_tile[i][0])))
                if self.film_type_is_negative:
                    self.edge_position_tile[i] = self.edge_result_tile[i][3]
                    self.edge_slope_tile[i] = int(self.edge_result_tile[i][1])
                    if enable_chart:
                        self.chart_slope_tile[i].append(self.edge_result_tile[i][1])
                else:
                    self.edge_position_tile[i] = self.edge_result_tile[i][2]
                    self.edge_slope_tile[i] = int(self.edge_result_tile[i][0])

            self._fps_counter_proc_image += 1
            if (datetime.now() - self.fps_proc_image_time).seconds >= 10:
                self.fps_proc_image = self._fps_counter_proc_image // 10
                self.fps_proc_image_time = datetime.now()
                self._fps_counter_proc_image = 0
            if int(self.proc_image_edge_data.getExposureTime().total_seconds() * 1000000) != self.exposure_time:
                self.exposure_time.value = int(self.proc_image_edge_data.getExposureTime().total_seconds() * 1000000)
            if self.proc_image_edge_data.getLensPosition() != self.lens_position:
                self.lens_position = self.proc_image_edge_data.getLensPosition()
                self.log_info_vSensor("lens-position changed to: {}".format(self.lens_position))
            if self.autoFocusEnabled:
                if (datetime.now() - self.af_start_time).seconds > 2:
                    self.autoFocusEnabled = False
                    self.setFocusValue(self.lens_position)
                    self.log_info_vSensor("disable AutoFocus")
                    config_init.set(self.name, "lens_position", str(self.lens_position))
                    writeInitConfig()
            self.calc_edge_position()
            return True
        else:
            return False

    @staticmethod
    def maxPairDiff(arr):
        vmin = arr[0]
        dmax = 0
        for i in range(len(arr)):
            if (arr[i] < vmin):
                vmin = arr[i]
            elif (arr[i] - vmin > dmax):
                dmax = arr[i] - vmin
        return dmax

    def calc_edge_position(self):
        self.edge_position.value = -1

        self.edge_position_tile_diff = self.maxPairDiff(self.edge_position_tile)
        self.total_edge_slope = sum(self.edge_slope_tile) // len(self.edge_slope_tile)

        if self.edge_position_tile_diff < max_edge_position_diff:
            if not self.enable_lcm:
                if self.total_edge_slope > edge_detection_slope_std:
                    self.edge_position.value = sum(self.edge_position_tile) // len(self.edge_position_tile)
            else:
                if self.total_edge_slope > edge_detection_slope_lcm:
                    self.edge_position.value = sum(self.edge_position_tile) // len(self.edge_position_tile)

        if self.edge_position.value > 0:
            self.edge_detected.value = True
        else:
            self.edge_detected.value = False

        if self.edge_position.value > self.stop_position_min:
            self.edge_is_in_position.value = True
        else:
            self.edge_is_in_position.value = False

        self.last_edge_position = self.edge_position

        if enable_chart and self.is_front_sensor:
            self.chart_slope.append(self.total_edge_slope)
            self.chart_edge_diff.append(self.edge_position_tile_diff)
            self.cur_image = self.cur_image + 1
            self.chart_x_image_nr.append(self.cur_image)
            self.chart_edge_position.append(self.edge_position.value // 2)
            # print(self.chart_slope)
            if self.edge_detected.value:
                self.chart_edge_detected.append(150)
            else:
                self.chart_edge_detected.append(0)
            if self.cur_image % 10 == 0:
                ax.set_xlim(min(self.chart_x_image_nr), max(self.chart_x_image_nr))
                ax.set_ylim(0, 200)
                line_slope.set_data(self.chart_x_image_nr, self.chart_slope)
                # line_edge_detected.set_data(self.chart_x_image_nr, self.chart_edge_detected)
                # line_edge_position.set_data(self.chart_x_image_nr, self.chart_edge_position)
                # line_slope_tile2.set_data(self.chart_x_image_nr, self.chart_slope_tile[1])
                # line_slope_tile1.set_data(self.chart_x_image_nr, self.chart_slope_tile[0])

    def check_video(self):
        self.jpg_image_data = self.video_queue.tryGet()
        if self.jpg_image_data is not None:
            self.jpg_image = self.jpg_image_data.getData()
            self.jpg_video_base64 = base64.b64encode(self.jpg_image)
            self._fps_counter_jpg_image += 1
            if (datetime.now() - self.fps_jpg_image_time).seconds >= 10:
                self.fps_jpg_image = self._fps_counter_jpg_image // 10
                self.fps_jpg_image_time = datetime.now()
                self._fps_counter_jpg_image = 0
                # logInfoGeneral("video-FPS: {}".format(str(self.fps_jpg_image)))
            return True
        return False

    def create_image_info_tile(self):
        if self.proc_image_edge is not None:
            self.image_info_tile = cv2.cvtColor(self.proc_image_edge, cv2.COLOR_GRAY2RGB)
            img_height, img_width, dim = self.image_info_tile.shape
            cv2.line(self.image_info_tile, (self.crop_image_left, 0), (self.crop_image_left, img_height), line_color_blue, 1)
            cv2.line(self.image_info_tile, (img_width - self.crop_image_right, 0), (img_width - self.crop_image_right, img_height), line_color_blue, 1)
            center_image_line = ((img_width - self.crop_image_right - self.crop_image_left) // 2) + self.crop_image_left
            cv2.line(self.image_info_tile, (center_image_line, 0), (center_image_line, img_height), line_color_red, 1)
            if self.edge_is_in_position.value:
                line_color = line_color_green
            else:
                line_color = line_color_orange
            cv2.line(self.image_info_tile, (0, self.edge_position.value), (img_width, self.edge_position.value), line_color, 2)

    def create_image_info_fullsize(self, create_thumbnail):

        self.image_jpg_info = jpeg.decode(self.jpg_image)
        img_height, img_width, dim = self.image_jpg_info.shape

        if self.is_front_sensor:
            edge_position = self.edge_position.value
        else:
            edge_position = img_width - self.edge_position.value

        if self.edge_is_in_position.value:
            line_color = line_color_green
        else:
            line_color = line_color_orange

        cv2.line(self.image_jpg_info, (edge_position, 0), (edge_position, img_height), line_color, 2)

        if create_thumbnail:
            tn_width = int(self.image_jpg_info.shape[1] * self.tn_scale_percent / 100)
            tn_height = int(self.image_jpg_info.shape[0] * self.tn_scale_percent / 100)
            self.image_info_tn = cv2.resize(self.image_jpg_info, (tn_width, tn_height))
            self.jpg_info_tn = jpeg.encode(self.image_info_tn, quality=80)
            self.jpg_info_tn_base64 = base64.b64encode(self.jpg_info_tn)

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
config_init.read(config_file_init)


def writeInitConfig():
    logInfoGeneral("write init-config file..")
    with open(config_file_init, 'w') as configfile:
        config_init.write(configfile)


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
        writeInitConfig()

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


@tl.job(interval=timedelta(seconds=10))
def get_fps():
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_fps, vs_front.fps_proc_image)
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_fps, vs_rear.fps_proc_image)


tl.start()

vs_front = VisionSensor(device_info_front_sensor, True, vs_front_config_name)
vs_rear = VisionSensor(device_info_rear_sensor, False, vs_rear_config_name)

vs_front.enable_lcm = vs_enable_low_contrast_mode.value
vs_rear.enable_lcm = vs_enable_low_contrast_mode.value

if enable_chart:
    ed_value = 0
    plt.ion()
    fig, ax = plt.subplots()
    ax.grid()
    line_slope, = ax.plot(vs_front.chart_x_image_nr, vs_front.chart_slope, lw=2, label="slope", color="red")
    # line_edge_detected, = ax.plot(vs_front.chart_x_image_nr, vs_front.chart_edge_detected)
    # line_edge_position, = ax.plot(vs_front.chart_x_image_nr, vs_front.chart_edge_position)
    line_slope_tile1, = ax.plot(vs_front.chart_x_image_nr, vs_front.chart_slope_tile[0], lw=1)
    line_slope_tile2, = ax.plot(vs_front.chart_x_image_nr, vs_front.chart_slope_tile[1], lw=1)


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
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(vs_enable_low_contrast_mode.value))

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
            write_config = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_enable_low_contrast_mode):
            if ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_enable_low_contrast_mode)):
                vs_front.enable_lcm = True
                vs_rear.enable_lcm = True
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(True))
            else:
                vs_front.enable_lcm = False
                vs_rear.enable_lcm = False
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(False))
            logInfoGeneral("set enable_lcm to: {}".format(vs_front.enable_lcm))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_captureImage) or mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_captureImage):
            send_thumbnail.value = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setCropLeft):
            vs_front.crop_image_left = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setCropLeft))
            config_init.set(vs_front_config_name, "crop_left_position", str(vs_front.crop_image_left))
            write_config = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setCropRight):
            vs_front.crop_image_right = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setCropRight))
            config_init.set(vs_front_config_name, "crop_right_position", str(vs_front.crop_image_right))
            write_config = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setCropLeft):
            vs_rear.crop_image_left = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setCropLeft))
            config_init.set(vs_rear_config_name, "crop_left_position", str(vs_rear.crop_image_left))
            write_config = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setCropRight):
            vs_rear.crop_image_right = int(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setCropRight))
            config_init.set(vs_rear_config_name, "crop_right_position", str(vs_rear.crop_image_right))
            write_config = True

        # Process Front-Sensor
        if vs_front.calculate_edge_parameter(0):
            if showOutput:
                vs_front.create_image_info_tile()
                cv2.imshow("Sensor-Front - Processing", vs_front.image_info_tile)

        if vs_front.check_video():

            if vs_front_enable_live_view.value:
                vs_front.create_image_info_fullsize(True)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageDataTn, vs_front.jpg_info_tn_base64)

            if showOutput:
                pass

        # Process Rear-Sensor
        if vs_rear.calculate_edge_parameter(0):
            if showOutput:
                vs_rear.create_image_info_tile()
                cv2.imshow("Sensor-Rear - Processing", vs_rear.image_info_tile)

        if vs_rear.check_video():
            if vs_rear_enable_live_view.value:
                vs_rear.create_image_info_fullsize(True)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageDataTn, vs_rear.jpg_info_tn_base64)

            if showOutput:

                if vs_front.image_jpg_info is not None and vs_rear.image_jpg_info is not None:
                    i0_complete = np.concatenate((vs_rear.image_jpg_info, vs_front.image_jpg_info), axis=1)
                    cv2.imshow("Video / Sensor-Front", i0_complete)

                if vs_front.image_info_tn is not None and vs_rear.image_info_tn is not None:
                    i0_complete = np.concatenate((vs_rear.image_info_tn, vs_front.image_info_tn), axis=1)
                    cv2.imshow("VideoTN / Sensor-Front", i0_complete)

        if vs_front.edge_position.is_different() or vs_rear.edge_position.is_different():

            # VS-Front - edge-position
            if vs_front.edge_position.is_different():
                # logInfoGeneral("new edge position: {}".format(vs_front.edge_position.value))
                if (vs_front.edge_position.value % mqtt_report_position_steps) == 0:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front.edge_position.value)

            # VS-Front - edge-detected
            if vs_front.edge_detected.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgeDetected, ef.bool2Str(vs_front.edge_detected.value))

            # VS-Front - edge-is-in-position
            if vs_front.edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_pictureIsInPosition, ef.bool2Str(vs_front.edge_is_in_position.value))

            # VS-Rear - edge-position
            if vs_rear.edge_position.is_different():
                # logInfoGeneral("{} // {}".format(vs_rear.edge_position.value, vs_front.edge_position.value))
                if (vs_rear.edge_position.value % mqtt_report_position_steps) == 0:
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
                        # send_thumbnail = True

        if send_thumbnail.value:
            vs_front.create_image_info_fullsize(True)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageDataTn, vs_front.jpg_info_tn_base64)
            # mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageData, vs_front.jp)
            vs_rear.create_image_info_fullsize(True)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageDataTn, vs_rear.jpg_info_tn_base64)
            if vs_front.exposure_time.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive, vs_front.exposure_time.value)
            if vs_rear.exposure_time.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive, vs_rear.exposure_time.value)
            send_thumbnail.value = False

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

        if abs(vs_front.total_edge_slope - last_slope_value) > 10:
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_slope_tile1, vs_front.total_edge_slope)

        # last_slope_value = vs_front.total_edge_slope.value

        if write_config:
            writeInitConfig()
            write_config = False

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
