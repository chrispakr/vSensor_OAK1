#!/usr/bin/env python3
from typing import List

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

# import paho.mqtt.client as mqtt
import base64
from timeloop import Timeloop
import logging
from datetime import datetime, timedelta
import mqtt_communication_handler_v2
import libs.functions as ef
from configparser import ConfigParser
from turbojpeg import TurboJPEG

# import calc_mean as CalcMean

debug_vs = False

edge_detection_slope = 15

config_file_init = "config/init.ini"

vs_front_config_name = "vs_front"
vs_rear_config_name = "vs_rear"

line_color_red = (0, 0, 255)
line_color_green = (0, 255, 0)
line_color_orange = (0, 165, 255)

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

enable_live_view = True
# center_image = True

current_film_move_direction = 0
current_move_command = 0
last_current_move_command = 0
reset_program = False

picture_in_position = False
last_picture_in_position = False

mqtt_position_steps = 5


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


vs_front_enable_live_view = ValueHandler(True)
vs_rear_enable_live_view = ValueHandler(True)

image_is_centered = ValueHandler(False)

film_type_is_negative = ValueHandler(True)

vs_enable_low_contrast_mode = ValueHandler(False)

# vs_front_stop_position = ValueHandler(350)
# vs_rear_stop_position = ValueHandler(350)

vs_whitebalance = ValueHandler(0)
vs_whitebalance.value = 4000

vs_front_lens_position = ValueHandler(0)
vs_rear_lens_position = ValueHandler(0)

vs_front_lens_position.value = 160
vs_rear_lens_position.value = 160

vs_is_initialized = False

# vs_front_image_cutoff_position = vs_front_stop_position.value + cutoff_offset
# vs_rear_image_cutoff_position = vs_rear_stop_position.value + cutoff_offset

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


class VisionSensor(ValueHandler):
    edge_detection_range = 20
    stop_position = 350
    lens_position = 160
    number_of_tiles = 2
    tn_scale_percent = 70

    def __init__(self, device_info, is_front_sensor, vs_name):
        self.name = vs_name
        self.device_info = device_info
        self.is_front_sensor = is_front_sensor
        self.film_type_is_negative = True
        self.edge_position = ValueHandler(-1)
        self.edge_detected = ValueHandler(False)
        self.edge_is_in_position = ValueHandler(False)
        self.video_data = None
        self.jpg_video = None
        self.image_video_info = None
        self.image_info_tn = None
        self.jpg_info_tn = None
        self.jpg_info_tn_base64 = None
        # self.image_info_tn = None
        self.edge_position_tile = []
        self.slope_tile = []
        self.edge_result_tile = []
        self.tile_data = []
        self.image_tile = []
        self.tile_queue = []
        self.fps_tile_time = []
        self.fps_tile = []
        self._fps_counter_tile = []
        self.image_info_tile = []
        for i in range(self.number_of_tiles):
            self.edge_position_tile.append(ValueHandler(-1))
            self.slope_tile.append(ValueHandler(0))
            self.edge_result_tile.append(None)
            self.tile_data.append(None)
            self.image_tile.append(None)
            self.fps_tile_time.append(datetime.now())
            self.fps_tile.append(0)
            self._fps_counter_tile.append(0)
            self.image_info_tile.append(None)
        self.exposure_time = ValueHandler(0)
        self.jpg_video_base64 = None
        self._fps_counter_video = 0
        self.fps_video = 0
        self.fps_video_time = datetime.now()
        self.camCtrl = None
        self.stop_position_min = self.stop_position - (self.edge_detection_range // 2)
        self.stop_position_max = self.stop_position + (self.edge_detection_range // 2)
        self.pipeline = None
        self.getPipeline()
        self.device = dai.Device(self.pipeline, self.device_info)
        self.tile_queue.append(self.device.getOutputQueue(name="out_image_tile1", maxSize=4, blocking=True))
        self.tile_queue.append(self.device.getOutputQueue(name="out_image_tile2", maxSize=4, blocking=True))
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=30, blocking=True)
        self.camera_control_queue = self.device.getInputQueue('control')
        self.af_start_time = 0
        self.autoFocusEnabled = False

    def getPipeline(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.crop_isp_image = self.pipeline.create((dai.node.ImageManip))
        self.videoEnc = self.pipeline.create(dai.node.VideoEncoder)
        self.tile1_crop = self.pipeline.create(dai.node.ImageManip)
        self.tile2_crop = self.pipeline.create(dai.node.ImageManip)
        self.full_crop = self.pipeline.create(dai.node.ImageManip)
        self.reduce_frame = self.pipeline.create(dai.node.Script)
        self.reduce_frame.setScript(s_half_frame_rate)
        self.reduce_frame.inputs['image'].setBlocking(True)
        self.reduce_frame.inputs['image'].setQueueSize(1)
        self.full_rotate = self.pipeline.create(dai.node.ImageManip)
        self.x_out_image1 = self.pipeline.create(dai.node.XLinkOut)
        self.x_out_image2 = self.pipeline.create(dai.node.XLinkOut)
        self.x_out_video = self.pipeline.create(dai.node.XLinkOut)

        self.controlIn = self.pipeline.create(dai.node.XLinkIn)

        self.x_out_image1.setStreamName('out_image_tile1')
        self.x_out_image2.setStreamName('out_image_tile2')
        self.x_out_video.setStreamName('video')
        self.controlIn.setStreamName('control')

        # Properties
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setPreviewSize(preview_width, preview_height)
        self.camRgb.setFps(camera_fps)
        self.camRgb.initialControl.setAutoFocusLensRange(120, 180)
        self.camRgb.initialControl.setManualFocus(int(config_init.get(self.name, "lens_position")))
        # self.camRgb.initialControl.setManualWhiteBalance(4000)
        # camRgb.setCropRect(cropX, cropY, 0, 0)
        # camRgb.setIspScale(2, 3)
        self.camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        self.camRgb.setInterleaved(False)
        maxFrameSize = self.camRgb.getPreviewWidth() * self.camRgb.getPreviewHeight() * 3

        self.crop_isp_image.initialConfig.setCropRect(0.15, 0, 0.85, 0.5)
        # self.crop_isp_image.setResizeThumbnail(800, 600)
        # self.crop_isp_image.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        self.tile1_crop.initialConfig.setCropRect(0, 0, 0.5, (self.stop_position_max + 20) / preview_height)
        # self.tile1_crop.setMaxOutputFrameSize(maxFrameSize)
        self.tile1_crop.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        self.tile2_crop.initialConfig.setCropRect(0.5, 0, 1, (self.stop_position_max + 20) / preview_height)
        # self.tile2_crop.setMaxOutputFrameSize(maxFrameSize)
        self.tile2_crop.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        # self.full_crop.initialConfig.setCropRect(0.1, 0, 1, preview_height_crop_factor)
        # self.full_crop.initialConfig.setResizeThumbnail(thumbnail_width, thumbnail_height)
        # full_crop.setMaxOutputFrameSize(maxFrameSize)

        # full_rotate.setMaxOutputFrameSize(maxFrameSize)
        self.rr = dai.RotatedRect()
        self.rr.center.x, self.rr.center.y = preview_width // 2, preview_height_crop // 2
        self.rr.size.width, self.rr.size.height = preview_height_crop, preview_width
        if self.is_front_sensor:
            self.rr.angle = 90
        else:
            self.rr.angle = 270
        self.full_rotate.initialConfig.setCropRotatedRect(self.rr, False)
        self.full_rotate.initialConfig.setFrameType(dai.RawImgFrame.Type.NV12)

        self.videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)

        # Links
        self.camRgb.isp.link(self.crop_isp_image.inputImage)
        # self.camRgb.isp.link(self.tile2_crop.inputImage)
        self.camRgb.preview.link(self.reduce_frame.inputs['input'])

        self.crop_isp_image.preview.link(self.tile1_crop.inputImage)
        self.crop_isp_image.preview.link(self.tile2_crop.inputImage)

        self.tile1_crop.out.link(self.x_out_image1.input)
        self.tile2_crop.out.link(self.x_out_image2.input)

        self.reduce_frame.outputs['out'].link(self.full_rotate.inputImage)
        # self.full_crop.out.link(self.full_rotate.inputImage)
        self.full_rotate.out.link(self.videoEnc.input)

        self.videoEnc.bitstream.link(self.x_out_video.input)

        self.controlIn.out.link(self.camRgb.inputControl)

        # return pipeline

    def check_tile(self, tile_id):
        self.tile_data[tile_id] = self.tile_queue[tile_id].tryGet()
        if self.tile_data[tile_id] is not None:
            # txt = f"[{self.tile1_data.getSequenceNum()}] "
            # txt += f"Exposure: {self.tile1_data.getExposureTime().total_seconds() * 1000:.3f} ms, "
            # txt += f"ISO: {self.tile1_data.getSensitivity()}, "
            # txt += f"Lens position: {self.tile1_data.getLensPosition()}, "
            # txt += f"Color temp: {self.tile1_data.getColorTemperature()} K"
            # print(txt)
            self.image_tile[tile_id] = self.tile_data[tile_id].getCvFrame()
            self.edge_result_tile[tile_id] = self.calc_edge_parameter(self.image_tile[tile_id])
            if self.film_type_is_negative:
                self.edge_position_tile[tile_id].value = self.edge_result_tile[tile_id][3]
                self.slope_tile[tile_id].value = int(self.edge_result_tile[tile_id][1])
            else:
                self.edge_position_tile[tile_id].value = self.edge_result_tile[tile_id][2]
                self.slope_tile[tile_id].value = int(self.edge_result_tile[tile_id][0])
            self._fps_counter_tile[tile_id] += 1
            if (datetime.now() - self.fps_tile_time[tile_id]).seconds >= 10:
                self.fps_tile[tile_id] = self._fps_counter_tile[tile_id] // 10
                self.fps_tile_time[tile_id] = datetime.now()
                self._fps_counter_tile[tile_id] = 0
            if int(self.tile_data[tile_id].getExposureTime().total_seconds() * 1000000) != self.exposure_time:
                self.exposure_time.value = int(self.tile_data[tile_id].getExposureTime().total_seconds() * 1000000)
            if self.tile_data[tile_id].getLensPosition() != self.lens_position:
                self.lens_position = self.tile_data[tile_id].getLensPosition()
                self.log_info_vSensor("lens-position changed to: {}".format(self.lens_position))
            if self.autoFocusEnabled:
                if (datetime.now() - self.af_start_time).seconds > 2:
                    self.autoFocusEnabled = False
                    self.setFocusValue(self.lens_position)
                    self.log_info_vSensor("disable AutoFocus")
                    config_init.set(self.name, "lens_position", str(self.lens_position))
                    writeInitConfig()
            return True
        else:
            return False

    def calc_edge_position(self):
        self.edge_position.value = -1
        if abs(self.edge_position_tile[0].value - self.edge_position_tile[1].value) < 15:
            if (self.slope_tile[0].value + self.slope_tile[1].value) > edge_detection_slope:
                self.edge_position.value = (self.edge_position_tile[0].value + self.edge_position_tile[1].value) // 2

        if self.edge_position.value > 0:
            self.edge_detected.value = True
        else:
            self.edge_detected.value = False

        if self.edge_position.value > self.stop_position_min:
            self.edge_is_in_position.value = True
        else:
            self.edge_is_in_position.value = False

    def check_video(self):
        self.video_data = self.video_queue.tryGet()
        if self.video_data is not None:
            self.jpg_video = self.video_data.getData()
            self.jpg_video_base64 = base64.b64encode(self.jpg_video)
            self._fps_counter_video += 1
            if (datetime.now() - self.fps_video_time).seconds >= 10:
                self.fps_video = self._fps_counter_video // 10
                self.fps_video_time = datetime.now()
                self._fps_counter_video = 0
                # logInfoGeneral("video-FPS: {}".format(str(self.fps_jpg_image)))
            return True
        return False

    def create_image_info_tile(self, tile_id):
        if self.image_tile[tile_id] is not None:
            self.image_info_tile[tile_id] = cv2.cvtColor(self.image_tile[tile_id], cv2.COLOR_GRAY2RGB)
            img_height, img_width, dim = self.image_info_tile[tile_id].shape
            if self.edge_is_in_position.value:
                line_color = line_color_green
            else:
                line_color = line_color_orange
            cv2.line(self.image_info_tile[tile_id], (0, self.edge_position.value), (img_width, self.edge_position.value), line_color, 2)

    def create_image_info_fullsize(self, create_thumbnail):
        self.log_info_vSensor("create image fullsize - start")

        self.image_video_info = jpeg.decode(self.jpg_video)
        img_height, img_width, dim = self.image_video_info.shape

        if self.is_front_sensor:
            edge_position = self.edge_position.value
        else:
            edge_position = img_width - self.edge_position.value

        if self.edge_is_in_position.value:
            line_color = line_color_green
        else:
            line_color = line_color_orange

        cv2.line(self.image_video_info, (edge_position, 0), (edge_position, img_height), line_color, 2)
        if create_thumbnail:
            tn_width = int(self.image_video_info.shape[1] * self.tn_scale_percent / 100)
            tn_height = int(self.image_video_info.shape[0] * self.tn_scale_percent / 100)
            self.image_info_tn = cv2.resize(self.image_video_info, (tn_width, tn_height))
            self.jpg_info_tn = jpeg.encode(self.image_info_tn, quality=80)
            self.jpg_info_tn_base64 = base64.b64encode(self.jpg_info_tn)
        self.log_info_vSensor("create image fullsize - end")

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


@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsController_heartbeat, 1)


@tl.job(interval=timedelta(seconds=10))
def get_fps():
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_fps, vs_front.fps_tile[0])
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_fps, vs_rear.fps_tile[1])


tl.start()


with contextlib.ExitStack() as stack:

    vs_front = VisionSensor(device_info_front_sensor, True, vs_front_config_name)
    vs_rear = VisionSensor(device_info_rear_sensor, False, vs_rear_config_name)

    while True:

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView):
            vs_front_enable_live_view.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsFront_liveViewIsEnabled, ef.bool2Str(vs_front_enable_live_view.value))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView):
            vs_rear_enable_live_view.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView))
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsRear_liveViewIsEnabled, ef.bool2Str(vs_rear_enable_live_view.value))

        # vs_enable_low_contrast_mode.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_enable_low_contrast_mode))

        # if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setStopPosition):
        #     vs_front_stop_position.value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setStopPosition)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getStopPosition, vs_front_stop_position.value)
        #
        # if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setStopPosition):
        #     vs_rear_stop_position.value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setStopPosition)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getStopPosition, vs_rear_stop_position.value)

        # if vs_enable_low_contrast_mode.is_different():
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(vs_enable_low_contrast_mode.value))

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
            writeInitConfig()

        # if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_centerImage):
        #     center_image = ef.str2bool(mqtt.sTopics_vsController.get_vsCtrl_centerImage)
        #     mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_centerImage, ef.bool2Str(center_image))
        #     logInfoGeneral("set centerImage to: {}".format(center_image))

        # Process Front-Sensor
        if vs_front.check_tile(0):
            if showOutput:
                vs_front.create_image_info_tile(0)

        if vs_front.check_tile(1):
            vs_front.calc_edge_position()

            if showOutput:
                vs_front.create_image_info_tile(1)
                if vs_front.image_info_tile[0] is not None and vs_front.image_info_tile[1] is not None:
                    img_complete_front = np.concatenate((vs_front.image_info_tile[0], vs_front.image_info_tile[1]), axis=1)
                    cv2.imshow("Sensor-Front - Processing", img_complete_front)

        if vs_front.check_video():

            if vs_front_enable_live_view.value:
                vs_front.create_image_info_fullsize(True)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageDataTn, vs_front.jpg_info_tn_base64)

            if showOutput:
                pass

        # Process Rear-Sensor
        if vs_rear.check_tile(0):
            if showOutput:
                vs_rear.create_image_info_tile(0)

        if vs_rear.check_tile(1):
            vs_rear.calc_edge_position()
            if showOutput:
                vs_rear.create_image_info_tile(1)
                if vs_rear.image_info_tile[0] is not None and vs_rear.image_info_tile[1] is not None:
                    img_complete_rear = np.concatenate((vs_rear.image_info_tile[0], vs_rear.image_info_tile[1]), axis=1)
                    cv2.imshow("Sensor-Rear - Processing", img_complete_rear)

        if vs_rear.check_video():
            if vs_rear_enable_live_view.value:
                vs_rear.create_image_info_fullsize(True)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageDataTn, vs_rear.jpg_info_tn_base64)

            if showOutput:

                if vs_front.image_video_info is not None and vs_rear.image_video_info is not None:
                    i0_complete = np.concatenate((vs_rear.image_video_info, vs_front.image_video_info), axis=1)
                    cv2.imshow("Video / Sensor-Front", i0_complete)

                if vs_front.image_info_tn is not None and vs_rear.image_info_tn is not None:
                    i0_complete = np.concatenate((vs_rear.image_info_tn, vs_front.image_info_tn), axis=1)
                    cv2.imshow("VideoTN / Sensor-Front", i0_complete)

        if vs_front.exposure_time.is_different():
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getExposureTimeLive, vs_front.exposure_time.value)

        if vs_rear.exposure_time.is_different():
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getExposureTimeLive, vs_rear.exposure_time.value)

        if vs_front.edge_position.is_different() or vs_rear.edge_position.is_different():

            # VS-Front - edge-position
            if vs_front.edge_position.is_different():
                if (vs_front.edge_position.value % mqtt_position_steps) == 0:
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
                if (vs_rear.edge_position.value % mqtt_position_steps) == 0:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_rear.edge_position.value)

            # VS-Rear - edge-detected
            if vs_rear.edge_detected.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgeDetected, ef.bool2Str(vs_rear.edge_detected.value))

            # VS-Rear - edge-is-in-position
            if vs_rear.edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_pictureIsInPosition, ef.bool2Str(vs_rear.edge_is_in_position.value))

            # Check if Image is Centered
            if vs_front.edge_position.value > -1 and vs_rear.edge_position.value > -1:

                image_is_centered.value = abs(vs_front.edge_position.value - vs_rear.edge_position.value) < 10 \
                                          and vs_rear.edge_is_in_position.value and vs_front.edge_is_in_position.value



                if image_is_centered.is_different():
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, image_is_centered.value)
                    if image_is_centered.value:
                        logInfoGeneral("IMAGE CENTERED!! #################################")
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front.edge_position.value)
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_front.edge_position.value)
                        vs_front.create_image_info_fullsize(True)
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageDataTn, vs_front.jpg_info_tn_base64)
                        vs_rear.create_image_info_fullsize(True)
                        mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageDataTn, vs_rear.jpg_info_tn_base64)

        if debug_vs:
            if vs_front_edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_slope_tile1, vs_front_slope_tile1.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_slope_tile2, vs_front_slope_tile2.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_tile1, vs_front_edge_position_tile1.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition_tile2, vs_front_edge_position_tile2.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_pip_debug, ef.bool2Str(vs_front_edge_detected.value))

            if vs_rear_edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_slope_tile1, vs_rear_slope_tile1.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_slope_tile2, vs_rear_slope_tile2.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_tile1, vs_rear_edge_position_tile1.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition_tile2, vs_rear_edge_position_tile2.value)
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_pip_debug, ef.bool2Str(vs_rear_edge_detected.value))

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

        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.001)
