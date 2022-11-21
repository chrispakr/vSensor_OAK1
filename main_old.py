#!/usr/bin/env python3

import cv2
import io
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
from datetime import timedelta
import mqtt_communication_handler_v2
import libs.functions as ef
from configparser import ConfigParser

# import calc_mean as CalcMean

showOutput = True
debug_vs = True

fps_sensor_front = 0
fps_edge1_sensor_front = 0
fps_edge2_sensor_front = 0
fps_preview_sensor_front = 0

fps_sensor_rear = 0
fps_edge1_sensor_rear = 0
fps_edge2_sensor_rear = 0
fps_preview_sensor_rear = 0

# edge_detection_slope_std = 35
edge_detection_slope = 15


config_file_init = "config/init.ini"

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

vs_front_image_tile1 = None
vs_front_image_tile2 = None

vs_rear_image_tile1 = None
vs_rear_image_tile2 = None

enable_live_view = True
center_image = True

edge_parameter_rear_tile1 = [0, 0, 0, 0]
edge_parameter_front_tile1 = [0, 0, 0, 0]

current_film_move_direction = 0
current_move_command = 0
last_current_move_command = 0
reset_program = False

picture_in_position = False
last_picture_in_position = False

sensor_fw = False

cur_fmc_state = 0
last_fmc_State = 0

# vs_front_stop_position = 350 - 10
# vs_rear_stop_position = 350 - 10

mqtt_position_steps = 5

# Step size ('W','A','S','D' controls)
STEP_SIZE = 8
# Manual exposure/focus/white-balance set step
EXP_STEP = 500  # us
ISO_STEP = 50
LENS_STEP = 3
WB_STEP = 200


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

vs_front_edge_is_in_position = ValueHandler(False)
vs_rear_edge_is_in_position = ValueHandler(False)

vs_front_edge_position = ValueHandler(-1)
vs_rear_edge_position = ValueHandler(0)

vs_front_edge_detected = ValueHandler(False)
vs_rear_edge_detected = ValueHandler(False)

image_is_centered = ValueHandler(False)

film_type_is_negative = ValueHandler(True)

vs_front_slope_tile1 = ValueHandler(0)
vs_front_slope_tile2 = ValueHandler(0)
vs_rear_slope_tile1 = ValueHandler(0)
vs_rear_slope_tile2 = ValueHandler(0)

vs_front_edge_position_tile1 = ValueHandler(0)
vs_front_edge_position_tile2 = ValueHandler(0)
vs_rear_edge_position_tile1 = ValueHandler(0)
vs_rear_edge_position_tile2 = ValueHandler(0)

vs_enable_low_contrast_mode = ValueHandler(False)

vs_front_auto_focus = ValueHandler(False)
vs_rear_auto_focus = ValueHandler(False)

edge_detection_range = 20
cutoff_offset = 30

vs_front_stop_position = ValueHandler(350)
vs_rear_stop_position = ValueHandler(350)

vs_whitebalance = ValueHandler(0)
vs_whitebalance.value = 4000

vs_front_lens_position = ValueHandler(0)
vs_rear_lens_position = ValueHandler(0)

vs_front_lens_position.value = 160
vs_rear_lens_position.value = 160

vs_is_initialized = False

vs_front_stop_position_min = vs_front_stop_position.value - (edge_detection_range // 2)
vs_rear_stop_position_min = vs_rear_stop_position.value - (edge_detection_range // 2)

vs_front_stop_position_max = vs_front_stop_position.value + (edge_detection_range // 2)
vs_rear_stop_position_max = vs_rear_stop_position.value + (edge_detection_range // 2)

# vs_front_image_cutoff_position = vs_front_stop_position.value + cutoff_offset
# vs_rear_image_cutoff_position = vs_rear_stop_position.value + cutoff_offset

s_half_frame_rate = """
    divide = 4
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

mqtt = mqtt_communication_handler_v2.MqttHandler(logger_enabled=False, client_type="vsController")
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_swapSensors)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_focusCamera)
mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_focusCamera)

tl = Timeloop()


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[logging.FileHandler("vSensor.log"),
              logging.StreamHandler(sys.stdout)]
)


class VisionSensor:
    def __init__(self, device_info, is_front_sensor):
        self.device_info = device_info
        self.is_front_sensor = is_front_sensor
        if self.is_front_sensor:
            self.pipeline = getPipeline(is_front_sensor=True)
        else:
            self.pipeline = getPipeline(is_front_sensor=False)
        self.device = dai.Device(self.pipeline, self.device_info)
        self.tile1_queue = self.device.getOutputQueue(name="out_image_tile1", maxSize=4, blocking=True)
        self.tile2_queue = self.device.getOutputQueue(name="out_image_tile2", maxSize=4, blocking=True)
        self.video_queue = selfdevice.getOutputQueue(name="video", maxSize=30, blocking=True)
        self.camera_control_queue = device.getInputQueue('control')




def logInfoGeneral(message):
    log_message = "[general]" + " - " + message
    logging.info(log_message)


def detect_edge_on_image(image, roi=None, img_v_flip=False, img_h_mirror=False, img_transpose=False):
    bri1 = 0
    bri2 = 0
    if roi is not None:
        extract = cv2.get_image_from_roi(image, roi)
        extract1 = cv2.replace(extract, img_h_mirror, img_v_flip, img_transpose)
        reduced = cv2.mean_pool(extract1)
    else:
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


def get_edge_position(image1, image2, film_is_negative=False):
    edge_position = -1
    edge_position_tile1 = -1
    edge_position_tile2 = -1
    slope_tile1 = 0
    slope_tile2 = 0
    if image1 is not None and image2 is not None:
        edge1 = detect_edge_on_image(image1)
        edge2 = detect_edge_on_image(image2)

        if film_is_negative:
            if abs(edge1[3] - edge2[3]) < 15:
                if (edge1[1] + edge2[1]) > edge_detection_slope:
                    edge_position = (edge1[3] + edge2[3]) // 2
            slope_tile1 = int(edge1[1])
            slope_tile2 = int(edge2[1])
            edge_position_tile1 = edge1[3]
            edge_position_tile2 = edge2[3]
        else:
            if abs(edge1[2] - edge2[2]) < 15:
                if (edge1[0] + edge1[0]) > edge_detection_slope:
                    edge_position = (edge1[2] + edge2[2]) // 2
            slope_tile1 = int(edge1[0])
            slope_tile2 = int(edge2[0])
            edge_position_tile1 = edge1[2]
            edge_position_tile2 = edge2[2]

    return edge_position, edge_position_tile1, edge_position_tile2, slope_tile1, slope_tile2


# Read Init-Configuration
logInfoGeneral("read init-config file..")
config_init = ConfigParser()
config_init.read(config_file_init)


def writeInitConfig(vs_rear_serial, vs_front_serial):
    logInfoGeneral("write init-config file..")
    config_init.set("vs_front", "serial", vs_front_serial)
    config_init.set("vs_rear", "serial", vs_rear_serial)
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


found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_init.get("vs_front", "serial"))
found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_init.get("vs_rear", "serial"))

if not found_front_sensor or not found_rear_sensor:
    device = dai.Device.getAllAvailableDevices()
    if len(device) == 2:
        writeInitConfig(device[0].getMxId(), device[1].getMxId())

    found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_init.get("vs_front", "serial"))
    found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_init.get("vs_rear", "serial"))

if not found_rear_sensor:
    raise RuntimeError("RearSensor not found!")

if not found_rear_sensor:
    raise RuntimeError("RearSensor not found!")





@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsController_heartbeat, 1)
    # print("heartbeat")


@tl.job(interval=timedelta(seconds=10))
def get_fps():
    global fps_sensor_front
    global fps_sensor_rear
    global fps_edge1_sensor_front
    global fps_edge2_sensor_front
    global fps_preview_sensor_front
    global fps_edge1_sensor_rear
    global fps_edge2_sensor_rear
    global fps_preview_sensor_rear
    fps_sensor_front = int(fps_edge1_sensor_front / 10)
    fps_sensor_rear = int(fps_edge1_sensor_rear / 10)
    # print(f"fps: {fps_sensor_front} // {fps_sensor_rear} // {fps_preview_sensor_rear}")
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_fps, fps_sensor_front)
    mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_fps, fps_sensor_rear)
    fps_edge1_sensor_front = 0
    fps_edge2_sensor_front = 0
    fps_preview_sensor_front = 0
    fps_edge1_sensor_rear = 0
    fps_edge2_sensor_rear = 0
    fps_preview_sensor_rear = 0


tl.start()


def getPipeline(is_front_sensor):
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    tile1_crop = pipeline.create(dai.node.ImageManip)
    tile2_crop = pipeline.create(dai.node.ImageManip)
    full_crop = pipeline.create(dai.node.ImageManip)
    reduce_frame = pipeline.create(dai.node.Script)
    reduce_frame.setScript(s_half_frame_rate)
    reduce_frame.inputs['image'].setBlocking(True)
    reduce_frame.inputs['image'].setQueueSize(1)
    full_rotate = pipeline.create(dai.node.ImageManip)
    x_out_image1 = pipeline.create(dai.node.XLinkOut)
    x_out_image2 = pipeline.create(dai.node.XLinkOut)
    x_out_video = pipeline.create(dai.node.XLinkOut)

    controlIn = pipeline.create(dai.node.XLinkIn)

    x_out_image1.setStreamName('out_image_tile1')
    x_out_image2.setStreamName('out_image_tile2')
    x_out_video.setStreamName('video')
    controlIn.setStreamName('control')

    # Properties
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewSize(preview_width, preview_height)
    camRgb.setFps(camera_fps)
    # camRgb.setCropRect(cropX, cropY, 0, 0)
    # camRgb.setIspScale(2, 3)
    camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    camRgb.setInterleaved(False)
    maxFrameSize = camRgb.getPreviewWidth() * camRgb.getPreviewHeight() * 3

    if is_front_sensor:
        tile1_crop.initialConfig.setCropRect(0, 0, 0.5, (vs_front_stop_position_max + 20) / preview_height)
    else:
        tile1_crop.initialConfig.setCropRect(0, 0, 0.5, (vs_rear_stop_position_max + 20) / preview_height)
    tile1_crop.setMaxOutputFrameSize(maxFrameSize)
    tile1_crop.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

    if is_front_sensor:
        tile2_crop.initialConfig.setCropRect(0.5, 0, 1, (vs_front_stop_position_max + 20) / preview_height)
    else:
        tile2_crop.initialConfig.setCropRect(0.5, 0, 1, (vs_rear_stop_position_max + 20) / preview_height)
    tile2_crop.setMaxOutputFrameSize(maxFrameSize)
    tile2_crop.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

    full_crop.initialConfig.setCropRect(0.1, 0, 1, preview_height_crop_factor)
    full_crop.initialConfig.setResizeThumbnail(thumbnail_width, thumbnail_height)
    # full_crop.setMaxOutputFrameSize(maxFrameSize)

    # full_rotate.setMaxOutputFrameSize(maxFrameSize)
    rr = dai.RotatedRect()
    rr.center.x, rr.center.y = preview_width // 2, preview_height_crop // 2
    rr.size.width, rr.size.height = preview_height_crop, preview_width
    if is_front_sensor:
        rr.angle = 90
    else:
        rr.angle = 270
    full_rotate.initialConfig.setCropRotatedRect(rr, False)
    full_rotate.initialConfig.setFrameType(dai.RawImgFrame.Type.YUV400p)

    videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)

    # Links
    camRgb.preview.link(tile1_crop.inputImage)
    camRgb.preview.link(tile2_crop.inputImage)
    camRgb.preview.link(reduce_frame.inputs['input'])

    tile1_crop.out.link(x_out_image1.input)
    tile2_crop.out.link(x_out_image2.input)

    reduce_frame.outputs['out'].link(full_rotate.inputImage)
    full_crop.out.link(full_rotate.inputImage)
    full_rotate.out.link(videoEnc.input)

    videoEnc.bitstream.link(x_out_video.input)

    controlIn.out.link(camRgb.inputControl)

    return pipeline


def focus_cameras():
    # print("Autoexposure and AutoFocus Lock")
    ctrl = dai.CameraControl()
    # ctrl.setAutoExposureRegion(1800, 0, 180, 320)
    ctrl.setManualWhiteBalance(4000)
    ctrl.setAutoExposureEnable()
    ctrl.setAutoExposureLock(False)
    # ctrl.setAutoFocusRegion()
    # ctrl.setManualExposure(1000, 100)
    # ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
    ctrl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)
    # ctrl.setAutoFocusTrigger()
    ctrl.setManualFocus(200)
    vs_front_camera_control_queue.send(ctrl)
    vs_rear_camera_control_queue.send(ctrl)


with contextlib.ExitStack() as stack:
    # pipeline_front = getPipeline(is_front_sensor=True)
    # pipeline_rear = getPipeline(is_front_sensor=False)
    # print("   >>> Loading pipeline")
    #
    # # vs_front_device = device_info_front_sensor.startPipeline(pipeline_front)
    # vs_front_device = dai.Device(pipeline_front, device_info_front_sensor)
    # print("USB Connection speed: {}".format(vs_front_device.getUsbSpeed()))
    # vs_front_tile1_queue = vs_front_device.getOutputQueue(name="out_image_tile1", maxSize=4, blocking=True)
    # vs_front_tile2_queue = vs_front_device.getOutputQueue(name="out_image_tile2", maxSize=4, blocking=True)
    # vs_front_video_queue = vs_front_device.getOutputQueue(name="video", maxSize=30, blocking=True)
    # vs_front_camera_control_queue = vs_front_device.getInputQueue('control')
    # vs_front_device.startPipeline()
    #
    # vs_rear_device = dai.Device(pipeline_rear, device_info_rear_sensor)
    # print("USB Connection speed: {}".format(vs_rear_device.getUsbSpeed()))
    # vs_rear_tile1_queue = vs_rear_device.getOutputQueue(name="out_image_tile1", maxSize=4, blocking=True)
    # vs_rear_tile2_queue = vs_rear_device.getOutputQueue(name="out_image_tile2", maxSize=4, blocking=True)
    # vs_rear_video_queue = vs_rear_device.getOutputQueue(name="video", maxSize=30, blocking=True)
    # vs_rear_camera_control_queue = vs_rear_device.getInputQueue('control')
    # vs_rear_device.startPipeline()

    vs_front = VisionSensor(device_info_front_sensor, True)
    vs_rear = VisionSensor(device_info_rear_sensor, False)

    # focus_cameras()

    while True:
        vs_front_enable_live_view.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsFront_enableLiveView))
        vs_rear_enable_live_view.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_vsRear_enableLiveView))
        vs_enable_low_contrast_mode.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsCtrl_enable_low_contrast_mode))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsFront_setStopPosition):
            vs_front_stop_position.value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsFront_setStopPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getStopPosition, vs_front_stop_position.value)

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsRear_setStopPosition):
            vs_rear_stop_position.value = mqtt.getMqttValue(mqtt.sTopics_vsController.get_vsRear_setStopPosition)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getStopPosition, vs_rear_stop_position.value)

        if vs_front_enable_live_view.is_different():
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsFront_liveViewIsEnabled, ef.bool2Str(vs_front_enable_live_view.value))

        if vs_rear_enable_live_view.value:
            mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsRear_liveViewIsEnabled, ef.bool2Str(vs_rear_enable_live_view.value))

        if vs_enable_low_contrast_mode.is_different():
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_low_contrast_mode_enabled, ef.bool2Str(vs_enable_low_contrast_mode.value))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsFront_focusCamera):
            vs_front_auto_focus. value = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_vsRear_focusCamera):
            vs_rear_auto_focus.value = True

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_setFilmTypeIsNegative):
            film_type_is_negative.value = ef.str2bool(mqtt.getMqttValue(mqtt.sTopics_vsController.get_setFilmTypeIsNegative))
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_filmTypeIsNegative, ef.bool2Str(film_type_is_negative.value))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_swapSensors):
            writeInitConfig(config_init.get("vs_front", "serial"), config_init.get("vs_rear", "serial"))

        if mqtt.isNewMqttValueAvailable(mqtt.sTopics_vsController.get_vsCtrl_centerImage):
            center_image = ef.str2bool(mqtt.sTopics_vsController.get_vsCtrl_centerImage)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsCtrl_centerImage, ef.bool2Str(center_image))
            logInfoGeneral("set centerImage to: {}".format(center_image))

        # Process Front-Sensor
        vs_front_tile1_data = vs_front_tile1_queue.tryGet()
        if vs_front_tile1_data is not None:
            txt = f"[{vs_front_tile1_data.getSequenceNum()}] "
            txt += f"Exposure: {vs_front_tile1_data.getExposureTime().total_seconds() * 1000:.3f} ms, "
            txt += f"ISO: {vs_front_tile1_data.getSensitivity()}, "
            txt += f"Lens position: {vs_front_tile1_data.getLensPosition()}, "
            txt += f"Color temp: {vs_front_tile1_data.getColorTemperature()} K"
            print(txt)
            vs_front_image_tile1 = vs_front_tile1_data.getCvFrame()
            fps_edge1_sensor_front += 1

        vs_front_tile2_data = vs_front_tile2_queue.tryGet()
        if vs_front_tile2_data is not None:
            vs_front_image_tile2 = vs_front_tile2_data.getCvFrame()
            fps_edge2_sensor_front += 1
            vs_front_edge_position.value, vs_front_edge_position_tile1.value, \
            vs_front_edge_position_tile2.value, vs_front_slope_tile1.value, \
            vs_front_slope_tile2.value = get_edge_position(vs_front_image_tile1, vs_front_image_tile2, film_type_is_negative.value)

            # print(vs_front_edge_position.value)
            if showOutput:
                if vs_front_image_tile1 is not None:
                    img_edge1_front_color = cv2.cvtColor(vs_front_image_tile1, cv2.COLOR_GRAY2RGB)
                    cv2.line(img_edge1_front_color, (0, vs_front_edge_position_tile1.value), (600, vs_front_edge_position_tile1.value), (0, 0, 255), 2)
                    if vs_front_edge_is_in_position.value:
                        cv2.line(img_edge1_front_color, (0, vs_front_edge_position.value), (600, vs_front_edge_position.value), (0, 255, 0), 2)
                    else:
                        cv2.line(img_edge1_front_color, (0, vs_front_edge_position.value), (600, vs_front_edge_position.value), (0, 255, 0), 1)
                    cv2.imshow("Tile 1 / Sensor-Front", img_edge1_front_color)
                if vs_front_image_tile2 is not None:
                    img_edge2_front_color = cv2.cvtColor(vs_front_image_tile2, cv2.COLOR_GRAY2RGB)
                    cv2.line(img_edge2_front_color, (0, vs_front_edge_position_tile2.value), (600, vs_front_edge_position_tile2.value), (0, 0, 255), 2)
                    if vs_front_edge_is_in_position.value:
                        cv2.line(img_edge2_front_color, (0, vs_front_edge_position.value), (600, vs_front_edge_position.value), (0, 255, 0), 2)
                    else:
                        cv2.line(img_edge2_front_color, (0, vs_front_edge_position.value), (600, vs_front_edge_position.value), (0, 255, 0), 1)
                    cv2.imshow("Tile 2 / Sensor-Front", img_edge2_front_color)

        for vs_front_video_data in vs_front_video_queue.tryGetAll():
            fps_preview_sensor_front += 1
            if vs_front_enable_live_view.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsFront_liveViewIsEnabled, ef.bool2Str(vs_front_enable_live_view.value))
                print("set enableLiveView to: {}".format(vs_rear_enable_live_view.value))
                vs_front_enable_live_view.reset()

            if vs_front_enable_live_view.value:
                sensor_front_image_data_base64 = base64.b64encode(vs_front_video_data.getData())
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageDataTn, sensor_front_image_data_base64)

            if showOutput:
                inp_front = np.asarray(bytearray(vs_front_video_data.getData()), dtype=np.uint8)
                i0_front = cv2.imdecode(inp_front, cv2.IMREAD_COLOR)
                cv2.line(i0_front, (vs_front_edge_position.value, 0), (vs_front_edge_position.value, 800), (255, 0, 0), 2)
                cv2.imshow("video / Sensor-Front", i0_front)

        # Process Rear-Sensor
        vs_rear_tile1_data = vs_rear_tile1_queue.tryGet()
        if vs_rear_tile1_data is not None:
            vs_rear_image_tile1 = vs_rear_tile1_data.getCvFrame()
            fps_edge1_sensor_rear += 1

        vs_rear_tile2_data = vs_rear_tile2_queue.tryGet()
        if vs_rear_tile2_data is not None:
            vs_rear_image_tile2 = vs_rear_tile2_data.getCvFrame()
            fps_edge2_sensor_rear += 1
            vs_rear_edge_position.value, vs_rear_edge_position_tile1.value,\
            vs_rear_edge_position_tile2.value, vs_rear_slope_tile1.value, \
            vs_rear_slope_tile2.value = get_edge_position(vs_rear_image_tile1, vs_rear_image_tile2, film_type_is_negative.value)

            if showOutput:
                if vs_rear_image_tile1 is not None:
                    img_edge1_rear_color = cv2.cvtColor(vs_rear_image_tile1, cv2.COLOR_GRAY2RGB)
                    cv2.line(img_edge1_rear_color, (0, vs_rear_edge_position_tile1.value), (600, vs_rear_edge_position_tile1.value), (0, 0, 255), 2)
                    if vs_rear_edge_is_in_position.value:
                        cv2.line(img_edge1_rear_color, (0, vs_rear_edge_position.value), (600, vs_rear_edge_position.value), (0, 255, 0), 2)
                    else:
                        cv2.line(img_edge1_rear_color, (0, vs_rear_edge_position.value), (600, vs_rear_edge_position.value), (0, 255, 0), 1)
                    cv2.imshow("Tile 1 / Sensor-Rear", img_edge1_rear_color)
                if vs_rear_image_tile2 is not None:
                    img_edge2_rear_color = cv2.cvtColor(vs_rear_image_tile2, cv2.COLOR_GRAY2RGB)
                    cv2.line(img_edge2_rear_color, (0, vs_rear_edge_position_tile2.value), (600, vs_rear_edge_position_tile2.value), (0, 0, 255), 2)
                    if vs_rear_edge_is_in_position.value:
                        cv2.line(img_edge2_rear_color, (0, vs_rear_edge_position.value), (600, vs_rear_edge_position.value), (0, 255, 0), 2)
                    else:
                        cv2.line(img_edge2_rear_color, (0, vs_rear_edge_position.value), (600, vs_rear_edge_position.value), (0, 255, 0), 1)
                    cv2.imshow("Tile 2 / Sensor-Rear", img_edge2_rear_color)

        for vs_rear_video_data in vs_rear_video_queue.tryGetAll():
            fps_preview_sensor_rear += 1

            if vs_rear_enable_live_view.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsRear_liveViewIsEnabled, ef.bool2Str(vs_rear_enable_live_view.value))

            if vs_rear_enable_live_view.value:
                sensor_rear_image_data_base64 = base64.b64encode(vs_rear_video_data.getData())
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageDataTn, sensor_rear_image_data_base64)

            if showOutput:
                inp_rear = np.asarray(bytearray(vs_rear_video_data.getData()), dtype=np.uint8)
                i0_rear = cv2.imdecode(inp_rear, cv2.IMREAD_COLOR)
                cv2.line(i0_rear, (preview_height_crop - vs_rear_edge_position.value, 0), (preview_height_crop - vs_rear_edge_position.value, 800), (255, 0, 0), 2)
                cv2.imshow("video / Sensor-Rear", i0_rear)

        if vs_front_edge_position.is_different() or vs_rear_edge_position.is_different():

            # VS-Front - edge-position
            if vs_front_edge_position.is_different():
                if (vs_front_edge_position.value % mqtt_position_steps) == 0:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front_edge_position.value)

            if vs_rear_edge_position.is_different():
                if (vs_rear_edge_position.value % mqtt_position_steps) == 0:
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_rear_edge_position.value)

            # VS-Front - edge-detected
            if vs_front_edge_position.value > 0:
                vs_front_edge_detected.value = True
            else:
                vs_front_edge_detected.value = False

            if vs_front_edge_detected.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgeDetected, ef.bool2Str(vs_front_edge_detected.value))

            # VS-Rear - edge-detected
            if vs_rear_edge_position.value > 0:
                vs_rear_edge_detected.value = True
            else:
                vs_rear_edge_detected.value = False

            if vs_rear_edge_detected.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgeDetected, ef.bool2Str(vs_rear_edge_detected.value))

            # VS-Front - edge-is-in-position
            if vs_front_edge_position.value > vs_front_stop_position_min:
                vs_front_edge_is_in_position.value = True
            else:
                vs_front_edge_is_in_position.value = False

            if vs_front_edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_pictureIsInPosition,
                                  ef.bool2Str(vs_front_edge_is_in_position.value))

            # VS-Rear - edge-is-in-position
            if vs_rear_edge_position.value > vs_rear_stop_position_min:
                vs_rear_edge_is_in_position.value = True
            else:
                vs_rear_edge_is_in_position.value = False

            if vs_rear_edge_is_in_position.is_different():
                mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_pictureIsInPosition,
                                  ef.bool2Str(vs_rear_edge_is_in_position.value))

            # Check if Image is Centered
            if vs_front_edge_position.value > -1 and vs_rear_edge_position.value > -1:

                if abs(vs_front_edge_position.value - vs_rear_edge_position.value) < 10 and vs_rear_edge_is_in_position.value and vs_front_edge_is_in_position.value:
                    image_is_centered.value = True
                else:
                    image_is_centered.value = False

                if image_is_centered.is_different():
                    print("IMAGE CENTERED!! #################################")
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_pictureIsInPosition, image_is_centered.value)
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_edgePosition, vs_front_edge_position.value)
                    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_edgePosition, vs_front_edge_position.value)

        if vs_front_auto_focus.value:
            logInfoGeneral("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            vs_front_camera_control_queue.send(ctrl)
            vs_front_auto_focus.value = False
            vs_front_auto_focus.reset()

        if vs_rear_auto_focus.value:
            logInfoGeneral("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            vs_rear_camera_control_queue.send(ctrl)
            vs_rear_auto_focus.value = False
            vs_rear_auto_focus.reset()

        if vs_whitebalance.is_different():
            ctrl = dai.CameraControl()
            ctrl.setManualWhiteBalance(vs_whitebalance.value)
            vs_front_camera_control_queue.send(ctrl)
            vs_rear_camera_control_queue.send(ctrl)
            logInfoGeneral("set white-balance to: {}".format(vs_whitebalance.value))
            vs_whitebalance.reset()

        if vs_front_lens_position.is_different():
            ctrl1 = dai.CameraControl()
            ctrl1.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            ctrl1.setManualFocus(vs_front_lens_position.value)
            vs_front_camera_control_queue.send(ctrl1)
            vs_front_lens_position.reset()

        if vs_rear_lens_position.is_different():
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            ctrl.setManualFocus(vs_rear_lens_position.value)
            vs_rear_camera_control_queue.send(ctrl)
            vs_rear_lens_position.reset()

        if vs_front_slope_tile1.is_different():
            if abs(vs_front_slope_tile1.value - vs_front_slope_tile1.last_value) > 2:
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_slope_tile1, vs_front_slope_tile1.value)

        if vs_front_slope_tile2.is_different():
            if abs(vs_front_slope_tile2.value - vs_front_slope_tile2.last_value) > 2:
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsFront_slope_tile2, vs_front_slope_tile2.value)

        if vs_rear_slope_tile1.is_different():
            if abs(vs_rear_slope_tile1.value - vs_rear_slope_tile1.last_value) > 2:
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_slope_tile1, vs_rear_slope_tile1.value)

        if vs_rear_slope_tile2.is_different():
            if abs(vs_rear_slope_tile2.value - vs_rear_slope_tile2.last_value) > 2:
                mqtt.setMqttValue(mqtt.pTopics_vsController.get_vsCtrl_vsRear_slope_tile2, vs_rear_slope_tile2.value)

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

        vs_front_edge_position.reset()
        vs_rear_edge_position.reset()
        vs_front_edge_detected.reset()
        vs_rear_edge_detected.reset()
        vs_front_edge_is_in_position.reset()
        vs_rear_edge_is_in_position.reset()
        image_is_centered.reset()
        vs_front_slope_tile1.reset()
        vs_front_slope_tile2.reset()
        vs_rear_slope_tile1.reset()
        vs_rear_slope_tile2.reset()

        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.001)
