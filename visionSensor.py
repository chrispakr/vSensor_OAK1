from __future__ import annotations

import threading
from datetime import datetime
from typing import Tuple

from depthai import DeviceInfo
from numpy.core.defchararray import center

from libs.functions import ValueHandler, ValueHandlerInt
from collections import deque
import depthai as dai
import numpy as np
from turbojpeg import TurboJPEG
from enum import Enum
import platform
import base64
import cv2
import time
from loguru import logger
from nptyping import NDArray, Bool
import socket, pickle, struct
import numpy


class TilePositionData:
    def __init__(self, param_edge:EdgeProcessingParameter):
        self.edge_position:int = 0
        self.edge_slope:int = 0
        self.image_data = None
        self._edge_result = None
        self._param_edge = param_edge

    def calc_edge_parameter(self):
        if self.image_data is not None:
            self._edge_result = self._calc_edge_parameter(image_data=self.image_data)
            if self._param_edge.film_type_is_negative:
                self.edge_position = self._edge_result[3]
                self.edge_slope = self._edge_result[1]
            else:
                self.edge_position = self._edge_result[2]
                self.edge_slope = self._edge_result[0]

    @staticmethod
    def _calc_edge_parameter(image_data:NDArray) -> Tuple or None:
        if image_data is not None:
            reduced = np.mean(image_data, axis=1)
            # compute the slope values at +/-3
            slope = [(reduced[i + 3] - reduced[i - 3]) for i in range(3, len(reduced) - 3, 1)]
            # compute the minVal, minPos, maxVal, maxPos of the slopes

            min_slope = min(slope)
            max_slope = max(slope)
            min_pos = slope.index(min_slope)
            max_pos = slope.index(max_slope)

            # prepare slopes and positions for output
            slope_and_pos_output = (min_slope * -1, max_slope, min_pos, max_pos)
            return slope_and_pos_output
        else:
            return None

class CalculateContrast:
    def __init__(self, param_image:ImageProcessingParameter):
        self.tile_left:float = 0.0
        self.tile_right:float = 0.0
        self.total:float = 0.0
        self.param_image = param_image

    def calculate_contrast(self,
                           image_tile_left:NDArray,
                           image_tile_right:NDArray,
                           edge_position:int = 0):
        contrast_max_pos = edge_position - self.param_image.contrast_pic_edge_offset
        contrast_min_pos = contrast_max_pos - self.param_image.contrast_pic_height
        image_roi_left = image_tile_left[contrast_min_pos:contrast_max_pos, 0:self.param_image.tile_width]
        image_roi_right = image_tile_right[contrast_min_pos:contrast_max_pos, 0:self.param_image.tile_width]
        self.tile_left = np.median(image_roi_left)
        self.tile_right = np.median(image_roi_right)
        self.total = self.tile_left + self.tile_right


class ImageProcessingParameter:
    def __init__(self,
                 preview_width:int = 800,
                 tile_center_offset:int = 50,
                 tile_width:int = 350,
                 contrast_pic_height:int = 20,
                 contrast_pic_edge_offset:int = 10
                 ):
        self._preview_width = preview_width
        self.contrast_pic_height = contrast_pic_height
        self.contrast_pic_edge_offset = contrast_pic_edge_offset
        self._tile_center_offset = tile_center_offset
        self._tile_width = tile_width

    @property
    def tile_center_offset(self):
        return self._tile_center_offset

    @tile_center_offset.setter
    def tile_center_offset(self, value:int):
        self._tile_center_offset = value
        if (self._tile_center_offset + self._tile_width) > (self._preview_width // 2):
            self._tile_width = (self._preview_width // 2) - self._tile_center_offset
        logger.debug(f"set tile_center_offset to: {self._tile_center_offset}")

    @property
    def tile_width(self):
        return self._tile_width

    @tile_width.setter
    def tile_width(self, value:int):
        self._tile_width = value
        if (self._tile_width + self._tile_center_offset) > (self._preview_width // 2):
            self._tile_center_offset = (self._preview_width // 2) - self._tile_width
        logger.debug(f"set tile_width to: {self._tile_width}")

    @property
    def preview_width(self):
        return self._preview_width

    @preview_width.setter
    def preview_width(self, value:int):
        self._preview_width = value
        if (self._tile_width + self._tile_center_offset) > (self._preview_width // 2):
            self._tile_width = (self._preview_width // 2) - self._tile_center_offset
        logger.debug(f"set preview_width to: {self._preview_width}")


class EdgeProcessingParameter:
    def __init__(self,
                 stop_position:int = 350,
                 edge_detection_range:int = 12,
                 film_type_is_negative:bool = True,
                 threshold_slope: float = 30.0,
                 contrast_offset:float = 30.0
                 ):
        self.stop_position = stop_position
        self.edge_detection_range = edge_detection_range
        self.film_type_is_negative = film_type_is_negative
        self.threshold_slope = threshold_slope
        self.contrast_offset = contrast_offset


class ProcessImageEdgeParameters:
    def __init__(self, param_image:ImageProcessingParameter, param_edge:EdgeProcessingParameter):
        self.image_data = None
        self.image_width:int = 0
        self.image_height:int = 0
        self.edge_position: int = 0
        self.param_image = param_image
        self.param_edge = param_edge
        self.left_tile_data = TilePositionData(param_edge=self.param_edge)
        self.right_tile_data = TilePositionData(param_edge=self.param_edge)
        self._edge_position_tile_diff:int = 0
        self._total_edge_slope:float = 0.0
        self._total_contrast_offset: float = 0.0
        self._arr_slope_total_mean = deque(maxlen=40)
        self._slope_total_mean: float = 0.0
        self._slope_diff_rising:float = 0.0
        self._slope_diff_falling:float = 0.0
        self._new_edge_detected:bool = False
        self._edge_position:int = 0
        self._in_pic_contrast = CalculateContrast(param_image=self.param_image)
        self._out_pic_contrast = CalculateContrast(param_image=self.param_image)
        self._edge_detected: bool = False
        self._edge_in_position: bool = False

    def process_image(self, image_data:NDArray):
        if image_data is not None:
            self._edge_position = -1
            self.image_data = image_data
            self.image_height, self.image_width = self.image_data.shape[:2]

            self.left_tile_data.image_data, self.right_tile_data.image_data = self._get_image_tiles(
                image_data=self.image_data,
                stop_position=self.param_edge.stop_position,
                tile_width=self.param_image.tile_width,
                tile_center_offset=self.param_image.tile_center_offset
            )

            self.left_tile_data.calc_edge_parameter()
            self.right_tile_data.calc_edge_parameter()

            self._edge_position_tile_diff = abs(self.left_tile_data.edge_position - self.right_tile_data.edge_position)
            self._total_edge_slope = self.left_tile_data.edge_slope + self.right_tile_data.edge_slope

            self._arr_slope_total_mean.append(self._total_edge_slope)
            self._slope_total_mean = sum(self._arr_slope_total_mean) // len(self._arr_slope_total_mean)
            self._slope_diff_rising = self._total_edge_slope - min(self._arr_slope_total_mean)
            self._slope_diff_falling = self._slope_total_mean - max(self._arr_slope_total_mean)

            if (self._total_edge_slope - self._slope_total_mean) > 40 and self._new_edge_detected == 0:
                self._new_edge_detected = 200
                self._arr_slope_total_mean.clear()
                self._arr_slope_total_mean.append(self._total_edge_slope)

            if (self._slope_total_mean - self._total_edge_slope) > 40 and self._new_edge_detected == 200:
                self._new_edge_detected = 0
                self._arr_slope_total_mean.clear()
                self._arr_slope_total_mean.append(self._total_edge_slope)


            if self._total_edge_slope > self.param_edge.threshold_slope:
                self._edge_position = (self.left_tile_data.edge_position + self.right_tile_data.edge_position) // 2

            if self._edge_position > (self.param_image.contrast_pic_height + self.param_image.contrast_pic_edge_offset):
                self._in_pic_contrast.calculate_contrast(
                    image_tile_left=self.left_tile_data.image_data,
                    image_tile_right=self.right_tile_data.image_data,
                    edge_position=self._edge_position
                )

                self._out_pic_contrast.calculate_contrast(
                    image_tile_left=self.left_tile_data.image_data,
                    image_tile_right=self.right_tile_data.image_data,
                    edge_position=self._edge_position
                )

            if self.param_edge.film_type_is_negative:
                if self._in_pic_contrast.total + self.param_edge.contrast_offset < self._out_pic_contrast.total:
                    self._edge_detected = True
                else:
                    self._edge_detected = False
            else:
                if self._in_pic_contrast.total + self.param_edge.contrast_offset > self._out_pic_contrast.total:
                    self._edge_detected = True
                else:
                    self._edge_detected = False

            if (self.param_edge.stop_position - (self.param_edge.edge_detection_range // 2)) < self._edge_position < (self.param_edge.stop_position + (self.param_edge.edge_detection_range // 2)):
                self._edge_in_position = True
            else:
                self._edge_in_position = False

            if not self._edge_detected and not self._edge_in_position:
                self.edge_position = -1

            if self._edge_detected and not self._edge_in_position:
                self.edge_position = self._edge_position

            if self._edge_detected and self._edge_in_position:
                self.edge_position = self._edge_position

            self.edge_position = self._edge_position

    @staticmethod
    def _get_image_tiles(
            image_data:NDArray,
            stop_position:int,
            tile_width:int,
            tile_center_offset:int=50) -> Tuple[NDArray, NDArray]:
        img_height, img_width = image_data.shape[:2]
        np_image_tile_left = image_data[
                             0:stop_position + 50,
                             (img_width // 2) - tile_width:(img_width // 2) - tile_center_offset]

        np_image_tile_right = image_data[
                              0:stop_position + 50,
                              (img_width // 2) + tile_center_offset:(img_width // 2) + tile_width]
        return np_image_tile_left, np_image_tile_right


class VisionSensor:
    fps_report_time = 3
    _contrast_pic_height = 20
    _contrast_pic_edge_offset = 10

    def __init__(self,
                 device_info:DeviceInfo,
                 is_front_sensor:bool,
                 vs_name:str,
                 fps:int = 60,
                 capture_width:int = 860,
                 capture_height:int = 600,
                 image_center_position:int = 430,
                 lens_position:int = 130,
                 ):
        # general Variables
        super().__init__()
        self.capture_width = capture_width
        self.capture_height = capture_height
        self._image_center_position = image_center_position
        self._lens_position = lens_position
        self.set_fps = fps
        self.controlIn = None
        self.x_out_edge_detection = None
        self.manip_edge_detection = None
        self.camRgb = None

        self.is_front_sensor = is_front_sensor
        self.name = vs_name
        self.device_info = device_info

        self.param_image = ImageProcessingParameter()
        self.param_edge = EdgeProcessingParameter()

        self.result = ProcessImageEdgeParameters(
            param_image=self.param_image,
            param_edge=self.param_edge,
        )

        self._new_image_available = False

        # processing image_data variables
        self._input_image_data = None
        self._raw_input_image = None
        self.proc_image_centered = None

        self.image_info_jpg = None
        self.image_info_base64 = None
        self.captured_images = 0

        # fps variables
        self.fps_elapsed_time = datetime.now()
        self.fps_jpg_image_time = datetime.now()
        self.fps = 0
        self.fps_jpg_image = 0
        self._fps_counter = 0
        self._fps_counter_jpg_image = 0

        # camera control variables
        self.camCtrl = None
        self.exposure_time = 0

        self.af_start_time = datetime.now()
        self.autoFocusEnabled = False
        self.autoFocusFinished = False

        self.ae_start_time = datetime.now()
        self.autoExposureEnabled = False
        self.autoExposureFinished = False

        self.capture_time = time.time()

        self.thread_fps = threading.Thread(target=self._calc_fps)
        self.thread_fps.daemon = True
        self.thread_fps.start()

        self.pipeline = None
        self.device = None
        self.image_edge_queue = None
        self.camera_control_queue = None
        self._log_info_vsensor(f"init camera {self.device_info}")
        if platform.system() == "Windows":
            logger.info(f"set parameters for windows-system")
            self.jpeg = TurboJPEG("libs/libturbojpeg.dll")

        if platform.system() == "Linux":
            self._log_info_vsensor(f"set parameters for linux-system")
            self.jpeg = TurboJPEG()

        self.init_camera()

        self.camera_task = threading.Thread(target=self._process_image)
        self.camera_task.daemon = True
        self.camera_task.start()

    def init_camera(self):
        self.pipeline = None
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.manip_edge_detection = self.pipeline.create(dai.node.ImageManip)
        self.x_out_edge_detection = self.pipeline.create(dai.node.XLinkOut)

        self.controlIn = self.pipeline.create(dai.node.XLinkIn)

        self.x_out_edge_detection.setStreamName('image_edge_detection')
        self.controlIn.setStreamName('control')

        # Properties
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setPreviewSize(self.capture_width, self.capture_height)
        self.camRgb.setFps(self.set_fps)
        # self.camRgb.initialControl.setAutoFocusLensRange(120, 180)
        self.camRgb.initialControl.setManualFocus(150)
        self.camRgb.initialControl.setManualExposure(1200, 100)
        self.camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        self.camRgb.setInterleaved(False)
        max_frame_size = self.camRgb.getPreviewWidth() * self.camRgb.getPreviewHeight() * 3

        self.manip_edge_detection.initialConfig.setCropRect(0, 0, 1, 0.75)
        self.manip_edge_detection.setMaxOutputFrameSize(max_frame_size)
        self.manip_edge_detection.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        # Links
        self.camRgb.preview.link(self.manip_edge_detection.inputImage)
        self.manip_edge_detection.out.link(self.x_out_edge_detection.input)
        self.controlIn.out.link(self.camRgb.inputControl)

        self.device = dai.Device(self.pipeline, self.device_info)
        self.image_edge_queue = self.device.getOutputQueue(name="image_edge_detection", maxSize=4, blocking=True)
        self.camera_control_queue = self.device.getInputQueue('control')

    def _process_image(self):
        while True:
            self._input_image_data = self.image_edge_queue.tryGet()
            if self._input_image_data is not None:
                self._fps_counter += 1
                self.t_start = time.time()
                self.image_info_jpg = None
                self.image_info_base64 = None
                self.capture_time = time.time()
                self._raw_input_image = self._input_image_data.getCvFrame()
                (full_image_height, full_image_width) = self._raw_input_image.shape[:2]
                self.proc_image_centered = self._raw_input_image[
                                           0:full_image_height, self._image_center_position - (self.param_image.preview_width // 2):
                                                                self._image_center_position + (self.param_image.preview_width // 2)]

                self.result.process_image(
                    image_data=self.proc_image_centered
                )

                if int(self._input_image_data.getExposureTime().total_seconds() * 1000000) != self.exposure_time:
                    self.exposure_time = int(self._input_image_data.getExposureTime().total_seconds() * 1000000)
                if self._input_image_data.getLensPosition() != self._lens_position:
                    self._lens_position = self._input_image_data.getLensPosition()
                    logger.debug("lens-position changed to: {}".format(self._lens_position))
                if self.autoFocusEnabled:
                    if (datetime.now() - self.af_start_time).seconds > 2:
                        self.camCtrl = dai.CameraControl()
                        self.focus_position = self._lens_position
                        logger.debug("disable AutoFocus")
                        self.autoFocusEnabled = False
                        self.autoFocusFinished = True
                if self.autoExposureEnabled:
                    logger.debug("exposureTime: {}".format(str(self.exposure_time)))
                    if (datetime.now() - self.ae_start_time).seconds > 2:
                        self.camCtrl = dai.CameraControl()
                        self.camCtrl.setAutoExposureLock(True)
                        logger.debug("disable AutoExposure")
                        self.camera_control_queue.send(self.camCtrl)
                        self.autoExposureEnabled = False
                        self.autoExposureFinished = True

                self._new_image_available = True
            time.sleep(0.0001)

    @staticmethod
    def _get_image_tiles(self, image:NDArray, stop_position:int, proc_image_width:int, center_offset:int=50) -> Tuple[NDArray, NDArray]:
        img_height, img_width = image.shape[:2]
        np_image_tile_left = image[
                             0:stop_position + 50,
                             (img_width // 2) - proc_image_width:(img_width // 2) - center_offset]

        np_image_tile_right = image[
                              0:stop_position + 50,
                              (img_width // 2) + center_offset:(img_width // 2) + proc_image_width]
        return np_image_tile_left, np_image_tile_right

    def get_base64_image(self):
        try:
            logger.debug("create base64 image_data-data")
            image_np_color = cv2.cvtColor(self.proc_image_centered, cv2.COLOR_GRAY2RGB)
            self.image_info_jpg = self.jpeg.encode(image_np_color, quality=80)
            self.image_info_base64 = base64.b64encode(self.image_info_jpg)
            return self.image_info_base64
        except Exception as e:
            print(e)
            return None

    def calc_statistics(self):
        if self.result.left_tile_data.image_data is not None and self.result.right_tile_data.image_data is not None:
            tile_height, tile_width = self.result.left_tile_data.image_data.shape
            stat_image_tile_left = self.result.left_tile_data.image_data[0:self.result.edge_position - 20, 0:tile_width]
            stat_image_tile_right = self.result.right_tile_data.image_data[0:self.result.edge_position - 20, 0:tile_width]
            stat_image_full = np.concatenate((stat_image_tile_left, stat_image_tile_right), axis=1)

            vs_maximum_dn = 256  # for image_data depth of byte
            clipping_percent = 0.05  # in percent for clipping the histogram with 0.025% from left and 0.025% from right

            # computing histogram
            hist = cv2.calcHist([stat_image_full], [0], None, [vs_maximum_dn], [0, vs_maximum_dn])
            hist = hist.flatten()

            # Clipping the histogram by CLIPPING_PERCENT/2 % from bottom and top
            cutoff = stat_image_full.shape[0] * stat_image_full.shape[1] * clipping_percent / 2

            image_min, image_max, _, _ = cv2.minMaxLoc(stat_image_full)
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
            mean, std_dev = cv2.meanStdDev(stat_image_full)

            # flatten mean and std to obtain a vector and obtain the single value in it.
            print(image_min, image_max, clip_min, clip_max, round(mean.flatten()[0], 3), round(std_dev.flatten()[0], 3))
            return (image_min, image_max, clip_min, clip_max, round(mean.flatten()[0], 3), round(std_dev.flatten()[0], 3))
        else:
            return None

    def auto_focus_camera(self):
        self._log_info_vsensor("start AutoFocus on Camera...")
        self.autoFocusFinished = False
        self.autoFocusEnabled = True
        self.af_start_time = datetime.now()
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
        self.camCtrl.setAutoFocusTrigger()
        self.camera_control_queue.send(self.camCtrl)

    @property
    def new_image_available(self):
        if self._new_image_available:
            self._new_image_available = False
            return True
        else:
            return False

    @new_image_available.setter
    def new_image_available(self, value):
        pass

    @property
    def focus_position(self):
        # self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        return self._lens_position

    @focus_position.setter
    def focus_position(self, lens_position):
        self._log_info_vsensor("Set lens-position to: {}".format(lens_position))
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        self.camCtrl.setManualFocus(lens_position)
        self.camera_control_queue.send(self.camCtrl)

    @property
    def numpy_image_array(self):
        return self.proc_image_centered

    def set_exposure_value(self, exposure):
        self._log_info_vsensor("Set exposure to: {}".format(exposure))
        self.camCtrl = dai.CameraControl()
        # self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        self.camCtrl.setManualExposure(exposure, 100)
        self.camera_control_queue.send(self.camCtrl)

    def auto_exposure_camera(self):
        self._log_info_vsensor("Sensor Auto-Exposure...")
        self.autoExposureEnabled = True
        self.ae_start_time = datetime.now()
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoExposureLock(False)
        self.camCtrl.setAutoExposureEnable()
        self.camera_control_queue.send(self.camCtrl)

    @staticmethod
    def calc_edge_parameter(image):
        start_time = time.time()
        slope_and_pos_output = ([0, 0, 0, 0])
        if image is not None:
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
        else:
            return None

    @property
    def stop_position(self):
        return self._stop_position

    @stop_position.setter
    def stop_position(self, value):
        self._stop_position = value
        self._log_info_vsensor("change stop_position to: " + str(self._stop_position))

    # @property
    # def stop_offset_compensation(self):
    #     return self._stop_offset_compensation

    # @stop_offset_compensation.setter
    # def stop_offset_compensation(self, value):
    #     self._stop_offset_compensation = value
    #     logger.debug("change stop_offset_compensation to: " + str(self._stop_offset_compensation))

    # @property
    # def edge_detection_range(self):
    #     return self._edge_detection_range
    #
    # @edge_detection_range.setter
    # def edge_detection_range(self, value):
    #     self._edge_detection_range = value
    #     self._log_info_vsensor("change edge_detection_range to: " + str(self._edge_detection_range))

    @property
    def image_center_position(self):
        return self._image_center_position

    @image_center_position.setter
    def image_center_position(self, value):
        self._image_center_position = value
        self._log_info_vsensor("change image_center_position to: " + str(self._image_center_position))

    @property
    def lcm_slope(self):
        return self._lcm_slope

    @lcm_slope.setter
    def lcm_slope(self, value):
        self._lcm_slope = value
        self._log_info_vsensor("set lcm_slope to: " + str(self._lcm_slope))

    @property
    def lcm_contrast_offset(self):
        return self._lcm_contrast_offset

    @lcm_contrast_offset.setter
    def lcm_contrast_offset(self, value):
        self._lcm_contrast_offset = value
        self._log_info_vsensor("set lcm_contrast_offset to: " + str(self._lcm_contrast_offset))

    @property
    def enable_low_contrast_mode(self):
        return self._lcm_contrast_offset

    @enable_low_contrast_mode.setter
    def enable_low_contrast_mode(self, value):
        self._enabled_lcm = value
        self._log_info_vsensor("set enable_low_contrast_mode to: " + str(self._enabled_lcm))


    def _calc_fps(self):
        while True:
            self.fps = self._fps_counter
            self._fps_counter = 0
            time.sleep(1.0)

    def _log_info_vsensor(self, message):
        message = str(message)
        log_message = f"[{self.name}]" + " - " + message
        logger.info(log_message)
