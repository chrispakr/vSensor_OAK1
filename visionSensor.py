import threading
from datetime import datetime
from typing import Tuple
from depthai import DeviceInfo

from collections import deque
import depthai as dai
import numpy as np
from turbojpeg import TurboJPEG
import platform
import base64
import cv2
import time
from loguru import logger
from nptyping import NDArray
from icecream import ic


class VisionSensorSettings:
    def __init__(self,
                 preview_width:int = 800,
                 camera_center_position:int = 430,
                 edge_detection_range: int = 12,
                 film_type_is_negative: bool = True,
                 slope_threshold: float = 30.0,
                 contrast_offset: float = 30.0,
                 tile_center_offset:int = 50,
                 tile_width:int = 300,
                 tile_height:int = 350,
                 contrast_pic_height:int = 20,
                 contrast_pic_edge_offset:int = 10
                 ):
        self._preview_width = preview_width
        self._camera_center_position = camera_center_position
        self._edge_detection_range = edge_detection_range
        self._film_type_is_negative = film_type_is_negative
        self._slope_threshold = slope_threshold
        self._contrast_offset = contrast_offset
        self._contrast_pic_height = contrast_pic_height
        self._contrast_pic_offset = contrast_pic_edge_offset
        self._tile_center_offset = tile_center_offset
        self._tile_width = tile_width
        self._tile_height = tile_height

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
    def tile_height(self):
        return self._tile_height

    @tile_height.setter
    def tile_height(self, value: int):
        self._tile_height = value
        logger.debug(f"set tile_height to: {self.tile_height}")

    @property
    def preview_width(self):
        return self._preview_width

    @preview_width.setter
    def preview_width(self, value:int):
        self._preview_width = value
        if (self._tile_width + self._tile_center_offset) > (self._preview_width // 2):
            self._tile_width = (self._preview_width // 2) - self._tile_center_offset
        logger.debug(f"set preview_width to: {self._preview_width}")

    @property
    def slope_threshold(self):
        return self._slope_threshold

    @slope_threshold.setter
    def slope_threshold(self, value):
        self._slope_threshold = value
        logger.debug(f"set slope_threshold to: {self._slope_threshold}")

    @property
    def contrast_offset(self):
        return self._contrast_offset

    @contrast_offset.setter
    def contrast_offset(self, value):
        self._contrast_offset = value
        logger.debug(f"set contrast_offset to: {self._contrast_offset}")

    @property
    def contrast_pic_height(self):
        return self._contrast_pic_height

    @contrast_pic_height.setter
    def contrast_pic_height(self, value):
        self._contrast_pic_height = value
        logger.debug(f"set contrast_pic_height to: {self._contrast_pic_height}")

    @property
    def contrast_pic_offset(self):
        return self._contrast_pic_offset

    @contrast_pic_offset.setter
    def contrast_pic_offset(self, value):
        self._contrast_pic_offset = value
        logger.debug(f"set contrast_pic_offset to: {self._contrast_pic_offset}")

    @property
    def camera_center_position(self):
        return self._camera_center_position

    @camera_center_position.setter
    def camera_center_position(self, value):
        self._camera_center_position = value
        logger.debug(f"set image_center_position to: {self._camera_center_position}")

    @property
    def edge_detection_range(self):
        return self._edge_detection_range

    @edge_detection_range.setter
    def edge_detection_range(self, value):
        self._edge_detection_range = value
        logger.debug(f"set edge_detection_range to: {self._edge_detection_range}")

    @property
    def film_type_is_negative(self):
        return self._film_type_is_negative

    @film_type_is_negative.setter
    def film_type_is_negative(self, value):
        self._film_type_is_negative = value
        logger.debug(f"set edge_detection_range to: {self._film_type_is_negative}")


class TilePositionData:
    def __init__(self, vs_settings:VisionSensorSettings, image_data=None):
        self.image_data = None
        self.edge_position:int = 0
        self.edge_slope:int = 0
        self._edge_result = None
        self._vs_settings = vs_settings

    def calc_edge_parameter(self):
        if self.image_data is not None:
            self._edge_result = self._calc_edge_parameter(image_data=self.image_data)
            if self._vs_settings.film_type_is_negative:
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
    def __init__(self, vs_settings:VisionSensorSettings):
        self.tile_left:float = 0.0
        self.tile_right:float = 0.0
        self.total:float = 0.0
        self.vs_settings = vs_settings

    def calculate_contrast(self,
                           image_tile_left:NDArray,
                           image_tile_right:NDArray,
                           edge_position:int = 0):
        contrast_max_pos = edge_position - self.vs_settings.contrast_pic_offset
        contrast_min_pos = contrast_max_pos - self.vs_settings.contrast_pic_height
        image_roi_left = image_tile_left[contrast_min_pos:contrast_max_pos, 0:self.vs_settings.tile_width]
        image_roi_right = image_tile_right[contrast_min_pos:contrast_max_pos, 0:self.vs_settings.tile_width]
        self.tile_left = np.median(image_roi_left)
        self.tile_right = np.median(image_roi_right)
        self.total = self.tile_left + self.tile_right


class ProcessImageEdgeParameters:
    def __init__(self, vs_settings:VisionSensorSettings):
        self.image_data = None
        self.image_width:int = 0
        self.image_height:int = 0
        self.edge_position: int = 0
        self._vs_settings = vs_settings
        self.left_tile_data = TilePositionData(vs_settings=self._vs_settings)
        self.right_tile_data = TilePositionData(vs_settings=self._vs_settings)
        self._edge_position_tile_diff:int = 0
        self._total_edge_slope:float = 0.0
        self._total_contrast_offset: float = 0.0
        self._arr_slope_total_mean = deque(maxlen=40)
        self._slope_total_mean: float = 0.0
        self._slope_diff_rising:float = 0.0
        self._slope_diff_falling:float = 0.0
        self._new_edge_detected:bool = False
        self._edge_position:int = 0
        self._in_pic_contrast = CalculateContrast(vs_settings=self._vs_settings)
        self._out_pic_contrast = CalculateContrast(vs_settings=self._vs_settings)
        self._edge_detected: bool = False
        self._edge_in_position: bool = False

    def process_image(self, image_data:NDArray):
        if image_data is not None:
            self._edge_position = -1
            self.image_data = image_data
            self.image_height, self.image_width = self.image_data.shape[:2]

            self.left_tile_data.image_data, self.right_tile_data.image_data = self._get_image_tiles(
                image_data=self.image_data,
                vs_settings=self._vs_settings
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


            if self._total_edge_slope > self._vs_settings.slope_threshold:
                self._edge_position = (self.left_tile_data.edge_position + self.right_tile_data.edge_position) // 2

            if self._edge_position > (self._vs_settings.contrast_pic_height + self._vs_settings.contrast_offset):
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

            if self._vs_settings.film_type_is_negative:
                if self._in_pic_contrast.total + self._vs_settings.contrast_offset < self._out_pic_contrast.total:
                    self._edge_detected = True
                else:
                    self._edge_detected = False
            else:
                if self._in_pic_contrast.total + self._vs_settings.contrast_offset > self._out_pic_contrast.total:
                    self._edge_detected = True
                else:
                    self._edge_detected = False

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
            vs_settings:VisionSensorSettings,
    ) -> Tuple[NDArray, NDArray]:
        img_height, img_width = image_data.shape[:2]
        np_image_tile_left = image_data[
                             0:vs_settings.tile_height,
                             (img_width // 2) - vs_settings.tile_width:(img_width // 2) - vs_settings.tile_center_offset]

        np_image_tile_right = image_data[
                              0:vs_settings.tile_height,
                              (img_width // 2) + vs_settings.tile_center_offset:(img_width // 2) + vs_settings.tile_width]
        return np_image_tile_left, np_image_tile_right

class VisionSensor:
    fps_report_time = 3

    def __init__(self,
                 device_info:DeviceInfo,
                 is_front_sensor:bool,
                 vs_name:str,
                 camera_capture_width: int,
                 camera_capture_height: int,
                 image_center_position: int,
                 lens_position: int,
                 fps:int = 60,
                 ):
        # general Variables
        super().__init__()
        self.camera_capture_width = camera_capture_width
        self.camera_capture_height = camera_capture_height
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

        self.settings = VisionSensorSettings()

        self.result = ProcessImageEdgeParameters(
            vs_settings=self.settings,
        )

        self._new_image_available = False

        # processing image_data variables
        self._sensor_image_data = None
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
        self._exposure_time = 0
        self._iso = 100

        self._af_start_time = datetime.now()
        self.autofocus_in_progress = False
        self._cb_autofocus_finished = None

        self._ae_start_time = datetime.now()
        self.auto_exposure_in_progress = False
        self._cb_auto_exposure_finished = None

        self.thread_fps = threading.Thread(target=self._calc_fps)
        self.thread_fps.daemon = True
        self.thread_fps.start()

        self.pipeline = None
        self.device = None
        self.image_edge_queue = None
        self.camera_control_queue = None
        self._log_info_vsensor(f"init camera {self.device_info}")
        self._log_info_vsensor(self.settings.__dict__)
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
        self.camRgb.setPreviewSize(self.camera_capture_width, self.camera_capture_height)
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
            self._sensor_image_data = self.image_edge_queue.tryGet()
            if self._sensor_image_data is not None:
                self._fps_counter += 1
                self.t_start = time.time()
                self.image_info_jpg = None
                self.image_info_base64 = None
                self._raw_input_image = self._sensor_image_data.getCvFrame()
                (full_image_height, full_image_width) = self._raw_input_image.shape[:2]
                self.proc_image_centered = self._raw_input_image[
                                           0:full_image_height,
                                           self._image_center_position - (self.settings.preview_width // 2):
                                           self._image_center_position + (self.settings.preview_width // 2)
                                           ]

                self.result.process_image(
                    image_data=self.proc_image_centered
                )

                if int(self._sensor_image_data.getExposureTime().total_seconds() * 1000000) != self._exposure_time:
                    self._exposure_time = int(self._sensor_image_data.getExposureTime().total_seconds() * 1000000)
                if self._sensor_image_data.getLensPosition() != self._lens_position:
                    self._lens_position = self._sensor_image_data.getLensPosition()
                    self._log_info_vsensor("lens-position changed to: {}".format(self._lens_position))
                if self.autofocus_in_progress:
                    if (datetime.now() - self._af_start_time).seconds > 2:
                        self.camCtrl = dai.CameraControl()
                        self.lens_position = self._lens_position
                        logger.debug("disable AutoFocus")
                        self.autofocus_in_progress = False
                        if self._cb_autofocus_finished is not None:
                            self._cb_autofocus_finished()
                if self.auto_exposure_in_progress:
                    logger.debug("exposureTime: {}".format(str(self._exposure_time)))
                    if (datetime.now() - self._ae_start_time).seconds > 2:
                        self.camCtrl = dai.CameraControl()
                        self.camCtrl.setAutoExposureLock(True)
                        logger.debug("disable AutoExposure")
                        self.camera_control_queue.send(self.camCtrl)
                        self.auto_exposure_in_progress = False
                        if self._cb_auto_exposure_finished is not None:
                            self._cb_auto_exposure_finished()

                self._new_image_available = True
            time.sleep(0.0001)

    @staticmethod
    def _get_image_tiles(image:NDArray, stop_position:int, proc_image_width:int, center_offset:int=50) -> Tuple[NDArray, NDArray]:
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
            # print(image_min, image_max, clip_min, clip_max, round(mean.flatten()[0], 3), round(std_dev.flatten()[0], 3))
            return image_min, image_max, clip_min, clip_max, round(mean.flatten()[0], 3), round(std_dev.flatten()[0], 3)
        else:
            return None

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
    def lens_position(self):
        return self._lens_position

    @lens_position.setter
    def lens_position(self, value):
        self._lens_position = value
        logger.debug(f"Set lens-position to: {self._lens_position}")
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        self.camCtrl.setManualFocus(self._lens_position)
        self.camera_control_queue.send(self.camCtrl)

    @property
    def numpy_image_array(self):
        return self.proc_image_centered

    @property
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value:int):
        self._exposure_time = value
        self._log_info_vsensor(f"Set value to: {self._exposure_time}")
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setManualExposure(self._exposure_time, self._iso)
        self.camera_control_queue.send(self.camCtrl)

    def auto_exposure_camera(self, cb_auto_exposure_finished=None):
        self._log_info_vsensor("Sensor Auto-Exposure...")
        self._cb_auto_exposure_finished = cb_auto_exposure_finished
        self.auto_exposure_in_progress = True
        self._ae_start_time = datetime.now()
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoExposureLock(False)
        self.camCtrl.setAutoExposureEnable()
        self.camera_control_queue.send(self.camCtrl)

    def auto_focus_camera(self, cb_autofocus_finished=None):
        self._log_info_vsensor("start AutoFocus on Camera...")
        self._cb_autofocus_finished = cb_autofocus_finished
        # self._autofocus_finished = False
        self.autofocus_in_progress = True
        self._af_start_time = datetime.now()
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
        self.camCtrl.setAutoFocusTrigger()
        self.camera_control_queue.send(self.camCtrl)

    @staticmethod
    def calc_edge_parameter(image):
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

    def _calc_fps(self):
        while True:
            self.fps = self._fps_counter
            self._fps_counter = 0
            time.sleep(1.0)

    def _log_info_vsensor(self, message):
        message = str(message)
        log_message = f"[{self.name}]" + " - " + message
        logger.info(log_message)

    def _log_debug_vsensor(self, message):
        message = str(message)
        log_message = f"[{self.name}]" + " - " + message
        logger.debug(log_message)