from datetime import datetime
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

if platform.system() == "Windows":
    # log_info_general("set parameters for windows-system")
    jpeg = TurboJPEG("libs/libturbojpeg.dll")
    showOutput = True

if platform.system() == "Linux":
    # self._log_info_vsensor("set parameters for linux-system")
    jpeg = TurboJPEG()
    showOutput = False


class VisionSensorOperationMode(Enum):
        AUTO = 0
        FRONT_SENSOR = 1
        REAR_SENSOR = 2
        BOTH_SENSORS = 3

class VisionSensor:
    fps_report_time = 3
    _contrast_pic_height = 20
    _contrast_pic_edge_offset = 10

    def __init__(self, device_info, is_front_sensor, vs_name, fps, logger,
                 capture_width=860, capture_height=600, preview_width=800,
                 edge_detection_slope_lcm=30, edge_detection_slope_std=50,
                 lcm_contrast_offset=40, std_contrast_offset=100):
        # general Variables
        super().__init__()
        self.logger = logger
        self.controlIn = None
        self.x_out_edge_detection = None
        self.manip_edge_detection = None
        self.camRgb = None

        self.name = vs_name
        self.device_info = device_info
        self._is_front_sensor = is_front_sensor
        self.film_type_is_negative = True
        self._total_edge_slope = 0
        self._enabled_lcm = False
        self.edge_position_tile_diff = 0

        self.edge_position = 0
        self._edge_position = 0
        self.edge_state = 0
        self._edge_detected = False
        self._edge_in_position = False

        self._image_center_position = 450
        self.proc_image_width = 400
        self._stop_offset_compensation = 8
        self._edge_detection_range = 30
        # self._edge_status = 0
        self._stop_motor = False

        self.capture_width = capture_width
        self.capture_height = capture_height
        self.set_fps = fps
        self.preview_width = preview_width
        self._lcm_slope = edge_detection_slope_lcm
        self._std_slope = edge_detection_slope_std
        self._lcm_contrast_offset = lcm_contrast_offset
        self._std_contrast_offset = std_contrast_offset
        self.lcm_statistics = None

        # processing image variables
        self.input_image_data = None
        self.raw_input_image = None
        self.np_image_tile_left = None
        self.np_image_tile_right = None
        self.np_image_info_edge_line = None
        self.np_image_info_setup_lines = None
        self.proc_image_centered = None

        self._left_edge_position = 0
        self._right_edge_position = 0

        self.left_edge_slope = 0
        self.right_edge_slope = 0

        self.left_edge_results = None
        self.right_edge_results = None

        self._arr_slope_total_mean = deque(maxlen=40)
        self._slope_total_mean = 0
        self._new_edge_detected = 0
        self._slope_diff_rising = 0
        self._slope_diff_falling = 0

        self.image_info_jpg = None
        self.image_info_base64 = None
        self.captured_images = 0

        # preview image variables
        self._img_width = 0
        self._img_height = 0
        self.img_width = 0
        self.img_height = 0

        # fps variables
        self.fps_elapsed_time = datetime.now()
        self.fps_jpg_image_time = datetime.now()
        self.fps = 0
        self.fps_jpg_image = 0
        self._fps_counter = 0
        self._fps_counter_jpg_image = 0

        # statistics image variables
        self.stat_image_full = None
        self._in_pic_median_left = 0
        self._in_pic_median_right = 0
        self._out_pic_median_left = 0
        self._out_pic_median_right = 0
        self._in_pic_median_total = 0
        self._out_pic_median_total = 0

        # camera control variables
        self.camCtrl = None
        self.exposure_time = 0

        self.af_start_time = datetime.now()
        self.autoFocusEnabled = False
        self.autoFocusFinished = False

        self.ae_start_time = datetime.now()
        self.autoExposureEnabled = False
        self.autoExposureFinished = False

        # self.enable_live_view = False

        self.pipeline = None
        self._get_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)
        self.image_edge_queue = self.device.getOutputQueue(name="image_edge_detection", maxSize=4, blocking=True)
        self.camera_control_queue = self.device.getInputQueue('control')

        self.capture_time = time.time()

        self._stop_position = 350
        self._lens_position = 130

        self.line_color_red = (32, 43, 255)
        self.line_color_green = (0, 255, 0)
        self.line_color_orange = (0, 165, 255)
        self.line_color_proc_image = (255, 0, 0)
        self.line_color_stop_position = (32, 43, 255)
        self.line_color_center_position = (163, 136, 22)
        self.line_color_stop_offset = (170, 102, 255)

    @property
    def stop_position(self):
        return self._stop_position

    @stop_position.setter
    def stop_position(self, value):
        self._stop_position = value
        self._log_info_vsensor("change stop_position to: " + str(self._stop_position))

    @property
    def stop_offset_compensation(self):
        return self._stop_offset_compensation

    @stop_offset_compensation.setter
    def stop_offset_compensation(self, value):
        self._stop_offset_compensation = value
        self._log_info_vsensor("change stop_offset_compensation to: " + str(self._stop_offset_compensation))

    @property
    def edge_detection_range(self):
        return self._edge_detection_range

    @edge_detection_range.setter
    def edge_detection_range(self, value):
        self._edge_detection_range = value
        self._log_info_vsensor("change edge_detection_range to: " + str(self._edge_detection_range))

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

    def _get_pipeline(self):
        # Create pipeline
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

    def process_image(self):
        self.input_image_data = self.image_edge_queue.tryGet()
        if self.input_image_data is not None:
            self.np_image_info_setup_lines = None
            self.image_info_jpg = None
            self.image_info_base64 = None
            # self._log_info_vsensor("capture-time: " + str(time.time() - self.capture_time))
            self.capture_time = time.time()
            # self._edge_position.value = -1
            self.raw_input_image = self.input_image_data.getCvFrame()
            (full_image_height, full_image_width) = self.raw_input_image.shape[:2]
            if self._image_center_position < self.preview_width // 2:
                self._image_center_position = self.capture_width // 2
            self.proc_image_centered = self.raw_input_image[0:full_image_height, self._image_center_position - (self.preview_width // 2):self._image_center_position + (self.preview_width // 2)]
            self.np_image_tile_left = self.raw_input_image[0:self._stop_position + 50, self._image_center_position - self.proc_image_width:self._image_center_position-50]
            self.np_image_tile_right = self.raw_input_image[0:self._stop_position + 50, self._image_center_position+50:self._image_center_position + self.proc_image_width]

            self.left_edge_results = self.calc_edge_parameter(self.np_image_tile_left)
            self.right_edge_results = self.calc_edge_parameter(self.np_image_tile_right)

            if self.film_type_is_negative:
                self._left_edge_position = self.left_edge_results[3]
                self.left_edge_slope = int(self.left_edge_results[1])
                self._right_edge_position = self.right_edge_results[3]
                self.right_edge_slope = int(self.right_edge_results[1])
            else:
                self._left_edge_position = self.left_edge_results[2]
                self.left_edge_slope = int(self.left_edge_results[0])
                self._right_edge_position = self.right_edge_results[2]
                self.right_edge_slope = int(self.right_edge_results[0])

            self.edge_position_tile_diff = abs(self._left_edge_position - self._right_edge_position)
            self._total_edge_slope = self.left_edge_slope + self.right_edge_slope

            self._arr_slope_total_mean.append(self._total_edge_slope)
            self._slope_total_mean = sum(self._arr_slope_total_mean) // len(self._arr_slope_total_mean)
            self._slope_diff_rising = self._total_edge_slope - min(self._arr_slope_total_mean)
            self._slope_diff_falling = self._slope_total_mean - max(self._arr_slope_total_mean)
            # if self._is_front_sensor:
            #     self._slope_diff = self._total_edge_slope - self._slope_total_mean
            #     if self._slope_total_mean > 0:
            #         slope_div = self._total_edge_slope / self._slope_total_mean

            if (self._total_edge_slope - self._slope_total_mean) > 40 and self._new_edge_detected == 0:
                # if self._is_front_sensor:
                #     self._log_info_vsensor("edge-diff (rising): {}".format(self._slope_diff))
                self._new_edge_detected = 200
                self._arr_slope_total_mean.clear()
                self._arr_slope_total_mean.append(self._total_edge_slope)
            if (self._slope_total_mean - self._total_edge_slope) > 40 and self._new_edge_detected == 200:
                # if self._is_front_sensor:
                #     self._log_info_vsensor("edge-diff (falling): {}".format(self._slope_diff))
                self._new_edge_detected = 0
                self._arr_slope_total_mean.clear()
                self._arr_slope_total_mean.append(self._total_edge_slope)
            if self._enabled_lcm:
                if self._total_edge_slope > self._lcm_slope:
                    self._edge_position = (self._left_edge_position + self._right_edge_position) // 2
            else:
                if self._total_edge_slope > self._std_slope:
                    self._edge_position = (self._left_edge_position + self._right_edge_position) // 2

            if self._edge_position > (self._contrast_pic_height + self._contrast_pic_edge_offset):
                _in_pic_contrast_max_pos = self._edge_position - self._contrast_pic_edge_offset
                _in_pic_contrast_min_pos = _in_pic_contrast_max_pos - self._contrast_pic_height
                _in_pic_contrast_image_left = self.np_image_tile_left[_in_pic_contrast_min_pos:_in_pic_contrast_max_pos, 0:self.proc_image_width]
                _in_pic_contrast_image_right = self.np_image_tile_right[_in_pic_contrast_min_pos:_in_pic_contrast_max_pos, 0:self.proc_image_width]
                self._in_pic_median_left = np.median(_in_pic_contrast_image_left)
                self._in_pic_median_right = np.median(_in_pic_contrast_image_right)
                self._in_pic_median_total = self._in_pic_median_left + self._in_pic_median_right

                _out_pic_contrast_min_pos = self._edge_position + self._contrast_pic_edge_offset
                _out_pic_contrast_max_pos = _out_pic_contrast_min_pos + self._contrast_pic_height
                _out_pic_contrast_image_left = self.np_image_tile_left[_out_pic_contrast_min_pos:_out_pic_contrast_max_pos, 0:self.proc_image_width]
                _out_pic_contrast_image_right = self.np_image_tile_right[_out_pic_contrast_min_pos:_out_pic_contrast_max_pos, 0:self.proc_image_width]
                self._out_pic_median_left = np.median(_out_pic_contrast_image_left)
                self._out_pic_median_right = np.median(_out_pic_contrast_image_right)
                self._out_pic_median_total = self._out_pic_median_left + self._out_pic_median_right

            if not self._enabled_lcm:
                if self.film_type_is_negative:
                    if self._in_pic_median_total + self._std_contrast_offset < self._out_pic_median_total:
                        self._edge_detected = True
                    else:
                        self._edge_detected = False
                else:
                    if self._in_pic_median_total + self._std_contrast_offset > self._out_pic_median_total:
                        self._edge_detected = True
                    else:
                        self._edge_detected = False

            if self._enabled_lcm:
                if self.film_type_is_negative:
                    self.lcm_statistics = [self._total_edge_slope, self._out_pic_median_total - self._in_pic_median_total]
                    if self._in_pic_median_total + self._lcm_contrast_offset < self._out_pic_median_total:
                        self._edge_detected = True
                    else:
                        self._edge_detected = False
                else:
                    self.lcm_statistics = [self._total_edge_slope, self._in_pic_median_total - self._out_pic_median_total]
                    if self._in_pic_median_total + self._lcm_contrast_offset > self._out_pic_median_total:
                        self._edge_detected = True
                    else:
                        self._edge_detected = False

            if (self._stop_position - (self._edge_detection_range//2)) < self._edge_position < (self._stop_position + (self._edge_detection_range // 2)):
                self._edge_in_position = True
            else:
                self._edge_in_position = False

            if not self._edge_detected and not self._edge_in_position:
                self.edge_state = 0
                self.edge_position = -1

            if self._edge_detected and not self._edge_in_position:
                self.edge_state = 1
                self.edge_position = self._edge_position

            if self._edge_detected and self._edge_in_position:
                self.edge_state = 2
                self.edge_position = self._edge_position

            # if self._is_front_sensor:
            #     print(self.edge_position, self._edge_position, self._edge_detected, self._edge_in_position, self.edge_state)

            self.captured_images += 1
            self._fps_counter += 1

            if (datetime.now() - self.fps_elapsed_time).seconds >= self.fps_report_time:
                self.fps = self._fps_counter // self.fps_report_time
                self.fps_elapsed_time = datetime.now()
                # self._log_info_vsensor("FPS: {}".format(str(self.fps)))
                self._fps_counter = 0
            if int(self.input_image_data.getExposureTime().total_seconds() * 1000000) != self.exposure_time:
                self.exposure_time = int(self.input_image_data.getExposureTime().total_seconds() * 1000000)
            if self.input_image_data.getLensPosition() != self._lens_position:
                self._lens_position = self.input_image_data.getLensPosition()
                self._log_info_vsensor("lens-position changed to: {}".format(self._lens_position))
            if self.autoFocusEnabled:
                if (datetime.now() - self.af_start_time).seconds > 2:
                    self.camCtrl = dai.CameraControl()
                    self.focus_position = self._lens_position
                    self._log_info_vsensor("disable AutoFocus")
                    self.autoFocusEnabled = False
                    self.autoFocusFinished = True
            if self.autoExposureEnabled:
                self._log_info_vsensor("exposureTime: {}".format(str(self.exposure_time)))
                if (datetime.now() - self.ae_start_time).seconds > 2:
                    self.camCtrl = dai.CameraControl()
                    self.camCtrl.setAutoExposureLock(True)
                    self._log_info_vsensor("disable AutoExposure")
                    self.camera_control_queue.send(self.camCtrl)
                    self.autoExposureEnabled = False
                    self.autoExposureFinished = True
            return True
        else:
            return False

    def create_image_info(self):
        if self.proc_image_centered is not None:
            self.np_image_info_edge_line = cv2.cvtColor(self.proc_image_centered, cv2.COLOR_GRAY2RGB)
            self._img_height, self._img_width, dim = self.np_image_info_edge_line.shape
            self.img_width = self._img_width
            self.img_height = self._img_height
            if self.edge_state == 2:
                line_color = self.line_color_green
            else:
                line_color = self.line_color_orange

            cv2.line(self.np_image_info_edge_line, (0, self._stop_position), (100, self._stop_position), self.line_color_stop_position, 2)
            cv2.line(self.np_image_info_edge_line, (self._img_width-100, self._stop_position), (self._img_width, self._stop_position), self.line_color_stop_position, 2)
            cv2.line(self.np_image_info_edge_line, (self._img_width - 50, self._stop_position-self.stop_offset_compensation), (self._img_width, self._stop_position-self.stop_offset_compensation), self.line_color_stop_offset, 2)
            cv2.line(self.np_image_info_edge_line, (0, self._stop_position - self.stop_offset_compensation), (50, self._stop_position - self.stop_offset_compensation),
                     self.line_color_stop_offset, 2)
            cv2.line(self.np_image_info_edge_line, (0, self._edge_position), (self._img_width, self._edge_position), line_color, 2)
            self.np_image_info_setup_lines = self.np_image_info_edge_line
            cv2.line(self.np_image_info_setup_lines, (self._img_width // 2 - self.proc_image_width, 0),
                     (self._img_width // 2 - self.proc_image_width, self._img_height), self.line_color_proc_image, 1)
            cv2.line(self.np_image_info_setup_lines, (self._img_width // 2 + self.proc_image_width, 0),
                     (self._img_width // 2 + self.proc_image_width, self._img_height), self.line_color_proc_image, 1)
            cv2.line(self.np_image_info_setup_lines, (self._img_width // 2, 0), (self._img_width // 2, self._img_height), self.line_color_center_position, 1)
            cv2.line(self.np_image_info_setup_lines, (self._img_width // 2, self._img_height-60), (self._img_width // 2, self._img_height), self.line_color_center_position, 2)
            return True
        else:
            return False

    def create_image_info_jpg(self):
        if self.np_image_info_setup_lines is None:
            self.create_image_info()
        if self.np_image_info_setup_lines is not None:
            self.image_info_jpg = jpeg.encode(self.np_image_info_setup_lines, quality=80)
            self.image_info_base64 = base64.b64encode(self.image_info_jpg)
            return True
        else:
            return False

    def calc_statistics(self):
        if self.np_image_tile_left is not None and self.np_image_tile_right is not None:
            tile_height, tile_width = self.np_image_tile_left.shape
            stat_image_tile_left = self.np_image_tile_left[0:self._edge_position - 20, 0:tile_width]
            stat_image_tile_right = self.np_image_tile_right[0:self._edge_position - 20, 0:tile_width]
            self.stat_image_full = np.concatenate((stat_image_tile_left, stat_image_tile_right), axis=1)

    def auto_focus_camera(self):
        self._log_info_vsensor("Focus Camera...")
        self.autoFocusEnabled = True
        self.af_start_time = datetime.now()
        self.camCtrl = dai.CameraControl()
        self.camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
        self.camCtrl.setAutoFocusTrigger()
        self.camera_control_queue.send(self.camCtrl)

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

    def _log_info_vsensor(self, message):
        message = str(message)
        if self._is_front_sensor:
            log_message = "[vs-front]" + " - " + message
        else:
            log_message = "[vs-rear ]" + " - " + message
        self.logger.info(log_message)
