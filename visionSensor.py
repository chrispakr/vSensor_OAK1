import threading
from datetime import datetime
from typing import Tuple
from depthai import DeviceInfo
import libs.vs_process_image as vps
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
class VisionSensor:
    fps_report_time = 3
    def __init__(self,
                 device_info: DeviceInfo,
                 vs_name: str,
                 camera_capture_width: int,
                 camera_capture_height: int,
                 image_center_position: int,
                 lens_position: int,
                 raw_image_height: int = 500,
                 raw_image_width: int = 0,
                 crop_raw_image_bottom: int = 0,
                 crop_raw_image_left: int = 0,
                 fps: int = 60,
                 ):
        # general Variables
        super().__init__()
        self.settings = vps.VisionSensorSettings()
        self._camera_capture_width = camera_capture_width
        self._camera_capture_height = camera_capture_height
        self._raw_image_height = raw_image_height
        self._raw_image_width = raw_image_width
        self._raw_image_height_offset = crop_raw_image_bottom
        self._raw_image_width_offset = crop_raw_image_left
        self.settings.camera_center_position = image_center_position
        self._lens_position = lens_position
        self.set_fps = fps
        self.controlIn = None
        self._x_out_edge_detection = None
        self._manip_edge_detection = None
        self._camRgb = None

        self.name = vs_name
        self.device_info = device_info

        self.results = vps.ProcessImageEdgeParameters(
            vs_settings=self.settings,
        )

        self._new_image_available = False

        # processing image_np variables
        self._sensor_image_data = None
        self._raw_input_image = None
        self._proc_image_centered = None

        self._image_info_jpg = None
        self._image_info_base64 = None

        # fps variables
        self.fps = 0
        self._fps_counter = 0
        self._fps_elapsed_time = datetime.now()

        # camera control variables
        self._camCtrl = None
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

        self._pipeline = None
        self._device = None
        self._image_edge_queue = None
        self._camera_control_queue = None
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
        self._pipeline = None
        self._pipeline = dai.Pipeline()

        # Define sources and outputs
        self._camRgb = self._pipeline.create(dai.node.ColorCamera)
        self._manip_edge_detection = self._pipeline.create(dai.node.ImageManip)
        self._x_out_edge_detection = self._pipeline.create(dai.node.XLinkOut)

        self.controlIn = self._pipeline.create(dai.node.XLinkIn)

        self._x_out_edge_detection.setStreamName('image_edge_detection')
        self.controlIn.setStreamName('control')

        # Properties
        logger.info(f"set camera-properties")
        logger.info(f"camera-capture_width: {self._camera_capture_width}")
        logger.info(f"camera-capture_height: {self._camera_capture_height}")
        self._camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_2024X1520)
        self._camRgb.setPreviewSize(self._camera_capture_width, self._camera_capture_height)
        self._camRgb.setFps(self.set_fps)
        self._camRgb.initialControl.setManualFocus(self._lens_position)
        self._camRgb.initialControl.setManualExposure(self._exposure_time, self._iso)
        self._camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        self._camRgb.setInterleaved(False)
        _max_frame_size = self._camRgb.getPreviewWidth() * self._camRgb.getPreviewHeight() * 3

        self._manip_edge_detection.initialConfig.setCropRect(
            xmin=0.09, ymin=0.1, xmax=0.91, ymax=0.65
        )

        self._manip_edge_detection.setMaxOutputFrameSize(_max_frame_size)
        self._manip_edge_detection.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        warp_left = 35

        p1 = dai.Point2f(warp_left, 0)
        p2 = dai.Point2f(1024 - warp_left, 0)
        p3 = dai.Point2f(0, 520)
        p4 = dai.Point2f(1024, 520)
        self._manip_edge_detection.setWarpMesh([p1, p2, p3, p4], 2, 2)

        # Links
        self._camRgb.preview.link(self._manip_edge_detection.inputImage)
        self._manip_edge_detection.out.link(self._x_out_edge_detection.input)
        self.controlIn.out.link(self._camRgb.inputControl)

        self._device = dai.Device(self._pipeline, self.device_info)
        self._image_edge_queue = self._device.getOutputQueue(name="image_edge_detection", maxSize=4, blocking=True)
        self._camera_control_queue = self._device.getInputQueue('control')

    def _process_image(self):
        while True:
            self._sensor_image_data = self._image_edge_queue.tryGet()
            if self._sensor_image_data is not None:
                self._fps_counter += 1
                self._image_info_jpg = None
                self._image_info_base64 = None
                self._raw_input_image = self._sensor_image_data.getCvFrame()
                (full_image_height, full_image_width) = self._raw_input_image.shape[:2]
                self._proc_image_centered = self._raw_input_image[
                                           0 : full_image_height,
                                           self.settings.camera_center_position - (self.settings.preview_width // 2):
                                           self.settings.camera_center_position + (self.settings.preview_width // 2)
                                            ]

                (centered_image_height, centered_image_width) = self._proc_image_centered.shape[:2]
                start_height = (centered_image_height // 2) - self._raw_image_height_offset - (self._raw_image_height // 2)
                end_height = (centered_image_height // 2) - self._raw_image_height_offset + (self._raw_image_height // 2)
                start_width = (centered_image_width // 2) - self._raw_image_width_offset - (self._raw_image_width // 2)
                end_width = (centered_image_width // 2) - self._raw_image_width_offset + (self._raw_image_width // 2)
                self.proc_image_roi = self._proc_image_centered[start_height: end_height, start_width: end_width]

                self.results.process_image(
                    image_data=self.proc_image_roi
                )

                if int(self._sensor_image_data.getExposureTime().total_seconds() * 1000000) != self._exposure_time:
                    self._exposure_time = int(self._sensor_image_data.getExposureTime().total_seconds() * 1000000)
                if self._sensor_image_data.getLensPosition() != self._lens_position:
                    self._lens_position = self._sensor_image_data.getLensPosition()
                    self._log_info_vsensor("lens-position changed to: {}".format(self._lens_position))
                if self.autofocus_in_progress:
                    if (datetime.now() - self._af_start_time).seconds > 2:
                        self._camCtrl = dai.CameraControl()
                        self.lens_position = self._lens_position
                        self._log_info_vsensor("disable AutoFocus")
                        self.autofocus_in_progress = False
                        if self._cb_autofocus_finished is not None:
                            self._cb_autofocus_finished()
                if self.auto_exposure_in_progress:
                    self._log_debug_vsensor("exposureTime: {}".format(str(self._exposure_time)))
                    if (datetime.now() - self._ae_start_time).seconds > 2:
                        self._camCtrl = dai.CameraControl()
                        self._camCtrl.setAutoExposureLock(True)
                        self._log_info_vsensor("disable AutoExposure")
                        self._camera_control_queue.send(self._camCtrl)
                        self.auto_exposure_in_progress = False
                        if self._cb_auto_exposure_finished is not None:
                            self._cb_auto_exposure_finished()

                self._new_image_available = True
            time.sleep(0.0001)

    def get_base64_image(self):
        try:
            logger.debug("create base64 image_np-data")
            image_np_color = cv2.cvtColor(self._proc_image_centered, cv2.COLOR_GRAY2RGB)
            self._image_info_jpg = self.jpeg.encode(image_np_color, quality=80)
            self._image_info_base64 = base64.b64encode(self._image_info_jpg)
            return self._image_info_base64
        except Exception as e:
            logger.error(e)
            return None

    def calc_statistics(self):
        if self.results.left_tile_slope_data.image_data is not None and self.results.right_tile_slope_data.image_data is not None:
            logger.info("start calculating statistics")
            tile_height, tile_width = self.results.left_tile_slope_data.image_data.shape
            stat_image_tile_left = self.results.left_tile_slope_data.image_data[
                                   0 : self.results.result_mean.edge_position - 20,
                                   0 : tile_width
                                   ]
            stat_image_tile_right = self.results.right_tile_slope_data.image_data[
                                    0 : self.results.result_mean.edge_position - 20,
                                    0 : tile_width
                                    ]
            stat_image_full = np.concatenate((stat_image_tile_left, stat_image_tile_right), axis=1)

            vs_maximum_dn = 256  # for image_np depth of byte
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
            try:
                # computing the mean and standard deviation. Note that the returned values are two-dimensional
                mean, std_dev = cv2.meanStdDev(stat_image_full)
                mean_flatten = mean.flatten()[0]
                std_dev_flatten = std_dev.flatten()[0]
            except Exception as e:
                logger.error(e)

            # flatten mean and std to obtain a vector and obtain the single value in it.
            return image_min, image_max, clip_min, clip_max, round(mean_flatten, 3), round(std_dev_flatten, 3)
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
        self._camCtrl = dai.CameraControl()
        self._camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        self._camCtrl.setManualFocus(self._lens_position)
        self._camera_control_queue.send(self._camCtrl)

    @property
    def numpy_image_array(self):
        return self.proc_image_roi

    @property
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value:int):
        self._exposure_time = value
        self._log_info_vsensor(f"Set value to: {self._exposure_time}")
        self._camCtrl = dai.CameraControl()
        self._camCtrl.setManualExposure(self._exposure_time, self._iso)
        self._camera_control_queue.send(self._camCtrl)

    def auto_exposure_camera(self, cb_auto_exposure_finished=None):
        self._log_info_vsensor("Sensor Auto-Exposure...")
        self._cb_auto_exposure_finished = cb_auto_exposure_finished
        self.auto_exposure_in_progress = True
        self._ae_start_time = datetime.now()
        self._camCtrl = dai.CameraControl()
        self._camCtrl.setAutoExposureLock(False)
        self._camCtrl.setAutoExposureCompensation(-2)
        self._camCtrl.setAutoExposureEnable()
        self._camera_control_queue.send(self._camCtrl)

    def auto_focus_camera(self, cb_autofocus_finished=None):
        self._log_info_vsensor(f"start AutoFocus on Camera...")
        self._cb_autofocus_finished = cb_autofocus_finished
        # self._autofocus_finished = False
        self.autofocus_in_progress = True
        self._af_start_time = datetime.now()
        self._camCtrl = dai.CameraControl()
        self._camCtrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
        self._camCtrl.setAutoFocusTrigger()
        self._camera_control_queue.send(self._camCtrl)

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