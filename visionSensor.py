import threading
from datetime import datetime
from typing import Optional, Tuple
from depthai import DeviceInfo
import libs.vs_process_image as vps
import depthai as dai
import numpy as np
import cv2
import time
import logging
import traceback


class VisionSensor:
    fps_report_time = 3
    DEVICE_CONNECT_RETRIES = 5
    DEVICE_CONNECT_RETRY_DELAY_S = 2.0
    EXPOSURE_TOLERANCE_US = 50
    EXPOSURE_TOLERANCE_REL = 0.02
    EXPOSURE_RESEND_INTERVAL_S = 1.0

    def __init__(self,
                 device_info: DeviceInfo,
                 vs_name: str,
                 camera_capture_width: int,
                 camera_capture_height: int,
                 edge_processing_settings: vps.ProcessEdgeSettings,
                 lens_position: int,
                 camera_center_position: int,
                 exposure_time: int = 1200,
                 preview_width: int = 800,
                 preview_height: int = 600,
                 warp_factor: int = 55,  # 55
                 raw_image_crop_top: int = 0,
                 raw_image_height: int = 380,
                 raw_image_width: int = 640,
                 flip_image:bool = False,
                 fps: int = 60,
                 new_image_event: Optional[threading.Event] = None,
                 ):
        # general Variables
        super().__init__()
        self.logger = logging.getLogger("main." + vs_name)
        self._camera_capture_width = camera_capture_width
        self._camera_capture_height = camera_capture_height
        self.raw_image_crop_top = raw_image_crop_top
        self._raw_image_height = raw_image_height
        self._raw_image_width = raw_image_width
        self._flip_image = flip_image
        self._lens_position = lens_position
        self._exposure_time = exposure_time
        self._requested_exposure_time = exposure_time
        # set via initialControl in the pipeline
        self._sent_exposure_time = exposure_time
        self._exposure_sent_time = time.monotonic()
        self._preview_width = preview_width
        self._preview_height = preview_height
        self._camera_center_position = camera_center_position
        self.edge_processing_settings = edge_processing_settings
        self._warp_factor = warp_factor
        self.set_fps = fps
        self._manip_edge_detection = None
        self._camRgb = None

        self.name = vs_name
        self.device_info = device_info

        self.results = vps.ProcessImageEdgeParameters(
            edge_processing_settings=self.edge_processing_settings,
        )

        self._new_image_available = False
        self._has_image = False
        self._new_image_event = new_image_event
        self._clamp_warned = set()
        self.is_running = False

        # processing image_np variables
        self._sensor_image_data = None
        self._raw_input_image = None
        self._proc_image_centered = None

        # fps variables
        self.fps = 0
        self._fps_counter = 0
        self._fps_elapsed_time = datetime.now()

        # camera control variables
        self._iso = 100
        self._camera_control_lock = threading.Lock()

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
        self.logger.info(f"init camera {self.device_info}")
        # self.logger.info(self.settings.__dict__)
        self.init_camera()

        self.camera_task = threading.Thread(target=self._process_image)
        self.camera_task.daemon = True
        self.camera_task.start()


    def _build_pipeline(self):
        self._pipeline = dai.Pipeline()

        # Define sources and outputs
        self._camRgb = self._pipeline.create(dai.node.ColorCamera)
        self._manip_edge_detection = self._pipeline.create(dai.node.ImageManip)
        self._x_out_edge_detection = self._pipeline.create(dai.node.XLinkOut)

        self.controlIn = self._pipeline.create(dai.node.XLinkIn)

        self._x_out_edge_detection.setStreamName('image_edge_detection')
        self.controlIn.setStreamName('control')

        # Properties
        self.logger.info(f"set camera-properties")
        self.logger.info(f"camera-capture_width: {self._camera_capture_width}")
        self.logger.info(f"camera-capture_height: {self._camera_capture_height}")
        self.logger.info(f"camera_lens_position: {self._lens_position}")
        self.logger.info(f"camera_exposure_time: {self._exposure_time}")
        self.logger.info(f"camera_iso: {self._iso}")
        self._camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_2024X1520)
        self._camRgb.setPreviewSize(self._camera_capture_width, self._camera_capture_height)
        self._camRgb.setFps(self.set_fps)
        self._camRgb.initialControl.setManualFocus(self._lens_position)
        self._camRgb.initialControl.setManualExposure(self._exposure_time, self._iso)
        self._camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        if self._flip_image:
            self._camRgb.setImageOrientation(dai.CameraImageOrientation.VERTICAL_FLIP)
        self._camRgb.setInterleaved(False)
        _max_frame_size = self._camRgb.getPreviewWidth() * self._camRgb.getPreviewHeight() * 3

        self._manip_edge_detection.initialConfig.setCropRect(
            xmin=0.09, ymin=0.1, xmax=0.91, ymax=0.65
        )

        self._manip_edge_detection.setMaxOutputFrameSize(_max_frame_size)
        self._manip_edge_detection.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

        p1 = dai.Point2f(self._warp_factor, 0)
        p2 = dai.Point2f(1024 - self._warp_factor, 0)
        p3 = dai.Point2f(0, 520)
        p4 = dai.Point2f(1024, 520)
        self._manip_edge_detection.setWarpMesh([p1, p2, p3, p4], 2, 2)

        # Links
        self._camRgb.preview.link(self._manip_edge_detection.inputImage)
        self._manip_edge_detection.out.link(self._x_out_edge_detection.input)
        self.controlIn.out.link(self._camRgb.inputControl)

    def init_camera(self):
        self._device = None
        for attempt in range(1, self.DEVICE_CONNECT_RETRIES + 1):
            self._build_pipeline()
            try:
                self._device = dai.Device(self._pipeline, self.device_info)
                break
            except RuntimeError as e:
                self.logger.error(
                    f"Failed to connect to device (attempt {attempt}/{self.DEVICE_CONNECT_RETRIES}): {e}"
                )
                if attempt == self.DEVICE_CONNECT_RETRIES:
                    raise
                time.sleep(self.DEVICE_CONNECT_RETRY_DELAY_S)

        # only the newest frame is of interest - with a deeper blocking queue a slow host would process stale frames
        self._image_edge_queue = self._device.getOutputQueue(name="image_edge_detection", maxSize=1, blocking=False)
        self._camera_control_queue = self._device.getInputQueue('control')

    def _process_image(self):
        while True:
            if not self.is_running:
                time.sleep(0.01)
            else:
                # blocks until the next frame arrives (instead of busy polling with tryGet)
                self._sensor_image_data = self._image_edge_queue.get()
                if self._sensor_image_data is not None:
                    self._fps_counter += 1
                    self._raw_input_image = self._sensor_image_data.getCvFrame()

                    (full_image_height, full_image_width) = self._raw_input_image.shape[:2]
                    start_width, end_width = self._clamped_window(
                        center=(full_image_width // 2) - self._camera_center_position,
                        width=self._preview_width,
                        total=full_image_width,
                        what="camera_center_position",
                    )
                    self._proc_image_centered = self._raw_input_image[0 : full_image_height, start_width : end_width]

                    (centered_image_height, centered_image_width) = self._proc_image_centered.shape[:2]
                    start_height = 0
                    end_height = self._raw_image_height
                    start_width, end_width = self._clamped_window(
                        center=centered_image_width // 2,
                        width=self._raw_image_width,
                        total=centered_image_width,
                        what="raw_image_width",
                    )
                    self.proc_image_roi = self._proc_image_centered[start_height: end_height, start_width: end_width]

                    self.results.process_image(
                        image_data=self.proc_image_roi
                    )

                    self._exposure_time = int(self._sensor_image_data.getExposureTime().total_seconds() * 1000000)

                    if not self.auto_exposure_in_progress:
                        self._apply_requested_exposure()

                    if self._sensor_image_data.getLensPosition() != self._lens_position:
                        self._lens_position = self._sensor_image_data.getLensPosition()
                        self.logger.info("lens-position changed to: {}".format(self._lens_position))
                    if self.autofocus_in_progress:
                        if (datetime.now() - self._af_start_time).seconds > 2:
                            self.lens_position = self._lens_position
                            self.logger.info("disable AutoFocus")
                            self.autofocus_in_progress = False
                            if self._cb_autofocus_finished is not None:
                                self._cb_autofocus_finished()
                    if self.auto_exposure_in_progress:
                        self.logger.debug("exposureTime: {}".format(str(self._exposure_time)))
                        if (datetime.now() - self._ae_start_time).seconds > 2:
                            cam_ctrl = dai.CameraControl()
                            cam_ctrl.setAutoExposureLock(True)
                            self.logger.info("disable AutoExposure")
                            with self._camera_control_lock:
                                self._camera_control_queue.send(cam_ctrl)
                            self._requested_exposure_time = self._exposure_time
                            # the camera is already locked at this exposure - nothing to send
                            self._sent_exposure_time = self._exposure_time
                            self.auto_exposure_in_progress = False
                            if self._cb_auto_exposure_finished is not None:
                                self._cb_auto_exposure_finished()

                    self._new_image_available = True
                    self._has_image = True
                    if self._new_image_event is not None:
                        self._new_image_event.set()

    def _clamped_window(self, center: int, width: int, total: int, what: str) -> Tuple[int, int]:
        """Returns (start, end) of a `width` wide window around `center`, shifted to stay inside [0, total].

        Without clamping a negative start index would wrap around in numpy slicing and yield an
        empty or wrong image section.
        """
        start = center - (width // 2)
        clamped_start = max(0, min(start, total - width))
        end = min(total, clamped_start + width)
        if clamped_start != start:
            if what not in self._clamp_warned:
                self._clamp_warned.add(what)
                self.logger.warning(
                    f"image window out of range ({what}): start={start}, width={width}, image_width={total} "
                    f"- clamped to [{clamped_start}:{end}]"
                )
        else:
            self._clamp_warned.discard(what)
        return clamped_start, end

    def _apply_requested_exposure(self):
        """Sends the requested exposure to the camera once per change.

        The sensor quantizes the exposure time, so the measured value may never exactly match the
        requested one. Only if it stays off by more than the tolerance, the command is re-sent
        (at most every EXPOSURE_RESEND_INTERVAL_S) - e.g. if a control message got lost.
        """
        now = time.monotonic()
        tolerance = max(self.EXPOSURE_TOLERANCE_US, self._requested_exposure_time * self.EXPOSURE_TOLERANCE_REL)
        changed = self._requested_exposure_time != self._sent_exposure_time
        deviates = abs(self._exposure_time - self._requested_exposure_time) > tolerance
        resend_due = (now - self._exposure_sent_time) >= self.EXPOSURE_RESEND_INTERVAL_S

        if changed or (deviates and resend_due):
            cam_ctrl = dai.CameraControl()
            cam_ctrl.setManualExposure(self._requested_exposure_time, self._iso)
            cam_ctrl.setAutoExposureLock(True)
            with self._camera_control_lock:
                self._camera_control_queue.send(cam_ctrl)
            self._sent_exposure_time = self._requested_exposure_time
            self._exposure_sent_time = now

    @property
    def has_image(self) -> bool:
        """True once the first frame has been processed (image and results are valid)."""
        return self._has_image

    def calc_statistics(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Calculate image statistics including min/max values and clipped histogram bounds.

        Returns:
            Tuple of (image_min, image_max, clip_min, clip_max, mean, std_dev) or None if images unavailable
        """
        # Constants
        EDGE_MARGIN = 20
        MAX_DN_VALUE = 256  # for image_np depth of byte
        CLIP_PERCENT = 0.05  # percent for histogram clipping (0.025% from each side)

        if not self._validate_image_data():
            return None

        self.logger.info("Start calculating statistics")

        try:
            # Prepare image data
            stat_image_full = self._prepare_concatenated_image(EDGE_MARGIN)

            # Calculate basic image statistics
            image_min, image_max, _, _ = cv2.minMaxLoc(stat_image_full)

            # Calculate histogram clipping bounds
            clip_min, clip_max = self._calculate_histogram_bounds(
                stat_image_full, MAX_DN_VALUE, CLIP_PERCENT)

            # Calculate mean and standard deviation
            mean, std_dev = self._calculate_image_statistics(stat_image_full)

            return (
                image_min,
                image_max,
                clip_min,
                clip_max,
                round(mean, 3),
                round(std_dev, 3)
            )

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            traceback.print_exc()
            return None

    def _validate_image_data(self) -> bool:
        """Check if required image data is available."""
        return (self.results.left_tile_slope_data.image_data is not None and
                self.results.right_tile_slope_data.image_data is not None)

    def _prepare_concatenated_image(self, edge_margin: int) -> np.ndarray:
        """Prepare concatenated image from left and right tiles."""
        tile_height, tile_width = self.results.left_tile_slope_data.image_data.shape

        stat_image_tile_left = self.results.left_tile_slope_data.image_data[
                               0: self.results.result_mean.edge_position - edge_margin,
                               0: tile_width
                               ]

        stat_image_tile_right = self.results.right_tile_slope_data.image_data[
                                0: self.results.result_mean.edge_position - edge_margin,
                                0: tile_width
                                ]

        return np.concatenate((stat_image_tile_left, stat_image_tile_right), axis=1)

    @staticmethod
    def _calculate_histogram_bounds(image: np.ndarray, max_value: int,
                                    clip_percent: float) -> Tuple[float, float]:
        """Calculate histogram clipping bounds based on a given percentage."""
        hist = cv2.calcHist([image], [0], None, [max_value], [0, max_value]).flatten()
        cutoff = image.shape[0] * image.shape[1] * clip_percent / 2

        clip_min = 0
        clip_max = max_value - 1

        # Calculate left bound
        accumulate = 0
        for i in range(len(hist)):
            accumulate += hist[i]
            if accumulate >= cutoff:
                clip_min = i
                break

        # Calculate right bound
        accumulate = 0
        for i in range(len(hist) - 1, -1, -1):
            accumulate += hist[i]
            if accumulate >= cutoff:
                clip_max = i
                break

        return clip_min, clip_max

    @staticmethod
    def _calculate_image_statistics(image: np.ndarray) -> Tuple[float, float]:
        """Calculate mean and standard deviation of the image."""
        mean, std_dev = cv2.meanStdDev(image)
        return float(mean.flatten()[0]), float(std_dev.flatten()[0])

    @property
    def new_image_available(self):
        if self._new_image_available:
            self._new_image_available = False
            return True
        else:
            return False

    @property
    def raw_image_width(self):
        return self._raw_image_width

    @raw_image_width.setter
    def raw_image_width(self, value):
        self.logger.info(f"set raw_image_width to {value}")
        self._raw_image_width = value

    @property
    def raw_image_height(self):
        return self._raw_image_height

    @raw_image_height.setter
    def raw_image_height(self, value):
        self.logger.info(f"set raw_image_height to {value}")
        self._raw_image_height = value

    @new_image_available.setter
    def new_image_available(self, value):
        pass

    @property
    def lens_position(self):
        return self._lens_position

    @lens_position.setter
    def lens_position(self, value):
        self._lens_position = value
        self.logger.debug(f"Set lens-position to: {self._lens_position}")
        cam_ctrl = dai.CameraControl()
        cam_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        cam_ctrl.setManualFocus(self._lens_position)
        with self._camera_control_lock:
            self._camera_control_queue.send(cam_ctrl)

    @property
    def camera_center_position(self):
        return self._camera_center_position

    @camera_center_position.setter
    def camera_center_position(self, value: int):
        self._camera_center_position = int(value)
        self.logger.info(f"Set camera_center_position to: {self._camera_center_position}")

    @property
    def numpy_image_array(self):
        return self.proc_image_roi

    @property
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value:int):
        self._requested_exposure_time = int(value)
        self.logger.info(f"Requested exposure_time: {self._requested_exposure_time}")

    def auto_exposure_camera(self, cb_auto_exposure_finished=None):
        self.logger.info("Sensor Auto-Exposure...")
        self._cb_auto_exposure_finished = cb_auto_exposure_finished
        self.auto_exposure_in_progress = True
        self._ae_start_time = datetime.now()
        cam_ctrl = dai.CameraControl()
        cam_ctrl.setAutoExposureLock(False)
        cam_ctrl.setAutoExposureCompensation(-2)
        cam_ctrl.setAutoExposureEnable()
        with self._camera_control_lock:
            self._camera_control_queue.send(cam_ctrl)

    def auto_focus_camera(self, cb_autofocus_finished=None):
        self.logger.info(f"start AutoFocus on Camera...")
        self._cb_autofocus_finished = cb_autofocus_finished
        # self._autofocus_finished = False
        self.autofocus_in_progress = True
        self._af_start_time = datetime.now()
        cam_ctrl = dai.CameraControl()
        cam_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
        cam_ctrl.setAutoFocusTrigger()
        with self._camera_control_lock:
            self._camera_control_queue.send(cam_ctrl)

    def _calc_fps(self):
        while True:
            self.fps = self._fps_counter
            self._fps_counter = 0
            time.sleep(1.0)
