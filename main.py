#!/usr/bin/env python3
import os
import threading
from platform import platform

import depthai as dai
import time
from datetime import datetime
import contextlib
import sys
from timeloop import Timeloop
import libs.functions as helper
import signal
import atexit
from datetime import timedelta
import mqtt_handler.mqtt_communication_handler_v2 as mqtt_communication_handler
from ads_handler.ads_handler import AdsHandler, MachineType
from visionSensor import VisionSensor
from libs.socket_handler import SocketHandler
import log_handler.log_handler as log_handler
from libs.config_file_handler import ConfigFileHandler
from libs.vs_process_image import VisionSensorSettings

CONFIG_DIR = ".vSensorM4"
HOME_DIR = os.path.expanduser('~')
FULL_CONFIG_DIR = os.path.join(HOME_DIR, CONFIG_DIR)
FULL_LOG_DIR = os.path.join(FULL_CONFIG_DIR, "logs")
INIT_CONFIG_FILE = os.path.join(FULL_CONFIG_DIR, "init.json")

if not os.path.exists(FULL_LOG_DIR):
    os.makedirs(FULL_LOG_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

main_logger = log_handler.setup_logger(
    logger_name="main",
    logfile=os.path.join(FULL_LOG_DIR, f"log_{timestamp}.log"),
    # level="DEBUG",
)

main_logger.info("Start visionSensorM4")

helper.delete_old_files(directory=FULL_LOG_DIR)

main_logger.info(f"HOME DIRECTORY: {HOME_DIR}")
main_logger.info(f"INITIALIZATION FILE: {INIT_CONFIG_FILE}")

live_view_fps_divider = 2
vs_front_live_view_frame_nr = 0
vs_rear_live_view_frame_nr = 0

vs_front_send_mqtt_image = False
vs_rear_send_mqtt_image = False

config_handler = ConfigFileHandler(INIT_CONFIG_FILE)

main_logger.info(f"DephtAi-Version : {dai.__version__}")

LOCAL_HOST_IP = "192.168.0.30"
PLC_IP_ADDRESS = "192.168.0.10"

main_logger.info(f"Waiting for PLC ({PLC_IP_ADDRESS}) to respond to ping...")
helper.wait_for_ping(PLC_IP_ADDRESS, interval=2.0, logger=main_logger)
main_logger.info(f"PLC ({PLC_IP_ADDRESS}) is reachable - continue initialization")

plc_handler = AdsHandler(
    local_host_ip=LOCAL_HOST_IP,
    plc_ip_address=PLC_IP_ADDRESS,
    route_name="vSensorM4",
)
# plc_handler.connect_to_plc()

plc_handler.connect()

CONNECT_TIMEOUT_S = 15
connect_deadline = time.time() + CONNECT_TIMEOUT_S
while not plc_handler.connected and time.time() < connect_deadline:
    time.sleep(0.1)

if not plc_handler.connected:
    main_logger.error(f"Could not connect to PLC within {CONNECT_TIMEOUT_S}s - exit program")
    sys.exit(1)

found_front_sensor = False
found_rear_sensor = False

devices_found = dai.Device.getAllAvailableDevices()

device_info_front_sensor = dai.DeviceInfo()
device_info_rear_sensor = dai.DeviceInfo()

if len(devices_found) != 2:
    main_logger.info(f"found only one sensor... - exit program")
    exit()

for device in devices_found:
    main_logger.info(f"found sensor ({device.getMxId()}) on state: {device.state}")

if config_handler.vs_front.serial and config_handler.vs_rear.serial:
    found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_handler.vs_front.serial)
    found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_handler.vs_rear.serial)
else:
    if len(devices_found) == 2:
        config_handler.vs_front.serial = devices_found[0].getMxId()
        config_handler.vs_rear.serial = devices_found[1].getMxId()
        config_handler.save_config()
        found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(config_handler.vs_front.serial)
        found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(config_handler.vs_rear.serial)

if not found_front_sensor:
    raise RuntimeError("FrontSensor not found!")

if not found_rear_sensor:
    raise RuntimeError("RearSensor not found!")

# plc_handler.vs_ctrl.vsRightSerialNumber.value = device_info_front_sensor

vs_front_settings = VisionSensorSettings(
    camera_center_position=plc_handler.vs_ctrl.vsRightImageCenterOffset.value,
    stop_position=plc_handler.vs_ctrl.vsRightStopPosition.value,
    edge_detection_range=plc_handler.vs_ctrl.edgeDetectionRange.value,
    is_film_type_negative=plc_handler.vs_ctrl.isFilmTypeNegative.value,
    slope_threshold=plc_handler.vs_ctrl.slopeThreshold.value,
    contrast_offset=plc_handler.vs_ctrl.contrastOffset.value,
    tile_center_offset=plc_handler.vs_ctrl.settings.imageTileCenterOffset.value,
    tile_width=plc_handler.vs_ctrl.settings.imageTileWidth.value,
    tile_height=plc_handler.vs_ctrl.settings.imageTileHeight.value,
    contrast_pic_height=plc_handler.vs_ctrl.settings.contrastPicHeight.value,
    contrast_pic_edge_offset=plc_handler.vs_ctrl.settings.contrastPicEdgeOffset.value,
    exposure_time=plc_handler.vs_ctrl.exposureTime.value,
    lens_position=plc_handler.vs_ctrl.vsRightLensPosition.value,
)
vs_rear_settings = VisionSensorSettings(
    camera_center_position=plc_handler.vs_ctrl.vsLeftImageCenterOffset.value,
    stop_position=plc_handler.vs_ctrl.vsLeftStopPosition.value,
    edge_detection_range=plc_handler.vs_ctrl.edgeDetectionRange.value,
    is_film_type_negative=plc_handler.vs_ctrl.isFilmTypeNegative.value,
    slope_threshold=plc_handler.vs_ctrl.slopeThreshold.value,
    contrast_offset=plc_handler.vs_ctrl.contrastOffset.value,
    tile_center_offset=plc_handler.vs_ctrl.settings.imageTileCenterOffset.value,
    tile_width=plc_handler.vs_ctrl.settings.imageTileWidth.value,
    tile_height=plc_handler.vs_ctrl.settings.imageTileHeight.value,
    contrast_pic_height=plc_handler.vs_ctrl.settings.contrastPicHeight.value,
    contrast_pic_edge_offset=plc_handler.vs_ctrl.settings.contrastPicEdgeOffset.value,
    exposure_time=plc_handler.vs_ctrl.exposureTime.value,
    lens_position=plc_handler.vs_ctrl.vsLeftLensPosition.value,
)


cam_vs_front = VisionSensor(
    device_info_front_sensor,
    vs_name="vs_front",
    vs_settings=vs_front_settings,
    camera_capture_width=config_handler.vs_front.capture_width,
    camera_capture_height=config_handler.vs_front.capture_height,
    # image_center_position=config_handler.vs_front.center_offset,
    # lens_position=config_handler.vs_front.lens_position,
    raw_image_height=plc_handler.vs_ctrl.rawImageHeight.value,
    raw_image_width=plc_handler.vs_ctrl.rawImageWidth.value,
    flip_image=False,
    fps=config_handler.general.fps,
)

cam_vs_rear = VisionSensor(
    device_info=device_info_rear_sensor,
    vs_name="vs_rear",
    vs_settings=vs_rear_settings,
    camera_capture_width=config_handler.vs_rear.capture_width,
    camera_capture_height=config_handler.vs_rear.capture_height,
    raw_image_height=plc_handler.vs_ctrl.rawImageHeight.value,
    raw_image_width=plc_handler.vs_ctrl.rawImageWidth.value,
    flip_image=True,
    fps=config_handler.general.fps,
)

def cb_auto_exposure_cameras_finished():
    if not cam_vs_front.auto_exposure_in_progress and not cam_vs_rear.auto_exposure_in_progress:
        mean_exposure = (cam_vs_front.exposure_time + cam_vs_rear.exposure_time) // 2
        main_logger.info(f"AutoExposure Finished")
        main_logger.info(f"Exposure cam_front: {cam_vs_front.exposure_time}us")
        main_logger.info(f"Exposure cam_rear : {cam_vs_rear.exposure_time}us")
        main_logger.info(f"Set mean-exposure-time to: {mean_exposure}us")
        plc_handler.vs_ctrl.exposureTime.value = mean_exposure
        plc_handler.vs_ctrl.isConnected.value = True

def cb_auto_focus_finished():
    if not cam_vs_front.autofocus_in_progress and not cam_vs_rear.auto_exposure_in_progress:
        main_logger.info("AutoFocus Finished")
        main_logger.info(f"Lens-Position cam_front: {cam_vs_front.lens_position}")
        main_logger.info(f"Lens-Position cam_rear : {cam_vs_rear.lens_position}")
        plc_handler.vs_ctrl.vsRightLensPosition.value = cam_vs_front.lens_position
        plc_handler.vs_ctrl.vsLeftLensPosition.value = cam_vs_rear.lens_position
        plc_handler.vs_ctrl.isConnected.value = True

def cb_live_view(value):
    main_logger.info(f"set enable_live_view to: {type(value)}")

def poll_ads_symbols():
    vs_ctrl = plc_handler.vs_ctrl
    if vs_ctrl is None:
        return

    if vs_ctrl.rawImageHeight.new_value_available():
        main_logger.info(f"rawImageHeight: {vs_ctrl.rawImageHeight.value}")
        cam_vs_front.raw_image_height = vs_ctrl.rawImageHeight.value
        cam_vs_rear.raw_image_height = vs_ctrl.rawImageHeight.value

    if vs_ctrl.rawImageWidth.new_value_available():
        main_logger.info(f"rawImageWidth: {vs_ctrl.rawImageWidth.value}")
        cam_vs_front.raw_image_width = vs_ctrl.rawImageWidth.value
        cam_vs_rear.raw_image_width = vs_ctrl.rawImageWidth.value

    if vs_ctrl.rawImageCropTop.new_value_available():
        cam_vs_rear.raw_image_crop_top = vs_ctrl.rawImageCropTop.value

    if vs_ctrl.vsRightImageCenterOffset.new_value_available():
        cam_vs_front.settings.camera_center_position = int(vs_ctrl.vsRightImageCenterOffset.value)

    if vs_ctrl.vsLeftImageCenterOffset.new_value_available():
        cam_vs_rear.settings.camera_center_position = int(vs_ctrl.vsLeftImageCenterOffset.value)

    if vs_ctrl.vsRightStopPosition.new_value_available():
        cam_vs_front.settings.stop_position = int(vs_ctrl.vsRightStopPosition.value)

    if vs_ctrl.vsLeftStopPosition.new_value_available():
        cam_vs_rear.settings.stop_position = int(vs_ctrl.vsLeftStopPosition.value)

    if vs_ctrl.slopeThreshold.new_value_available():
        cam_vs_front.settings.slope_threshold = int(vs_ctrl.slopeThreshold.value)
        cam_vs_rear.settings.slope_threshold = int(vs_ctrl.slopeThreshold.value)

    if vs_ctrl.settings.imageTileCenterOffset.new_value_available():
        cam_vs_front.settings.tile_center_offset = int(vs_ctrl.settings.imageTileCenterOffset.value)
        cam_vs_rear.settings.tile_center_offset = int(vs_ctrl.settings.imageTileCenterOffset.value)

    if vs_ctrl.settings.imageTileHeight.new_value_available():
        cam_vs_front.settings.tile_height = int(vs_ctrl.settings.imageTileHeight.value)
        cam_vs_rear.settings.tile_height = int(vs_ctrl.settings.imageTileHeight.value)

    if vs_ctrl.settings.imageTileWidth.new_value_available():
        cam_vs_front.settings.tile_width = int(vs_ctrl.settings.imageTileWidth.value)
        cam_vs_rear.settings.tile_width = int(vs_ctrl.settings.imageTileWidth.value)

    if vs_ctrl.contrastOffset.new_value_available():
        cam_vs_front.settings.contrast_offset = int(vs_ctrl.contrastOffset.value)
        cam_vs_rear.settings.contrast_offset = int(vs_ctrl.contrastOffset.value)

    if vs_ctrl.swapCameras.new_value_available():
        if vs_ctrl.swapCameras.value:
            config_handler.swap_cameras()
            vs_ctrl.swapCameras.value = False

    if vs_ctrl.exposureTime.new_value_available():
        cam_vs_front.exposure_time = vs_ctrl.exposureTime.value
        cam_vs_rear.exposure_time = vs_ctrl.exposureTime.value

    if vs_ctrl.edgeDetectionRange.new_value_available():
        cam_vs_front.settings.edge_detection_range = int(vs_ctrl.edgeDetectionRange.value)
        cam_vs_rear.settings.edge_detection_range = int(vs_ctrl.edgeDetectionRange.value)

    if vs_ctrl._autoExposureCameras.new_value_available():
        def _apply():
            main_logger.info("Start AutoFocus Cameras...")
            vs_ctrl.isConnected.value = False
            cam_vs_front.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
            cam_vs_rear.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
            vs_ctrl._autoExposureCameras.value = False

        if vs_ctrl._autoExposureCameras.value:
            threading.Thread(target=_apply, daemon=True).start()

    if vs_ctrl._autoFocusCameras.new_value_available():
        def _apply():
            main_logger.info("Start AutoFocus Cameras...")
            vs_ctrl.isConnected.value = False
            cam_vs_front.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
            cam_vs_rear.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
            vs_ctrl._autoFocusCameras.value = False

        if vs_ctrl._autoFocusCameras.value:
            threading.Thread(target=_apply, daemon=True).start()

    if vs_ctrl.isFilmTypeNegative.new_value_available():
        value = vs_ctrl.isFilmTypeNegative.value
        main_logger.info(f"set film_type_is_negative to: {value}")
        cam_vs_front.settings.is_film_type_negative = value
        cam_vs_rear.settings.is_film_type_negative = value
        if value:
            cam_vs_front.exposure_time = vs_ctrl.settings.stdExposureTimeNegative.value
            cam_vs_rear.exposure_time = vs_ctrl.settings.stdExposureTimeNegative.value
        else:
            cam_vs_front.exposure_time = vs_ctrl.settings.stdExposureTimePositive.value
            cam_vs_rear.exposure_time = vs_ctrl.settings.stdExposureTimePositive.value

    if vs_ctrl.settings.contrastPicHeight.new_value_available():
        cam_vs_front.settings.contrast_pic_height = vs_ctrl.settings.contrastPicHeight.value
        cam_vs_rear.settings.contrast_pic_height = vs_ctrl.settings.contrastPicHeight.value

    if vs_ctrl.settings.contrastPicEdgeOffset.new_value_available():
        cam_vs_front.settings.contrast_pic_offset = vs_ctrl.settings.contrastPicEdgeOffset.value
        cam_vs_rear.settings.contrast_pic_offset = vs_ctrl.settings.contrastPicEdgeOffset.value

    if vs_ctrl.settings.stdExposureTimeNegative.new_value_available():
        cam_vs_front.settings.std_exposure_time_negative = vs_ctrl.settings.stdExposureTimeNegative.value
        cam_vs_rear.settings.std_exposure_time_negative = vs_ctrl.settings.stdExposureTimeNegative.value

    if vs_ctrl.settings.stdExposureTimePositive.new_value_available():
        cam_vs_front.settings.std_exposure_time_positive = vs_ctrl.settings.stdExposureTimePositive.value
        cam_vs_rear.settings.std_exposure_time_positive = vs_ctrl.settings.stdExposureTimePositive.value

    if vs_ctrl.vsRightStopOffset.new_value_available():
        cam_vs_front.settings.stop_position = vs_ctrl.vsRightStopOffset.value
        cam_vs_rear.settings.stop_position = vs_ctrl.vsRightStopOffset.value

    if vs_ctrl.vsLeftStopOffset.new_value_available():
        cam_vs_front.settings.stop_position = vs_ctrl.vsLeftStopOffset.value
        cam_vs_rear.settings.stop_position = vs_ctrl.vsLeftStopOffset.value

# endregion

main_logger.info(f"Connect to MQTT-Broker...")
mqtt = mqtt_communication_handler.MqttHandler(logger_enabled=True, client_type="vsController", client_id="vsController")

tl = Timeloop()

@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.pTopics_vsController.set_vsController_heartbeat.value = 3

tl.start()

socket_handler = SocketHandler(host=LOCAL_HOST_IP, port=4001)

cam_vs_rear.is_running = True
cam_vs_front.is_running = True

def exit_handler():
    plc_handler.close_plc_connection()
    main_logger.info("Cleaning up")

def kill_handler(*args):
    sys.exit(0)

atexit.register(exit_handler)
signal.signal(signal.SIGINT, kill_handler)
signal.signal(signal.SIGTERM, kill_handler)

interval_send_edge_position = helper.IntervalTimer(interval=0.01)
send_vs_front_edge_position = False

interval_poll_ads = helper.IntervalTimer(interval=0.1)

vs_front_send_mqtt_image = True
vs_rear_send_mqtt_image = True

with (contextlib.ExitStack() as stack):
    while True:

        ##################################################################################################################
        # ADS POLLING PART
        ##################################################################################################################

        if interval_poll_ads.is_time_to_update():
            poll_ads_symbols()

        ##################################################################################################################
        # IMAGE PROCESSING PART
        ##################################################################################################################

        if cam_vs_front.new_image_available:
            plc_handler.vs_ctrl.vsRightEdgeState.value = cam_vs_front.results.result_mean.edge_state
            if plc_handler.vs_ctrl.enableLiveView.value:
                if vs_front_live_view_frame_nr == live_view_fps_divider:
                    socket_handler.send_image(
                        vs_front_slope_data=cam_vs_front.results,
                        vs_rear_slope_data=cam_vs_rear.results,
                    )
                    vs_front_live_view_frame_nr = 0
                vs_front_live_view_frame_nr += 1
            plc_handler.vs_ctrl.vsRightSensorFps.value = cam_vs_front.fps

        if cam_vs_rear.new_image_available:
            plc_handler.vs_ctrl.vsLeftEdgeState.value = cam_vs_rear.results.result_mean.edge_state
            plc_handler.vs_ctrl.vsLeftSensorFps.value = cam_vs_rear.fps

        if interval_send_edge_position.is_time_to_update():
            if send_vs_front_edge_position:
                plc_handler.vs_ctrl.vsRightEdgePosition.value = cam_vs_front.results.result_mean.edge_position
            else:
                plc_handler.vs_ctrl.vsLeftEdgePosition.value = cam_vs_rear.results.result_mean.edge_position
            send_vs_front_edge_position = not send_vs_front_edge_position

        # if platform.system() == "Windows":
        #     pass


        # if plc_handler.vs_ctrl.captureImage.value:
        #     main_logger.debug(f"Send image_data data....")
        #     socket_handler.send_image(
        #         vs_front_slope_data=cam_vs_front.results,
        #         vs_rear_slope_data=cam_vs_rear.results,
        #     )
        #     vs_front_send_mqtt_image = True
        #     vs_rear_send_mqtt_image = True
        #     plc_handler.vs_ctrl.captureImage.value = False

        if vs_front_send_mqtt_image:
            mqtt.pTopics_vsController.set_vsFront_imageData.value = cam_vs_front.get_base64_image()
            mqtt.pTopics_vsController.set_vsFront_getImageWidth.value = cam_vs_front.results.image_width
            mqtt.pTopics_vsController.set_vsFront_getImageHeight.value = cam_vs_front.results.image_height
            mqtt.pTopics_vsController.set_vsFront_image_statistics.value = str(cam_vs_front.calc_statistics())
            vs_front_send_mqtt_image = False

        if vs_rear_send_mqtt_image:
            mqtt.pTopics_vsController.set_vsRear_imageData.value = cam_vs_rear.get_base64_image()
            mqtt.pTopics_vsController.set_vsRear_getImageWidth.value = cam_vs_rear.results.image_width
            mqtt.pTopics_vsController.set_vsRear_getImageHeight.value = cam_vs_rear.results.image_height
            mqtt.pTopics_vsController.set_vsRear_image_statistics.value = str(cam_vs_rear.calc_statistics())
            vs_rear_send_mqtt_image = False

        time.sleep(0.0001)

