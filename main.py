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

vs_front_last_edge_position = 0
vs_rear_last_edge_position = 0

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

main_logger.info("set init values to plc")

# plc_handler.vs_ctrl.vsRightLensPosition.value = config_handler.vs_front.lens_position
# plc_handler.vs_ctrl.vsRightSerialNumber.value = config_handler.vs_front.serial
# plc_handler.vs_ctrl.vsRightImageCenterOffset.value = config_handler.vs_front.center_offset
#
# plc_handler.vs_ctrl.vsLeftLensPosition.value = config_handler.vs_rear.lens_position
# plc_handler.vs_ctrl.vsLeftSerialNumber.value = config_handler.vs_rear.serial
# plc_handler.vs_ctrl.vsLeftImageCenterOffset.value = config_handler.vs_rear.center_offset
#
# plc_handler.vs_ctrl.settings.stdExposureTimePositive.value = config_handler.general.exposure_time_positive
# plc_handler.vs_ctrl.settings.stdExposureTimeNegative.value = config_handler.general.exposure_time_negative
# plc_handler.vs_ctrl.rawImageCropTop.value = config_handler.vs_front.raw_image_crop_top
# plc_handler.vs_ctrl.rawImageHeight.value = config_handler.vs_front.raw_image_height
# plc_handler.vs_ctrl.rawImageWidth.value = config_handler.vs_front.raw_image_width

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
        config_handler.vs_front.serial = devices_found[0].getDeviceId()
        config_handler.vs_rear.serial = devices_found[1].getDeviceId()
        config_handler.save_config()
        found_front_sensor, device_info_front_sensor = dai.Device.getDeviceById(config_handler.vs_front.serial)
        found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceById(config_handler.vs_rear.serial)

if not found_front_sensor:
    raise RuntimeError("FrontSensor not found!")

if not found_rear_sensor:
    raise RuntimeError("RearSensor not found!")


cam_vs_front = VisionSensor(
    device_info_front_sensor,
    vs_name="vs_front",
    camera_capture_width=config_handler.vs_front.capture_width,
    camera_capture_height=config_handler.vs_front.capture_height,
    image_center_position=config_handler.vs_front.center_offset,
    lens_position=config_handler.vs_front.lens_position,
    raw_image_height=plc_handler.vs_ctrl.rawImageHeight.value,
    raw_image_width=plc_handler.vs_ctrl.rawImageWidth.value,
    flip_image=False,
    fps=config_handler.general.fps,
)

cam_vs_rear = VisionSensor(
    device_info=device_info_rear_sensor,
    vs_name="vs_rear",
    camera_capture_width=config_handler.vs_rear.capture_width,
    camera_capture_height=config_handler.vs_rear.capture_height,
    image_center_position=config_handler.vs_rear.center_offset,
    lens_position=config_handler.vs_rear.lens_position,
    raw_image_height=plc_handler.vs_ctrl.rawImageHeight.value,
    raw_image_width=plc_handler.vs_ctrl.rawImageWidth.value,
    flip_image=True,
    fps=config_handler.general.fps,

)

def cb_film_type_is_negative(value):
    def _apply():
        main_logger.info(f"set film_type_is_negative to: {value}")
        cam_vs_front.settings.is_film_type_negative = value
        cam_vs_rear.settings.is_film_type_negative = value
        if value:
            cam_vs_front.exposure_time = config_handler.general.exposure_time_negative
            cam_vs_rear.exposure_time = config_handler.general.exposure_time_negative
        else:
            cam_vs_front.exposure_time = config_handler.general.exposure_time_positive
            cam_vs_rear.exposure_time = config_handler.general.exposure_time_positive
    threading.Thread(target=_apply, daemon=True).start()

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
        config_handler.vs_front.lens_position = cam_vs_front.lens_position
        config_handler.vs_rear.lens_position = cam_vs_rear.lens_position
        config_handler.save_config()
        plc_handler.vs_ctrl.isConnected.value = True

def cb_auto_exposure_cameras(value):
    def _apply():
        main_logger.info(f"Start AutoExposure...")
        plc_handler.vs_ctrl.isConnected.value = False
        cam_vs_front.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        cam_vs_rear.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        plc_handler.vs_ctrl._autoExposureCameras.value = False
    if value:
        threading.Thread(target=_apply, daemon=True).start()

def cb_auto_focus_cameras(value):
    def _apply():
        main_logger.info("Start AutoFocus Cameras...")
        plc_handler.vs_ctrl.isConnected.value = False
        cam_vs_front.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        cam_vs_rear.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        plc_handler.vs_ctrl._autoFocusCameras.value = False
    if value:
        threading.Thread(target=_apply, daemon=True).start()

def cb_swap_cameras(value):
    def _apply():
        config_handler.swap_cameras()
        plc_handler.vs_ctrl.swapCameras.value = False
    if value:
        threading.Thread(target=_apply, daemon=True).start()

def cb_slope_threshold(value):
    cam_vs_front.settings.slope_threshold = value
    cam_vs_rear.settings.slope_threshold = value

def cb_image_tile_center_offset(value):
    cam_vs_front.settings.tile_center_offset = value
    cam_vs_rear.settings.tile_center_offset = value

def cb_contrast_offset(value):
    cam_vs_front.settings.contrast_offset = value
    cam_vs_rear.settings.contrast_offset = value

def cb_image_tile_width(value):
    cam_vs_front.settings.tile_width = int(value)
    cam_vs_rear.settings.tile_width = int(value)

def cb_image_tile_height(value):
    cam_vs_front.settings.tile_height = int(value)
    cam_vs_rear.settings.tile_height = int(value)

# def cb_vs_front_image_center_position(value):
#     cam_vs_front.settings.camera_center_position = int(value)
#     config_handler.vs_front.center_offset = value
#     config_handler.save_config()

# def cb_vs_rear_image_center_position(value):
#     cam_vs_rear.settings.camera_center_position = int(value)
#     config_handler.vs_rear.center_offset = value
#     config_handler.save_config()

def cb_edge_detection_range(value):
    cam_vs_front.settings.edge_detection_range = int(value)
    cam_vs_rear.settings.edge_detection_range = int(value)

def cb_contrast_pic_height(value):
    cam_vs_front.settings.contrast_pic_height = int(value)
    cam_vs_rear.settings.contrast_pic_height = int(value)

def cb_contrast_pic_edge_offset(value):
    cam_vs_front.contrast_pic_edge_offset = int(value)
    cam_vs_rear.contrast_pic_edge_offset = int(value)

def cb_live_view(value):
    main_logger.info(f"set enable_live_view to: {type(value)}")

# def cb_raw_image_crop_top(value):
#     cam_vs_rear.raw_image_crop_top = value
#     config_handler.vs_rear.raw_image_crop_top = value
#     config_handler.save_config()

def cb_vs_front_raw_image_crop_top(value):
    cam_vs_front.raw_image_crop_top = value
    config_handler.vs_front.raw_image_crop_top = value
    config_handler.save_config()

# def cb_raw_image_height(value):
#     cam_vs_front.raw_image_height = value
#     cam_vs_rear.raw_image_height = value
#     config_handler.vs_front.raw_image_height = value
#     config_handler.vs_rear.raw_image_height = value
#     config_handler.save_config()
#
# def cb_raw_image_width(value):
#     cam_vs_front.raw_image_width = value
#     cam_vs_rear.raw_image_width = value
#     config_handler.vs_front.raw_image_width = value
#     config_handler.vs_rear.raw_image_width = value
#     config_handler.save_config()

def cb_vs_rear_raw_image_height_offset(value):
    cam_vs_rear._raw_image_height_offset = value
    config_handler.vs_rear.raw_image_height_offset = value
    config_handler.save_config()

def cb_vs_rear_raw_image_width_offset(value):
    cam_vs_rear._raw_image_width_offset = value
    config_handler.vs_rear.raw_image_width_offset = value
    config_handler.save_config()

def cb_vs_front_raw_image_width(value):
    cam_vs_front._raw_image_width = value
    cam_vs_rear._raw_image_width = value
    config_handler.vs_front.raw_image_width = value
    config_handler.vs_rear.raw_image_width = value
    config_handler.save_config()

def cb_vs_front_raw_image_height_offset(value):
    cam_vs_front._raw_image_height_offset = value
    config_handler.vs_front.raw_image_height_offset = value
    config_handler.save_config()

def cb_vs_front_stop_position(value):
    cam_vs_front.settings.stop_position = value

def cb_vs_rear_stop_position(value):
    cam_vs_rear.settings.stop_position = value

def cb_vs_front_stop_offset(value):
    cam_vs_front.settings.stop_offset = value

def cb_vs_rear_stop_offset(value):
    cam_vs_rear.settings.stop_offset = value

def cb_std_exposure_time_pos(value):
    config_handler.general.exposure_time_positive = value
    config_handler.save_config()

def cb_std_exposure_time_neg(value):
    config_handler.general.exposure_time_negative = value
    config_handler.save_config()

def cb_exposure_time(value):
    if value < 500:
        value = 1200
    cam_vs_front.exposure_time = value
    cam_vs_rear.exposure_time = value

# region ADS-VARIABLES (polled)
# NOTE: values are updated via polling (see poll_ads_symbols()) instead of ADS notifications,
# so that slow/blocking callbacks can never stall the pyads notification thread.
# Symbols are resolved from plc_handler.vs_ctrl fresh on every poll (not captured once here),
# because AdsHandler replaces vs_ctrl with a brand-new object on every (re)connect - holding
# on to the old AdsSymbol references would silently stop them from ever updating again.
polled_symbols = [
    (lambda vs: vs.isFilmTypeNegative, cb_film_type_is_negative),
    (lambda vs: vs._autoExposureCameras, cb_auto_exposure_cameras),
    (lambda vs: vs._autoFocusCameras, cb_auto_focus_cameras),
    (lambda vs: vs.slopeThreshold, cb_slope_threshold),
    (lambda vs: vs.swapCameras, cb_swap_cameras),
    (lambda vs: vs.contrastOffset, cb_contrast_offset),
    (lambda vs: vs.edgeDetectionRange, cb_edge_detection_range),
    (lambda vs: vs.exposureTime, cb_exposure_time),

    (lambda vs: vs.settings.imageTileWidth, cb_image_tile_width),
    (lambda vs: vs.settings.imageTileHeight, cb_image_tile_height),
    (lambda vs: vs.settings.imageTileCenterOffset, cb_image_tile_center_offset),
    (lambda vs: vs.settings.contrastPicHeight, cb_contrast_pic_height),
    (lambda vs: vs.settings.contrastPicEdgeOffset, cb_contrast_pic_edge_offset),
    (lambda vs: vs.settings.stdExposureTimePositive, cb_std_exposure_time_pos),
    (lambda vs: vs.settings.stdExposureTimeNegative, cb_std_exposure_time_neg),

    # (lambda vs: vs.rawImageCropTop, cb_raw_image_crop_top),
    # (lambda vs: vs.rawImageHeight, cb_raw_image_height),
    # (lambda vs: vs.rawImageWidth, cb_raw_image_width),

    # (lambda vs: vs.vsRightImageCenterOffset, cb_vs_front_image_center_position),
    # (lambda vs: vs.vsLeftImageCenterOffset, cb_vs_rear_image_center_position),
    (lambda vs: vs.vsLeftStopPosition, cb_vs_rear_stop_position),
    (lambda vs: vs.vsRightStopPosition, cb_vs_front_stop_position),
    (lambda vs: vs.vsLeftStopOffset, cb_vs_rear_stop_offset),
    (lambda vs: vs.vsRightStopOffset, cb_vs_front_stop_offset),
]

def poll_ads_symbols():
    vs_ctrl = plc_handler.vs_ctrl
    if vs_ctrl is None:
        return

    for get_symbol, callback in polled_symbols:
        symbol = get_symbol(vs_ctrl)
        if symbol.new_value_available():
            callback(symbol.last_value)

    if vs_ctrl.rawImageHeight.new_value_available():
        main_logger.info(f"rawImageHeight: {vs_ctrl.rawImageHeight.value}")
        cam_vs_front.raw_image_height = vs_ctrl.rawImageHeight.value
        cam_vs_rear.raw_image_height = vs_ctrl.rawImageHeight.value
        # config_handler.vs_front.raw_image_height = vs_ctrl.rawImageHeight.value
        # config_handler.vs_rear.raw_image_height = vs_ctrl.rawImageHeight.value
        # config_handler.save_config()

    if vs_ctrl.rawImageWidth.new_value_available():
        main_logger.info(f"rawImageWidth: {vs_ctrl.rawImageWidth.value}")
        cam_vs_front.raw_image_width = vs_ctrl.rawImageWidth.value
        cam_vs_rear.raw_image_width = vs_ctrl.rawImageWidth.value
        # config_handler.vs_front.raw_image_width = vs_ctrl.rawImageWidth.value
        # config_handler.vs_rear.raw_image_width = vs_ctrl.rawImageWidth.value
        # config_handler.save_config()

    if vs_ctrl.rawImageCropTop.new_value_available():
        cam_vs_rear.raw_image_crop_top = vs_ctrl.rawImageCropTop.value
        # config_handler.vs_rear.raw_image_crop_top = vs_ctrl.rawImageCropTop.value
        # config_handler.save_config()

    if vs_ctrl.vsRightImageCenterOffset.new_value_available():
        cam_vs_front.settings.camera_center_position = int(vs_ctrl.vsRightImageCenterOffset.value)

    if vs_ctrl.vsLeftImageCenterOffset.new_value_available():
        cam_vs_rear.settings.camera_center_position = int(vs_ctrl.vsLeftImageCenterOffset.value)

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

