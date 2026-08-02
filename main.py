#!/usr/bin/env python3
import os
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
    logfile=os.path.join(FULL_LOG_DIR, f"log_{timestamp}.log")
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

plc_handler = AdsHandler(
    local_host_ip="192.168.0.30",
    plc_ip_address="192.168.0.10",
    route_name="vSensorM4",
)
# plc_handler.connect_to_plc()

plc_handler.connect()

time.sleep(2)

main_logger.info("set init values to plc")

plc_handler.vs_ctrl.vsRightLensPosition.value = config_handler.vs_front.lens_position
plc_handler.vs_ctrl.vsRightSerialNumber.value = config_handler.vs_front.serial
plc_handler.vs_ctrl.vsRightImageCenterOffset.value = config_handler.vs_front.center_offset

plc_handler.vs_ctrl.vsLeftLensPosition.value = config_handler.vs_rear.lens_position
plc_handler.vs_ctrl.vsLeftSerialNumber.value = config_handler.vs_rear.serial
plc_handler.vs_ctrl.vsLeftImageCenterOffset.value = config_handler.vs_rear.center_offset

plc_handler.vs_ctrl.settings.stdExposureTimePositive.value = config_handler.general.exposure_time_positive
plc_handler.vs_ctrl.settings.stdExposureTimeNegative.value = config_handler.general.exposure_time_negative
plc_handler.vs_ctrl.rawImageCropTop.value = config_handler.vs_front.raw_image_crop_top
plc_handler.vs_ctrl.rawImageHeight.value = config_handler.vs_front.raw_image_height
plc_handler.vs_ctrl.rawImageWidth.value = config_handler.vs_front.raw_image_width

found_front_sensor = False
found_rear_sensor = False

devices_found = dai.Device.getAllAvailableDevices()

device_info_front_sensor = dai.DeviceInfo()
device_info_rear_sensor = dai.DeviceInfo()

if len(devices_found) != 2:
    main_logger.info(f"found only one sensor... - exit program")
    exit()

for device in devices_found:
    main_logger.info(f"found sensor ({device.getDeviceId()}) on state: {device.state}")

if config_handler.vs_front.serial and config_handler.vs_rear.serial:
    found_front_sensor, device_info_front_sensor = dai.Device.getDeviceById(config_handler.vs_front.serial)
    found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceById(config_handler.vs_rear.serial)
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
    raw_image_height=plc_handler.vs_ctrl.rawImageHeight,
    raw_image_width=plc_handler.vs_ctrl.rawImageWidth,
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
    raw_image_height=plc_handler.vs_ctrl.rawImageHeight,
    raw_image_width=plc_handler.vs_ctrl.rawImageWidth,
    flip_image=True,
    fps=config_handler.general.fps,

)

def cb_film_type_is_negative(value):
    main_logger.info(f"set film_type_is_negative to: {value}")
    cam_vs_front.settings.is_film_type_negative = value
    cam_vs_rear.settings.is_film_type_negative = value
    if value:
        cam_vs_front.exposure_time = config_handler.general.exposure_time_negative
        cam_vs_rear.exposure_time = config_handler.general.exposure_time_negative
    else:
        cam_vs_front.exposure_time = config_handler.general.exposure_time_positive
        cam_vs_rear.exposure_time = config_handler.general.exposure_time_positive

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
    if value:
        main_logger.info(f"Start AutoExposure...")
        plc_handler.vs_ctrl.isConnected.value = False
        cam_vs_front.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        cam_vs_rear.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        plc_handler.vs_ctrl._autoExposureCameras.value = False

def cb_auto_focus_cameras(value):
    if value:
        main_logger.info("Start AutoFocus Cameras...")
        plc_handler.vs_ctrl.isConnected.value = False
        cam_vs_front.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        cam_vs_rear.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        plc_handler.vs_ctrl._autoFocusCameras.value = False

def cb_swap_cameras(value):
    if value:
        config_handler.swap_cameras()
        plc_handler.vs_ctrl.swapCameras.value = False

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

def cb_vs_front_image_center_position(value):
    cam_vs_front.settings.camera_center_position = int(value)
    config_handler.vs_front.center_offset = value
    config_handler.save_config()

def cb_vs_rear_image_center_position(value):
    cam_vs_rear.settings.camera_center_position = int(value)
    config_handler.vs_rear.center_offset = value
    config_handler.save_config()

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

def cb_raw_image_crop_top(value):
    cam_vs_rear.raw_image_crop_top = value
    config_handler.vs_rear.raw_image_crop_top = value
    config_handler.save_config()

def cb_vs_front_raw_image_crop_top(value):
    cam_vs_front.raw_image_crop_top = value
    config_handler.vs_front.raw_image_crop_top = value
    config_handler.save_config()

def cb_raw_image_height(value):
    cam_vs_front.raw_image_height = value
    cam_vs_rear.raw_image_height = value
    config_handler.vs_front.raw_image_height = value
    config_handler.vs_rear.raw_image_height = value
    config_handler.save_config()

def cb_raw_image_width(value):
    cam_vs_front.raw_image_width = value
    cam_vs_rear.raw_image_width = value
    config_handler.vs_front.raw_image_width = value
    config_handler.vs_rear.raw_image_width = value
    config_handler.save_config()

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

# region GLOBAL ADS-VARIABLES
plc_handler.vs_ctrl.isFilmTypeNegative.on_new_value = cb_film_type_is_negative
plc_handler.vs_ctrl._autoExposureCameras.on_new_value = cb_auto_exposure_cameras
plc_handler.vs_ctrl._autoFocusCameras.on_new_value = cb_auto_focus_cameras
plc_handler.vs_ctrl.slopeThreshold.on_new_value = cb_slope_threshold
plc_handler.vs_ctrl.swapCameras.on_new_value = cb_swap_cameras
plc_handler.vs_ctrl.contrastOffset.on_new_value = cb_contrast_offset
plc_handler.vs_ctrl.edgeDetectionRange.on_new_value = cb_edge_detection_range
plc_handler.vs_ctrl.exposureTime.on_new_value = cb_exposure_time

plc_handler.vs_ctrl.settings.imageTileWidth.on_new_value = cb_image_tile_width
plc_handler.vs_ctrl.settings.imageTileHeight.on_new_value = cb_image_tile_height
plc_handler.vs_ctrl.settings.imageTileCenterOffset.on_new_value = cb_image_tile_center_offset
plc_handler.vs_ctrl.settings.contrastPicHeight.on_new_value = cb_contrast_pic_height
plc_handler.vs_ctrl.settings.contrastPicEdgeOffset.on_new_value = cb_contrast_pic_edge_offset
plc_handler.vs_ctrl.settings.stdExposureTimePositive.on_new_value = cb_std_exposure_time_pos
plc_handler.vs_ctrl.settings.stdExposureTimeNegative.on_new_value = cb_std_exposure_time_neg

plc_handler.vs_ctrl.rawImageCropTop.on_new_value = cb_raw_image_crop_top
plc_handler.vs_ctrl.rawImageHeight.on_new_value = cb_raw_image_height
plc_handler.vs_ctrl.rawImageWidth.on_new_value = cb_raw_image_width
# endregion

# region SPECIFIC ADS-VARIABLES
plc_handler.vs_ctrl.vsRightImageCenterOffset.on_new_value = cb_vs_front_image_center_position
plc_handler.vs_ctrl.vsLeftImageCenterOffset.on_new_value = cb_vs_rear_image_center_position
plc_handler.vs_ctrl.vsLeftStopPosition.on_new_value = cb_vs_rear_stop_position
plc_handler.vs_ctrl.vsRightStopPosition.on_new_value = cb_vs_front_stop_position
plc_handler.vs_ctrl.vsLeftStopOffset.on_new_value = cb_vs_rear_stop_offset
plc_handler.vs_ctrl.vsRightStopOffset.on_new_value =cb_vs_front_stop_offset
#endregion

main_logger.info(f"Connect to MQTT-Broker...")
mqtt = mqtt_communication_handler.MqttHandler(logger_enabled=True, client_type="vsController", client_id="vsController")

tl = Timeloop()

@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.pTopics_vsController.set_vsController_heartbeat.value = 3

tl.start()

socket_handler = SocketHandler(host="192.168.0.30", port=4001)

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

interval_send_edge_position = helper.IntervalTimer(interval=0.05)
send_vs_front_edge_position = False

vs_front_send_mqtt_image = True
vs_rear_send_mqtt_image = True

with (contextlib.ExitStack() as stack):
    while True:

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

