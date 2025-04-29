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
import traceback
from datetime import timedelta
import mqtt_handler.mqtt_communication_handler_v2 as mqtt_communication_handler
from ads_handler.ads_handler import AdsHandler
from visionSensor import VisionSensor
from libs.socket_handler import SocketHandler
import log_handler.log_handler as log_handler
from libs.config_file_handler import ConfigFileHandler

CONFIG_DIR = ".vSensorM4"
HOME_DIR = os.path.expanduser('~')
FULL_CONFIG_DIR = os.path.join(HOME_DIR, CONFIG_DIR)
FULL_LOG_DIR = os.path.join(FULL_CONFIG_DIR, "logs")
INIT_CONFIG_FILE = os.path.join(FULL_CONFIG_DIR, "init.json")

if not os.path.exists(FULL_CONFIG_DIR):
    os.makedirs(FULL_CONFIG_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

main_logger = log_handler.setup_logger(
    logger_name="base",
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


plc_handler = AdsHandler(
    local_host_ip="192.168.0.30",
    plc_ip_address="192.168.0.10",
    route_name="vSensor",
)
# plc_handler.connect_to_plc()

plc_handler.vs_ctrl.capture_image.init()
plc_handler.vs_ctrl.enable_live_view.init()

plc_handler.vs_ctrl.vs_front_lens_position.init()
plc_handler.vs_ctrl.vs_front_serial_nr.init()
plc_handler.vs_ctrl.vs_front_image_center_offset.init()

plc_handler.vs_ctrl.vs_rear_lens_position.init()
plc_handler.vs_ctrl.vs_rear_serial_nr.init()
plc_handler.vs_ctrl.vs_rear_image_center_offset.init()

plc_handler.vs_ctrl.std_exposure_time_pos.init()
plc_handler.vs_ctrl.std_exposure_time_neg.init()
plc_handler.vs_ctrl.raw_image_crop_top.init()
plc_handler.vs_ctrl.raw_image_height.init()
plc_handler.vs_ctrl.raw_image_width.init()

time.sleep(2)

main_logger.info("set init values to plc")

plc_handler.vs_ctrl.vs_front_lens_position.value = config_handler.vs_front.lens_position
plc_handler.vs_ctrl.vs_front_serial_nr.value = config_handler.vs_front.serial
plc_handler.vs_ctrl.vs_front_image_center_offset.value = config_handler.vs_front.center_offset

plc_handler.vs_ctrl.vs_rear_lens_position.value = config_handler.vs_rear.lens_position
plc_handler.vs_ctrl.vs_rear_serial_nr.value = config_handler.vs_rear.serial
plc_handler.vs_ctrl.vs_rear_image_center_offset.value = config_handler.vs_rear.center_offset

plc_handler.vs_ctrl.std_exposure_time_pos.value = config_handler.general.exposure_time_positive
plc_handler.vs_ctrl.std_exposure_time_neg.value = config_handler.general.exposure_time_negative
plc_handler.vs_ctrl.raw_image_crop_top.value = config_handler.vs_front.raw_image_crop_top
plc_handler.vs_ctrl.raw_image_height.value = config_handler.vs_front.raw_image_height
plc_handler.vs_ctrl.raw_image_width.value = config_handler.vs_front.raw_image_width

plc_handler.vs_ctrl.vs_front_fps.init()
plc_handler.vs_ctrl.vs_rear_fps.init()
plc_handler.vs_ctrl.vs_front_edge_state.init()
plc_handler.vs_ctrl.vs_rear_edge_state.init()
plc_handler.vs_ctrl.vs_front_edge_position.init()
plc_handler.vs_ctrl.vs_rear_edge_position.init()

plc_handler.vs_ctrl.is_film_type_negative.init()
plc_handler.vs_ctrl.auto_exposure_cameras.init()
plc_handler.vs_ctrl.auto_focus_cameras.init()
plc_handler.vs_ctrl.slope_threshold.init()
plc_handler.vs_ctrl.swap_cameras.init()
plc_handler.vs_ctrl.image_tile_center_offset.init()
plc_handler.vs_ctrl.contrast_offset.init()
plc_handler.vs_ctrl.image_tile_width.init()
plc_handler.vs_ctrl.image_tile_height.init()

plc_handler.vs_ctrl.edge_detection_range.init()
plc_handler.vs_ctrl.contrast_pic_height.init()
plc_handler.vs_ctrl.contrast_pic_edge_offset.init()
plc_handler.vs_ctrl.exposure_time.init()
plc_handler.vs_ctrl.vs_rear_stop_position.init()
plc_handler.vs_ctrl.vs_rear_stop_offset.init()
plc_handler.vs_ctrl.vs_front_stop_position.init()
plc_handler.vs_ctrl.vs_front_stop_offset.init()

plc_handler.vs_ctrl.is_ready.init()

plc_handler.vs_ctrl.auto_focus_cameras.value = False
plc_handler.vs_ctrl.auto_exposure_cameras.value = False

cam_vs_front = VisionSensor(
    device_info_front_sensor,
    vs_name="vs_front",
    camera_capture_width=config_handler.vs_front.capture_width,
    camera_capture_height=config_handler.vs_front.capture_height,
    image_center_position=config_handler.vs_front.center_offset,
    lens_position=config_handler.vs_front.lens_position,
    raw_image_height=plc_handler.vs_ctrl.raw_image_height,
    raw_image_width=plc_handler.vs_ctrl.raw_image_width,
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
    raw_image_height=plc_handler.vs_ctrl.raw_image_height,
    raw_image_width=plc_handler.vs_ctrl.raw_image_width,
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
        plc_handler.vs_ctrl.exposure_time.value = mean_exposure
        plc_handler.vs_ctrl.is_ready.value = True

def cb_auto_focus_finished():
    if not cam_vs_front.autofocus_in_progress and not cam_vs_rear.auto_exposure_in_progress:
        main_logger.info("AutoFocus Finished")
        main_logger.info(f"Lens-Position cam_front: {cam_vs_front.lens_position}")
        main_logger.info(f"Lens-Position cam_rear : {cam_vs_rear.lens_position}")
        plc_handler.vs_ctrl.vs_front_lens_position.value = cam_vs_front.lens_position
        plc_handler.vs_ctrl.vs_rear_lens_position.value = cam_vs_rear.lens_position
        config_handler.vs_front.lens_position = cam_vs_front.lens_position
        config_handler.vs_rear.lens_position = cam_vs_rear.lens_position
        config_handler.save_config()
        plc_handler.vs_ctrl.is_ready.value = True

def cb_auto_exposure_cameras(value):
    if value:
        main_logger.info(f"Start AutoExposure...")
        plc_handler.vs_ctrl.is_ready.value = False
        cam_vs_front.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        cam_vs_rear.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        plc_handler.vs_ctrl.auto_exposure_cameras.value = False

def cb_auto_focus_cameras(value):
    if value:
        main_logger.info("Start AutoFocus Cameras...")
        plc_handler.vs_ctrl.is_ready.value = False
        cam_vs_front.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        cam_vs_rear.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        plc_handler.vs_ctrl.auto_focus_cameras.value = False

def cb_swap_cameras(value):
    if value:
        config_handler.swap_cameras()
        plc_handler.vs_ctrl.swap_cameras.value = False

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
plc_handler.vs_ctrl.is_film_type_negative.on_new_value = cb_film_type_is_negative
plc_handler.vs_ctrl.auto_exposure_cameras.on_new_value = cb_auto_exposure_cameras
plc_handler.vs_ctrl.auto_focus_cameras.on_new_value = cb_auto_focus_cameras
plc_handler.vs_ctrl.slope_threshold.on_new_value = cb_slope_threshold
plc_handler.vs_ctrl.swap_cameras.on_new_value = cb_swap_cameras
plc_handler.vs_ctrl.image_tile_center_offset.on_new_value = cb_image_tile_center_offset
plc_handler.vs_ctrl.contrast_offset.on_new_value = cb_contrast_offset
plc_handler.vs_ctrl.image_tile_width.on_new_value = cb_image_tile_width
plc_handler.vs_ctrl.image_tile_height.on_new_value = cb_image_tile_height
plc_handler.vs_ctrl.edge_detection_range.on_new_value = cb_edge_detection_range
plc_handler.vs_ctrl.contrast_pic_height.on_new_value = cb_contrast_pic_height
plc_handler.vs_ctrl.contrast_pic_edge_offset.on_new_value = cb_contrast_pic_edge_offset
plc_handler.vs_ctrl.std_exposure_time_pos.on_new_value = cb_std_exposure_time_pos
plc_handler.vs_ctrl.std_exposure_time_neg.on_new_value = cb_std_exposure_time_neg
plc_handler.vs_ctrl.exposure_time.on_new_value = cb_exposure_time
plc_handler.vs_ctrl.raw_image_crop_top.on_new_value = cb_raw_image_crop_top
plc_handler.vs_ctrl.raw_image_height.on_new_value = cb_raw_image_height
plc_handler.vs_ctrl.raw_image_width.on_new_value = cb_raw_image_width
# endregion

# region SPECIFIC ADS-VARIABLES
plc_handler.vs_ctrl.vs_front_image_center_offset.on_new_value = cb_vs_front_image_center_position
plc_handler.vs_ctrl.vs_rear_image_center_offset.on_new_value = cb_vs_rear_image_center_position
plc_handler.vs_ctrl.vs_rear_stop_position.on_new_value = cb_vs_rear_stop_position
plc_handler.vs_ctrl.vs_front_stop_position.on_new_value = cb_vs_front_stop_position
plc_handler.vs_ctrl.vs_rear_stop_offset.on_new_value = cb_vs_rear_stop_offset
plc_handler.vs_ctrl.vs_front_stop_offset.on_new_value =cb_vs_front_stop_offset
#endregion

main_logger.info(f"Connect to MQTT-Broker...")
mqtt = mqtt_communication_handler.MqttHandler(logger_enabled=True, client_type="vsController", client_id="vsController")

tl = Timeloop()

@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsController_heartbeat, 3)

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
            plc_handler.vs_ctrl.vs_front_edge_state.value = cam_vs_front.results.result_mean.edge_state
            if plc_handler.vs_ctrl.enable_live_view.value:
                if vs_front_live_view_frame_nr == live_view_fps_divider:
                    socket_handler.send_image(
                        vs_front_slope_data=cam_vs_front.results,
                        vs_rear_slope_data=cam_vs_rear.results,
                    )
                    vs_front_live_view_frame_nr = 0
                vs_front_live_view_frame_nr += 1
            plc_handler.vs_ctrl.vs_front_fps.value = cam_vs_front.fps

        if cam_vs_rear.new_image_available:
            plc_handler.vs_ctrl.vs_rear_edge_state.value = cam_vs_rear.results.result_mean.edge_state
            plc_handler.vs_ctrl.vs_rear_fps.value = cam_vs_rear.fps

        if interval_send_edge_position.is_time_to_update():
            if send_vs_front_edge_position:
                plc_handler.vs_ctrl.vs_front_edge_position.value = cam_vs_front.results.result_mean.edge_position
            else:
                plc_handler.vs_ctrl.vs_rear_edge_position.value = cam_vs_rear.results.result_mean.edge_position
            send_vs_front_edge_position = not send_vs_front_edge_position

        if plc_handler.vs_ctrl.capture_image.value:
            main_logger.debug(f"Send image_data data....")
            socket_handler.send_image(
                vs_front_slope_data=cam_vs_front.results,
                vs_rear_slope_data=cam_vs_rear.results,
            )
            vs_front_send_mqtt_image = True
            vs_rear_send_mqtt_image = True
            plc_handler.vs_ctrl.capture_image.value = False

        if vs_front_send_mqtt_image:
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_imageData, cam_vs_front.get_base64_image())
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageWidth, cam_vs_front.results.image_width)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_getImageHeight, cam_vs_front.results.image_height)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsFront_image_statistics, str(cam_vs_front.calc_statistics()))
            vs_front_send_mqtt_image = False

        if vs_rear_send_mqtt_image:
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_imageData, cam_vs_rear.get_base64_image())
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageWidth, cam_vs_rear.results.image_width)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_getImageHeight, cam_vs_rear.results.image_height)
            mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsRear_image_statistics, str(cam_vs_rear.calc_statistics()))
            vs_rear_send_mqtt_image = False

        time.sleep(0.0001)

