#!/usr/bin/env python3

import os
import depthai as dai
import time
import contextlib
import sys
from timeloop import Timeloop
import logging
from datetime import timedelta
import mqtt_handler.mqtt_communication_handler_v2 as mqtt_communication_handler
from ads_handler.ads_handler import AdsHandler
from visionSensor import VisionSensor
from libs.socket_handler import SocketHandler
from loguru import logger
from libs.init_file_handler import InitFileHandler

logger.add(sys.stderr, format="{time} {level} | {message}", filter="my_module", level="INFO")
logger.add("visionSensor.log")

logger.info("Start visionSensorM4")

home_dir = os.environ['HOME']

init_config_file = os.path.join(home_dir, ".vSensor", "init.ini")
# settings_config_file = "../config/vs_settings.ini"

logger.info(f"HOME DIRECTORY: {home_dir}")
logger.info(f"INITIALIZATION FILE: {init_config_file}")

live_view_fps_divider = 2
vs_front_live_view_frame_nr = 0
vs_rear_live_view_frame_nr = 0

vs_front_send_mqtt_image = False
vs_rear_send_mqtt_image = False

vs_front_last_edge_position = 0
vs_rear_last_edge_position = 0

init_config = InitFileHandler(init_config_file)

logger.info(f"DephtAi-Version : {dai.__version__}")

found_front_sensor = False
found_rear_sensor = False

devices_found = dai.Device.getAllAvailableDevices()

if len(devices_found) != 2:
    logger.info(f"found only one sensor... - exit program")
    exit()

for device in devices_found:
    logger.info(f"found sensor ({device.getMxId()}) on state: {device.state}")

if init_config.vs_front.serial and init_config.vs_rear.serial:
    found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(init_config.vs_front.serial)
    found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(init_config.vs_rear.serial)
else:
    if len(devices_found) == 2:
        init_config.vs_front.serial = devices_found[0].getMxId()
        init_config.vs_rear.serial = devices_found[1].getMxId()
        init_config.save_config()
        found_front_sensor, device_info_front_sensor = dai.Device.getDeviceByMxId(init_config.vs_front.serial)
        found_rear_sensor, device_info_rear_sensor = dai.Device.getDeviceByMxId(init_config.vs_rear.serial)
        if not found_front_sensor:
            raise RuntimeError("FrontSensor not found!")

        if not found_rear_sensor:
            raise RuntimeError("RearSensor not found!")


plc_handler = AdsHandler(local_host_ip="192.168.0.30", route_name="vSensor")
plc_handler.connect_to_plc()

time.sleep(2)

plc_handler.stop_film = False
plc_handler.vs_ctrl.vs_front_image_center_position.value = init_config.vs_front.center_position
plc_handler.vs_ctrl.vs_front_lens_position.value = init_config.vs_front.lens_position
plc_handler.vs_ctrl.vs_front_serial_nr.value = init_config.vs_front.serial
plc_handler.vs_ctrl.vs_front_raw_image_height.value = init_config.vs_front.raw_image_height
plc_handler.vs_ctrl.vs_front_raw_image_width.value = init_config.vs_front.raw_image_width
plc_handler.vs_ctrl.vs_front_raw_image_height_offset.value = init_config.vs_front.raw_image_height_offset
plc_handler.vs_ctrl.vs_front_raw_image_width_offset.value = init_config.vs_front.raw_image_width_offset


plc_handler.vs_ctrl.vs_rear_image_center_position.value = init_config.vs_rear.center_position
plc_handler.vs_ctrl.vs_rear_lens_position.value = init_config.vs_rear.lens_position
plc_handler.vs_ctrl.vs_rear_serial_nr.value = init_config.vs_rear.serial
plc_handler.vs_ctrl.vs_rear_raw_image_height.value = init_config.vs_rear.raw_image_height
plc_handler.vs_ctrl.vs_rear_raw_image_width.value = init_config.vs_rear.raw_image_width
plc_handler.vs_ctrl.vs_rear_raw_image_height_offset.value = init_config.vs_rear.raw_image_height_offset
plc_handler.vs_ctrl.vs_rear_raw_image_width_offset.value = init_config.vs_rear.raw_image_width_offset



cam_vs_front = VisionSensor(
    device_info_front_sensor,
    vs_name="vs_front",
    camera_capture_width=init_config.vs_front.capture_width,
    camera_capture_height=init_config.vs_front.capture_height,
    image_center_position=init_config.vs_front.center_position,
    lens_position=init_config.vs_front.lens_position,
    fps=init_config.general.fps,
)

cam_vs_rear = VisionSensor(
    device_info=device_info_rear_sensor,
    vs_name="vs_rear",
    camera_capture_width=init_config.vs_rear.capture_width,
    camera_capture_height=init_config.vs_rear.capture_height,
    image_center_position=init_config.vs_rear.center_position,
    lens_position=init_config.vs_rear.lens_position,
    fps=init_config.general.fps,

)


def cb_film_type_is_negative(value):
    logger.info(f"set film_type_is_negative to: {value}")
    cam_vs_front.film_type_is_negative = value
    cam_vs_rear.film_type_is_negative = value
    if value:
        cam_vs_front.exposure_time = init_config.general.exposure_time_negative
        cam_vs_rear.exposure_time = init_config.general.exposure_time_negative
    else:
        cam_vs_front.exposure_time = init_config.general.exposure_time_positive
        cam_vs_rear.exposure_time = init_config.general.exposure_time_positive

def cb_auto_exposure_cameras_finished():
    if not cam_vs_front.auto_exposure_in_progress and not cam_vs_rear.auto_exposure_in_progress:
        mean_exposure = (cam_vs_front.exposure_time + cam_vs_rear.exposure_time) // 2
        logger.info(f"AutoExposure Finished")
        logger.info(f"Exposure cam_front: {cam_vs_front.exposure_time}us")
        logger.info(f"Exposure cam_rear : {cam_vs_rear.exposure_time}us")
        logger.info(f"Set mean-exposure-time to: {mean_exposure}us")
        plc_handler.vs_ctrl.exposure_time.value = mean_exposure
        plc_handler.vs_ctrl.is_ready.value = True

def cb_auto_focus_finished():
    if not cam_vs_front.autofocus_in_progress and not cam_vs_rear.auto_exposure_in_progress:
        logger.info("AutoFocus Finished")
        logger.info(f"Lens-Position cam_front: {cam_vs_front.lens_position}")
        logger.info(f"Lens-Position cam_rear : {cam_vs_rear.lens_position}")
        plc_handler.vs_ctrl.vs_front_lens_position.value = cam_vs_front.lens_position
        plc_handler.vs_ctrl.vs_rear_lens_position.value = cam_vs_rear.lens_position
        init_config.vs_front.lens_position = cam_vs_front.lens_position
        init_config.vs_rear.lens_position = cam_vs_rear.lens_position
        init_config.save_config()
        plc_handler.vs_ctrl.is_ready.value = True

def cb_auto_exposure_cameras(value):
    if value:
        logger.info(f"Start AutoExposure...")
        plc_handler.vs_ctrl.is_ready.value = False
        cam_vs_front.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        cam_vs_rear.auto_exposure_camera(cb_auto_exposure_finished=cb_auto_exposure_cameras_finished)
        plc_handler.vs_ctrl.auto_exposure_cameras.value = False

def cb_auto_focus_cameras(value):
    if value:
        logger.info("Start AutoFocus Cameras...")
        plc_handler.vs_ctrl.is_ready.value = False
        # cam_vs_front.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        cam_vs_rear.auto_focus_camera(cb_autofocus_finished=cb_auto_focus_finished)
        plc_handler.vs_ctrl.auto_focus_cameras.value = False

def cb_swap_cameras(value):
    if value:
        init_config.swap_cameras()
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

def cb_vs_front_image_tile_width(value):
    cam_vs_front.settings.tile_width = int(value)

def cb_vs_rear_image_tile_width(value):
    cam_vs_rear.settings.tile_width = int(value)

def cb_vs_front_image_tile_height(value):
    cam_vs_front.settings.tile_height = int(value)

def cb_vs_rear_image_tile_height(value):
    cam_vs_rear.settings.tile_height = int(value)

def cb_vs_front_image_center_position(value):
    cam_vs_front.settings.camera_center_position = int(value)
    init_config.vs_front.center_position = cam_vs_front.settings.camera_center_position
    init_config.save_config()

def cb_vs_rear_image_center_position(value):
    cam_vs_rear.settings.camera_center_position = int(value)
    init_config.vs_rear.center_position = cam_vs_rear.settings.camera_center_position
    init_config.save_config()

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
    logger.info(f"set enable_live_view to: {type(value)}")

def cb_vs_rear_raw_image_height(value):
    cam_vs_front._raw_image_height = value
    cam_vs_rear._raw_image_height = value
    init_config.vs_front.raw_image_height = value
    init_config.vs_rear.raw_image_height = value
    init_config.save_config()

def cb_vs_rear_raw_image_width(value):
    cam_vs_front._raw_image_width = value
    cam_vs_rear._raw_image_width = value
    init_config.vs_front.raw_image_width = value
    init_config.vs_rear.raw_image_width = value
    init_config.save_config()

def cb_vs_rear_raw_image_height_offset(value):
    cam_vs_rear._raw_image_height_offset = value
    init_config.vs_rear.raw_image_height_offset = value
    init_config.save_config()

def cb_vs_rear_raw_image_width_offset(value):
    cam_vs_rear._raw_image_width_offset = value
    init_config.vs_rear.raw_image_width_offset = value
    init_config.save_config()

def cb_vs_front_raw_image_height(value):
    cam_vs_front._raw_image_height = value
    cam_vs_rear._raw_image_height = value
    init_config.vs_front.raw_image_height = value
    init_config.vs_rear.raw_image_height = value
    init_config.save_config()

def cb_vs_front_raw_image_width(value):
    cam_vs_front._raw_image_width = value
    cam_vs_rear._raw_image_width = value
    init_config.vs_front.raw_image_width = value
    init_config.vs_rear.raw_image_width = value
    init_config.save_config()

def cb_vs_front_raw_image_height_offset(value):
    cam_vs_front._raw_image_height_offset = value
    init_config.vs_front.raw_image_height_offset = value
    init_config.save_config()

def cb_vs_front_raw_image_width_offset(value):
    cam_vs_front._raw_image_width_offset = value
    init_config.vs_front.raw_image_width_offset = value
    init_config.save_config()

plc_handler.vs_ctrl.film_type_is_negative.set_cb_new_value(
    t_cb_new_value=cb_film_type_is_negative
)
plc_handler.vs_ctrl.auto_exposure_cameras.set_cb_new_value(
    t_cb_new_value=cb_auto_exposure_cameras
)
plc_handler.vs_ctrl.auto_focus_cameras.set_cb_new_value(
    t_cb_new_value=cb_auto_focus_cameras
)
plc_handler.vs_ctrl.slope_threshold.set_cb_new_value(
    t_cb_new_value=cb_slope_threshold
)
plc_handler.vs_ctrl.swap_cameras.set_cb_new_value(
    t_cb_new_value=cb_swap_cameras
)
plc_handler.vs_ctrl.image_tile_center_offset.set_cb_new_value(
    t_cb_new_value=cb_image_tile_center_offset
)
plc_handler.vs_ctrl.contrast_offset.set_cb_new_value(
    t_cb_new_value=cb_contrast_offset
)
plc_handler.vs_ctrl.vs_front_image_tile_width.set_cb_new_value(
    t_cb_new_value=cb_vs_front_image_tile_width
)
plc_handler.vs_ctrl.vs_rear_image_tile_width.set_cb_new_value(
    t_cb_new_value=cb_vs_rear_image_tile_width
)
plc_handler.vs_ctrl.vs_front_image_tile_height.set_cb_new_value(
    t_cb_new_value=cb_vs_front_image_tile_height
)
plc_handler.vs_ctrl.vs_rear_image_tile_height.set_cb_new_value(
    t_cb_new_value=cb_vs_rear_image_tile_height
)
plc_handler.vs_ctrl.vs_front_image_center_position.set_cb_new_value(
    t_cb_new_value=cb_vs_front_image_center_position
)
plc_handler.vs_ctrl.vs_rear_image_center_position.set_cb_new_value(
    t_cb_new_value=cb_vs_rear_image_center_position
)
plc_handler.vs_ctrl.edge_detection_range.set_cb_new_value(
    t_cb_new_value=cb_edge_detection_range
)
plc_handler.vs_ctrl.contrast_pic_height.set_cb_new_value(
    t_cb_new_value=cb_contrast_pic_height
)
plc_handler.vs_ctrl.contrast_pic_edge_offset.set_cb_new_value(
    t_cb_new_value=cb_contrast_pic_edge_offset
)
plc_handler.vs_ctrl.enable_live_view.set_cb_new_value(
    t_cb_new_value=cb_live_view
)

plc_handler.vs_ctrl.vs_rear_raw_image_height.set_cb_new_value(
    t_cb_new_value=cb_vs_rear_raw_image_height
)

plc_handler.vs_ctrl.vs_rear_raw_image_width.set_cb_new_value(
    t_cb_new_value=cb_vs_rear_raw_image_width
)

plc_handler.vs_ctrl.vs_rear_raw_image_height_offset.set_cb_new_value(
    t_cb_new_value=cb_vs_rear_raw_image_height_offset
)

plc_handler.vs_ctrl.vs_rear_raw_image_width_offset.set_cb_new_value(
    t_cb_new_value=cb_vs_rear_raw_image_width_offset
)

plc_handler.vs_ctrl.vs_front_raw_image_height.set_cb_new_value(
    t_cb_new_value=cb_vs_front_raw_image_height
)

plc_handler.vs_ctrl.vs_front_raw_image_width.set_cb_new_value(
    t_cb_new_value=cb_vs_front_raw_image_width
)

plc_handler.vs_ctrl.vs_front_raw_image_height_offset.set_cb_new_value(
    t_cb_new_value=cb_vs_front_raw_image_height_offset
)

plc_handler.vs_ctrl.vs_front_raw_image_width_offset.set_cb_new_value(
    t_cb_new_value=cb_vs_front_raw_image_width_offset
)


logger.info(f"Connect to MQTT-Broker...")
mqtt = mqtt_communication_handler.MqttHandler(logger_enabled=True, client_type="vsController", client_id="vsController", external_logger=logging)

tl = Timeloop()

@tl.job(interval=timedelta(seconds=3))
def mqtt_heartbeat():
    mqtt.setMqttValue(mqtt.pTopics_vsController.set_vsController_heartbeat, 3)
    logger.debug(f"{cam_vs_rear.proc_image_roi.shape[:2]} // {cam_vs_front.proc_image_roi.shape[:2]}")

tl.start()

socket_handler = SocketHandler(host="192.168.0.30", port=4001)


with (contextlib.ExitStack() as stack):
    while True:

        ##################################################################################################################
        # IMAGE PROCESSING PART
        ##################################################################################################################

        if cam_vs_front.new_image_available:
            if plc_handler.vs_ctrl.enable_live_view.value:
                if vs_front_live_view_frame_nr == live_view_fps_divider:
                    socket_handler.send_image(
                        vs_front_slope_data=cam_vs_front.results,
                        vs_rear_slope_data=cam_vs_rear.results,
                    )
                    vs_front_live_view_frame_nr = 0
                vs_front_live_view_frame_nr += 1
            if abs(cam_vs_front.results.result_mean.edge_position - vs_front_last_edge_position) > 1:
                vs_front_last_edge_position = cam_vs_front.results.result_mean.edge_position
            if plc_handler.vs_ctrl.vs_front_update_edge_state.value:
                plc_handler.vs_ctrl.vs_front_edge_state = cam_vs_front.results.result_mean.edge_state
                plc_handler.vs_ctrl.vs_front_edge_position.value = cam_vs_front.results.result_mean.edge_position
            plc_handler.vs_ctrl.vs_front_fps.value = cam_vs_front.fps

        if cam_vs_rear.new_image_available:
            if abs(cam_vs_rear.results.result_mean.edge_position - vs_rear_last_edge_position) > 1:
                plc_handler.vs_ctrl.vs_rear_edge_position.value = cam_vs_rear.results.result_mean.edge_position
                vs_rear_last_edge_position = cam_vs_rear.results.result_mean.edge_position
            if plc_handler.vs_ctrl.vs_rear_update_edge_state.value:
                plc_handler.vs_ctrl.vs_rear_edge_state = cam_vs_rear.results.result_mean.edge_state
                plc_handler.vs_ctrl.vs_rear_edge_position.value = cam_vs_rear.results.result_mean.edge_position
            plc_handler.vs_ctrl.vs_rear_fps.value = cam_vs_rear.fps

        if plc_handler.vs_ctrl.capture_image.value:
            logger.debug(f"Send image_data data....")
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
