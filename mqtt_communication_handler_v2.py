import threading
import time

import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import logging
import sys


class PublishTopicItem:
    def __init__(self, mqtt_address, description="", log_output=True, retain=True, qos=2):
        self.address = mqtt_address
        self.description = description
        self._value = "0"
        self.retain = retain
        self.qos = qos
        self.log_output = log_output


class SubscribeTopicItem:
    def __init__(self, mqtt_address, description="", log_output=True, enabled=True):
        self.address = mqtt_address
        self.description = description
        self.new_value_available = False
        self._value = None
        self.log_output = log_output
        self.enabled = enabled

    def getValue(self):
        self.new_value_available = False
        return self._value

    def setValue(self, value):
        self.new_value_available = True
        self._value = value

    value = property(getValue, setValue)


class GeneralPublishTopics:
    def __init__(self):
        # FilmMoveController (fmCtrl)
        self.set_fmCtrl_startInitFilm = PublishTopicItem("filmMoveController/filmStatus/startInitFilm")
        self.set_fmCtrl_startLoadFilm = PublishTopicItem("filmMoveController/filmStatus/startLoadFilm")
        self.set_fmCtrl_stopLoadFilm = PublishTopicItem("filmMoveController/filmStatus/stopLoadFilm")
        self.set_fmCtrl_filmLoadFastForward = PublishTopicItem("filmMoveController/filmStatus/filmLoadFastForward")
        self.set_fmCtrl_automaticModeOn = PublishTopicItem("filmMoveController/setAutomaticModeOn")
        self.set_fmCtrl_moveCommand = PublishTopicItem("filmMoveController/setMoveCommand")
        self.set_fmCtrl_increaseTotalPicCounter = PublishTopicItem("filmMoveController/pictureCounter/totalPics/incPicCounter")
        self.set_fmCtrl_increaseBwPicCounter = PublishTopicItem("filmMoveController/pictureCounter/bw/incPicCounter")
        self.set_fmCtrl_increaseColorPicCounter = PublishTopicItem("filmMoveController/pictureCounter/color/incPicCounter")
        self.set_fmCtrl_increaseRadiometricCalibrationPicCounter = PublishTopicItem(
            "filmMoveController/pictureCounter/radiometricCalibration/incPicCounter")
        self.set_fmCtrl_increaseGeometricCalibrationPicCounter = PublishTopicItem(
            "filmMoveController/pictureCounter/geometricCalibration/incPicCounter")

        self.set_fmCtrl_parameter_moveToPicture_distanceFast = PublishTopicItem(
            "filmMoveController/setParameter/moveToPicture_distanceFast")
        self.set_fmCtrl_parameter_moveToPicture_endVelocity = PublishTopicItem("filmMoveController/setParameter/moveToPicture_endVelocity")
        self.set_fmCtrl_parameter_moveToPicture_fixedDistance = PublishTopicItem(
            "filmMoveController/setParameter/moveToPicture_fixedDistance")
        self.set_fmCtrl_parameter_moveFilm_velocity = PublishTopicItem("filmMoveController/setParameter/moveFilm_velocity")
        self.set_fmCtrl_parameter_lockMotorsIfNoFilmIsLoaded = PublishTopicItem(
            "filmMoveController/setParameter/lockMotorsIfNoFilmIsLoaded")
        self.set_fmCtrl_parameter_autoUnloadFilm = PublishTopicItem("filmMoveController/setParameter/autoUnloadFilm")
        self.set_fmCtrl_parameter_vsSlowDownPosition = PublishTopicItem("filmMoveController/setParameter/vsSlowDownPosition")
        self.set_fmCtrl_enableFreeRun = PublishTopicItem("filmMoveController/enableFreeRun")
        self.set_fmCtrl_disableFreeRun = PublishTopicItem("filmMoveController/disableFreeRun")

        # LightController (lightCtrl)
        self.set_lightCtrl_ledBrightnessWhite = PublishTopicItem("lightTable/setConfig/ledBrightnessWhite")
        self.set_lightCtrl_ledBrightnessRed = PublishTopicItem("lightTable/setConfig/ledBrightnessRed")
        self.set_lightCtrl_ledBrightnessGreen = PublishTopicItem("lightTable/setConfig/ledBrightnessGreen")
        self.set_lightCtrl_ledBrightnessBlue = PublishTopicItem("lightTable/setConfig/ledBrightnessBlue")
        self.set_lightCtrl_ledColor = PublishTopicItem("lightTable/setColor")

        # GlassLifterController (glCtrl)
        self.set_glCtrl_initDh = PublishTopicItem("downHolder/initDh")
        self.set_glCtrl_dhPosition = PublishTopicItem("downHolder/setDhPosition")
        self.set_glCtrl_parameter_dh1NullPosition = PublishTopicItem("downHolder/setConfig/dh1NullPos")
        self.set_glCtrl_parameter_dh2NullPosition = PublishTopicItem("downHolder/setConfig/dh2NullPos")
        self.set_glCtrl_parameter_dh3NullPosition = PublishTopicItem("downHolder/setConfig/dh3NullPos")
        self.set_glCtrl_parameter_dh4NullPosition = PublishTopicItem("downHolder/setConfig/dh4NullPos")
        self.set_glCtrl_parameter_dhSaveConfig = PublishTopicItem("downHolder/setConfig/saveConfig")

        # OLD VISION-SENSOR (COGNEX)
        self.cognex_setVsJobById = PublishTopicItem("visionSensor/setVsJobById")

        # VisionSensorController (vsCtrl)
        self.set_vsCtrl_filmTypeIsNegative = PublishTopicItem("vsController/setFilmTypeIsNegative")
        self.set_vsCtrl_parameter_sensorExposureTime = PublishTopicItem("vsController/setConfig/sensorExposureTime")
        self.set_vsCtrl_enable_low_contrast_mode = PublishTopicItem("vsController/lcm_enabled")
        self.set_vsCtrl_centerImage = PublishTopicItem("vsController/setConfig/centerImage")

        self.set_vsCtrl_vsFront_enableLiveView = PublishTopicItem("vsController/vsFront/enableLiveView")
        self.set_vsCtrl_vsFront_focusCamera = PublishTopicItem("vsController/vsFront/focusCamera")
        self.set_vsCtrl_parameter_vsFront_stopPosition = PublishTopicItem("vsController/vsFront/setConfig/stopPosition")
        self.set_vsCtrl_parameter_vsFront_sensorCropTop = PublishTopicItem("vsController/vsFront/setConfig/cropTop")
        self.set_vsCtrl_parameter_vsFront_sensorCropRight = PublishTopicItem("vsController/vsFront/setConfig/cropRight")
        self.set_vsCtrl_parameter_vsFront_sensorCropLeft = PublishTopicItem("vsController/vsFront/setConfig/cropLeft")
        self.set_vsCtrl_parameter_vsFront_sensorCropBottom = PublishTopicItem("vsController/vsFront/setConfig/cropBottom")
        self.get_vsCtrl_parameter_vsFront_centerPosition = PublishTopicItem("vsController/vsFront/setConfig/centerPosition")
        self.get_vsCtrl_parameter_vsFront_procImageWidth = PublishTopicItem("vsController/vsFront/setConfig/procImageWidth")

        self.set_vsCtrl_vsRear_enableLiveView = PublishTopicItem("vsController/vsRear/enableLiveView")
        self.set_vsCtrl_vsRear_focusCamera = PublishTopicItem("vsController/vsRear/focusCamera")
        self.set_vsCtrl_parameter_vsRear_stopPosition = PublishTopicItem("vsController/vsRear/setConfig/stopPosition")
        self.set_vsCtrl_parameter_vsRear_sensorCropTop = PublishTopicItem("vsController/vsRear/setConfig/cropTop")
        self.set_vsCtrl_parameter_vsRear_sensorCropRight = PublishTopicItem("vsController/vsRear/setConfig/cropRight")
        self.set_vsCtrl_parameter_vsRear_sensorCropLeft = PublishTopicItem("vsController/vsRear/setConfig/cropLeft")
        self.set_vsCtrl_parameter_vsRear_sensorCropBottom = PublishTopicItem("vsController/vsRear/setConfig/cropBottom")
        self.get_vsCtrl_parameter_vsRear_centerPosition = PublishTopicItem("vsController/vsRear/setConfig/centerPosition")
        self.get_vsCtrl_parameter_vsRear_procImageWidth = PublishTopicItem("vsController/vsRear/setConfig/procImageWidth")

        # AutoFocusController (afCtrl)
        self.set_afCtrl_selectServo = PublishTopicItem("afController/control/selectServo")
        self.set_afCtrl_startInit = PublishTopicItem("afController/control/startInit")
        self.set_afCtrl_setPosition = PublishTopicItem("afController/control/setPosition")
        self.set_afCtrl_reboot = PublishTopicItem("afController/control/reboot")
        self.set_afCtrl_unloadLens = PublishTopicItem("afController/control/unload")


class GeneralSubscribeTopics:
    def __init__(self):
        # FilmMoveController
        self.get_fmCtrl_heartbeat = SubscribeTopicItem("filmMoveController/isOnline")
        self.get_fmCtrl_moveCommand = SubscribeTopicItem("filmMoveController/getMoveCommand")
        self.get_fmCtrl_filmMoveController_swVersionMajor = SubscribeTopicItem("filmMoveController/swVersionMinor")
        self.get_fmCtrl_filmMoveController_swVersionMinor = SubscribeTopicItem("filmMoveController/swVersionMinor")
        self.get_fmCtrl_picToPicTime = SubscribeTopicItem("filmMoveController/picToPicTime")
        self.get_fmCtrl_picToPicDistance = SubscribeTopicItem("filmMoveController/picToPicDistance")
        self.get_fmCtrl_fmcState = SubscribeTopicItem("filmMoveController/fmc_State")
        self.get_fmCtrl_filmMoveDirection = SubscribeTopicItem("filmMoveController/curFilmMoveDirection")
        self.get_fmCtrl_automaticModeIsOn = SubscribeTopicItem("filmMoveController/automaticModeIsOn")
        self.get_fmCtrl_filmPosition = SubscribeTopicItem("filmMoveController/curFilmPosition")
        self.get_fmCtrl_spoolDiameterFront = SubscribeTopicItem("filmMoveController/frontSpool/curSpoolDiameter")
        self.get_fmCtrl_spoolDiameterRear = SubscribeTopicItem("filmMoveController/rearSpool/curSpoolDiameter")
        self.get_fmCtrl_spoolTorqueFront = SubscribeTopicItem("filmMoveController/frontSpool/curTorque")
        self.get_fmCtrl_spoolTorqueRear = SubscribeTopicItem("filmMoveController/rearSpool/curTorque")
        self.get_fmCtrl_totalPicCounter = SubscribeTopicItem("filmMoveController/pictureCounter/totalPics/curValue")
        self.get_fmCtrl_bwPicCounter = SubscribeTopicItem("filmMoveController/pictureCounter/bw/curValue")
        self.get_fmCtrl_colorPicCounter = SubscribeTopicItem("filmMoveController/pictureCounter/color/curValue")
        self.get_fmCtrl_radiometricCalibrationPicCounter = SubscribeTopicItem(
            "filmMoveController/pictureCounter/radiometricCalibration/curValue")
        self.get_fmCtrl_geometricCalibrationPicCounter = SubscribeTopicItem(
            "filmMoveController/pictureCounter/geometricCalibration/curValue")

        self.get_fmCtrl_parameter_moveToPicture_distanceFast = SubscribeTopicItem(
            "filmMoveController/getParameter/moveToPicture_distanceFast")
        self.get_fmCtrl_parameter_moveToPicture_endVelocity = SubscribeTopicItem(
            "filmMoveController/getParameter/moveToPicture_endVelocity")
        self.get_fmCtrl_parameter_moveToPicture_fixedDistance = SubscribeTopicItem(
            "filmMoveController/getParameter/moveToPicture_fixedDistance")
        self.get_fmCtrl_parameter_moveFilm_velocity = SubscribeTopicItem("filmMoveController/getParameter/moveFilm_velocity")
        self.get_fmCtrl_parameter_lockMotorsIfNoFilmIsLoaded = SubscribeTopicItem(
            "filmMoveController/getParameter/lockMotorsIfNoFilmIsLoaded")
        self.get_fmCtrl_parameter_autoUnloadFilm = SubscribeTopicItem("filmMoveController/getParameter/autoUnloadFilm")
        self.get_fmCtrl_parameter_vsSlowDownPosition = SubscribeTopicItem("filmMoveController/getParameter/vsSlowDownPosition")

        self.get_fmCtrl_plcInfo_hardwareModel = SubscribeTopicItem("filmMoveController/plcInfo/hardwareModel")
        self.get_fmCtrl_plcInfo_hardwareSerialNo = SubscribeTopicItem("filmMoveController/plcInfo/hardwareSerialNo")
        self.get_fmCtrl_plcInfo_hardwareVersion = SubscribeTopicItem("filmMoveController/plcInfo/hardwareVersion")
        self.get_fmCtrl_plcInfo_hardwareDate = SubscribeTopicItem("filmMoveController/plcInfo/hardwareDate")
        self.get_fmCtrl_plcInfo_hardwareCPU = SubscribeTopicItem("filmMoveController/plcInfo/hardwareCPU")
        self.get_fmCtrl_plcInfo_amsNetId = SubscribeTopicItem("filmMoveController/plcInfo/amsNetId")
        self.get_fmCtrl_plcInfo_twinCATVersion = SubscribeTopicItem("filmMoveController/plcInfo/twinCATVersion")
        self.get_fmCtrl_plcInfo_twinCATRevision = SubscribeTopicItem("filmMoveController/plcInfo/twinCATRevision")
        self.get_fmCtrl_plcInfo_twinCATBuild = SubscribeTopicItem("filmMoveController/plcInfo/twinCATBuild")

        # HMI-Display-Info
        self.get_hmiDisplaySwVersion = SubscribeTopicItem("hmiDisplay/swVersion")
        self.get_hmiDisplayBuildDate = SubscribeTopicItem("hmiDisplay/buildDate")
        self.get_hmiDisplayBuildTime = SubscribeTopicItem("hmiDisplay/buildTime")

        # LightController
        self.get_ledBrightnessWhite = SubscribeTopicItem("lightTable/getConfig/ledBrightnessWhite")
        self.get_ledBrightnessRed = SubscribeTopicItem("lightTable/getConfig/ledBrightnessRed")
        self.get_ledBrightnessGreen = SubscribeTopicItem("lightTable/getConfig/ledBrightnessGreen")
        self.get_ledBrightnessBlue = SubscribeTopicItem("lightTable/getConfig/ledBrightnessBlue")
        self.get_lightTableIsClosed = SubscribeTopicItem("lightTable/isClosed")
        self.get_ledColor = SubscribeTopicItem("lightTable/getColor")

        self.get_mCtrl_swVersion = SubscribeTopicItem("mainController/swVersion")
        self.get_mCtrl_buildTime = SubscribeTopicItem("mainController/buildTime")
        self.get_mCtrl_heartbeat = SubscribeTopicItem("mainController/heartbeat")

        # GlassLifterController (glCtrl)
        self.get_glCtrl_dhPosition = SubscribeTopicItem("downHolder/getDhPosition")
        self.get_glCtrl_parameter_dh1NullPosition = SubscribeTopicItem("downHolder/getConfig/dh1NullPos")
        self.get_glCtrl_parameter_dh2NullPosition = SubscribeTopicItem("downHolder/getConfig/dh2NullPos")
        self.get_glCtrl_parameter_dh3NullPosition = SubscribeTopicItem("downHolder/getConfig/dh3NullPos")
        self.get_glCtrl_parameter_dh4NullPosition = SubscribeTopicItem("downHolder/getConfig/dh4NullPos")

        # OLD VISION-SENSOR (COGNEX)
        self.get_cognex_getVsJobList = SubscribeTopicItem("visionSensor/getVsJobList")
        self.get_cognex_getCurJobName = SubscribeTopicItem("visionSensor/getCurJobName")
        self.get_cognex_getVsJobId = SubscribeTopicItem("visionSensor/getVsJobId")

        # VisionSensorController (vsCtrl)
        self.get_vsCtrl_heartbeat = SubscribeTopicItem("vsController/heartbeat")
        self.set_vsCtrl_vsController_sensorVersion = SubscribeTopicItem("vsController/sensorVersion")
        self.get_vsCtrl_filmTypeIsNegative = SubscribeTopicItem("vsController/getFilmTypeIsNegative")
        self.get_vsCtrl_general_pictureIsInPosition = SubscribeTopicItem("vsController/pictureIsInPosition")
        self.get_vsCtrl_low_contrast_mode_enabled = SubscribeTopicItem("vsController/lcmEnabled")
        self.get_vsCtrl_parameter_sensorExposureTime = SubscribeTopicItem("vsController/getConfig/sensorExposureTime")
        self.get_vsCtrl_centerImage = SubscribeTopicItem("vsController/getConfig/centerImage")

        self.get_vsCtrl_vsFront_pictureIsInPosition = SubscribeTopicItem("vsController/vsFront/pictureIsInPosition")
        self.get_vsCtrl_vsFront_edgePosition = SubscribeTopicItem("vsController/vsFront/edgePosition")
        self.get_vsCtrl_vsFront_liveViewIsEnabled = SubscribeTopicItem("vsController/vsFront/liveViewIsEnabled")
        self.get_vsCtrl_vsFront_fps = SubscribeTopicItem("vsController/vsFront/fps")
        self.get_vsCtrl_vsFront_slope_tile1 = SubscribeTopicItem("vsController/vsFront/slope_tile1")
        self.get_vsCtrl_vsFront_slope_tile2 = SubscribeTopicItem("vsController/vsFront/slope_tile2")
        self.get_vsCtrl_vsFront_imageData = SubscribeTopicItem("vsController/vsFront/imageData")
        self.get_vsCtrl_vsFront_imageDataTn = SubscribeTopicItem("vsController/vsFront/imageDataTn")
        self.get_vsCtrl_vsFront_getImageWidth = SubscribeTopicItem("vsController/vsFront/getConfig/imageWidth")
        self.get_vsCtrl_vsFront_getImageHeight = SubscribeTopicItem("vsController/vsFront/getConfig/imageHeight")
        self.get_vsCtrl_vsFront_parameter_sensorCropTop = SubscribeTopicItem("vsController/vsFront/getConfig/cropTop")
        self.get_vsCtrl_vsFront_parameter_sensorCropRight = SubscribeTopicItem("vsController/vsFront/getConfig/cropRight")
        self.get_vsCtrl_vsFront_parameter_sensorCropLeft = SubscribeTopicItem("vsController/vsFront/getConfig/cropLeft")
        self.get_vsCtrl_vsFront_parameter_sensorCropBottom = SubscribeTopicItem("vsController/vsFront/getConfig/cropBottom")
        self.get_vsCtrl_vsFront_parameter_centerPosition = SubscribeTopicItem("vsController/vsFront/getConfig/centerPosition")
        self.get_vsCtrl_vsFront_parameter_procImageWidth = SubscribeTopicItem("vsController/vsFront/getConfig/procImageWidth")
        self.get_vsCtrl_vsFront_parameter_stopPosition = SubscribeTopicItem("vsController/vsFront/getConfig/stopPosition")
        self.get_vsCtrl_vsFront_liveViewIsEnabled = SubscribeTopicItem("vsController/vsFront/liveViewIsEnabled")

        self.get_vsCtrl_vsRear_pictureIsInPosition = SubscribeTopicItem("vsController/vsRear/pictureIsInPosition")
        self.get_vsCtrl_vsRear_edgePosition = SubscribeTopicItem("vsController/vsRear/edgePosition")
        self.get_vsCtrl_vsRear_liveViewIsEnabled = SubscribeTopicItem("vsController/vsRear/liveViewIsEnabled")
        self.get_vsCtrl_vsRear_fps = SubscribeTopicItem("vsController/vsRear/fps")
        self.get_vsCtrl_vsRear_slope_tile1 = SubscribeTopicItem("vsController/vsRear/slope_tile1")
        self.get_vsCtrl_vsRear_slope_tile2 = SubscribeTopicItem("vsController/vsRear/slope_tile2")
        self.get_vsCtrl_vsRear_imageData = SubscribeTopicItem("vsController/vsRear/imageData")
        self.get_vsCtrl_vsRear_imageDataTn = SubscribeTopicItem("vsController/vsRear/imageDataTn")
        self.get_vsCtrl_vsRear_getImageWidth = SubscribeTopicItem("vsController/vsRear/getConfig/imageWidth")
        self.get_vsCtrl_vsRear_getImageHeight = SubscribeTopicItem("vsController/vsRear/getConfig/imageHeight")
        self.get_vsCtrl_vsRear_parameter_sensorCropTop = SubscribeTopicItem("vsController/vsRear/getConfig/cropTop")
        self.get_vsCtrl_vsRear_parameter_sensorCropRight = SubscribeTopicItem("vsController/vsRear/getConfig/cropRight")
        self.get_vsCtrl_vsRear_parameter_sensorCropLeft = SubscribeTopicItem("vsController/vsRear/getConfig/cropLeft")
        self.get_vsCtrl_vsRear_parameter_sensorCropBottom = SubscribeTopicItem("vsController/vsRear/getConfig/cropBottom")
        self.get_vsCtrl_vsRear_parameter_centerPosition = SubscribeTopicItem("vsController/vsRear/getConfig/centerPosition")
        self.get_vsCtrl_vsRear_parameter_procImageWidth = SubscribeTopicItem("vsController/vsRear/getConfig/procImageWidth")
        self.get_vsCtrl_vsRear_parameter_stopPosition = SubscribeTopicItem("vsController/vsRear/getConfig/stopPosition")

        # AutoFocusController (afCtrl)
        self.get_afCtrl_selectedServo = SubscribeTopicItem("afController/control/selectedServo")
        self.get_afCtrl_initInProgress = SubscribeTopicItem("afController/control/initInProgress")
        self.get_afCtrl_getPosition = SubscribeTopicItem("afController/control/getPosition")
        self.get_afCtrl_isInPosition = SubscribeTopicItem("afController/control/isInPosition")
        self.get_afCtrl_maxRange = SubscribeTopicItem("afController/control/maxRange")
        self.get_afCtrl_readyForCommand = SubscribeTopicItem("afController/control/readyForCommand")
        self.get_afCtrl_unloadLensInProgress = SubscribeTopicItem("afController/control/unloadInProgress")
        self.get_afCtrl_connectedServos = SubscribeTopicItem("afController/control/connectedServos")
        self.get_afCtrl_temperature = SubscribeTopicItem("afController/control/temperature")
        self.get_afCtrl_humidity = SubscribeTopicItem("afController/control/humidity")


class VsControllerPublishTopics:
    def __init__(self):
        self.set_vsController_heartbeat = PublishTopicItem("vsController/heartbeat")
        self.set_vsCtrl_vsController_sensorVersion = PublishTopicItem("vsController/sensorVersion")
        self.set_vsFront_imageData = PublishTopicItem("vsController/vsFront/imageData", qos=0)
        self.set_vsRear_imageData = PublishTopicItem("vsController/vsRear/imageData", qos=0)

        self.set_vsFront_getExposureTimeLive = PublishTopicItem("vsController/vsFront/sensorExposureTime", qos=0)
        self.set_vsRear_getExposureTimeLive = PublishTopicItem("vsController/vsRear/sensorExposureTime", qos=0)

        self.set_vsCtrl_low_contrast_mode_enabled = PublishTopicItem("vsController/lcmEnabled")

        self.set_vsCtrl_filmTypeIsNegative = PublishTopicItem("vsController/getFilmTypeIsNegative")
        self.set_vsCtrl_centerImage = PublishTopicItem("vsController/getConfig/centerImage")

        self.get_vsCtrl_vsFront_slope_tile1 = PublishTopicItem("vsController/vsFront/slope_tile1")
        self.get_vsCtrl_vsFront_slope_tile2 = PublishTopicItem("vsController/vsFront/slope_tile2")

        self.get_vsCtrl_vsRear_slope_tile1 = PublishTopicItem("vsController/vsRear/slope_tile1")
        self.get_vsCtrl_vsRear_slope_tile2 = PublishTopicItem("vsController/vsRear/slope_tile2")

        self.get_vsCtrl_vsFront_pip_debug = PublishTopicItem("vsController/vsFront/pip_debug")
        self.get_vsCtrl_vsRear_pip_debug = PublishTopicItem("vsController/vsRear/pip_debug")

        self.set_vsFront_imageDataTn = PublishTopicItem("vsController/vsFront/imageDataTn", qos=0)
        self.set_vsRear_imageDataTn = PublishTopicItem("vsController/vsRear/imageDataTn", qos=0)

        self.get_vsFront_liveViewIsEnabled = PublishTopicItem("vsController/vsFront/liveViewIsEnabled")
        self.get_vsRear_liveViewIsEnabled = PublishTopicItem("vsController/vsRear/liveViewIsEnabled")

        self.get_vsCtrl_vsFront_fps = PublishTopicItem("vsController/vsFront/fps")
        self.get_vsCtrl_vsRear_fps = PublishTopicItem("vsController/vsRear/fps")

        self.set_pictureIsInPosition = PublishTopicItem("vsController/pictureIsInPosition")
        self.set_vsFront_pictureIsInPosition = PublishTopicItem("vsController/vsFront/pictureIsInPosition")
        self.set_vsRear_pictureIsInPosition = PublishTopicItem("vsController/vsRear/pictureIsInPosition")

        self.set_vsFront_edgeDetected = PublishTopicItem("vsController/vsFront/edgeDetected")
        self.set_vsRear_edgeDetected = PublishTopicItem("vsController/vsRear/edgeDetected")

        self.set_vsFront_edgePosition = PublishTopicItem("vsController/vsFront/edgePosition", qos=0)
        self.set_vsRear_edgePosition = PublishTopicItem("vsController/vsRear/edgePosition", qos=0)

        self.set_vsFront_edgePosition_tile1 = PublishTopicItem("vsController/vsFront/edgePosition_tile1", qos=0)
        self.set_vsFront_edgePosition_tile2 = PublishTopicItem("vsController/vsFront/edgePosition_tile2", qos=0)
        self.set_vsRear_edgePosition_tile1 = PublishTopicItem("vsController/vsRear/edgePosition_tile1", qos=0)
        self.set_vsRear_edgePosition_tile2 = PublishTopicItem("vsController/vsRear/edgePosition_tile2", qos=0)

        self.set_vsFront_getImageWidth = PublishTopicItem("vsController/vsFront/getConfig/imageWidth")
        self.set_vsFront_getImageHeight = PublishTopicItem("vsController/vsFront/getConfig/imageHeight")
        self.set_vsFront_getCropTop = PublishTopicItem("vsController/vsFront/getConfig/cropTop")
        self.set_vsFront_getCropRight = PublishTopicItem("vsController/vsFront/getConfig/cropRight")
        self.set_vsFront_getCropLeft = PublishTopicItem("vsController/vsFront/getConfig/cropLeft")
        self.set_vsFront_getCropBottom = PublishTopicItem("vsController/vsFront/getConfig/cropBottom")
        self.set_vsFront_getCenterPosition = PublishTopicItem("vsController/vsFront/getConfig/centerPosition")
        self.get_vsFront_getProcImageWidth = PublishTopicItem("vsController/vsFront/getConfig/procImageWidth")
        self.set_vsFront_getStopPosition = PublishTopicItem("vsController/vsFront/getConfig/stopPosition")

        self.set_vsRear_getImageWidth = PublishTopicItem("vsController/vsRear/getConfig/imageWidth")
        self.set_vsRear_getImageHeight = PublishTopicItem("vsController/vsRear/getConfig/imageHeight")
        self.set_vsRear_getCropTop = PublishTopicItem("vsController/vsRear/getConfig/cropTop")
        self.set_vsRear_getCropRight = PublishTopicItem("vsController/vsRear/getConfig/cropRight")
        self.set_vsRear_getCropLeft = PublishTopicItem("vsController/vsRear/getConfig/cropLeft")
        self.set_vsRear_getCropBottom = PublishTopicItem("vsController/vsRear/getConfig/cropBottom")
        self.set_vsRear_getCenterPosition = PublishTopicItem("vsController/vsRear/getConfig/centerPosition")
        self.get_vsRear_getProcImageWidth = PublishTopicItem("vsController/vsRear/getConfig/procImageWidth")
        self.set_vsRear_getStopPosition = PublishTopicItem("vsController/vsRear/getConfig/stopPosition")

        self.set_getExposureTime = PublishTopicItem("vsController/getConfig/sensorExposureTime")


class VsControllerSubscribeTopics:
    def __init__(self):
        self.get_fmCtrl_moveCommand = SubscribeTopicItem("filmMoveController/getMoveCommand")
        self.get_fmCtrl_filmMoveDirection = SubscribeTopicItem("filmMoveController/curFilmMoveDirection")

        self.get_fmCtrl_filmMoveDirection = SubscribeTopicItem("filmMoveController/curFilmMoveDirection")

        self.get_vsCtrl_enable_low_contrast_mode = SubscribeTopicItem("vsController/lcm_enabled")

        self.get_vsCtrl_swapSensors = SubscribeTopicItem("vsController/swapSensors")
        self.get_vsCtrl_centerImage = SubscribeTopicItem("vsController/setConfig/centerImage")

        self.get_vsFront_initSensor = SubscribeTopicItem("vsController/vsFront/initSensor")
        self.get_vsRear_initSensor = SubscribeTopicItem("vsController/vsRear/initSensor")
        self.get_setFilmTypeIsNegative = SubscribeTopicItem("vsController/setFilmTypeIsNegative")
        self.get_setExposureTime = SubscribeTopicItem("vsController/setConfig/sensorExposureTime")
        self.get_vsCtrl_vsFront_enableLiveView = SubscribeTopicItem("vsController/vsFront/enableLiveView")
        self.get_vsCtrl_vsRear_enableLiveView = SubscribeTopicItem("vsController/vsRear/enableLiveView")

        self.get_vsCtrl_vsFront_focusCamera = SubscribeTopicItem("vsController/vsFront/focusCamera")
        self.get_vsCtrl_vsRear_focusCamera = SubscribeTopicItem("vsController/vsRear/focusCamera")

        self.get_vsFront_setCropTop = SubscribeTopicItem("vsController/vsFront/setConfig/cropTop")
        self.get_vsFront_setCropRight = SubscribeTopicItem("vsController/vsFront/setConfig/cropRight")
        self.get_vsFront_setCropLeft = SubscribeTopicItem("vsController/vsFront/setConfig/cropLeft")
        self.get_vsFront_setCropBottom = SubscribeTopicItem("vsController/vsFront/setConfig/cropBottom")
        self.get_vsFront_setCenterPosition = SubscribeTopicItem("vsController/vsFront/setConfig/centerPosition")
        self.get_vsFront_setProcImageWidth = SubscribeTopicItem("vsController/vsFront/setConfig/procImageWidth")
        self.get_vsFront_setStopPosition = SubscribeTopicItem("vsController/vsFront/setConfig/stopPosition")

        self.get_vsRear_setCropTop = SubscribeTopicItem("vsController/vsRear/setConfig/cropTop")
        self.get_vsRear_setCropRight = SubscribeTopicItem("vsController/vsRear/setConfig/cropRight")
        self.get_vsRear_setCropLeft = SubscribeTopicItem("vsController/vsRear/setConfig/cropLeft")
        self.get_vsRear_setCropBottom = SubscribeTopicItem("vsController/vsRear/setConfig/cropBottom")
        self.get_vsRear_setCenterPosition = SubscribeTopicItem("vsController/vsRear/setConfig/centerPosition")
        self.get_vsRear_setProcImageWidth = SubscribeTopicItem("vsController/vsRear/setConfig/procImageWidth")
        self.get_vsRear_setStopPosition = SubscribeTopicItem("vsController/vsRear/setConfig/stopPosition")

        self.get_vsFront_captureImage = SubscribeTopicItem("vsController/vsFront/captureImage")
        self.get_vsRear_captureImage = SubscribeTopicItem("vsController/vsRear/captureImage")


class MqttHandler(threading.Thread):
    _externalLogger = None
    connected = False

    def setLogger(self, logger):
        self._externalLogger = logger

    def setMqttValue(self, mqtt_item, value):
        result = self._client_mqtt.publish(mqtt_item.address, value, retain=mqtt_item.retain, qos=mqtt_item.qos)
        if result.rc == mqtt.MQTT_ERR_QUEUE_SIZE:
            raise ValueError('Message is not queued due to ERR_QUEUE_SIZE')
        if mqtt_item.log_output:
            self._logInfoMqttPublishMessage(mqtt_item.address, value, result.rc)

    def enableLogger(self, enable):
        self.loggerEnabled = enable

    def getMqttValue(self, mqtt_item):
        return mqtt_item.value

    def isNewMqttValueAvailable(self, mqtt_item):
        value = mqtt_item.new_value_available
        mqtt_item.new_value_available = False
        return value

    def _onConnect(self, client, userdata, flags, rc):
        self._logInfoMqtt("Connected with result code " + str(rc))
        if self.client_type is None:
            for key, item in self.sTopics.__dict__.items():
                if item.enabled:
                    client.subscribe(item.address)
                    self._logInfoMqtt("subscribe to topic: " + item.address)
        if self.client_type == "vsController":
            for key, item in self.sTopics_vsController.__dict__.items():
                if item.enabled:
                    client.subscribe(item.address)
                    self._logInfoMqtt("subscribe to topic: " + item.address)

        # print(mqtt_topics.printMqttTopics())
        self.connected = True

        # self.publishMqttMessage(self.pTopics.set_dhPosition, "Up")
        # self.publishMqttMessage(self.pTopics.set_ledColor, "Off")

    def _onDisconnect(self, client, userdata, msg):
        self.connected = False
        self._logInfoMqtt("disconnected from mqtt-broker")

    def _onMessage(self, client, userdata, msg):
        cur_message = msg.payload.decode("utf-8").lower()
        if self.client_type is None:
            for key, item in self.sTopics.__dict__.items():
                if item.enabled:
                    if msg.topic == item.address:
                        item.value = cur_message
                        if item.log_output:
                            self._logInfoMqttReceiveMessage(msg.topic, cur_message)

        if self.client_type == "vsController":
            for key, item in self.sTopics_vsController.__dict__.items():
                if item.enabled:
                    if msg.topic == item.address:
                        item.value = cur_message
                        if item.log_output:
                            self._logInfoMqttReceiveMessage(msg.topic, cur_message)

    def _str2bool(self, value):
        value = str(value)
        return value.lower() in ("yes", "true", "t", "1")

    def _logInfoMqttPublishMessage(self, topic, message, rc):
        if self.loggerEnabled:
            message = str(message)
            if len(message) > 10:
                message = message[0:10]
            log_message = "[MQTT-CL][OUT]" + " - " + topic + " [" + str(message) + "]" + " [" + str(rc) + "]"
            if self._externalLogger is None:
                logging.info(log_message)
            else:
                self._externalLogger.info(log_message)

    def _logInfoMqttReceiveMessage(self, topic, message):
        if self.loggerEnabled:
            message = str(message)
            if len(message) > 10:
                message = message[0:10]
            log_message = "[MQTT-CL][IN ]" + " - " + topic + " [" + str(message) + "]"
            if self._externalLogger is None:
                logging.info(log_message)
            else:
                self._externalLogger.info(log_message)

    def _logInfoMqtt(self, message):
        if self.loggerEnabled:
            log_message = "[MQTT-CL]" + " - " + message
            if self._externalLogger is None:
                logging.info(log_message)
            else:
                self._externalLogger.info(log_message)

    def disconnect(self):
        self._client_mqtt.disconnect()

    def __init__(self, mqtt_broker_ip="192.168.0.5", client_id=None, external_logger=None, logger_enabled=True, client_type=None):
        threading.Thread.__init__(self)

        if external_logger is None:
            logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                                handlers=[logging.FileHandler("mqttLog.log"), logging.StreamHandler(sys.stdout)])
        else:
            self._externalLogger = external_logger

        self.loggerEnabled = logger_enabled
        self.client_type = client_type

        if self.client_type == "vsController":
            self.pTopics_vsController = VsControllerPublishTopics()
            self.sTopics_vsController = VsControllerSubscribeTopics()
        else:
            self.pTopics = GeneralPublishTopics()
            self.sTopics = GeneralSubscribeTopics()
        self._client_mqtt = mqtt.Client(client_id=client_id)
        self._client_mqtt.message_retry_set(0.1)
        self._client_mqtt.on_connect = self._onConnect
        self._client_mqtt.on_message = self._onMessage
        self._client_mqtt.on_disconnect = self._onDisconnect
        self._client_mqtt.connect_async(mqtt_broker_ip, 1883, 60)
        self._client_mqtt.loop_start()
        while not self.connected:
            self._logInfoMqtt("try to connect to mqtt-broker...")
            time.sleep(0.2)
