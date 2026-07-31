from ads_handler.ads_handler import AdsHandler, MachineType

plc_handler = AdsHandler(
    local_host_ip="192.168.0.30",
    plc_ip_address="192.168.0.10",
    route_name="vSensorM4",
)
# plc_handler.connect_to_plc()

plc_handler.connect()

while plc_handler.connected:
    print(f"wait for connection...{plc_handler.connected}")
    print("Waiting for connection...")
    # plc_handler.vs_ctrl.enableLiveView.init()

    plc_handler.vs_ctrl.vsRightLensPosition.init()
    plc_handler.vs_ctrl.vsRightSerialNumber.init()
    plc_handler.vs_ctrl.vsRightImageCenterOffset.init()

    plc_handler.vs_ctrl.vsLeftLensPosition.init()
    plc_handler.vs_ctrl.vsLeftSerialNumber.init()
    plc_handler.vs_ctrl.vsLeftSerialNumber.init()

    plc_handler.vs_ctrl.settings.stdExposureTimePositive.init()
    plc_handler.vs_ctrl.settings.stdExposureTimeNegative.init()
    (plc_handler.vs_ctrl.rawImageCropTop.init())
    plc_handler.vs_ctrl.rawImageHeight.init()
    plc_handler.vs_ctrl.rawImageWidth.init()

    plc_handler.vs_ctrl.vsRightSensorFps.init()
    plc_handler.vs_ctrl.vsLeftSensorFps.init()
    plc_handler.vs_ctrl.vsRightEdgeState.init()
    plc_handler.vs_ctrl.vsLeftEdgeState.init()
    plc_handler.vs_ctrl.vsRightEdgePosition.init()
    plc_handler.vs_ctrl.vsLeftEdgePosition.init()

    plc_handler.vs_ctrl.isFilmTypeNegative.init()
    plc_handler.vs_ctrl.slopeThreshold.init()
    plc_handler.vs_ctrl.contrastOffset.init()

    plc_handler.vs_ctrl.swapCameras.init()
    plc_handler.vs_ctrl.settings.imageTileCenterOffset.init()

    plc_handler.vs_ctrl.settings.imageTileWidth.init()
    plc_handler.vs_ctrl.settings.imageTileHeight.init()

    plc_handler.vs_ctrl.edgeDetectionRange.init()
    plc_handler.vs_ctrl.settings.contrastPicHeight.init()
    plc_handler.vs_ctrl.settings.contrastPicEdgeOffset.init()
    plc_handler.vs_ctrl.exposureTime.init()
    plc_handler.vs_ctrl.vsLeftStopPosition.init()
    plc_handler.vs_ctrl.vsRightStopPosition.init()
    plc_handler.vs_ctrl.vsLeftStopOffset.init()
    plc_handler.vs_ctrl.vsRightStopOffset.init()

    plc_handler.vs_ctrl.isConnected.init()
    break

while True:
    pass