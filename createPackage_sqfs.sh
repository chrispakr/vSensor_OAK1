#!/bin/bash

echo "Select Program:"
select option in "hmiDisplayM4" "vSensorM4" "systemController" "Exit"
do
    case $option in
        "hmiDisplayM4")
            programName="hmiDisplayM4"
            break
            ;;
        "vSensorM4")
            programName="vSensorM4"
            break
            ;;
        "systemController")
            programName="systemController"
            break
            ;;
        "Exit")
            break
            ;;
        *)
            echo "unsupported choice..."
            ;;
    esac
done


#programName="hmiDisplayM4"
venvPath="venv_pi"
echo "#############################################"
echo "Start Deploy Script for $programName"
echo "#############################################"
workingDir="$PWD"
inputDir="/root/source/${programName}"
programDir="${inputDir}/program"
outputPath="/root/packages/${programName}"
fileExtension="sqfs"
buildDate=$(date +"%Y-%m-%d_%H%M")
filename="${programName}_${buildDate}.${fileExtension}"
outputFile="${outputPath}/${filename}"

HOST="web171.dogado.net"
USER="h118407_sw_update"
PASSWD="wMLNKugwDc3WKwghGmELAk2Fj"
uploadPath="ftp://$HOST/_program_images/$programName/"
mountPath="/media/${programName}"

systemctl start systemd-timesyncd.service
echo "stopping service: ${programName}.service..."
systemctl stop "${programName}.service"

rm "${programDir}/build.dt"
echo "${buildDate}" >> "${programDir}/build.dt"

echo "Filename   : $filename"
echo "Dist-Path  : $outputPath"
echo "Version    : $buildDate"
echo "############################################"
echo "create sqfs app-package"
echo "############################################"

mksquashfs "${inputDir}" "${outputFile}"
#pyinstaller --clean -y vSensorM4.spec

echo "############################################"
echo "create app-package finished"
echo "############################################"

read -p "Do you want to activate the new image? (y/n): " antwort
if [[ $antwort == "y" ]] || [[ $antwort == "Y" ]]; then
    echo "unmount folder: ${mountPath})"
    umount "${mountPath}"
    echo "activate new image..."
    cp "${outputFile}" "${outputPath}/active.sqfs"
    echo "start service: ${programName}.service"
    systemctl start "${programName}.service"
else
    echo "new image is not activated..."
fi

read -p "Do you want to publish the image? (y/n): " antwort
if [[ $antwort == "y" ]] || [[ $antwort == "Y" ]]; then
    echo "upload image(${outputFile}) to: ${uploadPath}"
    curl -T "${outputFile}" "${uploadPath}" --user "$USER:$PASSWD"
else
    echo "no image upload done..."
fi


#curl -T "${deployFile}" "ftp://$HOST/$programName/" --user "$USER:$PASSWD"

