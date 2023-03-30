#!/bin/bash
programName="vSensor3"
venvPath="venv_pi"
echo "#############################################"
echo "Start Deploy Script for $programName"
echo "#############################################"
workingDir="$PWD"
distPath="${workingDir}/dist/${programName}/"
fullVenvPath=${workingDir}/${venvPath}/bin/activate
deployPath="/mnt/updates/${programName}"
fileExtension="tgz"
buildDate=$(date +"%Y-%m-%d_%H%M")
filename="${programName}_${buildDate}.${fileExtension}"
deployFile="${deployPath}/${filename}"

HOST="web171.dogado.net"
USER="h118407_sw_update"
PASSWD="wMLNKugwDc3WKwghGmELAk2Fj"
FILE="build.dt"
REMOTEPATH="/geodyn_sw_update"

rm build.dt
echo "${buildDate}" >> build.dt

echo "Filename   : $filename"
echo "Dist-Path  : $distPath"
echo "Version    : $buildDate"
echo "Python-venv: $fullVenvPath"
echo "############################################"
echo "create binary package"
echo "############################################"

#source "$fullVenvPath"
#pyinstaller --clean -y main.spec

echo "############################################"
echo "create binary package finished"
echo "############################################"

if [ -d "$distPath" ]; then
  echo "path found: $distPath"
else
  echo "path not found: $distPath"
  exit
fi
echo "change to dist folder: ${distPath}"
cd "${distPath}"
#ls
echo "compress folder to: ${deployFile}"
tar  -czf "${deployFile}" .

curl -T "${deployFile}" "ftp://$HOST/$programName/" --user "$USER:$PASSWD"

