# Android App

This is the Android demo app project folder. Import this project to Android Studio directly.

## Dependency

[OpenCV Android SDK](https://opencv.org/android/)

[Snapdragon Neural Processing Engine SDK](https://developer.qualcomm.com/sites/default/files/docs/snpe/overview.html)

## Usage

**NOTICE: This demo can only run on Qualcomm GPU.**

Download model files from: [Google Drive](https://drive.google.com/drive/folders/1jEL7o55bmsSCmJX9ihByaHyBNZYX0CF7?usp=sharing). There should be three .dlc files.

Push all three .dlc model files to device folder `/data/local/tmp/`, etc. `adb push face_detection.dlc /data/local/tmp/`.   

Compile the project using Android Studio or simply download and install the apk from [here](https://drive.google.com/file/d/19Xl8yoTZ_pnrcyQekoIRv913e0QM9ktv/view?usp=sharing), and enjoy the demo!
