# Sauron's Eye

## Install

```bash
git clone https://github.com/ruthwikdasyam/Saurons_eye.git
cd Saurons_eye

sudo apt install ros-humble-realsense2-camera ros-humble-rtabmap-ros

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run — capture (3 terminals, ROS sourced in each)

```bash
# 1. Camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

```bash
# 2. SLAM + GUI
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/camera/color/camera_info \
  approx_sync:=false rtabmap_viz:=true frame_id:=camera_link
```

```bash
# 3. Python subscriber
python -m capture.run_rtabmap
```

## Run — headset

```bash
python -m headset.server
# On Quest: browser → https://<laptop-ip>:8443  (accept self-signed cert, tap "Enter AR")
```

## Run — segmentation pipeline (standalone, no ROS)

```bash
python -m capture.run_segment --vr --camera-on-headset
```

With drone pose:

```bash
python -m capture.run_segment --vr --pose-source drone --drone-port /dev/ttyUSB0
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```
