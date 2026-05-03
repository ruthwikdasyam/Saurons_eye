# Sauron's Eye

**See through walls. Real sensors, real soldiers, real time.**

A multi-agent reconnaissance system that gives a dismounted soldier "X-ray vision" through walls before entering a room. Drones equipped with RGB-D cameras stream into a shared world model; the soldier sees the reconstructed geometry overlaid in their AR headset, locked to the real world.

---

## Updates

Running log of decisions, scope changes, and notes. Newest first.

- **2026-05-03** — Working checkpoint. End-to-end stack confirmed running: RealSense → RTAB-Map (RGB-D, no IMU due to iio permissions — see Setup) → `rtabmap_viz` GUI for live map; separately, headset HTTPS server pushes a stand-in scene to the Quest browser. README rewritten to match: architecture diagram, stack table, Setup, Quick Start, Demo Script, Acknowledgements all reflect RTAB-Map + rtabmap_viz (Rerun is gone; librealsense direct path is gone).
- **2026-05-02** — Headset viewer scaffold landed. [headset/server.py](headset/server.py) is an aiohttp HTTPS server on `:8443` (auto-generates a self-signed cert into `headset/certs/` on first run) that serves [headset/web/index.html](headset/web/index.html) and pushes a JSON scene over `/ws`. The Quest browser opens the page, taps "Enter AR", and three.js renders the scene as wireframes against the WebXR `local-floor` reference frame in passthrough. Stand-in scene is `random_cubes()` in [headset/scene.py](headset/scene.py); `POST /regen` reshuffles for all connected clients. Hook for the live SLAM map: replace `random_cubes()` with a function that emits the same `Cube` list — wire format and renderer don't change.
- **2026-05-02** — Dropped Rerun. Visualisation now via `rtabmap_viz` (RTAB-Map's built-in GUI) or rviz. The Python pipeline becomes a data subscriber for future downstream consumers (wire protocol, agentic layer). Added IMU fusion to the SLAM stack: realsense2_camera publishes IMU, `imu_filter_madgwick` filters to orientation, RTAB-Map subscribes via `imu_topic` for tighter pose tracking and faster init. Cleaned out unused modules (`viz.py`, `run.py`, `realsense.py`, `pose.py`, `pointcloud_clean_check.py`, `iphone.py`, `run_iphone.py`, `tests/test_colors.py`).
- **2026-05-02** — RTAB-Map integration via ROS bridge. New modules: [capture/realsense_ros.py](capture/realsense_ros.py) (camera frames over ROS topics with message-filter sync), [capture/pose_rtabmap.py](capture/pose_rtabmap.py) (`RtabmapPose` Protocol implementation, subscribes to `/rtabmap/odom` in a background thread, time-interpolates poses for camera frame timestamps), [capture/run_rtabmap.py](capture/run_rtabmap.py) (entrypoint). Result: rate jumps from 3 Hz → 25–35 Hz, drift over 30 s of motion + return drops from ~1.8 m → ~3 cm, `pose` cost per frame drops from 200 ms → <1 ms. Also: deleted unused TSDF baselines (`mapper.py`, `tsdf_check.py`, `tsdf_odom_check.py`, `pointcloud_check.py`).
- **2026-05-02** — Adopted three patterns from dimos: column carving (evict stale voxels in re-observed XY columns), GPU-style sparse occupancy via Open3D `o3c.HashMap` (replaces TSDF), and per-axis matplotlib LUT colouring. Tests-first under `tests/`. Pose still bottleneck (~200 ms/frame on RGB-D odometry); mapping side trivial.
- **2026-05-02** — Capture refactored into a layered library so the pose source is swappable. Modules: `realsense.py` (camera + filters + frame iterator), `pose.py` (`PoseSource` Protocol + `RgbdOdometryPose` / `IdentityPose` / `MavlinkPose` stub), `mapper.py` (TSDF wrapper), `occupancy.py` (sparse hashmap mapper), `viz.py` (Rerun helpers), `run.py` (entrypoint). On real drone, only the line `pose_source = RgbdOdometryPose(...)` changes to `MavlinkPose(...)`. Run with `python -m capture.run`. Old flat scripts remain as diagnostic baselines.
- **2026-05-02** — Capture pipeline diagnostic stack working. Five scripts under `capture/` build up from device check → per-frame cloud → cleaned voxelised cloud → stationary TSDF → TSDF + RGB-D odometry. The working "clean voxels" baseline is [pointcloud_clean_check.py](capture/pointcloud_clean_check.py): RealSense filters → Open3D cloud → 5 cm voxel downsample → statistical outlier removal → uniform-sphere display in Rerun, ~10 Hz, no colour. [tsdf_odom_check.py](capture/tsdf_odom_check.py) layers RGB-D odometry + TSDF fusion at 5 cm / 6 m on top.
- **2026-05-02** — Capture pipeline cooking. Step 1 ([pointcloud_check.py](capture/pointcloud_check.py)) streams per-frame Open3D clouds → Rerun. Step 3 stationary ([tsdf_check.py](capture/tsdf_check.py)) fuses RGB-D into a 2 cm TSDF with identity pose; ~7 K points stable from a single viewpoint. Spatial + temporal + hole-fill filters on raw depth before alignment.
- **2026-05-02** — Repo skeleton + specs in place ([protocol.md](shared/protocol.md), [frames.md](shared/frames.md)).

---

## Roadmap / future work

Captured here so they don't get lost. Independent of current scope.

### Agentic layer

Wrap the capture pipeline in an agent loop that accepts natural-language commands from the soldier and steers what gets surfaced in the headset. Voice in, world-model queries out.

Example commands the agent should handle:
- "Show me the next person." — cycle highlight to the next tracked `person` detection.
- "Focus on the weapon on the table." — re-rank detections, pin highlight on the matching object.
- "Ignore the chair." — class/instance suppression for the rest of the engagement.
- "How many people in the room?" — query the current world model, speak/display the answer.
- "Mark this as cleared." — annotate a region of the point cloud as friendly-checked.

Sketch of where it sits:

```
voice (Quest mic) ──► STT ──► agent (tool-using LLM) ──► tools:
                                                        ├─ query_detections()
                                                        ├─ set_highlight(id)
                                                        ├─ suppress_class(name)
                                                        ├─ annotate_region(bbox, label)
                                                        └─ describe_scene()
                                                              │
                                                              ▼
                                                       TTS / overlay update
```

Open questions:
- Where does the agent run — laptop (latency: STT round-trip over Wi-Fi) or Quest (limited compute)? Probably laptop, with Quest just shipping audio frames.
- Tool surface: extend the wire protocol with a client→server channel (currently server-push only). Likely needs a `commands` message type and a bump to `saurons-eye/2`.
- Model choice: low-latency model for tool-calling, escalate to a larger model for scene-description queries.

### Other directions

- **Drop in RTAB-Map as the L source.** Real SLAM (loop closure + RGB-D + IMU fusion) replacing our hand-rolled `RgbdOdometryPose`. Apt-installable on Ubuntu (`apt install ros-humble-rtabmap-ros`); plug in as `RtabmapPose` (ROS `/rtabmap/odom` subscriber → `PoseSource`). Drift goes to zero on revisits, RGB-D + IMU fusion improves accuracy materially over our frame-to-frame odometry. ORB-SLAM3 is the alternative but a build-time nightmare; skip.
- **Migrate to a robotics framework like dimos.** dimos provides the GPU `VoxelBlockGrid` mapper (we copied the concept), mature `PointCloud2` / `PoseStamped` / `Odometry` message types with `to_rerun()` baked in, an LCM pubsub + Rerun bridge that auto-logs anything with a `to_rerun()` method, and Module/StreamModule/Transformer abstractions. Patterns we've already borrowed: column carving, per-axis matplotlib colormap LUTs, `PoseSource` Protocol style. The natural target once the architecture stabilises.
- **Real drone integration via MAVLink (concrete plan).** Swap the hand-carried RealSense for an actual pocket drone (Black Hornet / Skydio class). The architecture already supports this — write a `MavlinkPose(PoseSource)` (~80 lines): pymavlink subscriber to `ODOMETRY`, buffer poses in the existing `PoseBuffer`, interpolate to camera timestamp, multiply by rigid `T_drone_camera` extrinsic, return T_WC. Then in the entrypoint flip *two* lines:
  ```python
  pose_source = MavlinkPose(connection_string="udpin:0.0.0.0:14540", T_drone_camera=...)
  mapper     = OccupancyVoxelGrid(voxel_size=0.05, carve_columns=True)
  ```
  Everything else — camera (still `RealSenseCapture`), per-frame cleaning recipe, viz, wire format, headset — stays. Drone's onboard VIO replaces our weak L entirely; `OccupancyVoxelGrid` handles M (no RTAB-Map needed because the drone's pose is already drift-corrected). New entrypoint `capture/run_mavlink.py` is mostly a copy of `run_rtabmap.py` with the two swaps.
- **iPhone Pro as a sensor stack.** Apple's iPhone Pro (12 Pro and later) has LiDAR ToF depth, ARKit (drift-corrected pose with loop closure), IMU, and high-quality cameras — in many ways a better RGB-D + L stack than the D435i. Integration paths:
  1. **Record3D / NeRFCapture iPhone apps** stream RGB + depth + pose over WiFi to a desktop. The `record3d` Python package handles the receiver side. We'd write `ArkitPose(PoseSource)` and `ArkitCapture` adapters; downstream pipeline unchanged.
  2. **ARKit `ARMeshAnchor`** for free dense mapping — the iPhone literally publishes a triangle mesh of the scanned environment, drift-corrected, no integration needed on our side. Would slot in as `ArkitMeshMap` (analogous to `RtabmapDenseMap`).
  3. **Range/resolution tradeoff**: iPhone LiDAR is 256×192 at ~5m (ToF) vs D435i's 640×480 at ~3–6m (stereo). LiDAR wins on textureless surfaces and indoor lighting; stereo wins on resolution. Both fine for room-scale.
  4. **Use case**: ideal development sensor when there's no drone (better than the hand-carried D435i metaphor). Not flying any time soon, but for bench iteration on the L→M→viz pipeline, it's the cleanest option after RealSense+RTAB-Map.
- **Multi-drone fusion.** Multiple drones, multiple capture nodes, one merged world model. Protocol already reserves `frame: "world"` for this.
- **Edge inference on the drone.** Move YOLO + Open3D off the laptop onto the drone's compute (Jetson Orin Nano class) so the link only carries deltas.
- **Threat classification beyond COCO.** Fine-tune a detector on weapons, IEDs, tripwires — the classes that actually matter for room clearing.
- **Persistent world model across engagements.** Save the reconstructed building between rooms; the soldier walks back in tomorrow and the map is already there.
- **GPU TSDF / dense mapping.** Move from sparse occupancy to a full GPU TSDF (Open3D Tensor `VoxelBlockGrid`, or Voxblox, or ESDF for collision-aware planning) for smoother surfaces and richer queries.
- **Light IMU fusion (rotation prior).** Read the D435i's BMI055 gyro, integrate over the 33 ms inter-frame gap, feed as init guess to `compute_rgbd_odometry`. Helps with fast rotation. Not a substitute for proper VIO.

---

## Quick Start

Tested on Ubuntu 22.04 / Python 3.10+. macOS works for everything except Quest deployment.

```bash
git clone https://github.com/ruthwikdasyam/Saurons_eye.git
cd Saurons_eye

# Create + activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Run the capture side (laptop tethered to RealSense). Three terminals, ROS Humble sourced in each.

```bash
# 1. RealSense camera (RGB-D only — IMU has Linux iio permission issues, see Setup)
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 2. RTAB-Map SLAM with built-in viewer (rtabmap_viz GUI is the main visualisation)
ros2 launch rtabmap_launch rtabmap.launch.py rgb_topic:=/camera/camera/color/image_raw depth_topic:=/camera/camera/aligned_depth_to_color/image_raw camera_info_topic:=/camera/camera/color/camera_info approx_sync:=false rtabmap_viz:=true frame_id:=camera_link

# 3. Our Python data subscriber (collects pose + dense map; place to plug downstream consumers)
python -m capture.run_rtabmap
```

Run the headset side (Quest browser, served via HTTPS — WebXR requires a secure origin):

```bash
python -m headset.server
# On Quest: open browser → https://<laptop-ip>:8443
# Accept the self-signed cert warning, then tap "Enter AR".
```

See [Setup](#setup) for the full breakdown.

---

## The Problem

Room clearing is the deadliest task in modern infantry combat. The "fatal funnel" — the doorway — kills soldiers because the defender knows where the attacker has to be, and the attacker doesn't know what's inside. The Army has spent over $20B on soldier-worn AR (IVAS) and bought tens of thousands of pocket drones (Black Hornet) trying to solve this. Nobody has cleanly fused the two into pre-entry X-ray vision.

Sauron's Eye does.

## The Concept

1. Soldier stacks on a wall. Unknown threat behind a closed door.
2. They deploy one or more pocket drones from their kit.
3. Drones fly into the next room, scanning RGB-D as they go.
4. A live point-cloud reconstruction of the room is built in a shared world frame.
5. Detected occupants and objects are highlighted (person / weapon / clear).
6. The soldier sees the entire reconstructed room overlaid through their AR optic, in correct world position — they look at the wall, and the room appears behind it.

## Current scope

The current implementation demonstrates the full pipeline end-to-end on real hardware. It is deliberately narrow.

**In scope:**
- Single hand-carried RealSense D435i acting as the "drone" (real drone integration is a future step)
- One operator wearing a Meta Quest 3 acting as the "soldier"
- A cardboard partition standing in for a wall, with a doorway
- Live SLAM (RTAB-Map RGB-D with loop closure) building a drift-corrected dense map
- `rtabmap_viz` GUI on the laptop for live visualisation + debugging
- World-frame anchoring so the map stays locked when the Quest user moves their head
- Object detection (person / chair / etc.) using YOLOv8 on the RGB stream, projected into 3D
- Aiohttp HTTPS server pushing scene JSON to the Quest browser; three.js renders in WebXR passthrough

**Future / not yet implemented:**
- Actual drone integration (currently a hand-carried sensor stand-in)
- Multi-agent fusion across two drones
- Edge inference on the drone itself (currently runs on laptop)
- Encrypted comms / mesh networking
- IMU-fused SLAM (Linux iio permissions block the D435i IMU; tracked as a Setup task)
- Live SLAM map → headset (currently the headset gets a stand-in scene; wire protocol exists, plug-in pending)

## Architecture

```
┌──────────────────────────────────────┐         ┌──────────────────────────────┐
│   "Drone" (laptop + D435i)           │         │   "Soldier" (Meta Quest 3)   │
│                                      │         │                              │
│   realsense2_camera (ROS) ─┐         │         │     ┌──────────────────┐     │
│                            ▼         │         │     │  three.js        │     │
│              rtabmap_ros (RGB-D SLAM)│         │     │  + WebXR         │     │
│              ├─► /rtabmap/odom       │         │     │  + passthrough   │     │
│              ├─► /rtabmap/cloud_map  │         │     └────────▲─────────┘     │
│              └─► rtabmap_viz GUI     │         │              │               │
│                            │         │         │              │               │
│                            ▼         │         │              │               │
│         capture.run_rtabmap (Python) │         │              │               │
│         RtabmapPose + RtabmapDenseMap│         │              │               │
│                            │         │         │              │               │
│                            ▼         │         │              │               │
│              headset.server (aiohttp HTTPS) ───┼──── /ws (JSON scene) ───────┤
│                            │         │         │              │               │
│                            ▼         │         │              │               │
│              YOLOv8 (planned) ───────┤         │     overlay  │               │
│                                      │         │     locked   │               │
│                                      │         │     to world │               │
│                                      │         │     frame    │               │
└──────────────────────────────────────┘         └──────────────────────────────┘
```

**Coordinate frames.** Both devices initialize at a known origin (ArUco marker on the partition, planned). The Quest sees the marker on first launch and computes its head pose relative to it. RTAB-Map keeps the laptop side anchored via its own SLAM (loop closure brings drift to ~3 cm over 30 s of motion). Both sides then maintain pose via their own VIO — RTAB-Map for the camera, Meta Insight for the headset.

**Pluggable pose source.** The Python pipeline uses a `PoseSource` Protocol so the localisation half is swappable. Today: `RtabmapPose` (subscribes to `/rtabmap/odom`). On a real drone: `MavlinkPose` subscribes to the autopilot's pose topic and the rest of the pipeline doesn't change. Mapping, wire format, and headset rendering are pose-source-agnostic. See [Roadmap](#roadmap--future-work) for the MAVLink plug.

## Stack

| Component | Choice | Why |
|---|---|---|
| Depth camera | Intel RealSense D435i | RGB-D + IMU, mature drivers |
| Camera bridge | `realsense2_camera` (ROS Humble) | Publishes RGB / depth / camera_info / IMU as ROS topics |
| SLAM (L+M) | RTAB-Map (RGB-D, with loop closure) | Drift-corrected pose + dense map, apt-installable, dedicated GUI |
| Visualisation | `rtabmap_viz` (RTAB-Map's GUI) | Purpose-built SLAM viewer; rviz also works against the same topics |
| Python consumer | `capture.run_rtabmap` (rclpy) | Subscribes to `/rtabmap/odom` + `/rtabmap/cloud_map`; plug-in point for wire protocol / agentic layer / detection |
| Object detection | YOLOv8 (Ultralytics) — planned | Best off-the-shelf detector, COCO classes cover our needs |
| Wire format | aiohttp HTTPS + WebSocket carrying JSON scene | HTTPS required by WebXR; JSON for fast iteration (move to msgpack later) |
| Headset runtime | WebXR (three.js) in Quest browser | Avoids Unity build pipeline |
| Marker tracking | OpenCV ArUco — planned | Trivial integration both sides |

We are deliberately **not** using a simulator. All sensor data is real.

## Repository Layout

```
Saurons_eye/
├── README.md                      ← you are here
├── CLAUDE.md                      ← collaboration + project conventions
├── capture/                       ← runs on laptop tethered to RealSense
│   ├── realsense_ros.py           ← ROS bridge: camera frames + intrinsics via topics
│   ├── pose_rtabmap.py            ← RtabmapPose: subscribes to /rtabmap/odom
│   ├── rtabmap_map.py             ← RtabmapDenseMap: subscribes to /rtabmap/cloud_map
│   ├── occupancy.py               ← sparse voxel hashmap (alternative mapper)
│   ├── run_rtabmap.py             ← entrypoint: data subscriber for downstream consumers
│   ├── realsense_check.py         ← low-level RealSense smoke test (no ROS, no Open3D)
│   └── _smoke_rtabmap_pose.py     ← verifies RtabmapPose end-to-end
├── headset/                       ← served to Quest browser
│   ├── server.py                  ← aiohttp HTTPS + /ws JSON push; auto-generates self-signed cert
│   ├── scene.py                   ← `Cube` dataclass + stand-in `random_cubes()`
│   └── web/index.html             ← three.js + WebXR scene; renders cubes as wireframes in passthrough
├── shared/
│   ├── protocol.md                ← wire format spec
│   └── frames.md                  ← coordinate frame conventions
├── tests/                         ← pytest suite (carving / voxel grid / pose / ros helpers / dense map)
├── demo/                          ← demo setup notes (TBD)
└── scripts/                       ← setup helpers (TBD)
```

## Wire Protocol

WebSocket on port 8765. Server (laptop) pushes, client (Quest) consumes. Messages are msgpack:

```
{
  "t": <float, seconds since epoch>,
  "type": "points" | "detections" | "pose",
  "frame": "world",
  "data": <see below>
}
```

- `points`: voxel-downsampled to 5 cm
- `detections`: per-object 3D bounding boxes with class + confidence
- `pose`: current camera pose in world frame, for debug overlay

Full spec in [shared/protocol.md](shared/protocol.md).

## Setup

Tested on Ubuntu 22.04 + Python 3.10 + ROS Humble.

**One-time:**
```bash
# ROS packages (RealSense bridge + SLAM)
sudo apt install ros-humble-realsense2-camera ros-humble-rtabmap-ros

# Project venv with access to system rclpy
git clone <repo>
cd Saurons_eye
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Per session — capture side (3 terminals, ROS sourced in each):**
```bash
# 1. Camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 2. SLAM + GUI
ros2 launch rtabmap_launch rtabmap.launch.py rgb_topic:=/camera/camera/color/image_raw depth_topic:=/camera/camera/aligned_depth_to_color/image_raw camera_info_topic:=/camera/camera/color/camera_info approx_sync:=false rtabmap_viz:=true frame_id:=camera_link

# 3. Python data subscriber
python -m capture.run_rtabmap
```

**Per session — headset side:**
```bash
python -m headset.server
# On Quest: open browser → https://<laptop-ip>:8443  (accept self-signed cert)
```

**Tests:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```

**Known issue — D435i IMU permissions:**
The IMU is exposed via Linux's `iio` subsystem and needs librealsense udev rules + `hid_sensor_*` modules blacklisted to be readable. Without that, the `enable_gyro:=true enable_accel:=true` flags fail with "Permission denied" / "Hid device is busy". RGB-D-only SLAM works fine without IMU; tracked as a Setup task.

## Demo Script (90 seconds)

1. **0:00–0:10** — Establishing shot: operator stacks on wall, drone visible in their kit.
2. **0:10–0:25** — Drone (RealSense) deployed through doorway. Quest user sees nothing through wall yet.
3. **0:25–0:50** — Drone scans interior. Dense map blooms in real time in `rtabmap_viz` (cutaway shot). Quest user, still outside, sees the room geometry materialize through the wall, locked in place.
4. **0:50–1:10** — A person stands behind the door. YOLO detects them. Red bounding outline appears in the Quest view, highlighting their position through the wall.
5. **1:10–1:25** — Quest user walks along the wall. Overlay stays locked to the real world. Parallax sells it.
6. **1:25–1:30** — Tagline.

## Pitch Opener

> In Tolkien, the Eye of Sauron saw everything, everywhere. We built one. For real. For soldiers.

## License

MIT.

## Team

[Names + roles to fill in]

## Acknowledgements

Built on RTAB-Map, ROS Humble, Open3D, three.js, aiohttp, and Intel RealSense. Architecture patterns adapted from [dimos](https://github.com/dimensional-OS/dimos).
