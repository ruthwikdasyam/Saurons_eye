# Sauron's Eye

**See through walls. Real sensors, real soldiers, real time.**

A multi-agent reconnaissance system that gives a dismounted soldier "X-ray vision" through walls before entering a room. Drones equipped with RGB-D cameras stream into a shared world model; the soldier sees the reconstructed geometry overlaid in their AR headset, locked to the real world.

---

## Updates

Running log of decisions, scope changes, and notes. Newest first.

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
- **Real drone integration.** Swap the hand-carried RealSense for an actual pocket drone (Black Hornet / Skydio class). Requires solving the airframe pose → world frame handoff cleanly: subscribe to MAVLink `ODOMETRY`, apply rigid `T_drone_camera` extrinsic, return as a `PoseSource`. Drops in via the existing interface. Drone's onboard VIO replaces our weak L entirely.
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

Run the capture side (laptop tethered to RealSense):

```bash
python -m capture.run        # WebSocket on :8765 + Rerun viewer
```

Run the headset side (Quest browser):

```bash
cd headset
python -m http.server 8000
# On Quest: open browser → http://<laptop-ip>:8000
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
- Live voxel-map reconstruction streamed from RealSense laptop → Quest
- World-frame anchoring so the cloud stays locked when the Quest user moves their head
- Object detection (person / chair / etc.) using YOLOv8 on the RGB stream, projected into 3D
- Rerun visualizer on the operator's laptop showing the live reconstruction (also serves as a backup demo channel)

**Future / not yet implemented:**
- Actual drone integration (currently a hand-carried sensor stand-in)
- Multi-agent fusion across two drones
- Edge inference on the drone itself (currently runs on laptop)
- Encrypted comms / mesh networking
- Production-grade SLAM with loop closure (see Roadmap)

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│   "Drone" (laptop + D435i)  │         │   "Soldier" (Meta Quest 3)   │
│                             │         │                              │
│   librealsense ──┐          │         │                              │
│                  ▼          │         │     ┌──────────────────┐     │
│           pose source       │         │     │  Three.js        │     │
│           (RGB-D odom now,  │         │     │  + WebXR         │     │
│            MAVLink later)   │         │     │  + passthrough   │     │
│                  │          │         │     └────────▲─────────┘     │
│                  ▼          │         │              │               │
│           occupancy mapper  │         │              │               │
│           (sparse voxels)   │         │              │               │
│                  │          │         │              │               │
│                  ├──────────┼─ WebSocket (LAN) ──────┤               │
│                  │          │         │              │               │
│                  ▼          │         │              │               │
│              YOLOv8 ────────┤         │     overlay  │               │
│              (person, etc.) │         │     locked   │               │
│                  │          │         │     to world │               │
│                  ▼          │         │     frame    │               │
│              Rerun viewer   │         │              │               │
│              (debug + demo) │         │              │               │
└─────────────────────────────┘         └──────────────────────────────┘
```

**Coordinate frames.** Both devices initialize at a known origin (ArUco marker on the partition). The RealSense computes its pose relative to the marker via OpenCV. The Quest sees the same marker on first launch and computes its head pose relative to it. From that moment, both devices maintain pose via their own VIO (visual-inertial odometry) — RealSense via Open3D's RGB-D odometry, Quest via Meta's Insight tracking. Drift is bounded by re-anchoring against the marker periodically.

**Pluggable pose source.** The capture pipeline uses a `PoseSource` Protocol so the localisation half of SLAM is swappable. Today: `RgbdOdometryPose` (Open3D dense alignment). On a real drone: `MavlinkPose` subscribes to the autopilot's pose topic and the rest of the pipeline doesn't change. Mapping (occupancy voxel grid), wire format, and headset rendering are pose-source-agnostic.

## Stack

| Component | Choice | Why |
|---|---|---|
| Depth camera | Intel RealSense D435i | RGB-D + IMU, mature drivers |
| Capture-side runtime | Python + librealsense | Fast iteration, ecosystem |
| Localisation (L) | Open3D RGB-D odometry (Steinbrücker / Kerl), pluggable behind `PoseSource` | Hackable, ~80 lines wrapper. Replaceable with RTAB-Map / MAVLink without touching the rest |
| Mapping (M) | Sparse occupancy voxel grid via Open3D `o3c.HashMap` | Cheaper than TSDF, GPU-capable, supports column carving for dynamic objects |
| Object detection | YOLOv8 (Ultralytics) | Best off-the-shelf detector, COCO classes cover our needs |
| Wire format | WebSocket carrying msgpack-encoded voxel deltas | Voxel-downsampled to 5 cm to fit Wi-Fi |
| Visualizer (laptop) | Rerun | Native point cloud + pose support, GPU rendering |
| Headset runtime | WebXR (Three.js) in Quest browser | Avoids Unity build pipeline |
| Marker tracking | OpenCV ArUco | Trivial integration both sides |

We are deliberately **not** using a simulator. All sensor data is real.

## Repository Layout

```
Saurons_eye/
├── README.md                      ← you are here
├── CLAUDE.md                      ← collaboration + project conventions
├── capture/                       ← runs on laptop tethered to RealSense
│   ├── realsense.py               ← librealsense direct: camera + filter chain + frame iterator
│   ├── realsense_ros.py           ← ROS bridge: same interface, camera frames via topics
│   ├── pose.py                    ← PoseSource Protocol + RGB-D odometry / Identity / Mavlink stub
│   ├── pose_rtabmap.py            ← RtabmapPose: subscribes to /rtabmap/odom
│   ├── occupancy.py               ← sparse voxel hashmap (the mapper)
│   ├── viz.py                     ← Rerun logging helpers
│   ├── run.py                     ← entrypoint (librealsense + RGB-D odometry)
│   ├── run_rtabmap.py             ← entrypoint (ROS + RTAB-Map for L)
│   ├── realsense_check.py         ← low-level RealSense smoke test (no ROS, no Open3D)
│   ├── pointcloud_clean_check.py  ← per-frame cleaning recipe baseline
│   └── _smoke_rtabmap_pose.py     ← verifies RtabmapPose end-to-end
├── headset/                       ← served to Quest browser (TBD)
├── shared/
│   ├── protocol.md                ← wire format spec
│   └── frames.md                  ← coordinate frame conventions
├── tests/                         ← pytest suite (colors / carving / voxel grid)
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

Tested on Ubuntu 22.04 + Python 3.10. macOS works for everything except Quest deployment.

```bash
git clone <repo>
cd Saurons_eye
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Capture side:
```bash
python -m capture.run
```

Quest side:
```bash
cd headset
python -m http.server 8000
# Then on Quest: open browser → http://<laptop-ip>:8000
```

Tests:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```

## Demo Script (90 seconds)

1. **0:00–0:10** — Establishing shot: operator stacks on wall, drone visible in their kit.
2. **0:10–0:25** — Drone (RealSense) deployed through doorway. Quest user sees nothing through wall yet.
3. **0:25–0:50** — Drone scans interior. Voxel map blooms in real time on Rerun (cutaway shot). Quest user, still outside, sees the room geometry materialize through the wall, locked in place.
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

Built on Open3D, Rerun, Three.js, Ultralytics, and Intel RealSense. Mapping and visualisation patterns adapted from [dimos](https://github.com/dimensional-OS/dimos).
