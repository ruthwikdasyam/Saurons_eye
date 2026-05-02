# WALLHACK

**See through walls. Real sensors, real soldiers, real time.**

A multi-agent reconnaissance system that gives a dismounted soldier "X-ray vision" through walls before entering a room. Drones equipped with RGB-D cameras stream into a shared world model; the soldier sees the reconstructed geometry overlaid in their AR headset, locked to the real world.

Built for the National Security Hackathon (Army xTech), May 2–3 2026, San Francisco.

Problem Statements addressed: **PS2 (Edge / Drones)** primary, **PS3 (C2)** and **PS1 (Sensor Fusion)** secondary.

---

## Updates

Running log of decisions, scope changes, and notes since kickoff. Newest first.

- **2026-05-02** — Repo skeleton + specs in place ([protocol.md](shared/protocol.md), [frames.md](shared/frames.md)). `capture/`, `headset/`, `demo/`, `scripts/` are empty stubs; build starts from here.

---

## Ideas / Post-v1

Captured here so they don't get lost during the 24-hour sprint. **None of these are in v1 scope** — they get attempted only after the H14 integration checkpoint passes and there is real slack.

### Agentic layer (high priority post-v1)

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

Open questions to resolve before building:
- Where does the agent run — laptop (latency: STT round-trip over Wi-Fi) or Quest (limited compute)? Probably laptop, with Quest just shipping audio frames.
- Tool surface: extend the wire protocol with a client→server channel (currently v1 is server-push only). Likely needs a `commands` message type and a bump to `saurons-eye/2`.
- Model choice: Claude Haiku for tool-calling latency, escalate to Sonnet for scene-description queries.

### Other ideas

- **Real drone integration.** Swap the hand-carried RealSense for an actual pocket drone (Black Hornet / Skydio class). Requires solving the airframe pose → world frame handoff cleanly.
- **Multi-drone fusion.** Two drones, two capture laptops, one merged world model. Protocol already reserves `frame: "world"` for this.
- **Edge inference on the drone.** Move YOLO + Open3D off the laptop onto the drone's compute (Jetson Orin Nano class) so the link only carries deltas.
- **Threat classification beyond COCO.** Fine-tune a detector on weapons, IEDs, tripwires — the classes that actually matter for room clearing.
- **Persistent world model across engagements.** Save the reconstructed building between rooms; the soldier walks back in tomorrow and the map is already there.

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
cd capture
python server.py        # WebSocket on :8765 + Rerun viewer
```

Run the headset side (Quest browser):

```bash
cd headset
python -m http.server 8000
# On Quest: open browser → http://<laptop-ip>:8000
```

See [Setup](#setup) further down for the full breakdown, and [Build Plan](#build-plan-24-hours) for the 24-hour workstream split.

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

## V1 Scope

V1 demonstrates the full pipeline end-to-end on real hardware in a controlled environment. It is deliberately narrow.

**In scope for v1:**
- Single hand-carried RealSense D435i acting as the "drone" (real drone integration is post-hackathon)
- One operator wearing a Meta Quest 3 acting as the "soldier"
- A cardboard partition standing in for a wall, with a doorway
- Live point-cloud reconstruction streamed from RealSense laptop → Quest
- World-frame anchoring so the cloud stays locked when the Quest user moves their head
- Object detection (person / chair / etc.) using YOLOv8 on the RGB stream, projected into 3D
- Rerun visualizer on the operator's laptop showing the live reconstruction (also serves as backup demo)

**Out of scope for v1 (stretch / post-hackathon):**
- Actual drone integration (we're using a hand-carried sensor as a stand-in)
- Multi-agent fusion across two drones
- Edge inference on the drone itself (currently runs on laptop)
- Encrypted comms / mesh networking
- Production-grade SLAM with loop closure

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│   "Drone" (laptop + D435i)  │         │   "Soldier" (Meta Quest 3)   │
│                             │         │                              │
│   librealsense ──┐          │         │                              │
│                  ▼          │         │     ┌──────────────────┐     │
│              SLAM (Open3D   │         │     │  Three.js        │     │
│              RGB-D odometry │         │     │  + WebXR         │     │
│              + TSDF fusion) │         │     │  + passthrough   │     │
│                  │          │         │     └────────▲─────────┘     │
│                  ▼          │         │              │               │
│          point cloud + pose │         │              │               │
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

**Coordinate frames.** Both devices initialize at a known origin (ArUco marker on the partition). The RealSense computes its pose relative to the marker via OpenCV. The Quest sees the same marker on first launch and computes its head pose relative to it. From that moment, both devices maintain pose via their own VIO (visual-inertial odometry) — RealSense via Open3D's RGB-D odometry, Quest via Meta's Insight tracking. Drift over a 60-second demo is small enough to be invisible.

## Stack

| Component | Choice | Why |
|---|---|---|
| Depth camera | Intel RealSense D435i | Best RGB-D + IMU in our budget, mature drivers |
| Capture-side runtime | Python + librealsense | Fastest path, everyone knows Python |
| SLAM / fusion | Open3D (RGB-D odometry + ScalableTSDFVolume) | Hackable, ~150 lines, no ROS |
| Object detection | YOLOv8 (Ultralytics) | Best off-the-shelf detector, COCO classes cover our needs |
| Wire format | WebSocket carrying msgpack-encoded point cloud deltas | Voxel-downsampled to fit Wi-Fi |
| Visualizer (laptop) | Rerun | Native RGB-D + point cloud + pose support, beautiful demos |
| Headset runtime | WebXR (Three.js) in Quest browser | Avoids Unity build pipeline; if passthrough doesn't work in 60min, fall back to Unity + Meta XR SDK |
| Marker tracking | OpenCV ArUco | Trivial integration both sides |

We are deliberately **not** using a simulator. All sensor data is real.

## Build Plan (24 Hours)

| Hours | Workstream A: Capture/SLAM | Workstream B: Quest/Render | Workstream C: Detection/Polish |
|---|---|---|---|
| 0–2 | RealSense streaming, Rerun hooked up | Get Quest browser to render a Three.js scene with WebXR + passthrough | YOLOv8 running on RGB frames |
| 2–6 | Open3D RGB-D odometry working, accumulating point cloud | WebSocket client in Three.js, render incoming points | Project 2D detections into 3D using depth |
| 6–10 | TSDF integration, downsampling for wire | ArUco marker pose recovery on Quest side | Highlight detected entities in cloud |
| 10–14 | End-to-end stream live to Quest | Coordinate frame alignment locked in | Visualize detections in Rerun |
| 14–18 | **Integration checkpoint** — full loop running | | |
| 18–21 | Polish: cardboard partition setup, lighting, demo rehearsal | | |
| 21–23 | Demo video shot, uploaded, edited | | |
| 23–24 | Submission, slack | | |

**Integration checkpoint at H14 is non-negotiable.** If the loop isn't running end-to-end by then, we cut scope (drop detection, drop marker-based alignment in favor of manual nudge).

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| WebXR passthrough doesn't work on our Quest firmware | Medium | 60-min timebox; fall back to Unity + Meta XR SDK |
| Coordinate frame alignment looks visibly wrong on stage | High | Build manual nudge controls (operator aligns once before demo) as fallback to ArUco |
| WebSocket bandwidth saturates Wi-Fi | Medium | Voxel-downsample aggressively; send deltas, not full clouds; bring our own router |
| SLAM drifts during the demo | Medium | Keep demo to <60s of capture; ArUco re-anchor every N frames |
| Demo hardware dies | Low but catastrophic | Rerun visualizer is the backup demo — looks great on its own |

## Repository Layout

```
Saurons_eye/
├── README.md                  ← you are here
├── capture/                   ← runs on laptop tethered to RealSense
│   ├── realsense_stream.py    ← librealsense → frames
│   ├── slam.py                ← Open3D RGB-D odometry + TSDF
│   ├── detect.py              ← YOLOv8 + 3D projection
│   ├── server.py              ← WebSocket server, msgpack encoder
│   └── rerun_viz.py           ← Rerun logging
├── headset/                   ← served to Quest browser
│   ├── index.html
│   ├── main.js                ← Three.js scene, WebXR, passthrough
│   ├── ws_client.js           ← WebSocket consumer
│   └── aruco.js               ← marker detection (fallback: manual align)
├── shared/
│   ├── protocol.md            ← wire format spec
│   └── frames.md              ← coordinate frame conventions
├── demo/
│   ├── partition.md           ← how to build the cardboard wall
│   └── checklist.md           ← pre-demo run-through
└── scripts/
    └── setup.sh               ← installs deps on a fresh Ubuntu/macOS box
```

## Wire Protocol (V1)

WebSocket on port 8765. Server (laptop) pushes, client (Quest) consumes. Messages are msgpack:

```
{
  "t": <float, seconds since epoch>,
  "type": "points" | "detections" | "pose",
  "frame": "world",
  "data": <see below>
}
```

- `points`: `{ "xyz": [[x,y,z],...], "rgb": [[r,g,b],...] }` — voxel-downsampled to 5cm
- `detections`: `[{ "class": "person", "conf": 0.87, "bbox3d": [[xmin,ymin,zmin],[xmax,ymax,zmax]] }, ...]`
- `pose`: current camera pose in world frame, for debug overlay

## Setup

Tested on Ubuntu 22.04 + Python 3.10. macOS works for everything except Quest deployment.

```bash
git clone <repo>
cd Saurons_eye
./scripts/setup.sh      # installs librealsense, open3d, ultralytics, rerun, ws server deps
```

Capture side:
```bash
cd capture
python server.py        # starts WebSocket on :8765 and Rerun viewer
```

Quest side:
```bash
cd headset
python -m http.server 8000   # serve the page
# Then on Quest: open browser → http://<laptop-ip>:8000
```

## Demo Script (90 seconds)

1. **0:00–0:10** — Establishing shot: operator stacks on wall, drone visible in their kit.
2. **0:10–0:25** — Drone (RealSense) deployed through doorway. Quest user sees nothing through wall yet.
3. **0:25–0:50** — Drone scans interior. Point cloud blooms in real time on Rerun (cutaway shot). Quest user, still outside, sees the room geometry materialize through the wall, locked in place.
4. **0:50–1:10** — A person stands behind the door. YOLO detects them. Red bounding outline appears in the Quest view, highlighting their position through the wall.
5. **1:10–1:25** — Quest user walks along the wall. Overlay stays locked to the real world. Parallax sells it.
6. **1:25–1:30** — Tagline.

## Pitch Opener

> In Tolkien, the Eye of Sauron saw everything, everywhere. We built one. For real. For soldiers. In 24 hours.

## License

MIT. Source must be public per hackathon rules.

## Team

[Names + roles to fill in]

## Acknowledgements

Army xTech, Cerebral Valley, Shack15, and the partner sponsors (Palantir, OpenAI, Danti). Built on Open3D, Rerun, Three.js, Ultralytics, and Intel RealSense.