# Coordinate Frames — V1

Three coordinate frames matter. Get them wrong and the overlay floats off the wall.

```
   world (W)        ←  the ArUco marker on the partition
       ▲
       │
   camera (C)       ←  the RealSense, moving
   head (H)         ←  the Quest headset, moving
```

All transforms are written `T_AB` meaning "the pose of frame B expressed in frame A", or equivalently "the matrix that takes a point in B coordinates and gives it in A coordinates". `p_A = T_AB · p_B`.

Units: **metres** everywhere. Angles: **radians**.

---

## World frame `W`

- **Origin:** centre of the ArUco marker glued to the cardboard partition.
- **+X:** along the marker's right edge (the marker's own +X).
- **+Y:** along the marker's top edge (the marker's own +Y).
- **+Z:** out of the marker, pointing into the room the operator is about to enter (i.e. away from the wall, toward where the "drone" goes). Right-handed.

This is the OpenCV ArUco convention by default. Don't change it.

The marker is **5 cm × 5 cm**, ArUco dictionary `DICT_4X4_50`, ID `0`. If we run multiple markers later (anti-occlusion), they get IDs 1, 2, 3 with hand-measured `T_W·marker_i` calibrations.

## Camera frame `C` (RealSense D435i)

The frame of the **colour stream** (not the depth stream — Open3D aligns depth into the colour frame on the capture side).

- **+X:** right in the image.
- **+Y:** down in the image.
- **+Z:** forward (out of the lens).

This is the standard OpenCV / pinhole convention. Open3D and OpenCV both use it natively, so no flips on the capture side.

## Head frame `H` (Quest)

WebXR reports head pose in a Y-up, **right-handed** frame, with:

- **+X:** right.
- **+Y:** up.
- **+Z:** **toward the user** (i.e. into the headset wearer's face). This is the WebXR convention.

This is **not** the same as the OpenCV convention. The Quest-side code must flip Y and Z when going between WebXR pose and our world frame. The conversion is:

```
T_W_WebXR = diag(1, -1, -1, 1)   # applied as a similarity, see headset/main.js
```

Apply once when the marker is first observed; cache.

---

## Initialisation

### Capture side (RealSense laptop)

1. On startup, run ArUco detection on the colour frame until marker ID 0 is detected with reprojection error < 1 px.
2. OpenCV `solvePnP` (or `estimatePoseSingleMarkers`) yields `T_CW` — pose of world in camera.
3. Invert: `T_WC = T_CW⁻¹`. This is the initial RealSense pose.
4. From there, Open3D RGB-D odometry maintains `T_WC(t)` incrementally:
   `T_WC(t) = T_WC(t-1) · T_C(t-1)C(t)⁻¹`
5. **Re-anchor every N=30 frames** if the marker is still visible: re-run PnP and trust the marker over the integrated odometry.

### Headset side (Quest)

1. On WebXR session start, ask the user to look at the partition until the marker is detected by the in-browser ArUco detector (`aruco.js`).
2. The detector returns `T_HW_initial` — pose of world in head, at marker-detection time.
3. Combine with the WebXR `XRReferenceSpace` pose to get `T_W_referenceSpace`. Cache it.
4. From there, the Quest's own VIO (`local-floor` reference space) keeps the head pose locked. We never run ArUco again on the Quest in v1.

---

## Drift budget

- Open3D RGB-D odometry: budget **2 cm / minute** of translation drift, **1° / minute** of rotation. Re-anchoring every 30 frames against the marker keeps cumulative error well below this.
- Quest Insight tracking: Meta's published spec is sub-centimetre over a typical room. We treat it as ground-truth and never re-anchor in v1.
- **Drift compounds with session length.** With Open3D's frame-to-frame RGB-D odometry and no re-anchoring, expect visible drift at 1–2 m overlay distance after roughly a minute of continuous motion. Mitigations: ArUco re-anchoring (planned), or replacing the L source with RTAB-Map / MAVLink (see README → Roadmap).

---

## Frame quick-reference

| Symbol       | Means                                 |
|--------------|---------------------------------------|
| `T_WC`       | Pose of camera in world.              |
| `T_CW`       | Pose of world in camera. `= T_WC⁻¹`.  |
| `T_WH`       | Pose of head in world.                |
| `p_W`        | A 3D point expressed in world coords. |
| `p_W = T_WC · p_C` | Transform a point from camera to world. |

When in doubt, write the subscript chain: `T_WH · T_HM · p_M` is unambiguous; "the pose" is not.
