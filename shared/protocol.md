# Wire Protocol — V1

The contract between the **capture side** (laptop running RealSense + SLAM + detection) and the **headset side** (Quest browser running WebXR). Anything that crosses the WebSocket lives here.

If you change a field, update this file in the same commit.

---

## Transport

- **Protocol:** WebSocket (RFC 6455), no TLS for v1.
- **Default endpoint:** `ws://<laptop-ip>:8765/`
- **Direction:** server (laptop) is the producer. Client (Quest) is the consumer. The client never sends payloads in v1; only its WebSocket-level handshake and pings.
- **Frame type:** binary frames only. Text frames must be ignored by the consumer.
- **Encoding:** [msgpack](https://msgpack.org/) (`msgpack` Python lib on the server, `@msgpack/msgpack` on the client). Use `use_bin_type=True` on the server.
- **Concurrency:** one connection per Quest. The server may hold multiple connections (e.g. Rerun is a separate process — it does **not** share this WebSocket).

## Lifecycle

1. Client opens the WebSocket. Server immediately sends one `pose` and one `points` (full state, not a delta) so the client can render something.
2. Server then streams `points` deltas, `detections`, and `pose` at independent rates (see [Cadence](#cadence)).
3. If the client disconnects, the server keeps running and serves the next connector with a fresh full-state on connect.
4. WebSocket-level pings every 10s. If 30s pass with no pong, the server drops the socket.

## Backpressure

If the client falls behind:

- The server measures `ws.transport.get_write_buffer_size()` (or equivalent).
- If buffered bytes exceed **1 MB**, the server **drops** the next `points` delta rather than queuing it. `pose` and `detections` are never dropped.
- Dropping is silent in v1 (no out-of-band notification). Add a counter in metrics later.

## Message envelope

Every message is a top-level msgpack map with these fields:

| Field   | Type     | Required | Notes |
|---------|----------|----------|-------|
| `t`     | float64  | yes      | Capture timestamp, seconds since UNIX epoch (server clock). |
| `seq`   | uint32   | yes      | Per-`type` sequence number, starts at 0, wraps at 2³². Lets the client detect drops. |
| `type`  | str      | yes      | `"points" \| "detections" \| "pose"`. |
| `frame` | str      | yes      | Always `"world"` in v1. The marker-anchored frame defined in [frames.md](frames.md). |
| `data`  | varies   | yes      | Payload, schema depends on `type`. |

Unknown `type` values must be ignored by the client (forward-compat).

## Payloads

### `points`

Voxel-downsampled point cloud, 5 cm voxels.

```python
{
  "kind": "delta" | "full",       # full only on first message after connect
  "xyz": bytes,                   # float32 little-endian, length = 3*N*4
  "rgb": bytes,                   # uint8, length = 3*N
  "n":   uint32,                  # number of points (sanity check)
  "removed": bytes | None,        # uint32 voxel indices to remove (delta only)
}
```

- `xyz` and `rgb` are packed binary, **not** lists of lists. Lists of lists balloon msgpack overhead ~10x and saturate Wi-Fi.
- Points are in the **world frame** (units: metres). See [frames.md](frames.md).
- A `delta` adds the points in `xyz/rgb` and removes any voxels listed in `removed`. The client maintains the cumulative cloud.
- A `full` is a full snapshot. The client must clear its cumulative cloud before applying it.

### `detections`

```python
{
  "items": [
    {
      "id":     uint32,           # stable across frames while track is alive
      "cls":    str,              # COCO class name, e.g. "person"
      "conf":   float32,          # 0.0–1.0
      "bbox3d": [
        [xmin, ymin, zmin],       # metres, world frame
        [xmax, ymax, zmax],
      ],
    },
    ...
  ]
}
```

- Empty `items` (no detections) is sent at the same cadence as detections — used by the client to clear stale highlights.
- `id` is the YOLO/ByteTrack track id when tracking is on; otherwise `0` and the client must treat each frame independently.

### `pose`

Current RealSense camera pose in the world frame. For debug overlay (showing where the "drone" is), not for transforming points (those are already in world).

```python
{
  "t_wc": [tx, ty, tz],            # camera origin in world, metres
  "q_wc": [qx, qy, qz, qw],        # camera orientation, world←camera, unit quaternion
}
```

Quaternion convention: Hamilton, scalar-last (`xyzw`), unit norm.

## Cadence

| Type         | Target rate | Hard cap |
|--------------|-------------|----------|
| `points`     | 5 Hz        | 10 Hz    |
| `detections` | 5 Hz        | 10 Hz    |
| `pose`       | 30 Hz       | 60 Hz    |

`pose` is small (~50 bytes) and the Quest needs it smooth for the debug glyph. Points are bandwidth-heavy and 5 Hz is enough for the wall-overlay illusion.

## Versioning

If the schema changes incompatibly, bump the WebSocket subprotocol string. v1 is `saurons-eye/1`. v2 would be `saurons-eye/2`. The client picks the highest one it supports during the handshake; the server honours it or rejects.

## Out of scope (v1)

- Authentication / TLS.
- Multi-drone fan-in (multiple capture laptops feeding one Quest).
- Server → server replication.
- Compression beyond voxel downsampling. (zstd was considered; the per-frame CPU cost on the Quest browser is not worth it for v1.)
