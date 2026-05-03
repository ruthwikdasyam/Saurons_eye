"""
Pixhawk pose reader over MAVLink (SiK radio link).

Subscribes to a ground-station-side MAVLink stream (direct serial to a SiK
USB radio, or UDP from a MAVProxy router) and surfaces drone pose at a
useful rate. Live pose is exposed two ways:

  - As a CLI: prints LOCAL_POSITION_NED + ATTITUDE + GPS at ~10 Hz so a
    human can sanity-check link health.
  - As a library: `DronePoseReader.run()` calls a user-supplied callback
    on every fresh `DronePoseSample`. This is the seam that downstream
    code (wire protocol, AR overlay) plugs into.

Frame note (READ THIS):
    LOCAL_POSITION_NED is in the **autopilot's** NED frame, with origin
    at the EKF / GPS-lock origin. This is NOT our world frame
    (shared/frames.md), which is ArUco-marker-anchored, OpenCV
    convention. Converting NED -> world requires a calibrated T_W_NED
    (e.g. fly the drone over the marker once, capture pose) and a
    NED -> RHS-Y-up axis swap. This module deliberately does no
    conversion. Consumers receive raw NED + autopilot quaternion.

Transports:
    --connection /dev/ttyUSB0   serial, direct USB SiK radio
    --connection udpin:127.0.0.1:14551
                                UDP, when MAVProxy is routing to QGC + us

Usage:
    # Direct serial (only one program can hold the radio)
    python -m drone.pose_reader --connection /dev/ttyUSB0 --baud 57600

    # Via MAVProxy (run QGC alongside)
    mavproxy.py --master=/dev/ttyUSB0 --baudrate=57600 \\
        --out=127.0.0.1:14550 --out=127.0.0.1:14551
    python -m drone.pose_reader --connection udpin:127.0.0.1:14551
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pymavlink import mavutil


# Per-message subscription rates (Hz). SiK telemetry on a stock 57600 bps
# link tops out around ~50–80 messages/sec total; budget conservatively.
DEFAULT_RATES_HZ: dict[str, int] = {
    "LOCAL_POSITION_NED": 30,
    "ATTITUDE": 30,
    "ATTITUDE_QUATERNION": 20,
    "GLOBAL_POSITION_INT": 5,
    "GPS_RAW_INT": 2,
    "ODOMETRY": 10,  # may not be emitted by all firmwares; harmless if absent
}


@dataclass
class DronePoseSample:
    """One snapshot of drone pose state. Fields are None until first seen."""
    # LOCAL_POSITION_NED (metres, m/s; NED frame, EKF origin)
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    pos_t_us: Optional[int] = None  # autopilot boot time of last LOCAL_POSITION_NED

    # ATTITUDE (radians; body wrt NED)
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None

    # ATTITUDE_QUATERNION (Hamilton, MAVLink order: w, x, y, z; body wrt NED)
    qw: Optional[float] = None
    qx: Optional[float] = None
    qy: Optional[float] = None
    qz: Optional[float] = None

    # GLOBAL_POSITION_INT
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_rel_m: Optional[float] = None  # relative to home

    # GPS_RAW_INT
    fix_type: Optional[int] = None     # 0/1=no fix, 2=2D, 3=3D, 4=DGPS, 5=RTK float, 6=RTK fixed
    sats: Optional[int] = None

    # Receiver clock (seconds since UNIX epoch) of last update of any field
    t_recv: float = field(default_factory=time.time)


class DronePoseReader:
    """Pulls MAVLink messages off the link and updates a DronePoseSample.

    Use as a library:

        reader = DronePoseReader("/dev/ttyUSB0", baud=57600)
        reader.connect()
        reader.request_streams()
        for sample in reader.iter_samples():
            ...  # consume
    """

    def __init__(
        self,
        connection: str,
        baud: int = 57600,
        rates_hz: Optional[dict[str, int]] = None,
    ) -> None:
        self.connection = connection
        self.baud = baud
        self.rates_hz = rates_hz or dict(DEFAULT_RATES_HZ)
        self.master: Optional[mavutil.mavfile] = None
        self.sample = DronePoseSample()
        self._autopilot_sysid: Optional[int] = None
        self._autopilot_compid: Optional[int] = None

    def connect(self, heartbeat_timeout_s: float = 10.0) -> None:
        """Open the link and wait for the first heartbeat from an autopilot."""
        if self.connection.startswith("/dev/") or self.connection.startswith("COM"):
            self.master = mavutil.mavlink_connection(self.connection, baud=self.baud)
        else:
            self.master = mavutil.mavlink_connection(self.connection)

        print(f"[drone] connecting via {self.connection}...", flush=True)

        # Hold out for an *autopilot* heartbeat (component == AUTOPILOT1 == 1).
        # On a multi-component vehicle (gimbal, companion, secondary FC) the
        # first heartbeat is often not the autopilot, and locking to the wrong
        # source means ATTITUDE flickers between two attitudes (the gimbal's
        # and the autopilot's). We want the FCU only.
        AUTOPILOT_COMP = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1  # = 1
        deadline = time.time() + heartbeat_timeout_s
        autopilot_sysid: Optional[int] = None
        while time.time() < deadline:
            hb = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if hb is None:
                continue
            sysid = hb.get_srcSystem()
            compid = hb.get_srcComponent()
            print(f"[drone]   saw HEARTBEAT from sys={sysid} comp={compid}", flush=True)
            if compid == AUTOPILOT_COMP:
                autopilot_sysid = sysid
                break

        if autopilot_sysid is None:
            raise SystemExit(
                f"[drone] no autopilot heartbeat in {heartbeat_timeout_s}s on {self.connection}.\n"
                "  - Is the SiK radio plugged in and paired?\n"
                "  - Is another program (QGC, MAVProxy) holding the port?\n"
                "  - Try `sudo systemctl stop ModemManager` and re-plug."
            )

        # Pin pymavlink's "target" so command_long_send() is addressed correctly,
        # AND stash the IDs so _absorb() can ignore non-autopilot traffic.
        self.master.target_system = autopilot_sysid
        self.master.target_component = AUTOPILOT_COMP
        self._autopilot_sysid = autopilot_sysid
        self._autopilot_compid = AUTOPILOT_COMP
        print(
            f"[drone] locked to autopilot system={autopilot_sysid} "
            f"component={AUTOPILOT_COMP} (ignoring other components)",
            flush=True,
        )

    def request_streams(self) -> None:
        """Ask the autopilot to push the messages we care about, at our rates."""
        assert self.master is not None
        for name, hz in self.rates_hz.items():
            msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if msg_id is None:
                print(f"[drone] WARN: unknown MAVLink message {name}, skipping")
                continue
            interval_us = int(1_000_000 / hz)
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                interval_us,
                0, 0, 0, 0, 0,
            )
            time.sleep(0.05)

    def _absorb(self, msg) -> bool:
        """Update self.sample from a single MAVLink message. Returns True if pose-bearing."""
        t = msg.get_type()
        s = self.sample

        if t == "LOCAL_POSITION_NED":
            s.x, s.y, s.z = msg.x, msg.y, msg.z
            s.vx, s.vy, s.vz = msg.vx, msg.vy, msg.vz
            s.pos_t_us = msg.time_boot_ms * 1000
        elif t == "ATTITUDE":
            s.roll, s.pitch, s.yaw = msg.roll, msg.pitch, msg.yaw
        elif t == "ATTITUDE_QUATERNION":
            # MAVLink order: q1=w, q2=x, q3=y, q4=z (Hamilton)
            s.qw, s.qx, s.qy, s.qz = msg.q1, msg.q2, msg.q3, msg.q4
        elif t == "GLOBAL_POSITION_INT":
            s.lat_deg = msg.lat / 1e7
            s.lon_deg = msg.lon / 1e7
            s.alt_rel_m = msg.relative_alt / 1000.0
        elif t == "GPS_RAW_INT":
            s.fix_type = msg.fix_type
            s.sats = msg.satellites_visible
        elif t == "ODOMETRY":
            # Some firmwares emit this with full pose+velocity in one shot.
            # ArduPilot stable usually does not. Treat as bonus.
            return True
        else:
            return False

        s.t_recv = time.time()
        return True

    def run(
        self,
        on_sample: Callable[[DronePoseSample], None],
        types: Optional[list[str]] = None,
        debug_traffic: bool = False,
        debug_period_s: float = 2.0,
    ) -> None:
        """Block forever, calling on_sample after every pose-bearing message.

        If debug_traffic=True, prints a periodic breakdown of (sys, comp, type)
        counts so you can see exactly who's emitting what on the link.
        """
        assert self.master is not None
        types = types or list(self.rates_hz.keys())
        # (sys, comp, type) -> count of messages received in this window
        traffic: dict[tuple[int, int, str], int] = {}
        last_dump = time.time()
        while True:
            msg = self.master.recv_match(type=types, blocking=True)
            if msg is None:
                continue
            sys_id = msg.get_srcSystem()
            comp_id = msg.get_srcComponent()
            mtype = msg.get_type()
            if debug_traffic:
                traffic[(sys_id, comp_id, mtype)] = traffic.get((sys_id, comp_id, mtype), 0) + 1
                now = time.time()
                if now - last_dump >= debug_period_s:
                    print()
                    print(f"[drone:traffic] last {debug_period_s:.0f}s:")
                    for (s, c, m), n in sorted(traffic.items()):
                        kept = (s == self._autopilot_sysid and c == self._autopilot_compid)
                        flag = "KEEP" if kept else "drop"
                        print(f"  {flag}  sys={s:<3} comp={c:<3} {m:<24} x{n}")
                    traffic.clear()
                    last_dump = now
            # Drop messages from other components on this same MAVLink link
            # (e.g. a gimbal or companion sending its own ATTITUDE). Without
            # this filter, ATTITUDE prints flicker between two sources.
            if sys_id != self._autopilot_sysid or comp_id != self._autopilot_compid:
                continue
            if self._absorb(msg):
                on_sample(self.sample)


def _format_sample(s: DronePoseSample) -> str:
    def f(v: Optional[float], fmt: str = "{:+.3f}") -> str:
        return fmt.format(v) if v is not None else "  ----"

    roll_deg = math.degrees(s.roll) if s.roll is not None else None
    pitch_deg = math.degrees(s.pitch) if s.pitch is not None else None
    yaw_deg = math.degrees(s.yaw) if s.yaw is not None else None

    return (
        f"NED x={f(s.x)} y={f(s.y)} z={f(s.z)} | "
        f"V {f(s.vx)} {f(s.vy)} {f(s.vz)} | "
        f"RPY° {f(roll_deg, '{:+6.1f}')} {f(pitch_deg, '{:+6.1f}')} {f(yaw_deg, '{:+6.1f}')} | "
        f"GPS fix={s.fix_type} sats={s.sats}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--connection",
        default="/dev/ttyUSB0",
        help="serial device (e.g. /dev/ttyUSB0) or pymavlink URL "
             "(e.g. udpin:127.0.0.1:14551)",
    )
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument(
        "--print-hz",
        type=float,
        default=10.0,
        help="rate-limit stdout printing; the underlying sample rate is independent",
    )
    parser.add_argument(
        "--debug-traffic",
        action="store_true",
        help="periodically print (sys, comp, type) message counts to identify "
             "extra senders on the bus (gimbal, companion, secondary FC).",
    )
    args = parser.parse_args()

    reader = DronePoseReader(args.connection, baud=args.baud)
    reader.connect()
    reader.request_streams()

    print_period = 1.0 / max(args.print_hz, 0.1)
    last_print = 0.0

    def on_sample(s: DronePoseSample) -> None:
        nonlocal last_print
        now = time.time()
        if now - last_print < print_period:
            return
        last_print = now
        print("\r" + _format_sample(s), end="", flush=True)

    print("[drone] streaming pose (Ctrl+C to stop)\n", flush=True)
    try:
        reader.run(on_sample, debug_traffic=args.debug_traffic)
    except KeyboardInterrupt:
        print("\n[drone] stopped.")


if __name__ == "__main__":
    main()
