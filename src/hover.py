#!/usr/bin/env python3
"""
PX4 hover script via SiK 915MHz telemetry radio (MAVLink over serial).

Flight sequence:
  connect → pre-stream setpoints → OFFBOARD mode → arm →
  climb to 1.5 m → hold 10 s → AUTO.LAND → disarm

PX4 OFFBOARD mode requires setpoints to arrive at >2 Hz BEFORE the mode
switch; this script starts streaming at the current ground position first,
then switches mode, then arms, then commands the climb.
"""

import glob
import signal
import sys
import threading
import time

from pymavlink import mavutil

# ---------------------------------------------------------------------------
# PX4 mode encoding
# custom_mode is a 32-bit field: bits 23-16 = main_mode, bits 31-24 = sub_mode
# ---------------------------------------------------------------------------
_MAIN_MODE_AUTO      = 4
_MAIN_MODE_OFFBOARD  = 6
_SUB_MODE_AUTO_LAND  = 6

PX4_MODE_OFFBOARD = _MAIN_MODE_OFFBOARD << 16                        # 0x00060000
PX4_MODE_AUTO_LAND = (_SUB_MODE_AUTO_LAND << 24) | (_MAIN_MODE_AUTO << 16)  # 0x06040000

# ---------------------------------------------------------------------------
# SET_POSITION_TARGET_LOCAL_NED type_mask
# bit=1 → IGNORE that field; bit=0 → USE that field
#
# We want position + yaw control:
#   use   x,y,z (bits 0-2 = 0)
#   ignore vx,vy,vz (bits 3-5 = 1)
#   ignore ax,ay,az (bits 6-8 = 1)
#   use   yaw angle (bit 10 = 0)
#   ignore yaw rate (bit 11 = 1)
# → 0b100111111000 = 0x09F8 = 2552
# ---------------------------------------------------------------------------
_TYPE_MASK_POS_YAW = 0x09F8

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
BAUD_RATE        = 57600
HOVER_ALTITUDE   = 1.5    # metres above takeoff point
HOVER_DURATION   = 10     # seconds to hold hover
SETPOINT_RATE_HZ = 20     # Hz; PX4 exits OFFBOARD if gap > ~500 ms
ALT_TOLERANCE_M  = 0.15   # metres; "close enough" to target altitude
ARM_TIMEOUT_S    = 10
MODE_TIMEOUT_S   = 10
CLIMB_TIMEOUT_S  = 25
LAND_TIMEOUT_S   = 40


# ---------------------------------------------------------------------------
# Serial port auto-detection
# ---------------------------------------------------------------------------

def find_serial_port() -> str:
    """Return the first plausible SiK radio port, raising if none found."""
    if sys.platform == "win32":
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        # Prefer a port whose description mentions SiK or FTDI
        for p in ports:
            if any(k in (p.description or "") for k in ("SiK", "FTDI", "USB Serial")):
                return p.device
        if ports:
            return ports[0].device
        raise RuntimeError(
            "No COM port found. Plug in the SiK radio USB dongle and retry, "
            "or pass --port COM3 (adjust number as needed)."
        )

    candidates = (
        glob.glob("/dev/ttyUSB*")        # Linux: CP210x / FTDI (most SiK radios)
        + glob.glob("/dev/ttyACM*")      # Linux: ACM
        + glob.glob("/dev/cu.usbserial-*")  # macOS: USB-serial adapters
        + glob.glob("/dev/cu.usbmodem*")    # macOS: ACM
    )
    if candidates:
        return candidates[0]
    raise RuntimeError(
        "No serial port found. Plug in the SiK radio and retry, "
        "or pass --port /dev/ttyUSB0 (adjust as needed).\n"
        "Hint: ls /dev/ttyUSB* /dev/ttyACM*"
    )


# ---------------------------------------------------------------------------
# DroneController
# ---------------------------------------------------------------------------

class DroneController:
    """
    Thin MAVLink wrapper around a PX4 flight controller.
    One background thread streams position setpoints; the main thread
    handles all inbound MAVLink reads so there are no receive-side races.
    """

    def __init__(self, port: str | None = None, baud: int = BAUD_RATE):
        self.port = port or find_serial_port()
        self.baud = baud
        self.master: mavutil.mavfile | None = None

        # Shared setpoint protected by a lock; updated by the main thread,
        # read by the background sender thread.
        self._setpoint = [0.0, 0.0, 0.0, 0.0]  # [x, y, z, yaw] NED metres/rad
        self._setpoint_lock = threading.Lock()
        self._stop_setpoints = threading.Event()
        self._setpoint_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        print(f"Connecting to {self.port} @ {self.baud} baud …")
        # mavlink_connection opens the serial port and auto-detects MAVLink dialect.
        # source_system=255 identifies us as a GCS.
        self.master = mavutil.mavlink_connection(
            self.port,
            baud=self.baud,
            source_system=255,
            source_component=190,
        )

        # HEARTBEAT is the MAVLink "I'm alive" beacon; PX4 emits it at 1 Hz.
        # We must receive one before sending any commands.
        print("Waiting for HEARTBEAT …")
        self.master.wait_heartbeat(timeout=30)
        print(
            f"  sysid={self.master.target_system} "
            f"compid={self.master.target_component}"
        )

        # Ask PX4 to push position data to us at useful rates.
        # REQUEST_DATA_STREAM is legacy but PX4 still honours it.
        self._request_data_streams()

        # Grab the first GPS fix so we have a reference altitude.
        self._wait_for_gps()
        print("Connected and GPS ready.")

    def _request_data_streams(self):
        # MAV_DATA_STREAM_POSITION delivers LOCAL_POSITION_NED + GLOBAL_POSITION_INT
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10, 1,  # 10 Hz, enable
        )
        # MAV_DATA_STREAM_EXTENDED_STATUS delivers EXTENDED_SYS_STATE (landed flag)
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
            5, 1,   # 5 Hz, enable
        )

    def _wait_for_gps(self):
        print("Waiting for GPS fix …")
        msg = self.master.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=30
        )
        if msg is None:
            raise RuntimeError("GPS fix timeout — is the M8N antenna connected?")
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.alt / 1e3
        print(f"  GPS: {lat:.6f}, {lon:.6f}  alt={alt:.1f} m MSL")

    # ------------------------------------------------------------------
    # Mode and arming commands
    # ------------------------------------------------------------------

    def set_mode(self, custom_mode: int, label: str = "") -> bool:
        """
        Switch PX4 flight mode using MAV_CMD_DO_SET_MODE.

        PX4 encodes its modes in the custom_mode field of SET_MODE:
          bits 23-16 = main flight mode (e.g. 6 = OFFBOARD)
          bits 31-24 = sub-mode within AUTO (e.g. 6 = AUTO.LAND)
        MAV_MODE_FLAG_CUSTOM_MODE_ENABLED tells PX4 to read custom_mode.
        """
        print(f"Setting mode: {label} (0x{custom_mode:08X}) …")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,  # confirmation byte (0 = first attempt)
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            float(custom_mode),
            0, 0, 0, 0, 0,
        )
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE, MODE_TIMEOUT_S)

    def arm(self) -> bool:
        """
        Arm motors via MAV_CMD_COMPONENT_ARM_DISARM.
        param1=1 → arm; param1=0 → disarm.
        PX4 will reject arming if pre-arm checks fail (GPS health, EKF, etc.).
        """
        print("Arming …")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,   # confirmation
            1,   # param1=1 → arm
            0, 0, 0, 0, 0, 0,
        )
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, ARM_TIMEOUT_S)

    def disarm(self) -> bool:
        """
        Disarm motors. param2=21196 is PX4's force-disarm magic number,
        which bypasses the in-air safety check (safe to use post-landing).
        """
        print("Disarming …")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,      # confirmation
            0,      # param1=0 → disarm
            21196,  # param2=21196 → force-disarm (PX4-specific)
            0, 0, 0, 0, 0,
        )
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, ARM_TIMEOUT_S)

    def _wait_for_ack(self, command: int, timeout: float) -> bool:
        """
        Block until COMMAND_ACK for `command` arrives or timeout expires.
        COMMAND_ACK.result == MAV_RESULT_ACCEPTED (0) means success.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1.0)
            if msg is None:
                continue
            if msg.command != command:
                continue  # ACK for a different command; discard
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"  ACK: accepted")
                return True
            # Look up the human-readable result name from the MAVLink enum table
            result_name = mavutil.mavlink.enums["MAV_RESULT"].get(msg.result, msg.result)
            print(f"  ACK: REJECTED — {result_name}")
            return False
        print(f"  ACK timeout for command {command}")
        return False

    # ------------------------------------------------------------------
    # Position setpoint streaming (background thread)
    # ------------------------------------------------------------------

    def start_setpoint_stream(self, x: float, y: float, z: float, yaw: float = 0.0):
        """
        Start (or update) the background thread that sends
        SET_POSITION_TARGET_LOCAL_NED at SETPOINT_RATE_HZ.

        PX4 requires continuous setpoints while in OFFBOARD mode;
        if they stop for ~500 ms PX4 exits OFFBOARD automatically.
        """
        with self._setpoint_lock:
            self._setpoint[:] = [x, y, z, yaw]

        if self._setpoint_thread and self._setpoint_thread.is_alive():
            return  # Already running; the lock update above is enough

        self._stop_setpoints.clear()
        self._setpoint_thread = threading.Thread(
            target=self._setpoint_loop, daemon=True
        )
        self._setpoint_thread.start()

    def update_setpoint(self, x: float, y: float, z: float, yaw: float | None = None):
        """Thread-safe in-place update of the streamed setpoint."""
        with self._setpoint_lock:
            self._setpoint[0] = x
            self._setpoint[1] = y
            self._setpoint[2] = z
            if yaw is not None:
                self._setpoint[3] = yaw

    def stop_setpoint_stream(self):
        self._stop_setpoints.set()
        if self._setpoint_thread:
            self._setpoint_thread.join(timeout=2.0)

    def _setpoint_loop(self):
        interval = 1.0 / SETPOINT_RATE_HZ
        while not self._stop_setpoints.is_set():
            with self._setpoint_lock:
                x, y, z, yaw = self._setpoint
            self._send_ned_setpoint(x, y, z, yaw)
            time.sleep(interval)

    def _send_ned_setpoint(self, x: float, y: float, z: float, yaw: float):
        """
        SET_POSITION_TARGET_LOCAL_NED — tell PX4 where to fly in NED frame.

        NED convention: x=North, y=East, z=Down.
        z is NEGATIVE for altitude above ground (z=-1.5 → 1.5 m up).

        type_mask=0x09F8: use position + yaw angle; ignore velocity,
        acceleration, and yaw rate.  See module header for bit breakdown.
        """
        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFF_FFFF,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            _TYPE_MASK_POS_YAW,
            x, y, z,       # target position (m)
            0.0, 0.0, 0.0, # velocity ignored
            0.0, 0.0, 0.0, # acceleration ignored
            yaw,            # yaw angle (rad, 0 = North)
            0.0,            # yaw rate ignored
        )

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def get_local_position(self) -> tuple[float, float, float] | None:
        """
        Read a LOCAL_POSITION_NED message and return (x, y, z) in metres.
        Returns None on timeout.
        """
        msg = self.master.recv_match(
            type="LOCAL_POSITION_NED", blocking=True, timeout=5.0
        )
        return (msg.x, msg.y, msg.z) if msg else None

    def get_attitude_yaw(self) -> float:
        """Return current yaw in radians from the ATTITUDE message."""
        msg = self.master.recv_match(type="ATTITUDE", blocking=True, timeout=5.0)
        return msg.yaw if msg else 0.0

    def wait_for_altitude(
        self,
        target_z_ned: float,
        tolerance: float = ALT_TOLERANCE_M,
        timeout: float = CLIMB_TIMEOUT_S,
    ) -> bool:
        """
        Poll LOCAL_POSITION_NED until |current_z - target_z_ned| < tolerance.
        target_z_ned is negative (e.g. -1.5 for 1.5 m altitude in NED).
        """
        target_alt = -target_z_ned  # convert NED-z to altitude for display
        print(f"Climbing to {target_alt:.1f} m …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            pos = self.get_local_position()
            if pos is None:
                continue
            current_alt = -pos[2]
            err = abs(pos[2] - target_z_ned)
            print(f"  alt={current_alt:.2f} m  (target={target_alt:.1f} m, err={err:.2f} m)", end="\r")
            if err < tolerance:
                print(f"\n  Reached {current_alt:.2f} m")
                return True
        print(f"\n  Altitude timeout — proceeded with partial climb")
        return False

    # ------------------------------------------------------------------
    # Landing
    # ------------------------------------------------------------------

    def land(self):
        """
        Stop setpoint streaming and switch to PX4 AUTO.LAND mode.
        AUTO.LAND handles descent rate, flare, and motor cut autonomously.
        Falls back to MAV_CMD_NAV_LAND if the mode switch is rejected.
        """
        print("Landing …")
        self.stop_setpoint_stream()

        ok = self.set_mode(PX4_MODE_AUTO_LAND, "AUTO.LAND")
        if not ok:
            # Fallback: explicit land command (param lat/lon NaN → current position)
            print("  Mode switch failed; using MAV_CMD_NAV_LAND fallback …")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0,              # abort altitude
                0,              # land mode (0=normal)
                float("nan"),   # latitude (NaN → current)
                float("nan"),   # longitude
                0,              # altitude
                float("nan"),   # yaw (NaN → current)
            )

        self._wait_for_landed()

    def _wait_for_landed(self, timeout: float = LAND_TIMEOUT_S):
        """
        Poll EXTENDED_SYS_STATE.landed_state until MAV_LANDED_STATE_ON_GROUND.
        EXTENDED_SYS_STATE is a PX4/MAVLink2 message that reports whether the
        vehicle believes it is on the ground, airborne, or in transition.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(
                type="EXTENDED_SYS_STATE", blocking=True, timeout=2.0
            )
            if msg and msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND:
                print("  Landed (EXTENDED_SYS_STATE confirmed).")
                return
            # Fallback: check altitude below threshold
            pos = self.get_local_position()
            if pos and abs(pos[2]) < 0.2:
                print("  Landed (altitude threshold).")
                return
        print("  Land timeout — assuming on ground.")

    def emergency_land(self):
        """Interrupt any active flight and land immediately."""
        print("\n[EMERGENCY] Landing now …")
        try:
            self.land()
            self.disarm()
        except Exception as exc:
            print(f"  Emergency land error: {exc}")
        finally:
            self.stop_setpoint_stream()

    def close(self):
        self.stop_setpoint_stream()
        if self.master:
            self.master.close()


# ---------------------------------------------------------------------------
# Top-level hover routine (importable by speech_hover.py)
# ---------------------------------------------------------------------------

def hover(
    port: str | None = None,
    baud: int = BAUD_RATE,
    stop_event: threading.Event | None = None,
) -> bool:
    """
    Execute a full autonomous hover mission:
      1. Connect via SiK radio
      2. Pre-stream ground setpoints (mandatory before OFFBOARD mode switch)
      3. Switch to OFFBOARD mode
      4. Arm
      5. Climb to HOVER_ALTITUDE metres
      6. Hold for HOVER_DURATION seconds (or until stop_event is set)
      7. Land + disarm

    Returns True on clean completion, False on any error.
    stop_event: if set externally (e.g. "Sauron Land" voice command), the
                hover hold phase exits early and landing begins.
    """
    drone = DroneController(port=port, baud=baud)

    def _sighandler(sig, frame):
        print("\n[SIGINT] Keyboard interrupt — landing …")
        drone.emergency_land()
        drone.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    try:
        drone.connect()

        # ---- Read initial position and heading ----
        pos = drone.get_local_position()
        if pos is None:
            raise RuntimeError("LOCAL_POSITION_NED not available")
        home_x, home_y, home_z = pos
        home_yaw = drone.get_attitude_yaw()

        # ---- Step 1: Pre-stream setpoints at ground position ----
        # PX4 will refuse to enter OFFBOARD mode unless setpoints are already
        # arriving.  We stream the current position (no movement commanded).
        print("Pre-streaming ground setpoints …")
        drone.start_setpoint_stream(home_x, home_y, home_z, home_yaw)
        time.sleep(1.0)  # give PX4 ~20 setpoints before the mode switch

        # ---- Step 2: Switch to OFFBOARD mode ----
        # MAV_CMD_DO_SET_MODE with custom_mode = (6 << 16) = OFFBOARD
        if not drone.set_mode(PX4_MODE_OFFBOARD, "OFFBOARD"):
            raise RuntimeError("OFFBOARD mode rejected — check pre-arm status in QGC")

        # ---- Step 3: Arm motors ----
        # PX4 arms if: pre-arm checks pass AND mode is armable (OFFBOARD qualifies)
        if not drone.arm():
            raise RuntimeError("Arm rejected — check pre-arm warnings in QGC")

        # ---- Step 4: Command climb to HOVER_ALTITUDE ----
        # NED z is negative upward: target_z = 0 - 1.5 = -1.5 m
        target_z = home_z - HOVER_ALTITUDE
        drone.update_setpoint(home_x, home_y, target_z, home_yaw)
        drone.wait_for_altitude(target_z)

        # ---- Step 5: Hold hover ----
        print(f"Hovering at {HOVER_ALTITUDE} m for {HOVER_DURATION} s …")
        elapsed = 0.0
        while elapsed < HOVER_DURATION:
            if stop_event and stop_event.is_set():
                print("Stop event received — ending hover early.")
                break
            time.sleep(0.25)
            elapsed += 0.25

        # ---- Step 6: Land + disarm ----
        drone.land()
        drone.disarm()
        print("Mission complete.")
        return True

    except Exception as exc:
        print(f"[ERROR] {exc}")
        drone.emergency_land()
        return False
    finally:
        drone.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PX4 hover via SiK 915 MHz telemetry")
    ap.add_argument("--port", default=None, help="Serial port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=BAUD_RATE, help=f"Baud rate (default {BAUD_RATE})")
    args = ap.parse_args()

    success = hover(port=args.port, baud=args.baud)
    sys.exit(0 if success else 1)
