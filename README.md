# Sauron Drone — Hackathon Hover Project

Voice-triggered autonomous hover using a Pixhawk 6C (PX4), SiK 915 MHz telemetry radio, and Google Speech-to-Text.

Say **"Sauron Deploy"** → drone takes off, hovers at 1.5 m for 10 s, lands.  
Say **"Sauron Land"** → aborts the hover and lands immediately.

---

## Hardware

| Component | Details |
|-----------|---------|
| Flight controller | Pixhawk 6C — PX4 firmware |
| GPS / compass | M8N |
| Telemetry (laptop link) | SiK 915 MHz on **TELEM1** — 57600 baud |
| RC receiver | RadioMaster RP3 on **TELEM2**, bound to TX16S at 250 Hz |

---

## Prerequisites

### System packages

**Linux (Ubuntu / Debian)**
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev
```

**macOS**
```bash
brew install portaudio
```

**Windows**  
No system package needed — use `pipwin` to install a pre-built pyaudio wheel (see below).

### Python

Python 3.10+ required.  The project ships with a `venv/` directory; activate it before running anything.

```bash
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows PowerShell
```

---

## Installation

```bash
# Activate venv first (see above), then:
pip install -r requirements.txt
```

**Windows pyaudio alternative** (skip the portaudio step):
```bash
pip install pipwin
pipwin install pyaudio
pip install SpeechRecognition pymavlink pyserial
```

---

## Finding the Correct Serial Port

The SiK radio USB dongle enumerates as a USB-serial adapter.

### Linux

```bash
ls /dev/ttyUSB*   # most common — CP210x or FTDI chip
ls /dev/ttyACM*   # less common — ACM device
```

Typical result: `/dev/ttyUSB0`

If you see `Permission denied` when running the script:
```bash
sudo usermod -aG dialout $USER   # add yourself to the dialout group
# then log out and back in
```

To confirm it's the right port, watch for MAVLink traffic:
```bash
python -c "
from pymavlink import mavutil
m = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
m.wait_heartbeat()
print('Got heartbeat!')
"
```

### macOS

```bash
ls /dev/cu.usbserial-*    # USB-serial adapters (SiK usually appears here)
ls /dev/cu.usbmodem*      # ACM devices
```

Typical result: `/dev/cu.usbserial-0001` (the suffix varies by cable)

### Windows

1. Open **Device Manager → Ports (COM & LPT)**
2. Plug in the SiK dongle — a new `COMx` entry appears
3. Note that port number (e.g. `COM4`) and pass it with `--port COM4`

```powershell
python src\hover.py --port COM4
```

---

## Running the Scripts

### Direct hover (no voice)

```bash
# Auto-detect serial port
python src/hover.py

# Specify port explicitly
python src/hover.py --port /dev/ttyUSB0 --baud 57600
```

What happens:
1. Connects to Pixhawk via SiK radio
2. Waits for GPS fix
3. Pre-streams position setpoints at ground level
4. Switches to OFFBOARD mode
5. Arms motors
6. Climbs to 1.5 m
7. Holds hover for 10 seconds
8. Switches to AUTO.LAND, waits for touchdown
9. Disarms

Press **Ctrl+C** at any point to trigger an immediate emergency land.

### Voice-triggered hover

```bash
# Auto-detect serial port
python src/speech_hover.py

# Specify port explicitly
python src/speech_hover.py --port /dev/ttyUSB0
```

The script calibrates for ambient noise for 1 second, then listens continuously.

| Say | Effect |
|-----|--------|
| **"Sauron Deploy"** | Starts full hover mission |
| **"Sauron Land"** | Aborts active hover, lands immediately |
| Anything else | Printed as unrecognized; ignored |

Requires an internet connection — speech is transcribed by Google STT (free tier).  
If your mic is quiet, try speaking clearly from ~30 cm away.

---

## Tuning

Key constants near the top of `src/hover.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `HOVER_ALTITUDE` | `1.5` m | Target hover height |
| `HOVER_DURATION` | `10` s | How long to hold position |
| `SETPOINT_RATE_HZ` | `20` Hz | Setpoint stream rate (PX4 needs >2 Hz) |
| `ALT_TOLERANCE_M` | `0.15` m | "Close enough" to target altitude |
| `BAUD_RATE` | `57600` | Must match SiK radio configuration |

---

## Troubleshooting

**"OFFBOARD mode rejected"**  
Open QGroundControl and check the pre-arm status panel. Common causes:
- GPS not locked (wait outdoors with clear sky view)
- EKF not converged (wait ~30 s after boot)
- RC not calibrated (TX16S + RP3 must be bound and calibrated)

**"Arm rejected"**  
Same as above — pre-arm checks must all pass before PX4 allows arming in OFFBOARD.

**pyaudio build error (portaudio.h not found)**  
```bash
sudo apt-get install -y portaudio19-dev   # Linux
brew install portaudio                    # macOS
```

**Google STT not recognizing speech**  
- Check internet connection (STT requires it)
- Run in a quieter environment or adjust `recognizer.energy_threshold` in `speech_hover.py`
- The free tier has a daily request limit; for heavy use consider a local STT engine

**Serial port permission denied (Linux)**  
```bash
sudo usermod -aG dialout $USER
# log out and back in
```
