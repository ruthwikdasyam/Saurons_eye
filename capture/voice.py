"""
Voice -> text helper for the YOLO-World tracker.

Spawns ``capture/_voice_worker.py`` as a fresh subprocess for each
capture(). The worker records audio via sounddevice and transcribes via
openai-whisper, then exits. We isolate whisper into a child process
because importing torch/whisper into the same process as cv2.imshow on
Mac arm64 corrupts the GUI runloop — the camera opens but no imshow
window ever appears. Subprocess isolation sidesteps it entirely.

Per-call cost is ~1.5 s extra for the whisper model load (vs an
in-process implementation). Worth it; it's the only Mac-stable path.

Usage:
    listener = VoiceListener(model_size="tiny.en", duration_s=3.0)
    query = listener.capture()
    if query:
        yolo_world.set_classes([query])

Whisper weights download to ~/.cache/whisper on first call (~75 MB for tiny.en).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "_voice_worker.py"


class VoiceListener:
    def __init__(
        self,
        model_size: str = "tiny.en",
        duration_s: float = 3.0,
        sample_rate: int = 16000,
    ) -> None:
        self._model_size = model_size
        self._duration = float(duration_s)
        self._sr = int(sample_rate)

    def capture(self, duration_s: float | None = None) -> str:
        """Spawn the worker; record + transcribe; return cleaned query string.

        Returns empty string on silence / failure / dep-missing. Caller
        should treat empty as "keep the previous query."
        """
        dur = float(duration_s) if duration_s is not None else self._duration
        cmd = [
            sys.executable, str(WORKER),
            "--model", self._model_size,
            "--duration", str(dur),
            "--sample-rate", str(self._sr),
        ]
        try:
            # First call may take ~10 s if it has to download the model;
            # give a generous timeout. Subsequent calls land in ~5–6 s.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=dur + 60,
            )
        except subprocess.TimeoutExpired:
            print("[mic] voice worker timed out", file=sys.stderr)
            return ""
        except FileNotFoundError as e:
            print(f"[mic] could not spawn worker: {e}", file=sys.stderr)
            return ""

        # Worker prints status updates to stderr in real time-ish; relay them.
        if result.stderr:
            sys.stderr.write(result.stderr)

        # Result is the last JSON-looking line on stdout.
        payload: dict = {}
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if "error" in payload:
            print(f"[mic] worker error: {payload['error']}", file=sys.stderr)
        return payload.get("text", "") or ""
