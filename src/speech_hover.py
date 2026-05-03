#!/usr/bin/env python3
"""
Voice-triggered drone hover controller.

Listens continuously via the default microphone and Google STT (free tier).

  "Sauron Deploy" → connects to drone, runs full hover mission
  "Sauron Land"   → signals the active mission to abort and land immediately

Any other speech → logged as unrecognized and ignored.

Usage:
  python src/speech_hover.py [--port /dev/ttyUSB0] [--baud 57600]
"""

import argparse
import sys
import threading

import speech_recognition as sr

from hover import BAUD_RATE, hover

TRIGGER_DEPLOY = "sauron deploy"
TRIGGER_LAND   = "sauron land"


class VoiceController:
    def __init__(self, port: str | None = None, baud: int = BAUD_RATE):
        self.port = port
        self.baud = baud

        # stop_event is set to interrupt the active hover mission early
        self._stop_event = threading.Event()
        self._mission_thread: threading.Thread | None = None
        self._mission_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mission management
    # ------------------------------------------------------------------

    def _is_flying(self) -> bool:
        with self._mission_lock:
            return self._mission_thread is not None and self._mission_thread.is_alive()

    def deploy(self):
        """Launch the hover mission in a background thread."""
        if self._is_flying():
            print("[VOICE] Hover already in progress — ignoring deploy.")
            return

        print("[VOICE] 'Sauron Deploy' heard — starting hover mission!")
        self._stop_event.clear()

        def _run():
            hover(port=self.port, baud=self.baud, stop_event=self._stop_event)
            print("[VOICE] Mission finished. Listening again …")

        with self._mission_lock:
            self._mission_thread = threading.Thread(target=_run, daemon=True)
            self._mission_thread.start()

    def land_now(self):
        """Signal the active mission to abort and land immediately."""
        if not self._is_flying():
            print("[VOICE] 'Sauron Land' heard — but no active flight.")
            return

        print("[VOICE] 'Sauron Land' heard — commanding immediate land!")
        # Setting stop_event causes hover()'s hold loop to exit, which
        # then falls through to land() + disarm() in hover.py.
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Microphone listener
    # ------------------------------------------------------------------

    def _listen_once(
        self, recognizer: sr.Recognizer, mic: sr.Microphone
    ) -> str | None:
        """
        Record one utterance and return lowercase transcription.
        Returns None if nothing was heard or speech was unintelligible.
        """
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                return None  # silence; loop and try again

        try:
            text = recognizer.recognize_google(audio).lower()
            print(f"[STT] Heard: '{text}'")
            return text
        except sr.UnknownValueError:
            print("[STT] (not recognized)")
            return None
        except sr.RequestError as exc:
            print(f"[STT] Google API error: {exc}")
            return None

    def run(self):
        """Main loop: calibrate mic, then listen and dispatch indefinitely."""
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True  # auto-adjust for ambient noise

        mic = sr.Microphone()

        print("Calibrating microphone for ambient noise (1 s) …")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        print(
            f"Ready.\n"
            f"  Say '{TRIGGER_DEPLOY}' to take off and hover.\n"
            f"  Say '{TRIGGER_LAND}'   to abort and land immediately.\n"
        )

        while True:
            text = self._listen_once(recognizer, mic)
            if text is None:
                continue

            if TRIGGER_DEPLOY in text:
                self.deploy()
            elif TRIGGER_LAND in text:
                self.land_now()
            else:
                print(f"[VOICE] Unrecognized command: '{text}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Voice-triggered PX4 hover (say 'Sauron Deploy' / 'Sauron Land')"
    )
    ap.add_argument("--port", default=None, help="Serial port for SiK radio (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=BAUD_RATE, help=f"Baud rate (default {BAUD_RATE})")
    args = ap.parse_args()

    controller = VoiceController(port=args.port, baud=args.baud)
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[VOICE] Shutting down — landing if airborne …")
        controller.land_now()
        # Give the mission thread a moment to land before the process exits
        if controller._mission_thread:
            controller._mission_thread.join(timeout=45)
        sys.exit(0)


if __name__ == "__main__":
    main()
