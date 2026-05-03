"""
Subprocess worker for voice capture. Records mic, transcribes, prints
JSON to stdout. Status messages go to stderr.

Run as a child of detections/voice.py. We isolate whisper into this process
so the parent (which runs cv2.imshow + ultralytics) never imports
torch/whisper directly — those imports break cv2's GUI runloop on Mac.

Stdout protocol (single line of JSON):
    {"text": "<query>"}    -- success or "no speech"
    {"text": "", "error": "<msg>"}  -- recoverable failure
"""

from __future__ import annotations

import argparse
import json
import sys


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny.en")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    args = p.parse_args()

    try:
        import numpy as np
        import sounddevice as sd
        import whisper
    except ImportError as e:
        _emit({"text": "", "error": f"missing dep: {e}"})
        sys.exit(1)

    print(f"[mic] listening {args.duration:.1f}s -- speak now...", file=sys.stderr, flush=True)
    try:
        audio = sd.rec(
            int(args.duration * args.sample_rate),
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:
        _emit({"text": "", "error": f"audio capture failed: {e}"})
        sys.exit(2)

    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4:
        print("[mic] silence — nothing to transcribe", file=sys.stderr, flush=True)
        _emit({"text": ""})
        return

    model = whisper.load_model(args.model, device="cpu")
    result = model.transcribe(audio, language="en", fp16=False)
    text = (result.get("text") or "").strip().rstrip(".!?,;:").strip().lower()
    if text:
        print(f"[mic] heard: {text!r}", file=sys.stderr, flush=True)
    else:
        print("[mic] no speech detected", file=sys.stderr, flush=True)
    _emit({"text": text})


if __name__ == "__main__":
    main()
