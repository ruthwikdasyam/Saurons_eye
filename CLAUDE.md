# Sauron's Eye — Working Instructions for Claude

A multi-agent reconnaissance system: drone-mounted RGB-D sensors stream into a shared world model so a dismounted soldier sees the reconstructed geometry overlaid in their AR headset, locked to the real world. See [README.md](README.md) for architecture, current scope, and Updates log. See [shared/protocol.md](shared/protocol.md) and [shared/frames.md](shared/frames.md) for the wire format and coordinate-frame contracts.

## How to collaborate

**Whenever you are unclear on things, ask me. Have alternative approaches — present them and ask which one to choose. We will keep choosing between approaches/concepts based on tradeoffs.**

This is the operating mode for the project. Don't decide architecture, library choices, file structure, or scope unilaterally when there's real tradeoff space. Lay out 2–3 options with their pros/cons and let me pick. Skip this only for trivial mechanical work with one sensible path — and when you do skip, name what you picked so it's easy to override.

## What we optimise for

**Accuracy and latency are the two axes that matter.** Map drift, ghost surfaces, pose error, end-to-end pipeline ms, per-stage ms — these are how we evaluate every option. Don't frame tradeoffs in terms of how long something takes to build, deadlines, or "ship by." If something takes long to implement, just say so plainly; the call is the user's, made on accuracy + latency grounds.

## Latency

The product is real-time AR. Every architectural choice — library selection, data structure, algorithm, what to log, update cadence, batch size — should weigh latency cost.

- When presenting options, include the per-stage latency (e.g. "X ms per frame, Y ms end-to-end").
- When something runs slow, treat it as a bug to diagnose with profiling, not a setting to accept.
- Don't guess — **measure**. Real per-stage measurements beat estimates. (We learned this hard: RGB-D odometry was estimated at 70–100 ms, measured at 270–350 ms — 3× off.)
- For new pipeline code, plan timing instrumentation in from the start so we can identify bottlenecks immediately.

## Accuracy

Pose drift, ghost surfaces, double-walls, hallucinated motion — these compound and they're what destroy the AR overlay illusion. When evaluating mapping/SLAM choices:

- Quantify drift in cm/sec or cm-over-N-seconds when possible.
- Watch for "voxel count exploding" as a leading indicator of pose drift creating duplicate geometry.
- Consider whether an option introduces or removes a class of error, not just whether it makes the per-frame number prettier.

## Specs are contracts

If you change the wire format, update [shared/protocol.md](shared/protocol.md) in the same change. If you change a coordinate frame convention, update [shared/frames.md](shared/frames.md). Treat these files as authoritative — if code drifts from spec, fix the code or update the spec, don't let them silently diverge.

## Updates log

When something material happens (decision, scope change, integration milestone reached, blocker discovered) add a one-line entry to [README.md → Updates](README.md#updates) with today's date. Newest first. Don't log it until it's actually true.

## Code style for this project

- Match what's in [capture/realsense_check.py](capture/realsense_check.py) — that's the model.
- Type hints on function signatures.
- Module-level docstring at the top with usage. Skip docstrings on obvious functions.
- Comments only when the WHY is non-obvious (e.g. the USB 2.x fallback).
- Python 3.10+, target Ubuntu 22.04. macOS works for everything except Quest deployment.
- Dependencies go in [requirements.txt](requirements.txt).
