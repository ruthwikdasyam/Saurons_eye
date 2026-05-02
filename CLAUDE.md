# Sauron's Eye — Working Instructions for Claude

24-hour hackathon project. See [README.md](README.md) for the pitch, architecture, build plan, and current Updates log. See [shared/protocol.md](shared/protocol.md) and [shared/frames.md](shared/frames.md) for the wire format and coordinate-frame contracts.

## How to collaborate

**Whenever you are unclear on things, ask me. Have alternative approaches — present them and ask which one to choose. We will keep choosing between approaches/concepts based on tradeoffs.**

This is the operating mode for the project. Don't decide architecture, library choices, file structure, or scope unilaterally when there's real tradeoff space. Lay out 2–3 options with their pros/cons and let me pick. Skip this only for trivial mechanical work with one sensible path — and when you do skip, name what you picked so it's easy to override.

## Scope discipline

V1 is deliberately narrow (see [README.md → V1 Scope](README.md#v1-scope)). Do not expand it without asking. Don't add features for hypothetical post-v1 needs — capture those in [README.md → Ideas / Post-v1](README.md#ideas--post-v1) instead.

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

## Time pressure

Hackathon runs May 2–3 2026. Integration checkpoint at H14 is non-negotiable per the README. Bias toward shipping a working slice over a complete one. Anything that's not on the critical path for the H14 checkpoint or the demo is a distraction — capture it in Ideas, don't build it.
