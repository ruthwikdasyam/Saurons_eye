"""
HTTPS + WebSocket server that streams a wireframe scene to the Quest 3
WebXR client at ``headset/web/index.html``.

Run:
    python -m headset.server

On the Quest browser open ``https://<laptop-ip>:8443/``, accept the
self-signed cert once, then tap "Enter AR". The scene is sent on every
WS connect; POST /regen reshuffles the cubes for all connected viewers:

    curl -k -X POST https://localhost:8443/regen

WebXR ``immersive-ar`` requires HTTPS, so a self-signed cert is generated
into ``headset/certs/`` on first run via the ``openssl`` CLI.

Later: replace ``random_cubes()`` with a function that converts the live
SLAM map into the same Cube list — the wire format and renderer stay the
same.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import socket
import ssl
import subprocess
from pathlib import Path

from aiohttp import WSMsgType, web

from headset.scene import random_cubes, to_message

logger = logging.getLogger("headset.server")

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
CERT_DIR = HERE / "certs"
CERT_FILE = CERT_DIR / "server.crt"
KEY_FILE = CERT_DIR / "server.key"


def _ensure_cert() -> None:
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    if shutil.which("openssl") is None:
        raise RuntimeError(
            "openssl CLI not found — install openssl, or drop server.crt / "
            "server.key into headset/certs/ manually."
        )
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
            "-days", "365", "-subj", "/CN=saurons-eye",
        ],
        check=True,
        capture_output=True,
    )
    logger.info("generated self-signed cert in %s", CERT_DIR)


def _lan_ip() -> str:
    # Best-effort: pick the laptop's outbound interface IP for the printed URL.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


class Hub:
    """Tracks connected WS clients and the current scene."""

    def __init__(self) -> None:
        self._clients: set[web.WebSocketResponse] = set()
        self._scene = to_message(random_cubes())
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def add(self, ws: web.WebSocketResponse) -> None:
        async with self._lock:
            self._clients.add(ws)
            scene = self._scene
        await ws.send_json(scene)

    async def remove(self, ws: web.WebSocketResponse) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def regenerate(self) -> int:
        scene = to_message(random_cubes())
        async with self._lock:
            self._scene = scene
            targets = list(self._clients)
        sent = 0
        for ws in targets:
            try:
                await ws.send_json(scene)
                sent += 1
            except (ConnectionResetError, RuntimeError):
                pass
        return sent


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "index.html")


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    hub: Hub = request.app["hub"]
    await hub.add(ws)
    logger.info("client connected (%d total)", hub.client_count)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT and msg.data == "regen":
                await hub.regenerate()
    finally:
        await hub.remove(ws)
        logger.info("client disconnected (%d total)", hub.client_count)
    return ws


async def regen(request: web.Request) -> web.Response:
    sent = await request.app["hub"].regenerate()
    return web.json_response({"ok": True, "sent_to": sent})


def make_app() -> web.Application:
    app = web.Application()
    app["hub"] = Hub()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/regen", regen)
    app.router.add_static("/static", WEB_DIR)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _ensure_cert()
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))

    print(f"\n  Open on Quest browser:  https://{_lan_ip()}:{args.port}/\n")
    web.run_app(make_app(), host=args.host, port=args.port, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
