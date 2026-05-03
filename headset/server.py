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

from headset.scene import Cube, PointCloud, Polyline, cubes_from_ply, random_cubes, to_message

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

    def __init__(self, ply_path: str | None = None) -> None:
        self._clients: set[web.WebSocketResponse] = set()
        self._ply_path = ply_path
        self._scene = to_message(self._build_cubes())
        self._lock = asyncio.Lock()

    def _build_cubes(self):
        if self._ply_path:
            logger.info("loading wireframe from %s", self._ply_path)
            cubes = cubes_from_ply(self._ply_path)
            logger.info("wireframe: %d cube(s) — %d floor / %d wall / %d object",
                        len(cubes),
                        sum(1 for c in cubes if c.color == 0x506070),
                        sum(1 for c in cubes if c.color == 0x00CCAA),
                        sum(1 for c in cubes if c.color == 0xFF8800))
            return cubes
        return random_cubes()

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
        scene = to_message(self._build_cubes())
        return await self._broadcast(scene)

    async def push_scene(
        self,
        cubes: list[Cube] | None = None,
        polylines: list[Polyline] | None = None,
        point_clouds: list[PointCloud] | None = None,
    ) -> int:
        """Replace the current scene from an external publisher and broadcast."""
        scene = to_message(cubes=cubes, polylines=polylines, point_clouds=point_clouds)
        return await self._broadcast(scene)

    async def _broadcast(self, scene: dict) -> int:
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


async def post_scene(request: web.Request) -> web.Response:
    """Live publisher endpoint.

    Body shape:
        {
          "cubes":     [{center, size, quat, color?}, ...],
          "polylines": [{points:[[x,y,z],...], color?, fill_color?, closed?}, ...]
        }

    Either or both arrays may be omitted.
    """
    payload = await request.json()
    raw_cubes = payload.get("cubes") or []
    raw_lines = payload.get("polylines") or []
    raw_clouds = payload.get("point_clouds") or []
    cubes = [
        Cube(
            center=tuple(c["center"]),
            size=tuple(c["size"]),
            quat=tuple(c["quat"]),
            color=int(c.get("color", 0x00FF88)),
            frame=str(c.get("frame", "world")),
        )
        for c in raw_cubes
    ]
    polylines = [
        Polyline(
            points=[tuple(p) for p in pl["points"]],
            color=int(pl.get("color", 0x00FF88)),
            fill_color=(int(pl["fill_color"]) if pl.get("fill_color") is not None else None),
            closed=bool(pl.get("closed", True)),
            frame=str(pl.get("frame", "world")),
        )
        for pl in raw_lines
    ]
    point_clouds = [
        PointCloud(
            points=[tuple(p) for p in pc["points"]],
            color=int(pc.get("color", 0x00FF88)),
            size=float(pc.get("size", 0.04)),
            frame=str(pc.get("frame", "world")),
        )
        for pc in raw_clouds
    ]
    sent = await request.app["hub"].push_scene(
        cubes=cubes, polylines=polylines, point_clouds=point_clouds,
    )
    return web.json_response({
        "ok": True, "sent_to": sent,
        "n_cubes": len(cubes), "n_polylines": len(polylines),
        "n_point_clouds": len(point_clouds),
    })


def make_app(ply_path: str | None = None) -> web.Application:
    app = web.Application()
    app["hub"] = Hub(ply_path=ply_path)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/regen", regen)
    app.router.add_post("/scene", post_scene)
    app.router.add_static("/static", WEB_DIR)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--ply", default=None,
                        help="Load wireframe from this .ply instead of random cubes.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _ensure_cert()
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))

    print(f"\n  Open on Quest browser:  https://{_lan_ip()}:{args.port}/\n")
    web.run_app(make_app(ply_path=args.ply), host=args.host, port=args.port, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
