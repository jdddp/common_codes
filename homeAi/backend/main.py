"""FastAPI 入口: REST / WebSocket / MJPEG + 生命周期管理。"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .core.events import EventBus, WsHub
from .core.pipeline import Pipeline
from .core.recorder import FrameRecorder
from .core.stream import CameraSource
from .plugins import PluginManager
from .storage import Storage

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
BOUNDARY = "frame"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    bus = EventBus()
    hub = WsHub(bus)

    storage = Storage(cfg, bus)
    recorder = FrameRecorder(cfg, bus)
    plugins = PluginManager(cfg, bus)

    def on_camera_status(status: str) -> None:
        bus.publish("camera_status", status=status, source="camera",
                    summary=f"摄像头状态: {status}", ts=time.time())

    camera = CameraSource(
        rtsp_url=str(cfg.get("camera", "rtsp_url", "")),
        reconnect_delay=float(cfg.get("camera", "reconnect_delay", 2)),
        max_reconnect_delay=float(cfg.get("camera", "max_reconnect_delay", 30)),
        watchdog_timeout=float(cfg.get("camera", "watchdog_timeout", 10)),
        on_status=on_camera_status,
    )

    pipeline = Pipeline(cfg, plugins, recorder)

    storage.start()
    recorder.start()
    plugins.load_all()
    camera.subscribe(pipeline.feed)
    camera.start()

    app.state.cfg = cfg
    app.state.hub = hub
    app.state.bus = bus
    app.state.camera = camera
    app.state.plugins = plugins
    app.state.storage = storage
    app.state.recorder = recorder

    hub_task = asyncio.create_task(_heartbeat_loop(hub))
    log.info("home-ai started, rtsp=%s", camera.status())

    yield

    hub_task.cancel()
    camera.stop()
    plugins.shutdown()
    recorder.stop()
    storage.stop()
    log.info("home-ai stopped")


app = FastAPI(title="Home AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


async def _heartbeat_loop(hub: WsHub) -> None:
    while True:
        await asyncio.sleep(20)
        hub.heartbeat()


def _media_paths() -> dict:
    return {
        "snapshot": Path(app.state.cfg.get("storage", "snapshot_dir", "data/snapshots")),
        "clip": Path(app.state.cfg.get("storage", "clips_dir", "data/clips")),
    }


def _safe_media_file(kind: str, path: str) -> Path:
    root = _media_paths()[kind].resolve()
    full = (root / path).resolve()
    if root not in full.parents and full != root:
        raise HTTPException(status_code=404, detail="not found")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return full


# ---------- 页面与静态 ----------
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


# ---------- MJPEG 实时流 ----------
@app.get("/cam/stream")
async def cam_stream():
    async def gen():
        while True:
            frame = app.state.camera.latest_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    yield (b"--" + BOUNDARY.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            await asyncio.sleep(0.06)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


# ---------- 快照 / 录像 ----------
@app.get("/snapshots/{path:path}")
async def snapshot(path: str):
    return FileResponse(_safe_media_file("snapshot", path))


@app.get("/clips/{path:path}")
async def clip(path: str):
    return FileResponse(_safe_media_file("clip", path))


# ---------- API ----------
@app.get("/api/status")
async def api_status():
    cam = app.state.camera
    return {
        "camera": {"status": cam.status(), "fps": round(cam.fps(), 1), "frame_id": cam.frame_id()},
        "plugins": app.state.plugins.status(),
        "ws_clients": app.state.hub.client_count(),
    }


@app.get("/api/plugins")
async def api_plugins():
    return app.state.plugins.status()


@app.post("/api/plugins/{name}/toggle")
async def api_plugin_toggle(name: str):
    plugin = app.state.plugins.plugin(name)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"plugin not found: {name}")
    plugin.set_enabled(not plugin.is_enabled())
    return {"name": name, "enabled": plugin.is_enabled()}


@app.get("/api/events")
async def api_events(
    after_id: int = Query(0),
    limit: int = Query(100, le=500),
    kind: str = Query(None),
    day: str = Query(None),
):
    events = app.state.storage.list_events(after_id=after_id, limit=limit, kind=kind, day=day)
    return {"events": events, "last_id": events[0]["id"] if events else after_id}


@app.get("/api/events/days")
async def api_events_days(limit: int = Query(30, le=90)):
    return {"days": app.state.storage.list_days(limit=limit)}


@app.delete("/api/events/{record_id}")
async def api_delete_event(record_id: int):
    if not app.state.storage.delete_record(record_id):
        raise HTTPException(status_code=404, detail="record not found")
    return {"ok": True}


# ---------- WebSocket ----------
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    hub = app.state.hub
    hub.bind_loop(asyncio.get_event_loop())
    hub.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws error")
    finally:
        hub.disconnect(ws)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    uvicorn.run(app, host=str(cfg.get("server", "host", "0.0.0.0")),
                port=int(cfg.get("server", "port", 8765)), log_level="info")


if __name__ == "__main__":
    main()