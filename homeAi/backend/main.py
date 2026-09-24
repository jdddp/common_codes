"""FastAPI 入口: REST / WebSocket / MJPEG + 生命周期管理(支持多摄像头)。"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .core.cameras import Cameras
from .core.events import EventBus, WsHub
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
    cameras = Cameras(cfg, bus)

    storage.start()
    cameras.start()

    app.state.cfg = cfg
    app.state.hub = hub
    app.state.bus = bus
    app.state.cameras = cameras
    app.state.storage = storage

    hub_task = asyncio.create_task(_heartbeat_loop(hub))
    ids = [f"{c.camera_id}({c.source.status()})" for c in cameras.units]
    log.info("home-ai started, cameras: %s", ", ".join(ids) or "(none)")

    yield

    hub_task.cancel()
    cameras.stop()
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
        "lapse": Path(app.state.cfg.get("storage", "lapse_dir", "data/lapse")),
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


@app.get("/playback", include_in_schema=False)
async def playback():
    return FileResponse(FRONTEND_DIR / "playback.html")


# ---------- MJPEG 实时流 ----------
def _unit_or_404(camera_id: str = ""):
    cameras = app.state.cameras
    unit = cameras.get(camera_id) if camera_id else cameras.default()
    if unit is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return unit


async def _mjpeg(unit):
    async def gen():
        while True:
            frame = unit.source.latest_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    yield (b"--" + BOUNDARY.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            await asyncio.sleep(0.06)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


@app.get("/cam/stream")
async def cam_stream():
    return await _mjpeg(_unit_or_404())


@app.get("/cam/{camera_id}/stream")
async def camera_stream(camera_id: str):
    return await _mjpeg(_unit_or_404(camera_id))


# ---------- 快照 / 录像 ----------
@app.get("/snapshots/{path:path}")
async def snapshot(path: str):
    return FileResponse(_safe_media_file("snapshot", path))


@app.get("/clips/{path:path}")
async def clip(path: str):
    return FileResponse(_safe_media_file("clip", path))


@app.get("/lapse/{camera_id}/{path:path}")
async def lapse(camera_id: str, path: str):
    root = (_media_paths()["lapse"] / camera_id).resolve()
    full = (root / path).resolve()
    if root != full.parent and root not in full.parents:
        raise HTTPException(status_code=404, detail="not found")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(full)


# ---------- API ----------
@app.get("/api/status")
async def api_status():
    cameras = app.state.cameras
    units = cameras.status()
    default = cameras.default()
    cam = {"status": "stopped", "fps": 0.0, "frame_id": 0}
    if default:
        cam = {"status": default.source.status(),
               "fps": round(default.source.fps(), 1),
               "frame_id": default.source.frame_id()}
    return {
        "camera": cam,  # 兼容旧字段(默认摄像头)
        "cameras": units,
        "plugins": cameras.all_plugins(),
        "ws_clients": app.state.hub.client_count(),
    }


@app.get("/api/plugins")
async def api_plugins():
    return app.state.cameras.all_plugins()


@app.post("/api/cameras/{camera_id}/plugins/{name}/toggle")
async def api_camera_plugin_toggle(camera_id: str, name: str):
    result = app.state.cameras.toggle_plugin(camera_id, name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"plugin not found: {camera_id}/{name}")
    return result


@app.post("/api/plugins/{name}/toggle")
async def api_plugin_toggle(name: str):
    result = app.state.cameras.toggle_plugin_first(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"plugin not found: {name}")
    return result


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