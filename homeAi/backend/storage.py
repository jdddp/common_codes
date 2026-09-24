"""记录形成模块: 系统内唯一有权写/删 SQLite 的模块。

- 订阅 snapshot(含帧)   : 编码 JPEG 落盘 → 落库 → record_created 信号
- 订阅 alert/clip/camera_status/system : 写 events 表 → record_created 信号
- 删除记录: 连同关联快照/录像文件一起清理 → record_deleted 信号
- 独立工作线程 + 队列, 不阻塞采集/插件线程
- 只此模块可写 DB; 插件/录像器只 emit, 不碰数据库
"""
import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

from .config import Config
from .core.events import EventBus

log = logging.getLogger(__name__)


class Storage:
    def __init__(self, config: Config, bus: EventBus):
        self._bus = bus
        self._snapshot_dir = str(config.get("storage", "snapshot_dir", "data/snapshots"))
        self._clips_dir = str(config.get("storage", "clips_dir", "data/clips"))
        self._lapse_dir = str(config.get("storage", "lapse_dir", "data/lapse"))
        self._db_path = str(config.get("storage", "db_path", "data/homeai.db"))
        self._retain_days = int(config.get("storage", "retain_days", 14))
        self._camera_id = str(config.get("camera", "camera_id", "main"))

        self._queue: Deque[tuple] = deque(maxlen=512)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_cleanup = 0.0

    # ---------- 生命周期 ----------
    def start(self) -> None:
        os.makedirs(self._snapshot_dir, exist_ok=True)
        self._init_db()
        self._bus.subscribe("snapshot", lambda e: self._queue.append(("snapshot", e)))
        for kind in ("alert", "clip", "camera_status", "system", "lapse"):
            self._bus.subscribe(kind, lambda e: self._queue.append(("record", e)))
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="storage", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ---------- 工作线程 ----------
    def _worker(self) -> None:
        while self._running:
            try:
                item = self._queue.popleft()
                if item[0] == "snapshot":
                    self._save_snapshot(item[1])
                else:
                    self._insert(item[1])
            except IndexError:
                self._cleanup_if_due()
                time.sleep(0.5)
            except Exception:
                log.exception("storage worker error")

    # ---------- 快照落盘 + 落库 ----------
    def _save_snapshot(self, event: Dict[str, Any]) -> None:
        frame = event.get("frame")
        if frame is None or not isinstance(frame, np.ndarray):
            return
        day = datetime.now().strftime("%Y%m%d")
        rel = f"{day}/{int(time.time() * 1000)}.jpg"
        path = os.path.join(self._snapshot_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            log.error("JPEG encode failed")
            return
        with open(path, "wb") as f:
            f.write(buf.tobytes())

        box = event.get("boxes")
        meta = {"boxes": box} if isinstance(box, list) else {}
        self._insert(
            {
                "kind": "snapshot",
                "ts": event.get("ts"),
                "source": event.get("source", "unknown"),
                "summary": event.get("summary", "快照"),
                "snapshot_path": rel,
                "meta": meta,
                "camera_id": event.get("camera_id"),
            }
        )

    # ---------- 落库(唯一写点) ----------
    def _insert(self, event: Dict[str, Any]) -> None:
        ts_ms = self._to_ms(event.get("ts") or time.time())
        meta = event.get("meta", {})
        if event.get("kind") == "lapse" and meta.get("lapse_path") and self._lapse_path_exists(
            event.get("camera_id") or self._camera_id, meta["lapse_path"]
        ):
            # 持续录像分片名全局唯一: 重启/重复检测不再重复入库
            return
        meta_str = json.dumps(meta, ensure_ascii=False) if meta else ""
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute(
                "INSERT INTO events (ts, camera_id, kind, source, summary, snapshot_path, meta) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    ts_ms,
                    event.get("camera_id") or self._camera_id,
                    event.get("kind", "event"),
                    event.get("source", ""),
                    event.get("summary", ""),
                    event.get("snapshot_path"),
                    meta_str,
                ),
            )
            conn.commit()
            record_id = cur.lastrowid
        finally:
            conn.close()
        self._bus.publish("record_created", record_id=record_id)

    def _lapse_path_exists(self, camera_id: str, lapse_path: str) -> bool:
        # meta 内以 "lapse_path": "文件名" 形式 json.dumps 存储(ensure_ascii=False)
        needle = '"lapse_path": "' + lapse_path + '"'
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT 1 FROM events WHERE kind='lapse' AND camera_id=? AND meta LIKE ? LIMIT 1",
                (camera_id, "%" + needle + "%"),
            ).fetchall()
            return bool(rows)
        finally:
            conn.close()

    # ---------- 查询 ----------
    def list_events(
        self,
        after_id: int = 0,
        limit: int = 100,
        kind: Optional[str] = None,
        day: Optional[str] = None,
    ) -> List[Dict]:
        where = ["id>?"]
        args: list = [after_id]
        if kind:
            where.append("kind=?")
            args.append(kind)
        else:
            # 事件记录默认不包含持续录像(lapse); 回放页用 kind=lapse 显式查询
            where.append("kind<>?")
            args.append("lapse")
        if day:
            start_d = datetime.strptime(day, "%Y-%m-%d")
            end_d = start_d + timedelta(days=1)
            where.append("ts>=? AND ts<?")
            args += [int(start_d.timestamp() * 1000), int(end_d.timestamp() * 1000)]
        sql = "SELECT * FROM events WHERE " + " AND ".join(where) + " ORDER BY id DESC LIMIT ?"
        args.append(limit)

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, args).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["meta"] = json.loads(d["meta"]) if d.get("meta") else {}
                out.append(d)
            return out
        finally:
            conn.close()

    def list_days(self, limit: int = 30) -> List[Dict]:
        """按本地日期统计事件数的降序列表, 供前端"按天筛选"(持续录像不计)。"""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT strftime('%Y-%m-%d', ts/1000.0, 'unixepoch', 'localtime') AS day, "
                "COUNT(*) AS c FROM events WHERE kind<>'lapse' GROUP BY day ORDER BY day DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [{"day": r[0], "count": r[1]} for r in rows]
        finally:
            conn.close()

    # ---------- 删除(连同关联记录与文件) ----------
    def delete_record(self, record_id: int) -> bool:
        """删除一条记录及其同一事件段(same source, ts 相近)的关联记录, 并清理所有媒体文件。"""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM events WHERE id=?", (record_id,)).fetchone()
            if row is None:
                return False
            # 同一事件段的告警/快照/录像: 同摄像头 + 相同 source + 触发时刻一致(毫秒级)
            related = conn.execute(
                "SELECT * FROM events WHERE camera_id=? AND source=? AND ts BETWEEN ? AND ?",
                (row["camera_id"], row["source"], row["ts"] - 1000, row["ts"] + 1000),
            ).fetchall()
            ids = [r["id"] for r in related]
            rows = [dict(r) for r in related]

            conn.execute(
                "DELETE FROM events WHERE id IN (%s)" % ",".join("?" * len(ids)), ids
            )
            conn.commit()
        finally:
            conn.close()

        # 清理关联媒体文件(所有关联记录的快照 + 录像 + 持续录像分片)
        files = set()
        for r in rows:
            if r.get("snapshot_path"):
                files.add(os.path.join(self._snapshot_dir, r["snapshot_path"]))
            meta = json.loads(r["meta"]) if r.get("meta") else {}
            clip = meta.get("clip_path")
            if clip:
                files.add(os.path.join(self._clips_dir, clip))
            lapse = meta.get("lapse_path")
            if lapse:
                files.add(os.path.join(self._lapse_dir, r.get("camera_id") or "", lapse))
        for path in files:
            self._remove_file(path)

        for rid in ids:
            self._bus.publish("record_deleted", record_id=rid)
        return True

    @staticmethod
    def _remove_file(path: str) -> None:
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except Exception:
            log.exception("remove file failed: %s", path)

    def last_id(self) -> int:
        conn = sqlite3.connect(self._db_path)
        try:
            return int(conn.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0])
        finally:
            conn.close()

    # ---------- SQLite 初始化/清理 ----------
    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            INTEGER,
                camera_id     TEXT,
                kind          TEXT,
                source        TEXT,
                summary       TEXT,
                snapshot_path TEXT,
                meta          TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _cleanup_if_due(self) -> None:
        now = time.time()
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now
        cutoff_ms = int((datetime.now() - timedelta(days=self._retain_days)).timestamp() * 1000)
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("DELETE FROM events WHERE ts < ?", (cutoff_ms,))
                conn.commit()
            finally:
                conn.close()

            for base in (self._snapshot_dir, self._clips_dir):
                if not os.path.isdir(base):
                    continue
                for day in os.listdir(base):
                    try:
                        day_date = datetime.strptime(day, "%Y%m%d")
                    except ValueError:
                        continue
                    if day_date < datetime.now() - timedelta(days=self._retain_days):
                        import shutil

                        shutil.rmtree(os.path.join(base, day), ignore_errors=True)
                        log.info("cleaned dir: %s", os.path.join(base, day))
        except Exception:
            log.exception("cleanup error")

    @staticmethod
    def _to_ms(v: Any) -> int:
        """输入秒级 float 或已经是毫秒, 统一成整数毫秒。"""
        v = float(v)
        if v > 1e12:  # 已是毫秒级
            return int(v)
        return int(v * 1000)