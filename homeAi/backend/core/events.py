"""事件总线(内部解耦) + WsHub(前端 WebSocket 信号广播)。

职责严格分离:
- EventBus : 内部事件发布/订阅, 插件只 emit, 感知不到浏览器
- WsHub    : 只管 WebSocket 连接管理与 record_created 信号广播
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Dict], Any]]] = {}

    def publish(self, kind: str, **payload: Any) -> Dict[str, Any]:
        event = {"kind": kind, **payload}
        subs = list(self._subscribers.get(kind, [])) + list(self._subscribers.get("*", []))
        for cb in subs:
            try:
                cb(event)
            except Exception:
                log.exception("event subscriber error: %s", kind)
        return event

    def subscribe(self, kind: str, callback: Callable[[Dict], Any]) -> None:
        self._subscribers.setdefault(kind, []).append(callback)


class WsHub:
    """只负责 WebSocket 连接管理与 record_created 信号广播。"""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._sockets: set = set()
        self._loop: asyncio.AbstractEventLoop = None
        bus.subscribe("record_created", self._dispatch)
        bus.subscribe("record_deleted", self._dispatch)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def connect(self, ws) -> None:
        self._sockets.add(ws)

    def disconnect(self, ws) -> None:
        self._sockets.discard(ws)

    def client_count(self) -> int:
        return len(self._sockets)

    def broadcast(self, payload: Dict) -> None:
        """把 JSON 安全的信号广播给所有已连接客户端(任意线程可调用)。"""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._broadcast, payload)

    def _broadcast(self, payload: Dict) -> None:
        for ws in list(self._sockets):
            asyncio.create_task(ws.send_json(payload))

    def heartbeat(self) -> None:
        self.broadcast({"type": "ping"})

    # -- record_created / record_deleted 订阅 --
    def _dispatch(self, event: Dict) -> None:
        signal = {"type": event["kind"], "record_id": event.get("record_id")}
        self.broadcast(signal)