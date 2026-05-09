"""WebSocket client — receives IMU quaternion from RPi0 minicam-api /ws/imu.

Thread-safe cache of the latest quaternion.  Reconnects automatically.

Usage::
    client = IMUClient("192.168.1.20:8000")
    client.start()
    q, ts_ms = client.get()   # [w,x,y,z], timestamp ms  (None if never received)
    client.stop()
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

import websocket  # websocket-client

log = logging.getLogger(__name__)

_RECONNECT_DELAY = 3.0   # seconds between reconnection attempts
_STALE_MS        = 5000  # after this delay without a message, report as disconnected


class IMUClient:
    def __init__(self, host: str = "192.168.1.20:8000") -> None:
        self._url     = f"ws://{host}/ws/imu"
        self._lock    = threading.Lock()
        self._q:  list[float]   = [1.0, 0.0, 0.0, 0.0]
        self._ts: Optional[int] = None   # last message timestamp (ms)
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="imu-client")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def get(self) -> tuple[list[float], Optional[int]]:
        """Return (q [w,x,y,z], timestamp_ms).  timestamp_ms is None if no data yet."""
        with self._lock:
            return list(self._q), self._ts

    @property
    def connected(self) -> bool:
        if self._ts is None:
            return False
        return (int(time.time() * 1000) - self._ts) < _STALE_MS

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    self._url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                )
                log.info("IMUClient connecting to %s", self._url)
                ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as exc:
                log.warning("IMUClient error: %s", exc)
            if not self._stop.is_set():
                log.info("IMUClient reconnecting in %.0fs", _RECONNECT_DELAY)
                self._stop.wait(_RECONNECT_DELAY)

    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            data = json.loads(raw)
            q  = data["q"]
            ts = data.get("t", int(time.time() * 1000))
            with self._lock:
                self._q  = q
                self._ts = ts
        except Exception as exc:
            log.debug("IMUClient bad message: %s — %s", raw[:80], exc)

    def _on_error(self, ws: websocket.WebSocketApp, exc: Exception) -> None:
        log.warning("IMUClient WS error: %s", exc)
