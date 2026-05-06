"""
MiniCam integration for RPiCamera2.

Three classes:
  MiniCamController     — set gain / exposure / WB via WS /ws/control
  MiniCamCaptureThread  — drop-in for AsyncCaptureThread, streams raw frames via WS /ws/raw
  MiniCamClient         — low-level WS helpers shared by both
"""
from __future__ import annotations

import json
import logging
import struct
import threading
import time
from typing import Any

import numpy as np
import websocket  # websocket-client (pip install websocket-client)

log = logging.getLogger(__name__)

# Default host when connected via USB gadget.
DEFAULT_HOST = "192.168.7.2:8000"

# Raw WS: number of frames to drop after a set_rate / initial connect before
# storing frames (lets the sensor pipeline settle).
_SETTLE_FRAMES = 2


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _ws_url(host: str, path: str) -> str:
    return f"ws://{host}{path}"


def _http_url(host: str, path: str) -> str:
    return f"http://{host}{path}"


# ---------------------------------------------------------------------------
# MiniCamController
# ---------------------------------------------------------------------------

class MiniCamController:
    """Send gain / exposure / WB commands to the MiniCam via WS /ws/control.

    Usage::
        ctrl = MiniCamController("192.168.7.2:8000")
        ctrl.connect()
        ctrl.set_gain(8.0)
        ctrl.set_exposure_ms(1000.0)
        ctrl.disconnect()
    """

    def __init__(self, host: str = DEFAULT_HOST) -> None:
        self.host = host
        self._ws: websocket.WebSocket | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            self._ws = self._try_connect()

    def _try_connect(self) -> "websocket.WebSocket | None":
        """Low-level connect — must be called with self._lock held."""
        try:
            ws = websocket.create_connection(
                _ws_url(self.host, "/ws/control"),
                timeout=5,
            )
            log.info("MiniCamController connected to %s", self.host)
            return ws
        except Exception as e:
            log.error("MiniCamController connect failed: %s", e)
            return None

    def disconnect(self) -> None:
        with self._lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None

    def _send(self, msg: dict) -> dict | None:
        with self._lock:
            # Reconnect if the connection was lost.
            if not self._ws:
                self._ws = self._try_connect()
            if not self._ws:
                return None
            try:
                self._ws.send(json.dumps(msg))
                raw = self._ws.recv()
                if not raw:
                    log.warning("MiniCamController: server closed connection (empty response to %r)", msg.get("cmd"))
                    self._ws = None
                    return None
                return json.loads(raw)
            except Exception as e:
                log.warning("MiniCamController send error (cmd=%r): %s", msg.get("cmd"), e)
                self._ws = None
                # One immediate reconnect attempt.
                self._ws = self._try_connect()
                if not self._ws:
                    return None
                try:
                    self._ws.send(json.dumps(msg))
                    raw = self._ws.recv()
                    return json.loads(raw) if raw else None
                except Exception as e2:
                    log.warning("MiniCamController retry failed (cmd=%r): %s", msg.get("cmd"), e2)
                    self._ws = None
                    return None

    def set_gain(self, value: float) -> None:
        self._send({"cmd": "set_gain", "value": float(value)})

    def set_exposure_ms(self, ms: float) -> None:
        self._send({"cmd": "set_exposure", "value_ms": float(ms)})

    def set_wb(self, red: float, blue: float) -> None:
        self._send({"cmd": "set_wb", "red": float(red), "blue": float(blue)})

    def status(self) -> dict:
        resp = self._send({"cmd": "status"})
        return resp or {}


# ---------------------------------------------------------------------------
# MiniCamCaptureThread
# ---------------------------------------------------------------------------

class MiniCamCaptureThread:
    """Drop-in replacement for AsyncCaptureThread using the MiniCam WS /ws/raw.

    The thread maintains a persistent WebSocket connection and continuously
    receives binary frames:
        [4 bytes big-endian JSON length][JSON metadata bytes][uint16 LE raw bytes]

    Raw frames are in "CSI-2 ×16" scale (uint16, 0–65520), identical to what
    picamera2 produces with unpacked=True — directly compatible with
    debayer_raw_array() in RPiCamera2.py.

    Usage mirrors AsyncCaptureThread::
        thread = MiniCamCaptureThread("192.168.7.2:8000")
        thread.start()
        frame, meta = thread.get_latest_frame()   # (ndarray | None, dict | None)
        thread.set_capture_params({'type': 'raw'})
        thread.stop()
    """

    def __init__(self, host: str = DEFAULT_HOST, fps: float = 5.0) -> None:
        self.host = host
        self.fps = fps

        self.running = False
        self.capturing = False
        self.capture_params: dict = {"type": "raw"}

        self._thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_metadata: dict | None = None
        self._cancel_current = False

    # ------------------------------------------------------------------
    # Public API (mirrors AsyncCaptureThread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            log.info("[MiniCam] Capture thread started (host=%s fps=%.1f)", self.host, self.fps)

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("[MiniCam] Capture thread stopped")

    def get_latest_frame(self) -> tuple[np.ndarray | None, dict | None]:
        """Non-blocking. Returns (frame, metadata) and clears the buffer."""
        with self._frame_lock:
            frame = self._latest_frame
            meta = self._latest_metadata
            self._latest_frame = None
            self._latest_metadata = None
        return frame, meta

    def set_capture_params(self, params: dict) -> None:
        """Called by RPiCamera2.py to switch between 'raw' and 'main' modes.

        MiniCam only has one stream type (raw Bayer), so this is a no-op for
        type='main' — callers should handle a 2-D frame as RAW regardless.
        """
        self.capture_params = params.copy()
        with self._frame_lock:
            self._latest_frame = None
            self._latest_metadata = None

    def cancel_capture(self) -> None:
        self._cancel_current = True

    def is_capturing(self) -> bool:
        return self.capturing

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self.running:
            ws = self._connect()
            if ws is None:
                time.sleep(2.0)
                continue
            try:
                self._stream(ws)
            except Exception as e:
                log.warning("[MiniCam] Stream error: %s — reconnecting", e)
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
            if self.running:
                time.sleep(1.0)

    def _connect(self) -> websocket.WebSocket | None:
        url = _ws_url(self.host, "/ws/raw")
        try:
            ws = websocket.create_connection(url, timeout=10)
            # Request desired frame rate.
            ws.send(json.dumps({"cmd": "set_rate", "fps": self.fps}))
            log.info("[MiniCam] WS /ws/raw connected")
            return ws
        except Exception as e:
            log.warning("[MiniCam] WS /ws/raw connect failed: %s", e)
            return None

    def _stream(self, ws: websocket.WebSocket) -> None:
        settle = _SETTLE_FRAMES
        ws.settimeout(30.0)

        while self.running:
            self.capturing = True
            self._cancel_current = False

            try:
                data = ws.recv()
            except Exception:
                self.capturing = False
                raise

            self.capturing = False

            if self._cancel_current:
                continue

            if not isinstance(data, (bytes, bytearray)) or len(data) < 4:
                log.warning("[MiniCam] invalid data received: type=%s len=%d", type(data), len(data) if data else 0)
                continue

            # Parse binary message: [4-byte JSON len][JSON][uint16 LE pixels]
            json_len = struct.unpack(">I", data[:4])[0]
            if len(data) < 4 + json_len:
                continue

            try:
                meta = json.loads(data[4:4 + json_len])
            except Exception as e:
                log.warning("[MiniCam] JSON parse error: %s", e)
                continue

            pixel_bytes = data[4 + json_len:]
            w = meta.get("w", 0)
            h = meta.get("h", 0)

            if w <= 0 or h <= 0 or len(pixel_bytes) != w * h * 2:
                log.warning("[MiniCam] geometry mismatch: w=%d h=%d bytes=%d expected=%d", w, h, len(pixel_bytes), w * h * 2)
                continue

            frame = np.frombuffer(pixel_bytes, dtype="<u2").reshape(h, w).copy()

            if settle > 0:
                settle -= 1
                continue

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_metadata = {
                    "ExposureTime": meta.get("ExposureTime", meta.get("exposure_us", 0)),
                    "AnalogueGain": meta.get("AnalogueGain", meta.get("gain", 1.0)),
                    "ColourGains": (1.0, 1.0),
                }


# ---------------------------------------------------------------------------
# Standalone helper: capture one raw frame (for plate solving)
# ---------------------------------------------------------------------------

def capture_one_raw_frame(
    host: str = DEFAULT_HOST,
    timeout: float = 15.0,
    settle: int = 3,
) -> tuple[np.ndarray | None, dict]:
    """Open a temporary /ws/raw connection and return one settled RAW frame.

    Returns (frame_uint16_CSI2x16, metadata) or (None, {}) on failure.
    The frame is in uint16 with CSI-2 ×16 scale (identical to picamera2 unpacked),
    directly compatible with debayer_raw_array() in RPiCamera2.py.
    """
    try:
        ws = websocket.create_connection(_ws_url(host, "/ws/raw"), timeout=10)
    except Exception as exc:
        log.warning("[MiniCam] capture_one_raw_frame connect failed: %s", exc)
        return None, {}

    try:
        ws.settimeout(timeout)
        remaining_settle = settle
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                data = ws.recv()
            except Exception as exc:
                log.warning("[MiniCam] capture_one_raw_frame recv error: %s", exc)
                return None, {}

            if not isinstance(data, (bytes, bytearray)) or len(data) < 4:
                continue

            json_len = struct.unpack(">I", data[:4])[0]
            if len(data) < 4 + json_len:
                continue
            try:
                meta = json.loads(data[4:4 + json_len])
            except Exception:
                continue

            pixel_bytes = data[4 + json_len:]
            w = meta.get("w", 0)
            h = meta.get("h", 0)
            if w <= 0 or h <= 0 or len(pixel_bytes) != w * h * 2:
                continue

            if remaining_settle > 0:
                remaining_settle -= 1
                continue

            frame = np.frombuffer(pixel_bytes, dtype="<u2").reshape(h, w).copy()
            return frame, {
                "ExposureTime": meta.get("ExposureTime", meta.get("exposure_us", 0)),
                "AnalogueGain": meta.get("AnalogueGain", meta.get("gain", 1.0)),
                "w": w,
                "h": h,
            }
    finally:
        try:
            ws.close()
        except Exception:
            pass

    log.warning("[MiniCam] capture_one_raw_frame: timeout without frame")
    return None, {}
