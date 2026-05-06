"""Minimal LX200-compatible TCP server — makes the RPi appear as a telescope mount.

Planetarium apps (Stellarium Mobile, SkySafari…) connect on port 4030 and send
GoTo commands (:Sr/:Sd/:MS#).  The server captures the target RA/Dec and fires
the on_goto(ra_deg, dec_deg) callback so RPiCamera2.py can drive push-to guidance.

Position queries (:GR#/:GD#) return the last plate-solve result so the planetarium
app can show where the telescope is currently pointing.
"""
from __future__ import annotations

import logging
import re
import socket
import threading
from typing import Callable, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_PORT = 4030


# ---------------------------------------------------------------------------
# Coord helpers
# ---------------------------------------------------------------------------

def _ra_deg_to_hms(ra_deg: float) -> str:
    ra_deg = ra_deg % 360.0
    h_total = ra_deg / 15.0
    h = int(h_total)
    m_total = (h_total - h) * 60.0
    m = int(m_total)
    s = (m_total - m) * 60.0
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _dec_deg_to_dms(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    d = abs(dec_deg)
    deg = int(d)
    m_total = (d - deg) * 60.0
    m = int(m_total)
    s = int((m_total - m) * 60.0)
    return f"{sign}{deg:02d}*{m:02d}'{s:02d}"


def _parse_hms(s: str) -> Optional[float]:
    """HH:MM:SS.s → degrees, or None."""
    m = re.match(r"(\d+)[:\s](\d+)[:\s]([\d.]+)", s.strip())
    if not m:
        return None
    h, mn, sc = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return (h + mn / 60.0 + sc / 3600.0) * 15.0


def _parse_dms(s: str) -> Optional[float]:
    """sDD*MM'SS or sDD:MM:SS → degrees, or None."""
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    m = re.match(r"(\d+)[*:\s](\d+)[':?\s]?([\d.]*)", s)
    if not m:
        return None
    d, mn = float(m.group(1)), float(m.group(2))
    sc = float(m.group(3)) if m.group(3) else 0.0
    return sign * (d + mn / 60.0 + sc / 3600.0)


# ---------------------------------------------------------------------------
# LX200Server
# ---------------------------------------------------------------------------

class LX200Server:
    """Single-client LX200 TCP server.

    Usage::
        srv = LX200Server(port=4030)
        srv.on_goto = lambda ra, dec: print(f"GoTo {ra:.4f} {dec:.4f}")
        srv.start()
        srv.set_position(ra_deg, dec_deg)   # update from plate solve
        target = srv.get_target()            # (ra, dec) or None
        srv.stop()
    """

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self.port = port
        self.on_goto: Optional[Callable[[float, float], None]] = None

        self._lock = threading.Lock()
        self._current_ra: float = 0.0
        self._current_dec: float = 0.0
        self._target_ra: Optional[float] = None
        self._target_dec: Optional[float] = None
        self._pending_ra: Optional[float] = None
        self._pending_dec: Optional[float] = None

        self._running = False
        self._client_connected = False
        self._thread: Optional[threading.Thread] = None
        self._server_sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True, name="LX200Server")
        self._thread.start()
        log.info("[LX200] Server started on port %d", self.port)

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
        self._client_connected = False
        log.info("[LX200] Server stopped")

    def set_position(self, ra_deg: float, dec_deg: float) -> None:
        """Update current pointing position (call after each successful plate solve)."""
        with self._lock:
            self._current_ra = ra_deg % 360.0
            self._current_dec = max(-90.0, min(90.0, dec_deg))

    def get_target(self) -> Optional[Tuple[float, float]]:
        """Return (ra_deg, dec_deg) of last GoTo, or None."""
        with self._lock:
            if self._target_ra is None:
                return None
            return self._target_ra, self._target_dec

    def clear_target(self) -> None:
        with self._lock:
            self._target_ra = None
            self._target_dec = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_connected(self) -> bool:
        return self._client_connected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _serve(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.listen(1)
            sock.settimeout(1.0)
            self._server_sock = sock
            log.info("[LX200] Listening on 0.0.0.0:%d", self.port)
        except Exception as e:
            log.error("[LX200] Bind failed: %s", e)
            self._running = False
            return

        while self._running:
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            log.info("[LX200] Client connected: %s", addr)
            self._client_connected = True
            try:
                self._handle_client(conn)
            except Exception as e:
                log.warning("[LX200] Client error: %s", e)
            finally:
                self._client_connected = False
                try:
                    conn.close()
                except Exception:
                    pass
            log.info("[LX200] Client disconnected: %s", addr)

        try:
            sock.close()
        except Exception:
            pass

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(30.0)
        buf = ""
        while self._running:
            try:
                raw = conn.recv(256)
            except socket.timeout:
                continue
            except Exception:
                break
            if not raw:
                break

            for byte in raw:
                if byte == 0x06:
                    # LX200 ACK: identify mount type → "A" = alt-az
                    log.debug("[LX200] ACK → A")
                    try:
                        conn.sendall(b"A")
                    except Exception:
                        return
                    continue

                ch = chr(byte)
                buf += ch
                if ch == "#":
                    cmd = buf.strip()
                    buf = ""
                    resp = self._dispatch(cmd)
                    if resp:
                        try:
                            conn.sendall(resp.encode("ascii"))
                        except Exception:
                            return

    def _dispatch(self, cmd: str) -> str:
        if not cmd.startswith(":"):
            return ""
        body = cmd[1:-1]  # strip ':' and '#'

        # Current position queries
        if body == "GR":
            with self._lock:
                return _ra_deg_to_hms(self._current_ra) + "#"
        if body == "GD":
            with self._lock:
                return _dec_deg_to_dms(self._current_dec) + "#"

        # Set target RA
        if body.startswith("Sr"):
            ra = _parse_hms(body[2:].strip())
            if ra is not None:
                with self._lock:
                    self._pending_ra = ra
            return "1"

        # Set target Dec
        if body.startswith("Sd"):
            dec = _parse_dms(body[2:].strip())
            if dec is not None:
                with self._lock:
                    self._pending_dec = dec
            return "1"

        # Slew / GoTo
        if body == "MS":
            target = None
            with self._lock:
                if self._pending_ra is not None and self._pending_dec is not None:
                    self._target_ra = self._pending_ra
                    self._target_dec = self._pending_dec
                    target = (self._target_ra, self._target_dec)
                    self._pending_ra = None
                    self._pending_dec = None
            if target is not None:
                log.info("[LX200] GoTo RA=%.4f Dec=%.4f", target[0], target[1])
                if self.on_goto:
                    try:
                        self.on_goto(target[0], target[1])
                    except Exception:
                        pass
            return "0"  # 0 = slewing, no error

        # Stop
        if body.startswith("Q"):
            return ""

        # Mount status: AT0 = Alt-Az, no tracking (push-to dobsonian)
        if body == "GW":
            return "AT0#"

        # Identity
        if body == "GVP":
            return "RPiCamera2#"
        if body in ("GVF", "GVN"):
            return "1.0#"
        if body in ("ACK", "P"):
            return "A"

        # Silently ignore unsupported settings (location, time, etc.)
        if body.startswith(("St", "Sg", "SG", "SL", "SC", "Sw", "SS")):
            return "1"

        log.debug("[LX200] Unknown: %s", cmd)
        return ""
