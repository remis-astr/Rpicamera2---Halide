"""Plate-solve client for the Finder mode.

Orchestrates:
  1. Capture one RAW frame from the mini-cam  (minicam.capture_one_raw_frame)
  2. Debayer → grayscale                       (cv2)
  3. Run ASTAP                                 (platesolve.PlateSolver)
  4. Return SolveResult + the IMU quaternion   (imu_client.IMUClient)

Solving runs in a background thread so the UI stays responsive.
Callback signature: callback(result: SolveResult, q_imu: list[float])
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from libastrostack.minicam import capture_one_raw_frame
from libastrostack.platesolve import PlateSolver, SolveResult
from libastrostack.finder.imu_client import IMUClient

log = logging.getLogger(__name__)

# IMX462 / IMX290 at 1920×1080 on mini-cam
# pixel_size_um=2.9, lens focal determined empirically (≈90 mm for ~3.5°×2° FOV)
_PIXEL_SIZE_UM = 2.9
_FOCAL_MM      = 90.0   # update after first successful solve via focal_mm_measured


class SolverClient:
    """Non-blocking plate-solve trigger.

    Usage::
        client = SolverClient(minicam_host="192.168.1.20:8000",
                              imu_client=imu)
        client.solve_async(callback=on_solve_done)
        # ... later ...
        def on_solve_done(result, q_imu):
            if result.success:
                engine.align(q_imu, result.ra_deg, result.dec_deg)
    """

    def __init__(
        self,
        minicam_host: str = "192.168.1.20:8000",
        imu_client:   Optional[IMUClient] = None,
        focal_mm:     float = _FOCAL_MM,
        pixel_size_um: float = _PIXEL_SIZE_UM,
        timeout_s:    float = 40.0,
        max_stars:    int   = 500,
    ) -> None:
        self._host      = minicam_host
        self._imu       = imu_client
        self._focal_mm  = focal_mm
        self._px_um     = pixel_size_um
        self._timeout   = timeout_s
        self._lock      = threading.Lock()
        self._thread:   Optional[threading.Thread] = None
        self._cancel    = threading.Event()
        self._solver    = PlateSolver(
            pixel_size_um = pixel_size_um,
            timeout_s     = timeout_s - 10,   # leave headroom for capture
            max_stars     = max(10, max_stars),
            downsample    = 0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_solving(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def solve_async(
        self,
        callback:        Callable[[SolveResult, list[float]], None],
        hint_ra:         Optional[float] = None,
        hint_dec:        Optional[float] = None,
        hint_radius_deg: Optional[float] = None,
    ) -> bool:
        """Start a background solve.  Returns False if one is already running."""
        if self.is_solving:
            log.warning("SolverClient: solve already in progress")
            return False
        self._cancel.clear()
        t = threading.Thread(
            target  = self._run,
            args    = (callback, hint_ra, hint_dec, hint_radius_deg),
            daemon  = True,
            name    = "finder-solver",
        )
        with self._lock:
            self._thread = t
        t.start()
        return True

    def cancel(self) -> None:
        """Request cancellation of the current solve."""
        self._cancel.set()
        self._solver.cancel()

    def update_params(self, focal_mm: Optional[float] = None, max_stars: Optional[int] = None) -> None:
        """Update focal length and/or max stars (safe to call while idle)."""
        if focal_mm is not None:
            self._focal_mm = focal_mm
        if max_stars is not None:
            self._solver.max_stars = max(10, max_stars)

    def solve_sync(
        self,
        hint_ra:         Optional[float] = None,
        hint_dec:        Optional[float] = None,
        hint_radius_deg: Optional[float] = None,
    ) -> tuple[SolveResult, list[float]]:
        """Blocking version — for tests and CLI use."""
        results: list = []
        done = threading.Event()

        def cb(r, q):
            results.append((r, q))
            done.set()

        self.solve_async(cb, hint_ra, hint_dec, hint_radius_deg)
        done.wait(timeout=self._timeout + 5)
        return results[0] if results else (SolveResult(success=False, error="timeout"), [1,0,0,0])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        callback:        Callable[[SolveResult, list[float]], None],
        hint_ra:         Optional[float],
        hint_dec:        Optional[float],
        hint_radius_deg: Optional[float],
    ) -> None:
        t0 = time.time()
        log.info("SolverClient: capturing frame from %s …", self._host)

        # 1 — Capture
        if self._cancel.is_set():
            return
        frame_raw, meta = capture_one_raw_frame(self._host, timeout=15.0, settle=3)

        if frame_raw is None:
            log.warning("SolverClient: frame capture failed")
            with self._lock:
                self._thread = None
            callback(SolveResult(success=False, error="capture failed"), [1,0,0,0])
            return

        if self._cancel.is_set():
            return

        # 2 — Snapshot IMU quaternion at the moment the frame arrived
        q_imu = [1.0, 0.0, 0.0, 0.0]
        if self._imu is not None:
            q_snap, ts = self._imu.get()
            if ts is not None:
                q_imu = q_snap
                log.debug("SolverClient: IMU q=%s", q_imu)

        # 3 — Debayer RAW (CSI-2 ×16 uint16) → BGR uint8
        image_bgr = self._debayer(frame_raw)
        log.info("SolverClient: frame %dx%d captured in %.1fs, solving …",
                 image_bgr.shape[1], image_bgr.shape[0], time.time() - t0)

        if self._cancel.is_set():
            return

        # 4 — Solve
        if hint_radius_deg is not None:
            hint_r = hint_radius_deg
        elif hint_ra is not None and hint_dec is not None:
            hint_r = 35.0
        else:
            hint_r = 180.0
        result = self._solver.solve(
            image_bgr,
            focal_mm        = self._focal_mm,
            hint_ra         = hint_ra,
            hint_dec        = hint_dec,
            hint_radius_deg = hint_r,
        )

        if result.success and result.focal_mm_measured > 0:
            self._focal_mm = result.focal_mm_measured
            log.info("SolverClient: focal updated to %.1f mm", self._focal_mm)

        log.info("SolverClient: %s  RA=%.3f Dec=%.3f  (%.1fs total)",
                 "OK" if result.success else f"FAIL: {result.error}",
                 result.ra_deg, result.dec_deg,
                 time.time() - t0)

        # Clear thread reference before the callback so that the callback can
        # call solve_async again (e.g. for a full-sky retry) without hitting
        # the "already solving" guard (is_solving checks self._thread.is_alive()).
        with self._lock:
            self._thread = None
        callback(result, q_imu)

    @staticmethod
    def _debayer(raw: np.ndarray) -> np.ndarray:
        """CSI-2 ×16 uint16 Bayer → BGR uint8.

        IMX290 Bayer pattern is RGGB.  We use BayerRG2BGR (same empirical result
        as IMX585 in the main RPiCamera2 pipeline — see memory/color_channel_order.md).
        """
        # Remove CSI-2 ×16 scale → 12-bit in uint16
        scaled = (raw >> 4).astype(np.uint16)
        # Debayer → BGR uint16
        bgr16 = cv2.cvtColor(scaled, cv2.COLOR_BayerRG2BGR)
        # Scale to 8-bit
        return (bgr16 >> 4).astype(np.uint8)
