"""
Polar alignment screen controller for RPiCamera2.

Machine à états: SETUP → CAPTURING → RESULT → LIVE_ALT → LIVE_AZ → VERIFY

Usage (from RPiCamera2.py)::

    if _polar_align_screen is None:
        _polar_align_screen = PolarAlignScreen(
            obs_lat=obs_lat_deg, obs_lon=obs_lon_deg,
            minicam_host=host,
            focal_mm=minicam_focal_mm, pixel_um=minicam_pixel_um,
            max_stars=minicam_solve_max_stars,
            capture_thread=_minicam_capture_thread,
        )
    _polar_align_screen.open()
    _polar_align_active = True

    # In main loop:
    stay = _polar_align_screen.update(surface, events)
    if not stay:
        _polar_align_screen.close()
        _polar_align_active = False
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import pygame

from ..finder.imu_client import IMUClient
from ..platesolve import PlateSolver
from .polar_alignment import (
    Phase, PolarAlignmentState, Capture, AlignmentResult, ImuState,
    compute_axis, compute_target_imu_state, live_deviation,
    imu_state_from_quaternion, format_angle, direction_hint, error_color,
    polaris_position,
)
from .polar_widgets import PolarVisualizationWidget


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
_BG    = (8,  10,  20)
_PANEL = (18, 22,  40)
_BORD  = (50, 70, 100)
_TXT   = (190, 210, 240)
_DIM   = (100, 120, 150)
_ACC   = (0,  160, 200)
_GREEN = (50, 200, 100)
_ORG   = (220, 140, 50)
_RED   = (210, 70,  70)
_WHITE = (230, 230, 230)


def _fnt(size: int) -> pygame.font.Font:
    return pygame.font.Font(None, size)


def _btn(surf: pygame.Surface, rect: pygame.Rect,
         label: str, color: tuple, font_size: int = 22,
         active: bool = False) -> None:
    """Draw a simple button."""
    bc = tuple(min(v + 30, 255) for v in color) if active else color
    pygame.draw.rect(surf, bc, rect, border_radius=6)
    pygame.draw.rect(surf, _WHITE, rect, 1, border_radius=6)
    f   = _fnt(font_size)
    txt = f.render(label, True, _WHITE)
    surf.blit(txt, txt.get_rect(center=rect.center))


def _gauge(surf: pygame.Surface, rect: pygame.Rect,
           value_am: float, max_am: float = 20.0) -> None:
    """Draw a horizontal deviation gauge. Center = target, cursor = current."""
    pygame.draw.rect(surf, (30, 35, 50), rect, border_radius=4)
    pygame.draw.rect(surf, _BORD, rect, 1, border_radius=4)

    # Color zones
    w3  = rect.w // 3
    g_r = pygame.Rect(rect.x, rect.y, w3, rect.h)
    o_r = pygame.Rect(rect.x + w3, rect.y, w3, rect.h)
    r_r = pygame.Rect(rect.x + 2*w3, rect.y, w3, rect.h)
    pygame.draw.rect(surf, (60, 200, 80, 60), g_r, border_radius=4)
    pygame.draw.rect(surf, (200, 140, 40, 60), o_r, border_radius=4)
    pygame.draw.rect(surf, (200, 60, 60, 60), r_r, border_radius=4)

    # Center mark (target)
    mid_x = rect.centerx
    pygame.draw.line(surf, _WHITE, (mid_x, rect.y + 4), (mid_x, rect.bottom - 4), 2)

    # Cursor position
    frac  = max(-1.0, min(1.0, value_am / max(0.001, max_am)))
    cur_x = rect.centerx + int(frac * rect.w / 2)
    cur_x = max(rect.x + 4, min(rect.right - 4, cur_x))
    col   = error_color(abs(value_am))
    pygame.draw.rect(surf, col, pygame.Rect(cur_x - 5, rect.y + 4, 10, rect.h - 8),
                     border_radius=3)


class PolarAlignScreen:
    def __init__(self,
                 obs_lat: float,
                 obs_lon: float,
                 minicam_host: str,
                 focal_mm: float,
                 pixel_um: float,
                 max_stars: int,
                 capture_thread=None) -> None:
        self._lat      = obs_lat
        self._lon      = obs_lon
        self._host     = minicam_host
        self._focal    = focal_mm
        self._pixel_um = pixel_um
        self._max_stars = max_stars
        self._cap_thread = capture_thread    # MiniCamCaptureThread or None

        self._imu    = IMUClient(minicam_host)
        self._solver = PlateSolver(pixel_size_um=pixel_um, max_stars=max_stars)
        self._state  = PolarAlignmentState()
        self._vis    = None    # PolarVisualizationWidget — created on first draw

        # Solve state
        self._solving       = False
        self._solve_error   = ""
        self._solve_thread: Optional[threading.Thread] = None
        self._cancelled     = False

        # Gyro timer (AZ phase)
        self._gyro_calib_ts: Optional[float] = None
        self._gyro_reliability_s = 90.0

        self._rects: dict[str, pygame.Rect] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._imu.start()
        self._state.phase = Phase.SETUP
        self._state.reset_cycle()
        self._solve_error = ""
        self._cancelled   = False

    def close(self) -> None:
        self._cancelled = True
        self._imu.stop()
        if self._solve_thread and self._solve_thread.is_alive():
            self._solve_thread.join(timeout=2.0)

    def update_params(self, focal_mm: float, max_stars: int,
                      capture_thread=None) -> None:
        self._focal     = focal_mm
        self._max_stars = max_stars
        if capture_thread is not None:
            self._cap_thread = capture_thread

    # ------------------------------------------------------------------
    # Main update (called every frame by RPiCamera2 main loop)
    # ------------------------------------------------------------------

    def update(self, surface: pygame.Surface, events: list) -> bool:
        """Draw and handle events. Returns False when user requests exit."""
        self._rects = {}
        w, h = surface.get_size()

        # Ensure vis widget is sized
        if self._vis is None or self._vis.rect.w != w // 2:
            ww = min(w // 2, h - 80)
            self._vis = PolarVisualizationWidget(
                pygame.Rect(20, 50, ww, ww),
                self._lat, self._lon,
            )

        surface.fill(_BG)

        # Title bar
        self._draw_title(surface, w)

        phase = self._state.phase
        if phase == Phase.SETUP:
            self._draw_setup(surface, w, h)
        elif phase == Phase.CAPTURING:
            self._draw_capturing(surface, w, h)
        elif phase == Phase.RESULT:
            self._draw_result(surface, w, h)
        elif phase in (Phase.LIVE_ALT, Phase.LIVE_AZ):
            self._draw_live(surface, w, h, phase)
        elif phase == Phase.VERIFY:
            self._draw_capturing(surface, w, h)

        # Handle events
        for ev in events:
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return False
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if not self._handle_click(ev.pos):
                    return False

        return True

    # ------------------------------------------------------------------
    # Title bar
    # ------------------------------------------------------------------

    def _draw_title(self, surf: pygame.Surface, w: int) -> None:
        # Back button
        back_r = pygame.Rect(8, 8, 60, 30)
        pygame.draw.rect(surf, (30, 40, 60), back_r, border_radius=5)
        pygame.draw.rect(surf, _BORD, back_r, 1, border_radius=5)
        f = _fnt(20)
        surf.blit(f.render("← Ret.", True, _DIM), back_r.move(8, 7))
        self._rects['back'] = back_r

        title = "Mise en station"
        if self._state.is_verify:
            title = "Vérification"
        f2 = _fnt(26)
        t  = f2.render(title, True, _TXT)
        surf.blit(t, t.get_rect(centerx=w // 2, y=10))

        # Phase indicator
        phases = ["Config", "Capture", "Résultat", "Live IMU"]
        idx_map = {Phase.SETUP: 0, Phase.CAPTURING: 1, Phase.VERIFY: 1,
                   Phase.RESULT: 2, Phase.LIVE_ALT: 3, Phase.LIVE_AZ: 3}
        cur = idx_map.get(self._state.phase, 0)
        px0 = w - len(phases) * 80 - 10
        for i, lbl in enumerate(phases):
            col = _ACC if i == cur else _BORD
            f3  = _fnt(16)
            t3  = f3.render(lbl, True, col)
            surf.blit(t3, (px0 + i * 80, 14))

    # ------------------------------------------------------------------
    # Phase: SETUP
    # ------------------------------------------------------------------

    def _draw_setup(self, surf: pygame.Surface, w: int, h: int) -> None:
        y = 55
        f = _fnt(20)

        # Lat / lon info
        lat_str = f"{abs(self._lat):.3f}° {'N' if self._lat >= 0 else 'S'}"
        lon_str = f"{abs(self._lon):.3f}° {'E' if self._lon >= 0 else 'O'}"
        surf.blit(f.render(f"Latitude : {lat_str}    Longitude : {lon_str}",
                            True, _TXT), (20, y))
        y += 30

        # Optimal zone hint
        surf.blit(_fnt(18).render(
            "Zone optimale : méridien ±30°, Déc ±20°  (objet brillant près de l'équateur)",
            True, _DIM), (20, y))
        y += 38

        # Separator
        pygame.draw.line(surf, _BORD, (20, y), (w - 20, y))
        y += 16

        # N captures selector
        surf.blit(_fnt(22).render("Nombre de captures :", True, _TXT), (20, y))
        y += 30

        n_opts = [3, 4, 5, 6]
        times  = ["~1 min", "~1.5 min", "~2 min", "~2.5 min"]
        btn_w, btn_h, gap = 70, 40, 14
        for i, (n, est) in enumerate(zip(n_opts, times)):
            bx = 20 + i * (btn_w + gap)
            r  = pygame.Rect(bx, y, btn_w, btn_h)
            active = (self._state.n_captures == n)
            _btn(surf, r, str(n), _ACC if active else (40, 50, 75),
                 font_size=26, active=active)
            surf.blit(_fnt(14).render(est, True, _DIM),
                      (bx, y + btn_h + 3))
            self._rects[f'n_{n}'] = r

        y += btn_h + 28

        # Precision indicator
        if self._state.n_captures == 3:
            note = "Mode rapide — formule 3-points exacte"
        else:
            note = f"Mode précision — fit moindres carrés ({self._state.n_captures} points)"
        surf.blit(_fnt(17).render(note, True, _DIM), (20, y))
        y += 30

        # Instructions
        lines = [
            "1. Pointez une étoile brillante proche du méridien et de l'équateur céleste.",
            "2. Bloquez le tube du Dobson (ne le bougez plus pendant tout le cycle).",
            "3. Appuyez sur COMMENCER — le programme vous guidera pour les 3 captures.",
        ]
        for line in lines:
            surf.blit(_fnt(17).render(line, True, _TXT), (20, y))
            y += 22
        y += 10

        # Start button
        start_r = pygame.Rect(w // 2 - 100, h - 80, 200, 48)
        _btn(surf, start_r, "COMMENCER", _ACC, font_size=24)
        self._rects['start'] = start_r

    # ------------------------------------------------------------------
    # Phase: CAPTURING / VERIFY
    # ------------------------------------------------------------------

    def _draw_capturing(self, surf: pygame.Surface, w: int, h: int) -> None:
        n   = self._state.n_captures
        done = len(self._state.captures)
        y   = 55

        # Progress bar
        bar_r = pygame.Rect(20, y, w - 40, 16)
        pygame.draw.rect(surf, (30, 40, 60), bar_r, border_radius=4)
        frac = done / n
        if frac > 0:
            fr_r = pygame.Rect(20, y, int((w - 40) * frac), 16)
            pygame.draw.rect(surf, _ACC, fr_r, border_radius=4)
        pygame.draw.rect(surf, _BORD, bar_r, 1, border_radius=4)
        y += 24

        # Capture number
        verb = "Vérification" if self._state.is_verify else "Capture"
        lbl = f"{verb} {done + 1} / {n}"
        surf.blit(_fnt(34).render(lbl, True, _TXT),
                  (w // 2 - 80, y))
        y += 50

        # Instructions
        if done == 0:
            instr = f"Plateau en position de départ (~0°)"
        else:
            step_deg = 14.0 / (n - 1) if n > 1 else 14.0
            instr = f"Déplacez le plateau d'environ {step_deg:.0f}° puis capturez."
        surf.blit(_fnt(20).render(instr, True, _DIM), (w // 2 - 200, y))
        y += 36

        # Solve status
        if self._solving:
            surf.blit(_fnt(20).render("⟳ Plate solve en cours…", True, _ORG),
                      (w // 2 - 120, y))
        elif self._solve_error:
            surf.blit(_fnt(20).render(f"✗ {self._solve_error}", True, _RED),
                      (w // 2 - 160, y))
        y += 32

        # IMU connected indicator
        q, ts = self._imu.get()
        imu_ok = ts is not None and (time.time() * 1000 - ts) < 5000
        imu_col = _GREEN if imu_ok else _DIM
        surf.blit(_fnt(16).render(f"IMU {'●' if imu_ok else '○'} {'connecté' if imu_ok else 'indisponible'}",
                                   True, imu_col), (20, h - 50))

        # Previous captures summary
        if self._state.captures:
            surf.blit(_fnt(17).render("Captures enregistrées :", True, _DIM),
                      (20, y))
            y += 20
            for i, c in enumerate(self._state.captures):
                s = f"  #{i+1}  RA {c.ra_deg:.3f}°  Déc {c.dec_deg:.3f}°"
                surf.blit(_fnt(16).render(s, True, _TXT), (20, y))
                y += 18

        # Capture button
        can_capture = not self._solving
        cap_col = _ACC if can_capture else (50, 60, 80)
        cap_r = pygame.Rect(w // 2 - 100, h - 80, 200, 48)
        _btn(surf, cap_r, "CAPTURER", cap_col, font_size=24)
        if can_capture:
            self._rects['capture'] = cap_r

        # Cancel button
        cancel_r = pygame.Rect(w - 140, h - 80, 120, 38)
        _btn(surf, cancel_r, "ANNULER", (70, 30, 30), font_size=20)
        self._rects['cancel'] = cancel_r

    # ------------------------------------------------------------------
    # Phase: RESULT
    # ------------------------------------------------------------------

    def _draw_result(self, surf: pygame.Surface, w: int, h: int) -> None:
        result = self._state.result
        if result is None:
            return

        # Left: visualization widget
        ww = min(w // 2 - 30, h - 100)
        self._vis.rect = pygame.Rect(15, 48, ww, ww)

        # Adjust scale based on error
        if result.total_error_arcmin > 30:
            self._vis.set_scale(90)
        else:
            self._vis.set_scale(60)

        self._vis.draw(surf, result, datetime.now(timezone.utc),
                       history=self._state.history)

        # Right: text panel
        rx = ww + 30
        ry = 55
        rw = w - rx - 15

        # Error value + color
        col = error_color(result.total_error_arcmin)
        f_big = _fnt(32)
        tot_s = f"Total : {result.total_error_arcmin:.1f}'"
        surf.blit(f_big.render(tot_s, True, col), (rx, ry))
        ry += 40

        # Quality hint
        if result.total_error_arcmin < 5:
            qual = "✓ Excellent — < 5'"
        elif result.total_error_arcmin < 15:
            qual = "⚠ Acceptable — < 15'"
        else:
            qual = "✗ Trop éloigné — > 15'"
        surf.blit(_fnt(18).render(qual, True, col), (rx, ry))
        ry += 30

        pygame.draw.line(surf, _BORD, (rx, ry), (rx + rw, ry))
        ry += 12

        # ΔAlt / ΔAz
        alt_hint, az_hint = direction_hint(result.delta_az_arcmin, result.delta_alt_arcmin)
        f_med = _fnt(22)
        f_sml = _fnt(18)

        surf.blit(f_med.render(f"ΔAlt = {format_angle(result.delta_alt_arcmin / 60)}", True, _TXT), (rx, ry))
        surf.blit(f_sml.render(f"       ({result.delta_alt_arcmin:+.1f}')", True, _DIM), (rx, ry + 22))
        surf.blit(f_sml.render(alt_hint, True, _ORG), (rx, ry + 42))
        ry += 70

        surf.blit(f_med.render(f"ΔAz  = {format_angle(result.delta_az_arcmin / 60)}", True, _TXT), (rx, ry))
        surf.blit(f_sml.render(f"       ({result.delta_az_arcmin:+.1f}')", True, _DIM), (rx, ry + 22))
        surf.blit(f_sml.render(az_hint, True, _ORG), (rx, ry + 42))
        ry += 70

        # Polaris reference
        _, dec_p = polaris_position()
        polaris_dist = (90.0 - dec_p) * 60.0
        pct = result.total_error_arcmin / polaris_dist * 100 if polaris_dist > 0 else 0
        surf.blit(_fnt(16).render(
            f"≈ {pct:.0f}% de la distance Polaris↔pôle ({polaris_dist:.0f}')",
            True, _DIM), (rx, ry))
        ry += 24

        # Residual (mode précision)
        if result.residual_rms_arcsec is not None:
            ry += 6
            rms = result.residual_rms_arcsec
            if rms < 30:
                rc, rl = _GREEN, f"Résidu : {rms:.0f}\"  ✓ excellent"
            elif rms < 120:
                rc, rl = _ORG, f"Résidu : {rms:.0f}\"  ⚠ flexion probable"
            else:
                rc, rl = _RED, f"Résidu : {rms:.0f}\"  ✗ mesure peu fiable"
            surf.blit(_fnt(17).render(rl, True, rc), (rx, ry))
            ry += 22

            if result.outliers_indices:
                surf.blit(_fnt(16).render(
                    f"Captures écartées : {result.outliers_indices}", True, _ORG), (rx, ry))
                ry += 20

        # Action buttons
        btn_y = h - 56
        bw = (rw - 20) // 3
        live_r  = pygame.Rect(rx,            btn_y, bw, 44)
        new_r   = pygame.Rect(rx + bw + 10,  btn_y, bw, 44)
        done_r  = pygame.Rect(rx + 2*bw + 20, btn_y, bw, 44)
        _btn(surf, live_r,  "Live IMU", (0, 80, 120), 20)
        _btn(surf, new_r,   "Nouveau",  (50, 80, 40), 20)
        _btn(surf, done_r,  "Terminer", (70, 30, 30), 20)
        self._rects['live_imu'] = live_r
        self._rects['new_cycle'] = new_r
        self._rects['done'] = done_r

    # ------------------------------------------------------------------
    # Phase: LIVE_ALT / LIVE_AZ
    # ------------------------------------------------------------------

    def _draw_live(self, surf: pygame.Surface, w: int, h: int, phase: Phase) -> None:
        is_alt = (phase == Phase.LIVE_ALT)
        q, ts  = self._imu.get()
        imu_ok = ts is not None and (time.time() * 1000 - ts) < 5000

        y = 55

        # Phase selector tabs
        for i, (label, ph) in enumerate([("ALTITUDE", Phase.LIVE_ALT),
                                          ("AZIMUT",   Phase.LIVE_AZ)]):
            tab_r = pygame.Rect(20 + i * 160, y, 150, 34)
            col   = _ACC if phase == ph else (30, 40, 60)
            pygame.draw.rect(surf, col, tab_r, border_radius=6)
            pygame.draw.rect(surf, _BORD, tab_r, 1, border_radius=6)
            surf.blit(_fnt(20).render(label, True, _WHITE), tab_r.move(14, 8))
            self._rects[f'tab_{"alt" if ph == Phase.LIVE_ALT else "az"}'] = tab_r
        y += 48

        # IMU status
        if not imu_ok:
            surf.blit(_fnt(20).render("⚠ IMU non disponible — reconnexion…", True, _ORG),
                      (20, y))
            y += 30

        # Compute deviation
        dev = None
        if imu_ok and self._state.target_imu is not None:
            imu_now = imu_state_from_quaternion(q, time.time())
            dev     = live_deviation(imu_now, self._state.target_imu)

        if dev is None:
            surf.blit(_fnt(20).render("En attente de données IMU…", True, _DIM), (20, y))
        else:
            val_am = dev.delta_alt_arcmin if is_alt else dev.delta_az_arcmin
            label  = "ΔAlt" if is_alt else "ΔAz"
            col    = error_color(abs(val_am))

            # Numeric display
            surf.blit(_fnt(36).render(f"{label} = {val_am:+.1f}'", True, col), (20, y))
            y += 42
            surf.blit(_fnt(22).render(f"       = {val_am / 60:+.3f}°", True, _DIM), (20, y))
            y += 34

            # Direction hint
            if is_alt:
                hint = "↓ Baisser côté Nord" if val_am > 0 else "↑ Lever côté Nord"
            else:
                hint = "← Vers l'Ouest" if val_am > 0 else "→ Vers l'Est"
            if abs(val_am) < 1.0:
                hint = "✓ OK — écart < 1'"
                col  = _GREEN
            surf.blit(_fnt(22).render(hint, True, col), (20, y))
            y += 38

            # Big gauge
            gauge_r = pygame.Rect(20, y, w - 40, 50)
            _gauge(surf, gauge_r, val_am, max_am=20.0)
            y += 62

        # AZ phase: gyro timer
        if not is_alt:
            if self._gyro_calib_ts is not None:
                elapsed = time.time() - self._gyro_calib_ts
                remain  = max(0, self._gyro_reliability_s - elapsed)
                bar_w   = int((w - 40) * remain / self._gyro_reliability_s)
                pygame.draw.rect(surf, (30, 40, 60), pygame.Rect(20, y, w - 40, 12),
                                 border_radius=4)
                pygame.draw.rect(surf, _ORG if remain < 30 else _GREEN,
                                 pygame.Rect(20, y, bar_w, 12), border_radius=4)
                surf.blit(_fnt(17).render(
                    f"Fiabilité gyro : {remain:.0f} s restantes", True, _DIM), (20, y + 14))
                y += 36

            # Recalibrate gyro button
            recal_r = pygame.Rect(20, y, 200, 36)
            _btn(surf, recal_r, "Recalibrer gyro", (60, 50, 20), 18)
            self._rects['recal_gyro'] = recal_r
            y += 46

        # Action buttons at bottom
        if is_alt:
            next_r = pygame.Rect(w - 200, h - 60, 180, 44)
            _btn(surf, next_r, "→ Phase AZ", _ACC, 20)
            self._rects['next_az'] = next_r
        else:
            verify_r = pygame.Rect(w - 220, h - 60, 200, 44)
            _btn(surf, verify_r, "Vérification", _GREEN, 20)
            self._rects['verify'] = verify_r

        result_r = pygame.Rect(20, h - 60, 160, 44)
        _btn(surf, result_r, "← Résultat", (40, 50, 80), 18)
        self._rects['back_result'] = result_r

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------

    def _handle_click(self, pos: tuple) -> bool:
        """Return False to signal exit."""
        for key, rect in self._rects.items():
            if not rect.collidepoint(pos):
                continue

            if key == 'back':
                return False

            elif key.startswith('n_'):
                n = int(key[2:])
                self._state.n_captures = n

            elif key == 'start':
                self._state.reset_cycle()
                self._state.is_verify = False
                self._state.phase = Phase.CAPTURING
                self._solve_error = ""

            elif key == 'capture':
                if not self._solving:
                    self._do_capture()

            elif key == 'cancel':
                self._cancelled = True
                self._state.phase = Phase.SETUP
                self._state.reset_cycle()
                self._solve_error = ""
                self._cancelled   = False

            elif key == 'live_imu':
                if self._state.result is not None:
                    self._state.phase = Phase.LIVE_ALT
                    self._gyro_calib_ts = None

            elif key == 'new_cycle':
                self._state.reset_cycle()
                self._state.is_verify = False
                self._state.phase = Phase.CAPTURING
                self._solve_error = ""

            elif key == 'done':
                return False

            elif key == 'tab_alt':
                self._state.phase = Phase.LIVE_ALT

            elif key == 'tab_az':
                self._state.phase = Phase.LIVE_AZ
                # Start gyro calibration timer when switching to AZ
                if self._gyro_calib_ts is None:
                    self._gyro_calib_ts = time.time()

            elif key == 'next_az':
                self._state.phase = Phase.LIVE_AZ
                self._gyro_calib_ts = time.time()

            elif key == 'recal_gyro':
                self._gyro_calib_ts = time.time()

            elif key == 'verify':
                self._state.reset_cycle()
                self._state.is_verify = True
                self._state.phase = Phase.VERIFY
                self._solve_error = ""

            elif key == 'back_result':
                self._state.phase = Phase.RESULT

            break   # one rect at a time

        return True

    # ------------------------------------------------------------------
    # Plate solve capture
    # ------------------------------------------------------------------

    def _do_capture(self) -> None:
        self._solving     = True
        self._solve_error = ""
        self._cancelled   = False
        self._solve_thread = threading.Thread(target=self._solve_worker, daemon=True)
        self._solve_thread.start()

    def _solve_worker(self) -> None:
        try:
            bgr = self._get_frame_bgr()
            if bgr is None:
                self._solve_error = "Impossible d'obtenir une image"
                return
            if self._cancelled:
                return

            # Plate solve
            result = self._solver.solve(bgr, focal_mm=self._focal)
            if self._cancelled:
                return

            if not result.success:
                self._solve_error = result.error or "Solve échoué"
                return

            # Record IMU state at solve time
            q_now, ts_ms = self._imu.get()
            ts_s  = ts_ms / 1000.0 if ts_ms else time.time()
            imu_s = imu_state_from_quaternion(q_now, ts_s) if ts_ms else None

            cap = Capture(
                ra_deg        = result.ra_deg,
                dec_deg       = result.dec_deg,
                timestamp_utc = datetime.now(timezone.utc),
                imu_state     = imu_s,
            )
            self._state.add_capture(cap)

            # Save last IMU as reference
            if imu_s:
                self._state.imu_reference = imu_s

            if self._state.is_cycle_complete():
                self._finish_cycle()

        except Exception as exc:
            self._solve_error = str(exc)
        finally:
            self._solving = False

    def _get_frame_bgr(self) -> Optional[np.ndarray]:
        """Get a BGR frame for plate solving.

        Prefers the running capture thread; falls back to a dedicated WS connection.
        """
        # Try capture thread first
        if self._cap_thread is not None and self._cap_thread.running:
            deadline = time.time() + 3.0
            while time.time() < deadline:
                frame, _ = self._cap_thread.get_latest_frame()
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
                time.sleep(0.05)

        # Dedicated connection (only if no active thread)
        if self._cap_thread is None or not self._cap_thread.running:
            try:
                from ..minicam import capture_one_raw_frame
                frame, _ = capture_one_raw_frame(self._host, timeout=10.0, settle=2)
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            except Exception as exc:
                print(f"[PolarAlign] capture_one_raw_frame failed: {exc}")

        return None

    def _finish_cycle(self) -> None:
        """Compute axis and transition to RESULT phase."""
        try:
            result = compute_axis(self._state.captures, self._lat, self._lon)
        except Exception as exc:
            self._solve_error = f"Calcul échoué: {exc}"
            return

        self._state.result = result
        self._state.history.append(result)

        # Compute target IMU state if reference is available
        if self._state.imu_reference is not None:
            self._state.target_imu = compute_target_imu_state(
                result, self._state.imu_reference)

        if self._state.is_verify:
            self._state.phase = Phase.RESULT
        else:
            self._state.phase = Phase.RESULT
