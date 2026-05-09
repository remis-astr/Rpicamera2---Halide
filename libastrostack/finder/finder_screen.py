"""Finder screen — main Pygame controller.

Designed to integrate with RPiCamera2.py's event loop:

    finder = FinderScreen(obs_lat=47.312, obs_lon=0.482,
                          minicam_host="192.168.1.20:8000")
    finder.open()
    while True:
        events = pygame.event.get()
        stay   = finder.update(windowSurfaceObj, events)
        if not stay:
            break   # user pressed Back → return to minicam mode
        pygame.display.update()

Can also run standalone (python -m libastrostack.finder.finder_screen).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

import numpy as np
import pygame

import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time

from libastrostack.finder.coordinates import CoordinateHelper
from libastrostack.finder.data_loader  import get_loader, CatalogEntry, StarEntry
from libastrostack.finder.imu_client   import IMUClient
from libastrostack.finder.pushto_engine import PushToEngine
from libastrostack.finder.solver_client import SolverClient
from libastrostack.finder.target_source import Target
from libastrostack.finder.ui           import theme
from libastrostack.finder.ui.theme     import p, get_font
from libastrostack.finder.ui           import widgets as W
from libastrostack.finder.ui           import compass_view as CV
from libastrostack.finder.calibration  import load_calibration, load_calib_anchor
from libastrostack.finder.ui.calibration_wizard import CalibrationWizard

_SCREEN_W  = 1024
_SCREEN_H  = 600
_BAR_TOP   = 40
_VIEW_H    = 460
_BAR_INFO  = 50   # info bar height
_BAR_ACT   = 50   # action bar height

_STAR_CACHE_INTERVAL = 2.0   # seconds between full star alt/az recompute
_AUTORESOLVE_DEFAULT = 180   # seconds


class FinderScreen:
    def __init__(
        self,
        obs_lat:      float = 47.312,
        obs_lon:      float = 0.482,
        obs_elev:     float = 0.0,
        minicam_host: str   = "192.168.1.20:8000",
        night_mode:   bool  = True,
        focal_mm:     float = 90.0,
        max_stars:    int   = 500,
    ) -> None:
        self._coord   = CoordinateHelper(obs_lat, obs_lon, obs_elev)
        self._loader  = get_loader()
        self._imu     = IMUClient(minicam_host)
        self._engine  = PushToEngine(self._coord)
        self._solver  = SolverClient(minicam_host, imu_client=self._imu,
                                     focal_mm=focal_mm, max_stars=max_stars)
        self._night   = night_mode
        theme.set_night(night_mode)

        self._target:  Optional[Target] = None
        self.on_target_changed = None  # Callable[[Target], None] | None
        self._fov_deg: float = 30.0
        self._autoresolve_s: int = _AUTORESOLVE_DEFAULT

        # Star / constellation alt/az cache
        self._star_altaz:  Optional[list[tuple[float, float]]] = None
        self._const_altaz: Optional[dict] = None
        self._cache_t: float = 0.0
        self._cache_refreshing: bool = False   # background thread running

        # Solving state
        self._solving:          bool  = False
        self._solve_msg:        str   = ""
        self._solve_flash:      float = 0.0   # timestamp of last successful solve
        self._solve_hint_radius: float = 180.0  # radius used for current/last solve

        # Target guidance (updated each frame in update())
        self._target_sep:      float = 0.0
        self._target_dalt:     float = 0.0
        self._target_daz:      float = 0.0
        self._target_imu_only: bool  = False  # True = IMU calib only, no plate solve

        # Sub-surface for compass view
        self._view_surf: Optional[pygame.Surface] = None

        # Button rects (populated in _draw_top_bar / _draw_action_bar)
        self._btn_back:    Optional[pygame.Rect] = None
        self._btn_night:   Optional[pygame.Rect] = None
        self._btn_settings:Optional[pygame.Rect] = None
        self._btn_fov_plus: Optional[pygame.Rect] = None
        self._btn_fov_minus:Optional[pygame.Rect] = None
        self._btn_search:  Optional[pygame.Rect] = None
        self._btn_solve:   Optional[pygame.Rect] = None
        self._btn_preview: Optional[pygame.Rect] = None

        self._calib_wizard: Optional[CalibrationWizard] = None

        # IMU relative tracking (before first solve)
        self._q_imu_ref: Optional[list] = None   # quaternion de référence session courante
        self._center_ref: tuple[float, float] = (0.0, 0.0)
        self._q_calib_anchor: Optional[list] = None  # q_north du wizard (ancre absolue)

        # Search modal
        self._show_search = False
        self._search_query = ""
        self._search_results: list[CatalogEntry] = []
        self._search_above_hz: list[bool] = []   # pre-computed, same order as _search_results
        self._search_selected = 0
        self._kbd_layout = self._build_kbd()
        self._kbd_rects: dict = {}
        self._filt_m       = True
        self._filt_ngc     = True
        self._filt_ic      = True
        self._filt_horizon = False
        self._filt_rects:  dict = {}   # populated in _draw_search_overlay
        self._search_result_rects: list[pygame.Rect] = []
        self._last_click_result: int = -1   # for double-click detection
        self._last_click_t: float   = 0.0

        # Settings overlay
        self._show_settings = False
        self._settings_rects: dict = {}    # populated in _draw_settings_overlay

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start IMU client.  Call before the first update()."""
        self._imu.start()
        self._q_imu_ref    = None
        self._q_calib_anchor = None
        q_calib = load_calibration()
        if q_calib is not None:
            self._engine.set_r_calib(q_calib.tolist())
            q_anchor = load_calib_anchor()
            if q_anchor is not None:
                self._q_calib_anchor = q_anchor.tolist()
                self._q_imu_ref  = self._q_calib_anchor
                self._center_ref = (0.0, 0.0)

    def close(self) -> None:
        """Stop background threads."""
        self._imu.stop()

    def set_target(self, target: Optional[Target]) -> None:
        """Set push-to target and notify RPiCamera2 if a callback is registered."""
        self._target = target
        if target is not None and callable(self.on_target_changed):
            self.on_target_changed(target)

    def update_solve_params(self, focal_mm: float, max_stars: int) -> None:
        """Sync focal length and max_stars from the MiniCam solve sliders."""
        self._solver.update_params(focal_mm=focal_mm, max_stars=max_stars)

    def notify_solve(self, ra_deg: float, dec_deg: float) -> None:
        """Receive coordinates from an external solve (MiniCam mode).

        Aligns the engine using the current IMU quaternion so the Finder
        immediately shows ALIGNED status.  Also stores the coords as a fallback
        hint for the next in-Finder Re-solve.
        """
        self._ext_solve_ra  = ra_deg
        self._ext_solve_dec = dec_deg
        q_now, ts = self._imu.get()
        if ts is not None:
            self._engine.align(q_now, ra_deg, dec_deg)
            self._solve_msg   = f"Aligned (MiniCam): {ra_deg:.2f}° {dec_deg:+.2f}°"
            self._solve_flash = time.monotonic()
            log.info("FinderScreen: aligné via solve MiniCam RA=%.3f Dec=%.3f", ra_deg, dec_deg)
        else:
            log.warning("FinderScreen: IMU indisponible — solve externe stocké comme hint seulement")

    # ------------------------------------------------------------------
    # Main update — called every frame from RPiCamera2's loop
    # ------------------------------------------------------------------

    def update(self, surf: pygame.Surface, events: list) -> bool:
        """Render one frame and process events.

        Returns True to keep the Finder open, False to exit.
        """
        # — Calibration wizard (takes over the whole surface when active) —
        if self._calib_wizard is not None:
            still_open = self._calib_wizard.update(surf, events)
            if not still_open:
                self._calib_wizard = None
                # Reload calibration immediately after wizard closes
                q_calib = load_calibration()
                if q_calib is not None:
                    self._engine.set_r_calib(q_calib.tolist())
                q_anchor = load_calib_anchor()
                if q_anchor is not None:
                    self._q_calib_anchor = q_anchor.tolist()
                    self._q_imu_ref  = self._q_calib_anchor
                    self._center_ref = (0.0, 0.0)
                    log.info("FinderScreen: ancre calibration chargée")
            return True

        # — Refresh star/constellation cache if stale —
        if time.monotonic() - self._cache_t > _STAR_CACHE_INTERVAL:
            self._refresh_sky_cache()

        # — Get current pointing —
        q_now, ts = self._imu.get()
        if self._engine.is_aligned:
            pointing = self._engine.get_current_pointing(q_now)
            cur_alt  = pointing[2] if pointing else 45.0
            cur_az   = pointing[3] if pointing else 0.0
        elif ts is not None:
            # Avant le premier solve : suivi relatif depuis la session courante
            if self._q_imu_ref is None:
                self._q_imu_ref = list(q_now)
            cur_alt, cur_az = self._relative_pointing(q_now)
        else:
            cur_alt, cur_az = 45.0, 180.0   # IMU non connecté

        # — Compute target delta —
        target_dalt = target_daz = target_sep = 0.0
        target_imu_only = False
        if self._target:
            if self._engine.is_aligned:
                target_sep, target_dalt, target_daz = self._engine.get_delta_to_target(
                    q_now, self._target.ra_deg, self._target.dec_deg)
            elif ts is not None and self._q_calib_anchor is not None:
                # IMU calibration done but no plate solve yet: compute delta
                # from relative pointing (alt/az from IMU) vs target sky position.
                try:
                    from libastrostack.finder.pushto_engine import (
                        _altaz_to_vec, _angular_dist_vec)
                    alt_now, az_now = self._relative_pointing(q_now)
                    alt_tgt, az_tgt = self._coord.altaz_now(
                        self._target.ra_deg, self._target.dec_deg)
                    v_now = _altaz_to_vec(alt_now, az_now)
                    v_tgt = _altaz_to_vec(alt_tgt, az_tgt)
                    target_sep  = _angular_dist_vec(v_now, v_tgt)
                    target_dalt = alt_tgt - alt_now
                    target_daz  = (az_tgt - az_now + 180.0) % 360.0 - 180.0
                    target_imu_only = True
                except Exception:
                    pass
        self._target_sep      = target_sep
        self._target_dalt     = target_dalt
        self._target_daz      = target_daz
        self._target_imu_only = target_imu_only

        # — Auto-resolve timer —
        if (self._autoresolve_s > 0
                and self._engine.is_aligned
                and self._engine.age_s > self._autoresolve_s
                and not self._solving):
            self._trigger_solve()

        # — Draw —
        surf.fill(p().bg)
        if self._view_surf is None or self._view_surf.get_size() != (_SCREEN_W, _VIEW_H):
            self._view_surf = pygame.Surface((_SCREEN_W, _VIEW_H))

        CV.render(
            self._view_surf,
            center_alt    = cur_alt,
            center_az     = cur_az,
            fov_deg       = self._fov_deg,
            stars         = self._loader.stars,
            constellations= self._loader.constellations,
            target        = self._search_results[self._search_selected]
                            if self._show_search and self._search_results
                            else (self._target_as_entry() if self._target else None),
            target_dalt   = target_dalt,
            target_daz    = target_daz,
            target_sep    = target_sep,
            star_altaz    = self._star_altaz,
            const_altaz   = self._const_altaz,
            is_aligned    = self._engine.is_aligned,
            night         = self._night,
        )

        surf.blit(self._view_surf, (0, _BAR_TOP))

        self._draw_top_bar(surf, cur_alt, cur_az)
        self._draw_info_bar(surf, cur_alt, cur_az)
        self._draw_action_bar(surf)

        if self._show_search:
            self._draw_search_overlay(surf)
        elif self._show_settings:
            self._draw_settings_overlay(surf)

        if self._solving:
            self._draw_solving_indicator(surf)

        # — Events —
        return self._handle_events(events, q_now)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_top_bar(self, surf, cur_alt, cur_az):
        pygame.draw.rect(surf, p().bg_bar, (0, 0, _SCREEN_W, _BAR_TOP))
        W.draw_hline(surf, _BAR_TOP)

        # Back button — far LEFT to avoid collision with MiniCam exit button (top-right)
        self._btn_back = W.draw_button(surf, "← Back", 4, 5, 70, 28, size=14)

        # Status (shifted right of Back button)
        age = self._engine.age_s
        age_str = ""
        if self._engine.is_aligned and age < 3600:
            m, s = divmod(int(age), 60)
            age_str = f"{m}:{s:02d}"
        W.draw_status(surf, self._engine.status.replace("_", " "),
                      age_str, x=82, y=10)

        # FOV
        fov_str = f"∠ {self._fov_deg:.0f}°"
        W.draw_text(surf, fov_str, _SCREEN_W // 2, 10,
                    size=15, color=p().text_val, mono=True, anchor="midtop")
        self._btn_fov_minus = W.draw_button(surf, "−",
            _SCREEN_W//2 - 80, 5, 28, 28, size=16)
        self._btn_fov_plus  = W.draw_button(surf, "+",
            _SCREEN_W//2 + 52, 5, 28, 28, size=16)

        # Right buttons (⚙ and 🌙 only — Back moved to left)
        x = _SCREEN_W - 6
        self._btn_night = W.draw_icon_btn(surf,
            "🌙" if not self._night else "☀",
            x - 32, 5, 28, font_size=14)
        x -= 36
        self._btn_settings = W.draw_icon_btn(surf, "⚙", x - 32, 5, 28, font_size=16)

    def _draw_info_bar(self, surf, cur_alt, cur_az):
        y0 = _BAR_TOP + _VIEW_H
        pygame.draw.rect(surf, p().bg_bar, (0, y0, _SCREEN_W, _BAR_INFO))
        W.draw_hline(surf, y0)

        if self._target:
            tgt = self._target
            src_icon = "⌖" if tgt.source == "stellarium" else "⌕"
            W.draw_text(surf, f"{src_icon} {tgt.code}  {tgt.name}",
                        8, y0 + 8, size=15, color=p().target)
            if self._engine.is_aligned or self._target_imu_only:
                sep, dalt, daz = self._target_sep, self._target_dalt, self._target_daz
                vals = f"Δ {sep:.1f}°   Az {daz:+.1f}°   Alt {dalt:+.1f}°"
                # Dimmer color when guidance is IMU-only (lower accuracy than plate solve)
                col = p().text_val if self._engine.is_aligned else p().text_dim
                W.draw_text(surf, vals, _SCREEN_W - 8, y0 + 8,
                            size=15, mono=True, color=col, anchor="topright")
        else:
            W.draw_text(surf, "Aucune cible — [Search] ou GoTo Stellarium",
                        8, y0 + 14, size=14, color=p().text_dim)

        W.draw_text(surf, f"Alt {cur_alt:.1f}°  Az {cur_az:.1f}°",
                    _SCREEN_W // 2, y0 + 8, size=13, mono=True,
                    color=p().text_dim, anchor="midtop")

    def _draw_action_bar(self, surf):
        y0 = _BAR_TOP + _VIEW_H + _BAR_INFO
        pygame.draw.rect(surf, p().bg_bar, (0, y0, _SCREEN_W, _BAR_ACT))
        W.draw_hline(surf, y0)
        bh = 34
        by = y0 + (_BAR_ACT - bh) // 2

        self._btn_search  = W.draw_button(surf, "⌕ Search",   8, by, 130, bh, size=15)
        self._btn_solve   = W.draw_button(surf, "↻ Re-solve",
            _SCREEN_W // 2 - 65, by, 130, bh, size=15,
            active=self._solving)
        preview_lbl = "Preview: OFF"
        self._btn_preview = W.draw_button(surf, preview_lbl,
            _SCREEN_W - 138, by, 130, bh, size=14)

    def _draw_solving_indicator(self, surf):
        msg = self._solve_msg or "Solving…"
        f   = get_font(18, bold=True)
        t   = f.render(msg, True, p().solve_flash)
        surf.blit(t, t.get_rect(center=(_SCREEN_W // 2, _BAR_TOP + _VIEW_H // 2)))

    def _draw_search_overlay(self, surf):
        ow, oh = 540, 450
        ox     = (_SCREEN_W - ow) // 2
        oy     = (_SCREEN_H - oh) // 2
        bg     = pygame.Surface((ow, oh), pygame.SRCALPHA)
        bg.fill((*p().bg_bar, 240))
        surf.blit(bg, (ox, oy))
        pygame.draw.rect(surf, p().btn_border, (ox, oy, ow, oh), 2, border_radius=6)

        # Title + close
        W.draw_text(surf, "Search target", ox + 10, oy + 8, size=17, bold=True)
        close_r = W.draw_icon_btn(surf, "✕", ox + ow - 36, oy + 6, 28)
        self._search_close_rect = close_r

        # Query field
        qy = oy + 38
        pygame.draw.rect(surf, p().btn_bg, (ox + 8, qy, ow - 16, 30), border_radius=4)
        pygame.draw.rect(surf, p().btn_border, (ox + 8, qy, ow - 16, 30), 1, border_radius=4)
        q_disp = f"  {self._search_query}|"
        W.draw_text(surf, q_disp, ox + 14, qy + 7, size=15)

        # Results (5 rows)
        ry = qy + 36
        self._search_result_rects = []
        for i, e in enumerate(self._search_results[:5]):
            row_y  = ry + i * 26
            row_bg = p().btn_hover if i == self._search_selected else p().btn_bg
            r = pygame.Rect(ox + 8, row_y, ow - 16, 24)
            pygame.draw.rect(surf, row_bg, r, border_radius=3)
            above = self._search_above_hz[i] if i < len(self._search_above_hz) else True
            mag_s = f"{e.mag:.1f}" if e.mag < 90 else "—"
            hz    = "" if above else "↓"
            row_t = f"{e.display_code:<9}{e.display_name:<24}{e.type:<14} {mag_s} {hz}"
            col   = p().target if i == self._search_selected else (
                    p().text_dim if not above else p().text)
            W.draw_text(surf, row_t.rstrip(), ox + 12, row_y + 5, size=13, color=col)
            self._search_result_rects.append(r)

        # Virtual keyboard
        kbd_y = ry + 5 * 26 + 4
        self._draw_kbd(surf, ox + 8, kbd_y, ow - 16)

        # Filter buttons row
        fy_filt = oy + oh - 64
        self._filt_rects = {}
        filters = [
            ("M",       self._filt_m),
            ("NGC",     self._filt_ngc),
            ("IC",      self._filt_ic),
            (">horiz.", self._filt_horizon),
        ]
        fw = 74
        fx = ox + 8
        for label, active in filters:
            r = W.draw_button(surf, ("✓ " if active else "  ") + label,
                              fx, fy_filt, fw, 24, active=active, size=13)
            self._filt_rects[label] = r
            fx += fw + 4

        # Cancel / Select buttons
        fy = oy + oh - 34
        self._search_cancel_rect = W.draw_button(surf, "Annuler", ox + 8, fy, 100, 28, size=14)
        self._search_select_rect = W.draw_button(
            surf, "Sélect. →", ox + ow - 114, fy, 106, 28, size=14,
            active=bool(self._search_results),
        )

    def _is_above_horizon(self, e: "CatalogEntry") -> bool:
        try:
            alt, _ = self._coord.altaz_now(e.ra_deg, e.dec_deg)
            return alt > 0.0
        except Exception:
            return True  # unknown → assume visible

    def _draw_settings_overlay(self, surf):
        ow, oh = 380, 320
        ox = (_SCREEN_W - ow) // 2
        oy = (_SCREEN_H - oh) // 2
        bg = pygame.Surface((ow, oh), pygame.SRCALPHA)
        bg.fill((*p().bg_bar, 245))
        surf.blit(bg, (ox, oy))
        pygame.draw.rect(surf, p().btn_border, (ox, oy, ow, oh), 2, border_radius=6)

        W.draw_text(surf, "Paramètres Finder", ox + 10, oy + 8, size=17, bold=True)
        self._settings_rects["close"] = W.draw_icon_btn(surf, "✕", ox + ow - 36, oy + 6, 28)
        W.draw_hline(surf, oy + 36, color=p().btn_border)

        y = oy + 46
        bw, bh = ow - 20, 34

        # Calibration IMU
        self._settings_rects["calib"] = W.draw_button(
            surf, "⊙ Calibration IMU…", ox + 10, y, bw, bh, size=15)
        y += bh + 8

        # Auto-resolve interval
        ar_lbl = f"↻ Auto-solve : {self._autoresolve_s} s" if self._autoresolve_s > 0 else "↻ Auto-solve : désactivé"
        self._settings_rects["autoresolve_dec"] = W.draw_button(
            surf, "−", ox + 10, y, 30, bh, size=16)
        self._settings_rects["autoresolve_inc"] = W.draw_button(
            surf, "+", ox + 10 + 30 + 4, y, 30, bh, size=16)
        W.draw_text(surf, ar_lbl, ox + 10 + 70, y + 9, size=14, color=p().text)
        y += bh + 8

        # FOV par défaut
        self._settings_rects["fov_dec"] = W.draw_button(
            surf, "−", ox + 10, y, 30, bh, size=16)
        self._settings_rects["fov_inc"] = W.draw_button(
            surf, "+", ox + 10 + 30 + 4, y, 30, bh, size=16)
        W.draw_text(surf, f"∠ FOV défaut : {self._fov_deg:.0f}°", ox + 10 + 70, y + 9, size=14, color=p().text)
        y += bh + 8

        # Night mode toggle
        nm_lbl = "🌙 Mode nuit : ON" if self._night else "☀ Mode nuit : OFF"
        self._settings_rects["night"] = W.draw_button(
            surf, nm_lbl, ox + 10, y, bw, bh, size=15, active=self._night)
        y += bh + 8

        # Manual alignment button
        self._settings_rects["manual_align"] = W.draw_button(
            surf, "✎ Alignement manuel…", ox + 10, y, bw, bh, size=15)

    def _draw_kbd(self, surf, x0, y0, total_w):
        rows = self._kbd_layout
        key_w = total_w // 10
        key_h = 26
        self._kbd_rects = {}
        for row_i, row in enumerate(rows):
            x_off = x0 + (total_w - len(row) * key_w) // 2
            for col_i, ch in enumerate(row):
                kx = x_off + col_i * key_w
                ky = y0 + row_i * (key_h + 3)
                r  = W.draw_button(surf, ch, kx, ky, key_w - 2, key_h, size=13)
                self._kbd_rects[ch] = r

    @staticmethod
    def _build_kbd():
        return [
            list("1234567890⌫"),
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM ↵"),
        ]

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self, events: list, q_now: list[float]) -> bool:
        for ev in events:
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if self._show_search:
                        self._show_search = False
                    elif self._show_settings:
                        self._show_settings = False
                    else:
                        return False
                elif not self._show_search and not self._show_settings:
                    if ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                        self._zoom_in()
                    elif ev.key == pygame.K_MINUS:
                        self._zoom_out()
                elif self._show_search:
                    self._handle_search_key(ev)

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if not self._handle_click(ev.pos, q_now):
                    return False
        return True

    def _handle_click(self, pos: tuple[int, int], q_now: list[float]) -> bool:
        if self._show_search:
            self._handle_search_click(pos)
            return True

        if self._show_settings:
            self._handle_settings_click(pos)
            return True

        if self._btn_back and self._btn_back.collidepoint(pos):
            return False

        if self._btn_fov_plus and self._btn_fov_plus.collidepoint(pos):
            self._zoom_in()
        elif self._btn_fov_minus and self._btn_fov_minus.collidepoint(pos):
            self._zoom_out()
        elif self._btn_night and self._btn_night.collidepoint(pos):
            self._night = not self._night
            theme.set_night(self._night)
        elif self._btn_settings and self._btn_settings.collidepoint(pos):
            self._show_settings = True
        elif self._btn_solve and self._btn_solve.collidepoint(pos):
            self._trigger_solve()
        elif self._btn_search and self._btn_search.collidepoint(pos):
            self._show_search  = True
            self._search_query = ""
            self._search_results = []

        return True

    def _handle_search_click(self, pos):
        # Close button
        if hasattr(self, "_search_close_rect") and self._search_close_rect.collidepoint(pos):
            self._show_search = False
            return
        # Cancel button
        if hasattr(self, "_search_cancel_rect") and self._search_cancel_rect.collidepoint(pos):
            self._show_search = False
            return
        # Select button
        if hasattr(self, "_search_select_rect") and self._search_select_rect.collidepoint(pos):
            self._confirm_search()
            return

        # Filter toggles
        for label, r in self._filt_rects.items():
            if r.collidepoint(pos):
                if label == "M":         self._filt_m       = not self._filt_m
                elif label == "NGC":     self._filt_ngc     = not self._filt_ngc
                elif label == "IC":      self._filt_ic      = not self._filt_ic
                elif label == ">horiz.": self._filt_horizon = not self._filt_horizon
                self._run_search()
                return

        # Virtual keyboard
        for ch, r in self._kbd_rects.items():
            if r.collidepoint(pos):
                if ch == "⌫":
                    self._search_query = self._search_query[:-1]
                elif ch == "↵":
                    self._confirm_search()
                    return
                elif ch == " ":
                    self._search_query += " "
                else:
                    self._search_query += ch
                self._run_search()
                return

        # Result rows — single click = select, double click = confirm
        for i, r in enumerate(self._search_result_rects):
            if r.collidepoint(pos):
                now = time.monotonic()
                if (i == self._last_click_result
                        and now - self._last_click_t < 0.5):
                    self._confirm_search()
                else:
                    self._search_selected = i
                    self._last_click_result = i
                    self._last_click_t = now
                return

    def _handle_settings_click(self, pos):
        r = self._settings_rects
        if r.get("close") and r["close"].collidepoint(pos):
            self._show_settings = False
        elif r.get("calib") and r["calib"].collidepoint(pos):
            self._show_settings = False
            self._calib_wizard = CalibrationWizard(
                self._imu, night_mode=self._night,
                on_done=lambda q: self._engine.set_r_calib(q.tolist()),
            )
        elif r.get("autoresolve_dec") and r["autoresolve_dec"].collidepoint(pos):
            self._autoresolve_s = max(0, self._autoresolve_s - 30)
        elif r.get("autoresolve_inc") and r["autoresolve_inc"].collidepoint(pos):
            self._autoresolve_s = min(600, self._autoresolve_s + 30)
        elif r.get("fov_dec") and r["fov_dec"].collidepoint(pos):
            self._zoom_out()
        elif r.get("fov_inc") and r["fov_inc"].collidepoint(pos):
            self._zoom_in()
        elif r.get("night") and r["night"].collidepoint(pos):
            self._night = not self._night
            theme.set_night(self._night)
        elif r.get("manual_align") and r["manual_align"].collidepoint(pos):
            self._show_settings = False
            # Manual alignment: use current IMU pointing as reference for last known RA/Dec
            if self._engine.align_ra is not None:
                q_now, _ = self._imu.get()
                self._engine.align(q_now, self._engine.align_ra, self._engine.align_dec)

    def _handle_search_key(self, ev):
        if ev.key == pygame.K_RETURN:
            self._confirm_search()
        elif ev.key == pygame.K_BACKSPACE:
            self._search_query = self._search_query[:-1]
            self._run_search()
        elif ev.key == pygame.K_DOWN:
            self._search_selected = min(self._search_selected + 1,
                                        len(self._search_results) - 1)
        elif ev.key == pygame.K_UP:
            self._search_selected = max(self._search_selected - 1, 0)
        elif ev.unicode and ev.unicode.isprintable():
            self._search_query += ev.unicode.upper()
            self._run_search()

    def _run_search(self):
        if len(self._search_query) < 1:
            return
        raw = self._loader.search(self._search_query, max_results=40)

        # Batch horizon check for the whole raw list (one astropy transform call)
        if raw:
            try:
                now   = Time.now()
                frame = AltAz(obstime=now, location=self._coord._location)
                ras   = np.array([e.ra_deg  for e in raw])
                decs  = np.array([e.dec_deg for e in raw])
                aa    = SkyCoord(ra=ras * u.deg, dec=decs * u.deg).transform_to(frame)
                raw_above = (aa.alt.deg > 0.0).tolist()
            except Exception:
                raw_above = [True] * len(raw)
        else:
            raw_above = []

        filtered:    list[CatalogEntry] = []
        filtered_hz: list[bool]         = []
        for e, above in zip(raw, raw_above):
            is_m   = bool(e.messier)
            is_ngc = (e.catalog == "NGC") and not is_m
            is_ic  = (e.catalog == "IC")
            if is_m   and not self._filt_m:   continue
            if is_ngc and not self._filt_ngc: continue
            if is_ic  and not self._filt_ic:  continue
            if self._filt_horizon and not above:   continue
            filtered.append(e)
            filtered_hz.append(above)

        self._search_results  = filtered[:5]
        self._search_above_hz = filtered_hz[:5]
        self._search_selected = 0

    def _confirm_search(self):
        if self._search_results:
            e = self._search_results[self._search_selected]
            from libastrostack.finder.target_source import target_from_catalog_entry
            self.set_target(target_from_catalog_entry(e, source="search"))
        self._show_search = False

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _trigger_solve(self):
        if self._solving:
            return
        self._solving   = True
        self._solve_msg = "Capturing…"

        hint_ra = hint_dec = None
        hint_radius = 180.0

        q_now, ts = self._imu.get()

        if self._engine.is_aligned and ts is not None:
            # Already aligned: use current IMU-estimated pointing (delta from last solve)
            # → much tighter search radius than the stale last-solve position
            pointing = self._engine.get_current_pointing(q_now)
            if pointing is not None:
                hint_ra, hint_dec = pointing[0], pointing[1]
                hint_radius = 15.0
                log.info("FinderScreen: hint IMU courant → RA=%.2f Dec=%.2f (r=15°)",
                         hint_ra, hint_dec)
            else:
                # RA/Dec cache not ready: fall back to last-solve position
                hint_ra, hint_dec = self._engine.align_ra, self._engine.align_dec
                hint_radius = 35.0

        elif self._q_calib_anchor is not None and ts is not None:
            # First solve: estimate RA/Dec from IMU + calibration anchor
            try:
                alt_est, az_est = self._relative_pointing(q_now)
                hint_ra, hint_dec = self._coord.altaz_to_radec(alt_est, az_est)
                hint_radius = 35.0
                log.info("FinderScreen: hint ancre IMU → alt=%.1f° az=%.1f°  "
                         "RA=%.2f Dec=%.2f (r=35°)", alt_est, az_est,
                         hint_ra, hint_dec)
            except Exception as exc:
                log.warning("FinderScreen: hint IMU échoué: %s", exc)

        # Fallback: use coordinates from the last MiniCam-mode solve if available
        if hint_ra is None and hasattr(self, "_ext_solve_ra"):
            hint_ra     = self._ext_solve_ra
            hint_dec    = self._ext_solve_dec
            hint_radius = 35.0
            log.info("FinderScreen: hint solve externe → RA=%.2f Dec=%.2f (r=35°)",
                     hint_ra, hint_dec)

        self._solve_hint_radius = hint_radius
        self._solver.solve_async(self._on_solve_done,
                                 hint_ra=hint_ra, hint_dec=hint_dec,
                                 hint_radius_deg=hint_radius)

    def _on_solve_done(self, result, q_imu):
        self._solving = False
        if result.success:
            self._engine.align(q_imu, result.ra_deg, result.dec_deg)
            self._solve_flash = time.monotonic()
            self._solve_msg   = f"Solved: {result.ra_deg:.2f}° {result.dec_deg:+.2f}°"
        elif self._solve_hint_radius < 180.0:
            # Tight-radius solve failed — retry full sky (new capture)
            log.info("FinderScreen: solve r=%.0f° échoué (%s) → retry full sky",
                     self._solve_hint_radius, result.error)
            self._solve_msg         = f"Retry full sky… ({result.error})"
            self._solving           = True
            self._solve_hint_radius = 180.0
            self._solver.solve_async(self._on_solve_done,
                                     hint_ra=None, hint_dec=None,
                                     hint_radius_deg=180.0)
        else:
            self._solve_msg = f"Solve failed: {result.error}"

    # ------------------------------------------------------------------
    # Sky cache
    # ------------------------------------------------------------------

    def _refresh_sky_cache(self):
        """Non-blocking: launch background thread if none is running.

        The thread updates _star_altaz / _const_altaz atomically when done.
        The UI keeps using the previous cache until the new one is ready.
        """
        if self._cache_refreshing:
            return
        self._cache_refreshing = True
        self._cache_t = time.monotonic()   # arm next trigger in 2s even if thread is slow
        threading.Thread(target=self._bg_refresh_sky_cache, daemon=True,
                         name="sky-cache").start()

    def _bg_refresh_sky_cache(self):
        try:
            loc   = self._coord._location
            now   = Time.now()
            frame = AltAz(obstime=now, location=loc)

            # Stars — single vectorised transform
            ras  = np.array([s.ra_deg  for s in self._loader.stars])
            decs = np.array([s.dec_deg for s in self._loader.stars])
            aa   = SkyCoord(ra=ras * u.deg, dec=decs * u.deg).transform_to(frame)
            star_altaz = list(zip(aa.alt.deg.tolist(), aa.az.deg.tolist()))

            # Constellations — collect all points, one transform call, then split back
            all_ra:  list[float] = []
            all_dec: list[float] = []
            struct:  list[tuple[str, list[int]]] = []   # (abbr, [seg_len, …])

            for abbr, data in self._loader.constellations.items():
                seg_lens = []
                for seg in data["lines"]:
                    seg_lens.append(len(seg))
                    for ra_d, dec_d in seg:
                        all_ra.append(ra_d)
                        all_dec.append(dec_d)
                struct.append((abbr, seg_lens))

            ca: dict = {}
            if all_ra:
                aa_c = SkyCoord(ra=np.array(all_ra) * u.deg,
                                dec=np.array(all_dec) * u.deg).transform_to(frame)
                alts = aa_c.alt.deg.tolist()
                azs  = aa_c.az.deg.tolist()
                idx  = 0
                for abbr, seg_lens in struct:
                    segs = []
                    for n in seg_lens:
                        segs.append([(alts[idx + i], azs[idx + i]) for i in range(n)])
                        idx += n
                    ca[abbr] = segs

            # Atomic swap — GIL guarantees the reference assignment is atomic
            self._star_altaz  = star_altaz
            self._const_altaz = ca
        except Exception as exc:
            log.warning("sky-cache refresh error: %s", exc)
        finally:
            self._cache_refreshing = False

    # ------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _relative_pointing(self, q_now: list) -> tuple[float, float]:
        """Calcule le pointage courant par delta IMU depuis _q_imu_ref.

        Utilisé avant le premier solve ASTAP pour montrer le mouvement de la caméra.
        Utilise la rotation monde q_cur ⊗ q_ref⁻¹ (pas body-frame) pour que
        _qrotate sur un vecteur ENU donne le bon résultat en azimut.
        """
        from libastrostack.finder.pushto_engine import (
            _qmul, _qinv, _qrotate, _altaz_to_vec, _vec_to_altaz,
        )
        import numpy as np
        q_ref = np.array(self._q_imu_ref, dtype=float)
        q_cur = np.array(q_now, dtype=float)

        q_world_rot = _qmul(q_cur, _qinv(q_ref))
        v_ref       = _altaz_to_vec(*self._center_ref)
        v_now       = _qrotate(q_world_rot, v_ref)
        return _vec_to_altaz(v_now)

    def _zoom_in(self):
        self._fov_deg = max(5.0, self._fov_deg * 0.7)

    def _zoom_out(self):
        self._fov_deg = min(180.0, self._fov_deg / 0.7)

    def _target_as_entry(self) -> Optional[CatalogEntry]:
        """Convert current Target to a minimal CatalogEntry for compass_view."""
        if self._target is None:
            return None
        e = self._loader.get_by_code(self._target.code) if self._target.code else None
        if e:
            return e
        # Manual target: fake entry
        from libastrostack.finder.data_loader import CatalogEntry
        return CatalogEntry(
            code=self._target.code or "TGT", catalog="M", messier=0,
            name=self._target.name, type=self._target.obj_type,
            ra_deg=self._target.ra_deg, dec_deg=self._target.dec_deg,
            mag=self._target.mag, const="",
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import sys
    pygame.init()
    screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H))
    pygame.display.set_caption("Finder — test standalone")
    clock  = pygame.time.Clock()

    finder = FinderScreen(
        obs_lat      = 47.312,
        obs_lon      = 0.482,
        minicam_host = "192.168.1.20:8000",
        night_mode   = True,
    )
    finder.open()

    # Pre-set a test target (M42, Orion Nebula) for visual check
    from libastrostack.finder.target_source import target_from_catalog_entry
    e = finder._loader.get_by_code("M42")
    if e:
        finder._target = target_from_catalog_entry(e)

    running = True
    while running:
        events = pygame.event.get()
        running = finder.update(screen, events)
        pygame.display.flip()
        clock.tick(30)

    finder.close()
    pygame.quit()


if __name__ == "__main__":
    main()
