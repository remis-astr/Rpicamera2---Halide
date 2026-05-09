"""Calibration wizard — Pygame UI for the 3-pose IMU calibration.

Displays one pose at a time:
  - Instructions (what to point at)
  - Live IMU attitude: α (azimuth) / β (elevation) / γ (roll)
  - Stability bar (gyro variance over the last 30 samples)
  - [Capturer] button (enabled when stable)
  - Progress indicator (pose 1/2/3)

After all 3 poses are captured it calls compute_r_calib(), saves the result,
and invokes an optional callback(q_calib).

Usage (standalone for testing)::
    from libastrostack.finder.ui.calibration_wizard import CalibrationWizard
    wiz = CalibrationWizard(imu_client, night_mode=True)
    wiz.run()          # blocking Pygame event loop (opens its own window)

Usage (embedded in FinderScreen)::
    wiz = CalibrationWizard(imu_client, night_mode=True,
                            on_done=lambda q: engine.set_r_calib(q))
    # In the main event loop:
    still_running = wiz.update(surf, events)
"""
from __future__ import annotations

import collections
import logging
import math
import time
from typing import Callable, Optional

import numpy as np
import pygame

log = logging.getLogger(__name__)

from libastrostack.finder.calibration import (
    POSES, PoseCapture, compute_r_calib, save_calibration
)
from libastrostack.finder.ui.theme import get_font, p, set_night
from libastrostack.finder.ui.widgets import draw_button, draw_text

# ---------------------------------------------------------------------------
# Stability parameters
# ---------------------------------------------------------------------------
_SAMPLE_WINDOW   = 40       # samples kept for variance calculation
_STABLE_VAR_THR  = 0.02     # ~8° entre deux échantillons — coarse, affiné par ASTAP
_CAPTURE_SAMPLES = 15       # samples averaged for each capture
_LOG_INTERVAL_S  = 2.0      # intervalle des logs console


class CalibrationWizard:
    """Three-step calibration wizard (embedded or standalone)."""

    def __init__(
        self,
        imu_client,
        night_mode: bool = True,
        on_done: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._imu      = imu_client
        self._on_done  = on_done
        self._night    = night_mode
        set_night(night_mode)

        self._step      = 0           # current pose index (0-2)
        self._captures: list[PoseCapture] = []
        self._cur_cap:  Optional[PoseCapture] = None   # accumulating

        self._q_history: collections.deque[list[float]] = collections.deque(
            maxlen=_SAMPLE_WINDOW
        )
        # We track angular velocity proxy: deviation of last q from mean
        self._gyro_buf: collections.deque[float] = collections.deque(
            maxlen=_SAMPLE_WINDOW
        )

        self._last_q:   Optional[list[float]] = None
        self._last_ts:  Optional[int]         = None
        self._status_msg: str = ""
        self._done      = False
        self._error_msg: str = ""
        self._last_log_t:  float = 0.0
        self._last_stable: Optional[bool] = None   # pour logguer les transitions

        # rects for hit-testing
        self._btn_capture: Optional[pygame.Rect] = None
        self._btn_force:   Optional[pygame.Rect] = None
        self._btn_skip:    Optional[pygame.Rect] = None
        self._btn_cancel:  Optional[pygame.Rect] = None

        self._flash_until: float = 0.0   # timestamp until capture flash shown

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self._done

    def update(self, surf: pygame.Surface, events: list) -> bool:
        """Draw + handle events.  Returns False when wizard should close."""
        self._poll_imu()
        self._draw(surf)
        return self._handle_events(events)

    def run(self) -> None:
        """Blocking standalone Pygame loop (opens its own 640×480 window)."""
        pygame.init()
        screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Calibration IMU")
        clock = pygame.time.Clock()
        running = True
        while running:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    running = False
            if not self.update(screen, events):
                running = False
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    # ------------------------------------------------------------------
    # IMU polling
    # ------------------------------------------------------------------

    def _poll_imu(self) -> None:
        q, ts = self._imu.get()
        if ts is None:
            now = time.monotonic()
            if now - self._last_log_t > _LOG_INTERVAL_S:
                print("[CalibWizard] IMU non connecté (ts=None)")
                self._last_log_t = now
            return
        if ts == self._last_ts:
            return
        self._last_q  = q
        self._last_ts = ts
        self._q_history.append(q)

        # Angular velocity proxy: angle between successive quaternions
        if len(self._q_history) >= 2:
            q1 = np.array(self._q_history[-2], dtype=float)
            q2 = np.array(self._q_history[-1], dtype=float)
            dot = abs(float(np.dot(q1, q2)))
            dot = min(dot, 1.0)
            angle = 2.0 * math.acos(dot)   # radians
            self._gyro_buf.append(angle * angle)

        # Log périodique : variance courante et seuil
        now = time.monotonic()
        if now - self._last_log_t > _LOG_INTERVAL_S:
            var       = self._gyro_variance()
            angle_deg = math.degrees(math.sqrt(var)) if var > 0 else 0.0
            thr_deg   = math.degrees(math.sqrt(_STABLE_VAR_THR))
            stable    = var < _STABLE_VAR_THR
            print(f"[CalibWizard] pose={self._step}  mouvement={angle_deg:.2f}°/sample"
                  f"  seuil={thr_deg:.2f}°  {'STABLE ✓' if stable else 'instable'}  buf={len(self._gyro_buf)}")
            if self._last_stable is not None and stable != self._last_stable:
                print(f"[CalibWizard] → {'STABLE ✓' if stable else 'instable'}")
            self._last_stable = stable
            self._last_log_t  = now

        # Accumulate capture samples if in progress
        if self._cur_cap is not None:
            self._cur_cap.q_samples.append(list(q))
            if len(self._cur_cap.q_samples) >= _CAPTURE_SAMPLES:
                self._finish_capture()

    def _gyro_variance(self) -> float:
        if len(self._gyro_buf) < 5:
            return 1.0   # pas encore assez d'échantillons → instable par défaut
        return float(np.mean(list(self._gyro_buf)))

    @property
    def _is_stable(self) -> bool:
        return self._gyro_variance() < _STABLE_VAR_THR

    # ------------------------------------------------------------------
    # Capture state machine
    # ------------------------------------------------------------------

    def _start_capture(self) -> None:
        var = self._gyro_variance()
        deg = math.degrees(math.sqrt(var)) if var > 0 else 0.0
        print(f"[CalibWizard] Capture démarrée — pose={self._step}  mouvement={deg:.2f}°/sample")
        self._cur_cap = PoseCapture(pose_index=self._step)
        self._status_msg = "Acquisition…"

    def _finish_capture(self) -> None:
        cap = self._cur_cap
        self._cur_cap  = None
        self._captures.append(cap)
        self._flash_until = time.monotonic() + 0.4
        qm = [round(v, 4) for v in (cap.q_mean.tolist() if cap.q_mean is not None else [])]
        print(f"[CalibWizard] Pose {self._step} capturée — {len(cap.q_samples)} échantillons  q_mean={qm}")
        self._status_msg = f"Pose {self._step + 1} enregistrée ✓"

        self._step += 1
        if self._step >= len(POSES):
            self._compute_and_save()

    def _compute_and_save(self) -> None:
        try:
            print(f"[CalibWizard] Calcul R_calib sur {len(self._captures)} poses…")
            q_calib = compute_r_calib(self._captures)
            path    = save_calibration(q_calib, self._captures)
            print(f"[CalibWizard] R_calib = {[round(v,5) for v in q_calib]}  → {path}")
            self._status_msg = f"Calibration sauvegardée"
            self._done = True
            if self._on_done:
                self._on_done(q_calib)
        except Exception as exc:
            import traceback
            print(f"[CalibWizard] ERREUR compute_r_calib: {exc}")
            traceback.print_exc()
            self._error_msg  = f"Erreur: {exc}"
            self._status_msg = "Calibration échouée — recommencez"
            self._step       = 0
            self._captures.clear()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, surf: pygame.Surface) -> None:
        w, h = surf.get_size()
        pal  = p()

        # Flash on capture
        if time.monotonic() < self._flash_until:
            surf.fill((60, 40, 10))
        else:
            surf.fill(pal.bg)

        if self._done:
            self._draw_done(surf, w, h)
            return
        if self._error_msg:
            self._draw_error(surf, w, h)
            return

        pose = POSES[self._step]
        y = 18

        # Title
        draw_text(surf, f"Calibration IMU — {pose.title}",
                  w // 2, y, size=18, bold=True, anchor="midtop")
        y += 32

        # Progress dots
        self._draw_progress(surf, w, y)
        y += 30

        # Instructions
        for line in pose.instruction.split("\n"):
            draw_text(surf, line, w // 2, y, size=15, anchor="midtop",
                      color=pal.text)
            y += 22
        y += 8

        # IMU live readout
        self._draw_imu_readout(surf, w, y)
        y += 60

        # Stability bar
        stable = self._is_stable
        self._draw_stability(surf, w, y, stable)
        y += 40

        # Status message
        if self._status_msg:
            col = pal.status_ok if "✓" in self._status_msg else pal.text_dim
            draw_text(surf, self._status_msg, w // 2, y, size=13,
                      color=col, anchor="midtop")
        y += 24

        # Capture button
        bw, bh = 180, 44
        bx, by = w // 2 - bw // 2, h - bh - 50
        capturing = self._cur_cap is not None
        enabled   = stable and not capturing

        self._btn_capture = draw_button(
            surf, "Capturer" if not capturing else "Acquisition…",
            bx, by, bw, bh,
            active     = capturing,
            color_bg   = pal.btn_hover if enabled else pal.btn_bg,
            color_text = pal.status_ok if enabled else pal.text_dim,
            size       = 17,
        )

        # Bouton "Forcer" si pas stable (bypass du seuil)
        self._btn_force = None
        if not stable and not capturing:
            self._btn_force = draw_button(
                surf, "Forcer ⚠", bx + bw + 8, by + 6, 90, bh - 12,
                size=13, color_text=(200, 160, 40),
            )

        # Cancel button (bottom-left)
        self._btn_cancel = draw_button(
            surf, "Annuler", 12, h - bh - 10, 110, bh - 10, size=14
        )

    def _draw_progress(self, surf: pygame.Surface, w: int, y: int) -> None:
        pal   = p()
        total = len(POSES)
        r     = 9
        gap   = 32
        x0    = w // 2 - (total - 1) * gap // 2
        for i in range(total):
            cx = x0 + i * gap
            if i < self._step:
                col = pal.status_ok
                pygame.draw.circle(surf, col, (cx, y), r)
            elif i == self._step:
                col = pal.target
                pygame.draw.circle(surf, col, (cx, y), r, 2)
                pygame.draw.circle(surf, col, (cx, y), r - 4)
            else:
                col = pal.btn_border
                pygame.draw.circle(surf, col, (cx, y), r, 2)

    def _draw_imu_readout(self, surf: pygame.Surface, w: int, y: int) -> None:
        pal = p()
        q   = self._last_q
        if q is None:
            draw_text(surf, "IMU non connecté", w // 2, y, size=14,
                      color=pal.status_err, anchor="midtop")
            return

        # Convert quaternion → Euler (ZYX: yaw=α, pitch=β, roll=γ)
        qw, qx, qy, qz = q
        # Roll (x-axis rotation)
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.degrees(math.atan2(sinr, cosr))
        # Pitch (y-axis)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.degrees(math.asin(sinp))
        # Yaw (z-axis)
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw  = math.degrees(math.atan2(siny, cosy))

        f   = get_font(15, mono=True)
        txt = f"Az {yaw:+7.2f}°   El {pitch:+7.2f}°   Roll {roll:+7.2f}°"
        t   = f.render(txt, True, pal.text_val)
        surf.blit(t, t.get_rect(centerx=w // 2, top=y))

        # Accumulation progress bar
        if self._cur_cap is not None:
            filled = len(self._cur_cap.q_samples)
            bw     = 240
            bx     = w // 2 - bw // 2
            by     = y + 28
            pygame.draw.rect(surf, pal.btn_bg,     (bx, by, bw, 10), border_radius=3)
            prog   = int(bw * filled / _CAPTURE_SAMPLES)
            pygame.draw.rect(surf, pal.status_ok,  (bx, by, prog, 10), border_radius=3)
            pygame.draw.rect(surf, pal.btn_border, (bx, by, bw, 10), 1, border_radius=3)

    def _draw_stability(self, surf: pygame.Surface, w: int, y: int, stable: bool) -> None:
        pal   = p()
        bw    = 200
        bx    = w // 2 - bw // 2

        var       = self._gyro_variance()
        angle_deg = math.degrees(math.sqrt(var)) if var > 0 else 0.0
        thr_deg   = math.degrees(math.sqrt(_STABLE_VAR_THR))

        label = f"Stabilité  {angle_deg:.2f}° / seuil {thr_deg:.2f}°"
        draw_text(surf, label, w // 2, y - 2, size=12, color=pal.text_dim, anchor="midtop")

        # Bar: variance mapped to 0-1 (0=very stable → bar full)
        ratio = min(1.0, var / _STABLE_VAR_THR)
        bar_w = int(bw * (1.0 - ratio))

        pygame.draw.rect(surf, pal.btn_bg,    (bx, y + 14, bw, 14), border_radius=4)
        col = pal.status_ok if stable else pal.arrow
        if bar_w > 0:
            pygame.draw.rect(surf, col,       (bx, y + 14, bar_w, 14), border_radius=4)
        pygame.draw.rect(surf, pal.btn_border,(bx, y + 14, bw, 14), 1, border_radius=4)

        status = "STABLE ✓" if stable else "Immobilisez le tube…"
        draw_text(surf, status, bx + bw + 8, y + 16, size=13,
                  color=pal.status_ok if stable else pal.text_dim)

    def _draw_done(self, surf: pygame.Surface, w: int, h: int) -> None:
        pal = p()
        draw_text(surf, "Calibration terminée", w // 2, h // 2 - 50,
                  size=22, bold=True, color=pal.status_ok, anchor="midtop")
        draw_text(surf, "R_calib sauvegardé.", w // 2, h // 2 - 12,
                  size=15, color=pal.text, anchor="midtop")
        draw_text(surf, "Vous pointez vers le Nord — c'est votre référence.", w // 2, h // 2 + 16,
                  size=13, color=pal.target, anchor="midtop")
        draw_text(surf, "Fermez cette fenêtre pour ouvrir le Finder.", w // 2, h // 2 + 40,
                  size=13, color=pal.text_dim, anchor="midtop")

    def _draw_error(self, surf: pygame.Surface, w: int, h: int) -> None:
        pal = p()
        draw_text(surf, "Erreur de calibration", w // 2, h // 2 - 30,
                  size=18, bold=True, color=pal.status_err, anchor="midtop")
        draw_text(surf, self._error_msg, w // 2, h // 2 + 2,
                  size=13, color=pal.text_dim, anchor="midtop")
        bw, bh = 180, 40
        self._btn_cancel = draw_button(
            surf, "Réessayer", w // 2 - bw // 2, h // 2 + 50, bw, bh, size=15
        )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self, events: list) -> bool:
        for event in events:
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self._error_msg:
                    # "Retry" acts as Cancel → reset
                    if self._btn_cancel and self._btn_cancel.collidepoint(pos):
                        self._error_msg = ""
                    return True
                if self._btn_cancel and self._btn_cancel.collidepoint(pos):
                    return False   # caller interprets False as "close"
                if (self._btn_capture and self._btn_capture.collidepoint(pos)
                        and self._is_stable and self._cur_cap is None):
                    self._start_capture()
                elif (self._btn_force and self._btn_force.collidepoint(pos)
                        and self._cur_cap is None):
                    print(f"[CalibWizard] ⚠ Capture FORCÉE (seuil ignoré) — pose={self._step}  var={self._gyro_variance():.5f}")
                    self._start_capture()
        return not self._done
