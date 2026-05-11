"""
Polar visualization widget for pygame.

Shows NCP (center), Polaris (rotating), and the measured axis on a polar chart.
Astronomical orientation: North up, East left.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import pygame

from .polar_alignment import (
    AlignmentResult, polaris_position, polaris_hour_angle_deg, error_color,
)


_C_BG        = (10,  12,  22)
_C_GRID      = (40,  50,  70)
_C_POLE      = (200, 220, 255)
_C_POLARIS   = (255, 220, 80)
_C_AXIS_OLD  = (80,  90, 110)
_C_SCALE_BAR = (130, 150, 180)
_C_TEXT      = (160, 180, 210)


class PolarVisualizationWidget:
    """Draws the polar alignment chart on a sub-rect of a pygame surface.

    Call draw() every frame (or at least every 10 s to refresh Polaris rotation).
    """

    def __init__(self, rect: pygame.Rect, observer_lat: float, observer_lon: float) -> None:
        self.rect = rect
        self.observer_lat = observer_lat
        self.observer_lon = observer_lon
        self.full_scale_arcmin = 90.0   # half-width of chart

    def set_scale(self, full_scale_arcmin: float) -> None:
        self.full_scale_arcmin = max(10.0, min(180.0, full_scale_arcmin))

    def _scale(self) -> float:
        """Pixels per arcmin."""
        half = min(self.rect.w, self.rect.h) / 2.0
        return half / self.full_scale_arcmin

    def draw(self, surface: pygame.Surface,
             result: Optional[AlignmentResult],
             t_utc: Optional[datetime] = None,
             history: Optional[list] = None) -> None:
        if t_utc is None:
            t_utc = datetime.now(timezone.utc)

        cx = self.rect.centerx
        cy = self.rect.centery
        sc = self._scale()

        # Background
        pygame.draw.rect(surface, _C_BG, self.rect)
        pygame.draw.rect(surface, _C_GRID, self.rect, 1)

        # Grid circles at 5', 15', 30', 60'
        for r_am in (5, 15, 30, 60, 90):
            r_px = int(r_am * sc)
            if r_px > min(self.rect.w, self.rect.h) // 2:
                break
            pygame.draw.circle(surface, _C_GRID, (cx, cy), r_px, 1)

        # Cardinal labels (N up, E left)
        self._draw_cardinals(surface, cx, cy, sc)

        # Scale bar (bottom of widget)
        self._draw_scale_bar(surface, sc)

        # Historical axis positions (greyed out)
        if history:
            for old in history[:-1]:
                self._draw_axis_point(surface, cx, cy, sc, old, _C_AXIS_OLD, radius=4)

        # Polaris — rotating with sky
        self._draw_polaris(surface, cx, cy, sc, t_utc)

        # Pole cross (always at center)
        self._draw_pole_cross(surface, cx, cy)

        # Current measured axis
        if result is not None:
            col = error_color(result.total_error_arcmin)
            self._draw_axis_point(surface, cx, cy, sc, result, col, radius=6)
            self._draw_axis_arrow(surface, cx, cy, sc, result, col)

    def _px(self, cx: int, cy: int, sc: float,
            daz_am: float, dalt_am: float,
            lat_rad: float | None = None) -> tuple[int, int]:
        """Convert (ΔAz arcmin, ΔAlt arcmin) offset from pole to pixel coords.

        East is left (negative x), North is up (negative y).
        cos(lat) factor applied to east-west for proper angular scale.
        """
        if lat_rad is None:
            lat_rad = math.radians(self.observer_lat)
        x = cx - int(daz_am * math.cos(lat_rad) * sc)
        y = cy - int(dalt_am * sc)
        return x, y

    def _draw_cardinals(self, surface: pygame.Surface,
                        cx: int, cy: int, sc: float) -> None:
        fnt = pygame.font.Font(None, 16)
        r   = int(self.full_scale_arcmin * sc * 0.88)
        for label, dx, dy in (("N", 0, -1), ("S", 0, 1), ("E", -1, 0), ("O", 1, 0)):
            tx = cx + dx * r
            ty = cy + dy * r
            t  = fnt.render(label, True, _C_GRID)
            surface.blit(t, t.get_rect(center=(tx, ty)))

    def _draw_scale_bar(self, surface: pygame.Surface, sc: float) -> None:
        # 10 arcmin bar at the bottom of the widget
        bar_am = 10
        bar_px = int(bar_am * sc)
        bx = self.rect.x + 10
        by = self.rect.bottom - 14
        pygame.draw.line(surface, _C_SCALE_BAR, (bx, by), (bx + bar_px, by), 2)
        pygame.draw.line(surface, _C_SCALE_BAR, (bx, by - 3), (bx, by + 3), 1)
        pygame.draw.line(surface, _C_SCALE_BAR, (bx + bar_px, by - 3), (bx + bar_px, by + 3), 1)
        fnt = pygame.font.Font(None, 14)
        lbl = fnt.render("10'", True, _C_SCALE_BAR)
        surface.blit(lbl, (bx + bar_px + 4, by - 6))

    def _draw_pole_cross(self, surface: pygame.Surface, cx: int, cy: int) -> None:
        s = 8
        pygame.draw.line(surface, _C_POLE, (cx - s, cy), (cx + s, cy), 2)
        pygame.draw.line(surface, _C_POLE, (cx, cy - s), (cx, cy + s), 2)

    def _draw_polaris(self, surface: pygame.Surface,
                      cx: int, cy: int, sc: float, t_utc: datetime) -> None:
        _, dec_p = polaris_position(t_utc)
        ha_deg   = polaris_hour_angle_deg(t_utc, self.observer_lon)
        ha_rad   = math.radians(ha_deg)
        rho_am   = (90.0 - dec_p) * 60.0   # angular distance from pole in arcmin

        # Position on widget: HA=0 → top, HA=90 → right (west)
        # East is left, so: x = +sin(HA) (west=right=+), y = -cos(HA) (top=-)
        px = cx + int(rho_am * sc * math.sin(ha_rad))
        py = cy - int(rho_am * sc * math.cos(ha_rad))

        # Dashed reference circle for Polaris orbit
        self._draw_dashed_circle(surface, cx, cy, int(rho_am * sc), _C_POLARIS, alpha=80)

        # 5-branch star symbol
        self._draw_star(surface, px, py, 6, _C_POLARIS)

        # Label
        fnt = pygame.font.Font(None, 15)
        lbl = fnt.render("Polaris", True, _C_POLARIS)
        surface.blit(lbl, (px + 8, py - 5))

    def _draw_axis_point(self, surface: pygame.Surface,
                         cx: int, cy: int, sc: float,
                         result: AlignmentResult,
                         color: tuple, radius: int = 6) -> None:
        lat_rad = math.radians(self.observer_lat)
        x, y = self._px(cx, cy, sc, result.delta_az_arcmin, result.delta_alt_arcmin, lat_rad)
        if self.rect.collidepoint(x, y):
            pygame.draw.circle(surface, color, (x, y), radius)
            pygame.draw.circle(surface, (255, 255, 255), (x, y), radius, 1)

    def _draw_axis_arrow(self, surface: pygame.Surface,
                         cx: int, cy: int, sc: float,
                         result: AlignmentResult,
                         color: tuple) -> None:
        lat_rad = math.radians(self.observer_lat)
        ax, ay = self._px(cx, cy, sc, result.delta_az_arcmin, result.delta_alt_arcmin, lat_rad)
        dx = cx - ax
        dy = cy - ay
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 4:
            return
        # Arrow from axis point toward pole, stopping 8px from pole center
        t = max(0.0, (dist - 8) / dist)
        ex, ey = int(ax + dx * t), int(ay + dy * t)
        pygame.draw.line(surface, color, (ax, ay), (ex, ey), 2)
        # Arrowhead
        angle = math.atan2(dy, dx)
        for da in (0.4, -0.4):
            hx = ex - int(10 * math.cos(angle + da))
            hy = ey - int(10 * math.sin(angle + da))
            pygame.draw.line(surface, color, (ex, ey), (hx, hy), 2)

        # Distance label along the arrow
        fnt  = pygame.font.Font(None, 15)
        dist_am = result.total_error_arcmin
        lbl  = fnt.render(f"{dist_am:.1f}'", True, color)
        mx   = (ax + ex) // 2 + 6
        my   = (ay + ey) // 2 - 10
        surface.blit(lbl, (mx, my))

    @staticmethod
    def _draw_star(surface: pygame.Surface, x: int, y: int,
                   r: int, color: tuple) -> None:
        """Draw a 5-branch star."""
        for i in range(5):
            angle = math.radians(i * 72 - 90)
            outer_x = x + int(r * math.cos(angle))
            outer_y = y + int(r * math.sin(angle))
            inner_angle = angle + math.radians(36)
            inner_x = x + int(r * 0.4 * math.cos(inner_angle))
            inner_y = y + int(r * 0.4 * math.sin(inner_angle))
            pygame.draw.line(surface, color, (x, y), (outer_x, outer_y), 2)
        pygame.draw.circle(surface, color, (x, y), 3)

    @staticmethod
    def _draw_dashed_circle(surface: pygame.Surface, cx: int, cy: int,
                             r: int, color: tuple, alpha: int = 120) -> None:
        """Draw a dashed circle."""
        n_segs = 36
        for i in range(0, n_segs, 2):
            a0 = 2 * math.pi * i / n_segs
            a1 = 2 * math.pi * (i + 1) / n_segs
            x0 = cx + int(r * math.cos(a0))
            y0 = cy + int(r * math.sin(a0))
            x1 = cx + int(r * math.cos(a1))
            y1 = cy + int(r * math.sin(a1))
            c  = tuple(int(v * alpha / 255) for v in color)
            pygame.draw.line(surface, c, (x0, y0), (x1, y1), 1)
