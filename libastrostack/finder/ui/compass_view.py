"""Compass view — the central 460 px area of the Finder screen.

Renders on a sub-surface (1024 × 460).  All inputs are passed in at render
time so the view stays stateless (easy to test and re-use).

Gnomonic (tangent-plane) projection centred on the current pointing direction.
Distortion < 1% within ±30° of centre — more than enough for any useful FOV.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import pygame

from libastrostack.finder.ui.theme import get_font, p
from libastrostack.finder.data_loader import CatalogEntry, StarEntry

# Pre-computed star surface cache (keyed by (mag_bucket, night_mode))
_star_surf_cache: dict[tuple, pygame.Surface] = {}

_FONT_CONST = None   # loaded lazily


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _altaz_to_vec(alt_deg: float, az_deg: float) -> np.ndarray:
    a, z = math.radians(alt_deg), math.radians(az_deg)
    return np.array([math.cos(a) * math.sin(z),
                     math.cos(a) * math.cos(z),
                     math.sin(a)])


def _project(
    v_center: np.ndarray,
    v_star:   np.ndarray,
    east:     np.ndarray,
    north:    np.ndarray,
    scale:    float,      # px per radian (tangent)
    cx: int, cy: int,
) -> Optional[tuple[int, int]]:
    """Gnomonic projection of v_star onto tangent plane at v_center.

    Returns (screen_x, screen_y) or None if behind the plane.
    """
    dot = float(np.dot(v_center, v_star))
    if dot <= 0.01:
        return None
    px = float(np.dot(v_star, east))  / dot
    py = float(np.dot(v_star, north)) / dot
    # Negate px: east vector is up×v_center = West in ENU; negating gives
    # the correct camera-view convention where East appears on the right.
    return int(cx - px * scale), int(cy - py * scale)


def _east_north(v_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute East and North unit vectors at v_center in ENU frame."""
    up = np.array([0.0, 0.0, 1.0])
    east = np.cross(up, v_center)
    norm = np.linalg.norm(east)
    if norm < 1e-6:
        east = np.array([1.0, 0.0, 0.0])
    else:
        east /= norm
    north = np.cross(v_center, east)
    north /= np.linalg.norm(north)
    return east, north


# ---------------------------------------------------------------------------
# Star dot surface (pre-rendered for performance)
# ---------------------------------------------------------------------------

def _star_surf(mag: float, night: bool) -> pygame.Surface:
    bucket = int(mag * 2) / 2   # 0.5-mag buckets
    key    = (bucket, night)
    if key in _star_surf_cache:
        return _star_surf_cache[key]

    radius = max(1, int(4.5 - min(mag, 4.0) * 0.8))
    size   = radius * 2 + 2
    surf   = pygame.Surface((size, size), pygame.SRCALPHA)
    col    = p().star_b if mag < 2.0 else p().star_d
    alpha  = max(80, 255 - int(max(0, mag - 1.5) * 50))
    pygame.draw.circle(surf, (*col, alpha), (radius + 1, radius + 1), radius)
    _star_surf_cache[key] = surf
    return surf


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(
    surf:          pygame.Surface,
    center_alt:    float,
    center_az:     float,
    fov_deg:       float,
    stars:         list[StarEntry],
    constellations: dict,
    target:        Optional[CatalogEntry],
    target_dalt:   float,
    target_daz:    float,
    target_sep:    float,
    star_altaz:    Optional[list[tuple[float, float]]],  # pre-computed, same order as stars
    const_altaz:   Optional[dict],                        # abbr → list of [(alt,az),(alt,az),…] segments
    is_aligned:    bool,
    night:         bool,
) -> None:
    """Draw everything onto surf.

    star_altaz / const_altaz are pre-computed by FinderScreen (cached, 2s refresh).
    If None, star/constellation rendering is skipped (first frame).
    """
    global _FONT_CONST
    if _FONT_CONST is None:
        _FONT_CONST = get_font(12)

    w, h   = surf.get_size()
    cx, cy = w // 2, h // 2

    # Scale: how many pixels per radian in gnomonic projection
    scale = (h / 2) / math.tan(math.radians(fov_deg / 2))

    v_center     = _altaz_to_vec(center_alt, center_az)
    east, north  = _east_north(v_center)

    # — Background —
    surf.fill(p().bg)

    # — Horizon line (alt = 0) —
    _draw_horizon(surf, v_center, east, north, scale, cx, cy, w)

    # — Constellation lines —
    if const_altaz:
        _draw_constellations(surf, const_altaz, v_center, east, north, scale, cx, cy, w, h)

    # — Stars —
    if star_altaz:
        _draw_stars(surf, stars, star_altaz, v_center, east, north, scale, cx, cy, w, h, night)

    # — Tolerance circles —
    _draw_circles(surf, scale, cx, cy, fov_deg)

    # — Crosshair (current pointing) —
    _draw_crosshair(surf, cx, cy, is_aligned)

    # — Target —
    if target is not None:
        tgt_alt = center_alt + target_dalt
        tgt_az  = (center_az  + target_daz) % 360.0
        _draw_target(surf, target, target_dalt, target_daz, target_sep,
                     tgt_alt, tgt_az,
                     v_center, east, north, scale, cx, cy, w, h)

    # — Scale indicator (bottom-left) —
    _draw_scale(surf, fov_deg, w, h, scale)


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _draw_horizon(surf, v_center, east, north, scale, cx, cy, w):
    """Draw the horizon as a thick line."""
    # Horizon = alt 0, sweep azimuth
    pts = []
    for az_d in range(0, 361, 5):
        v = _altaz_to_vec(0.0, float(az_d))
        p2 = _project(v_center, v, east, north, scale, cx, cy)
        if p2:
            pts.append(p2)
    if len(pts) > 2:
        pygame.draw.lines(surf, p().horizon, False, pts, 2)


def _draw_constellations(surf, const_altaz, v_center, east, north, scale, cx, cy, w, h):
    margin = 50
    clip   = pygame.Rect(-margin, -margin, w + 2*margin, h + 2*margin)
    col    = p().const_line
    for segments in const_altaz.values():
        for seg in segments:
            if len(seg) < 2:
                continue
            pts = []
            for alt, az in seg:
                v = _altaz_to_vec(alt, az)
                pp = _project(v_center, v, east, north, scale, cx, cy)
                pts.append(pp)
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i+1]
                if p1 and p2 and (clip.collidepoint(p1) or clip.collidepoint(p2)):
                    pygame.draw.line(surf, col, p1, p2, 1)


def _draw_stars(surf, stars, star_altaz, v_center, east, north, scale, cx, cy, w, h, night):
    margin = 20
    for i, (alt, az) in enumerate(star_altaz):
        if alt < -5.0:
            continue
        v  = _altaz_to_vec(alt, az)
        pp = _project(v_center, v, east, north, scale, cx, cy)
        if pp is None:
            continue
        sx, sy = pp
        if -margin <= sx <= w + margin and -margin <= sy <= h + margin:
            ss = _star_surf(stars[i].mag, night)
            r  = ss.get_width() // 2
            surf.blit(ss, (sx - r, sy - r))
            # Labels: all named stars up to mag 4
            if stars[i].mag < 4.0 and stars[i].name:
                fsize = 11 if stars[i].mag < 2.0 else 9
                col   = p().text_val if stars[i].mag < 2.0 else p().text_dim
                f = get_font(fsize)
                t = f.render(stars[i].name, True, col)
                surf.blit(t, (sx + r + 2, sy - 5))


def _draw_circles(surf, scale, cx, cy, fov_deg):
    """Tolerance rings: 5°, 1°, 0.2°."""
    col = p().circle_tol
    for angle_deg in [5.0, 1.0, 0.2]:
        r_px = int(scale * math.tan(math.radians(angle_deg)))
        if 4 < r_px < min(cx, cy) - 2:
            pygame.draw.circle(surf, col, (cx, cy), r_px, 1)


def _draw_crosshair(surf, cx, cy, is_aligned):
    col  = p().status_ok if is_aligned else p().crosshair
    size = 18
    t    = 2
    pygame.draw.line(surf, col, (cx - size, cy), (cx + size, cy), t)
    pygame.draw.line(surf, col, (cx, cy - size), (cx, cy + size), t)
    pygame.draw.circle(surf, col, (cx, cy), 6, 1)


def _draw_target(surf, target, dalt, daz, sep,
                 tgt_alt, tgt_az,
                 v_center, east, north, scale, cx, cy, w, h):
    # Arrow from screen centre toward target (when off-centre)
    _draw_arrow(surf, dalt, daz, sep, cx, cy)

    # Cross marker at the projected sky position of the target
    col = p().target
    v_tgt = _altaz_to_vec(tgt_alt, tgt_az)
    pp = _project(v_center, v_tgt, east, north, scale, cx, cy)
    if pp is not None:
        tx, ty = pp
        if -30 <= tx <= w + 30 and -30 <= ty <= h + 30:
            arm = 9
            lw  = 2
            pygame.draw.line(surf, col, (tx - arm, ty), (tx + arm, ty), lw)
            pygame.draw.line(surf, col, (tx, ty - arm), (tx, ty + arm), lw)
            pygame.draw.circle(surf, col, (tx, ty), arm + 3, 1)
            # Name label next to cross
            name = target.display_name or target.display_code
            f = get_font(12)
            t = f.render(name, True, col)
            surf.blit(t, (tx + arm + 4, ty - t.get_height() // 2))

    # Bottom bar: always visible (fallback when target is off-screen)
    label = f"⌖ {target.display_code}  {target.display_name}"
    font  = get_font(14)
    t     = font.render(label, True, col)
    surf.blit(t, t.get_rect(centerx=cx, bottom=h - 4))


def _draw_arrow(surf, dalt, daz, sep, cx, cy):
    """Directional arrow from centre toward target, size ∝ angular distance."""
    if sep < 0.15:
        # In tolerance: pulsing green ring
        t = time.monotonic()
        alpha = int(128 + 127 * math.sin(t * 4))
        a_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(a_surf, (*p().status_ok, alpha), (20, 20), 18, 3)
        surf.blit(a_surf, (cx - 20, cy - 20))
        return

    # Arrow direction vector (screen coords: +x=East, -y=up/North)
    angle_rad = math.atan2(daz, dalt)   # angle from "up" (North/altitude)
    dx =  math.sin(angle_rad)
    dy = -math.cos(angle_rad)

    # Arrow length: 30–120 px, proportional to log(sep)
    length = max(30, min(120, int(40 * math.log1p(sep * 2))))
    ax = cx + dx * length
    ay = cy + dy * length

    col = p().arrow
    pygame.draw.line(surf, col, (cx, cy), (int(ax), int(ay)), 3)
    # Arrowhead
    head = 12
    angle_head = math.atan2(ay - cy, ax - cx)
    for side in [0.4, -0.4]:
        hx = ax - head * math.cos(angle_head - side)
        hy = ay - head * math.sin(angle_head - side)
        pygame.draw.line(surf, col, (int(ax), int(ay)), (int(hx), int(hy)), 3)

    # Distance label near arrow tip
    f = get_font(13, mono=True)
    dist_str = f"{sep:.1f}°" if sep >= 1.0 else f"{sep*60:.0f}'"
    t = f.render(dist_str, True, p().text_val)
    surf.blit(t, (int(ax) + 6, int(ay) - 8))


def _draw_scale(surf, fov_deg, w, h, scale):
    """Draw a 1° scale bar in the bottom-left corner."""
    bar_deg = 1.0 if fov_deg > 3 else 0.2
    bar_px  = int(scale * math.tan(math.radians(bar_deg)))
    if bar_px < 3:
        return
    x0, y0 = 12, h - 14
    pygame.draw.line(surf, p().text_dim, (x0, y0), (x0 + bar_px, y0), 2)
    pygame.draw.line(surf, p().text_dim, (x0, y0 - 3), (x0, y0 + 3), 2)
    pygame.draw.line(surf, p().text_dim, (x0 + bar_px, y0 - 3), (x0 + bar_px, y0 + 3), 2)
    f = get_font(11)
    lbl = f"{bar_deg:.1f}°" if bar_deg >= 1 else f"{int(bar_deg*60)}'"
    t = f.render(lbl, True, p().text_dim)
    surf.blit(t, (x0 + bar_px // 2 - t.get_width() // 2, y0 - 14))
