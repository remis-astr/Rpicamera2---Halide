"""Reusable UI widgets for the Finder screen.

All draw functions take a pygame Surface and return a pygame.Rect (bounding box)
so callers can do hit-testing in the event loop.
"""
from __future__ import annotations

import pygame
from libastrostack.finder.ui.theme import get_font, p


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def draw_text(
    surf: pygame.Surface,
    text: str,
    x: int, y: int,
    size: int = 16,
    color: tuple | None = None,
    mono: bool = False,
    bold: bool = False,
    anchor: str = "topleft",   # "topleft" | "center" | "topright" | "midleft"
) -> pygame.Rect:
    font = get_font(size, mono=mono, bold=bold)
    col  = color or p().text
    surf_t = font.render(text, True, col)
    r = surf_t.get_rect()
    setattr(r, anchor, (x, y))
    surf.blit(surf_t, r)
    return r


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------

def draw_button(
    surf: pygame.Surface,
    text: str,
    x: int, y: int, w: int, h: int,
    active: bool = False,
    hover: bool = False,
    size: int = 16,
    color_text: tuple | None = None,
    color_bg:   tuple | None = None,
) -> pygame.Rect:
    r = pygame.Rect(x, y, w, h)
    if active:
        bg = color_bg or p().btn_hover
    elif hover:
        bg = p().btn_hover
    else:
        bg = color_bg or p().btn_bg
    pygame.draw.rect(surf, bg, r, border_radius=4)
    pygame.draw.rect(surf, p().btn_border, r, width=1, border_radius=4)
    col = color_text or p().text
    font = get_font(size)
    t = font.render(text, True, col)
    surf.blit(t, t.get_rect(center=r.center))
    return r


# ---------------------------------------------------------------------------
# Icon button (small square, single character)
# ---------------------------------------------------------------------------

def draw_icon_btn(
    surf: pygame.Surface,
    icon: str,
    x: int, y: int, size: int = 32,
    active: bool = False,
    font_size: int = 18,
) -> pygame.Rect:
    r = pygame.Rect(x, y, size, size)
    bg = p().btn_hover if active else p().btn_bg
    pygame.draw.rect(surf, bg, r, border_radius=4)
    pygame.draw.rect(surf, p().btn_border, r, width=1, border_radius=4)
    draw_text(surf, icon, r.centerx, r.centery, size=font_size, anchor="center")
    return r


# ---------------------------------------------------------------------------
# Separator line
# ---------------------------------------------------------------------------

def draw_hline(surf: pygame.Surface, y: int, color: tuple | None = None) -> None:
    w = surf.get_width()
    pygame.draw.line(surf, color or p().btn_border, (0, y), (w, y))


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------

def draw_status(
    surf: pygame.Surface,
    label: str,        # "ALIGNED" | "DRIFT" | "STALE" | "NOT ALIGNED"
    age_str: str,      # "1:42" — empty if NOT_ALIGNED
    x: int, y: int,
) -> pygame.Rect:
    if label == "ALIGNED":
        col = p().status_ok
    elif label == "DRIFT":
        col = p().status_warn
    else:
        col = p().status_err
    full = f"◎ {label}"
    if age_str:
        full += f" {age_str}"
    return draw_text(surf, full, x, y, size=15, color=col, bold=True)
