"""Finder UI theme — palettes and font loader.

Night mode (default): red-on-black, astronomically appropriate.
Day mode: white-on-dark-blue, for daytime debugging.
"""
from __future__ import annotations

import pygame

_FONT_SANS  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_FONT_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_font_cache: dict[tuple, pygame.font.Font] = {}


def get_font(size: int, mono: bool = False, bold: bool = False) -> pygame.font.Font:
    key = (size, mono, bold)
    if key not in _font_cache:
        if bold:
            path = _FONT_BOLD
        elif mono:
            path = _FONT_MONO
        else:
            path = _FONT_SANS
        try:
            _font_cache[key] = pygame.font.Font(path, size)
        except FileNotFoundError:
            _font_cache[key] = pygame.font.Font(None, size)
    return _font_cache[key]


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

class _Palette:
    def __init__(self, night: bool) -> None:
        if night:
            self.bg         = ( 10,   0,   0)
            self.bg_bar     = ( 20,   5,   5)
            self.star_b     = (255, 210, 190)   # bright stars
            self.star_d     = (120,  60,  50)   # dim stars
            self.const_line = ( 70,  25,  20)   # constellation lines
            self.const_name = ( 90,  40,  35)
            self.horizon    = ( 50,  20,  15)
            self.crosshair  = (180,  80,  60)
            self.circle_tol = ( 90,  40,  30)   # tolerance rings
            self.target     = (220, 100,  40)   # target circle
            self.arrow      = (255,  80,  60)
            self.text       = (200, 140, 120)
            self.text_dim   = (110,  65,  55)
            self.text_val   = (220, 180, 160)   # numeric values (mono)
            self.btn_bg     = ( 35,  12,  10)
            self.btn_hover  = ( 60,  22,  18)
            self.btn_border = ( 80,  35,  28)
            self.status_ok  = ( 80, 180,  80)   # ALIGNED
            self.status_warn= (200, 160,  40)   # DRIFT
            self.status_err = (200,  60,  40)   # STALE / NOT_ALIGNED
            self.solve_flash= (255, 200,  80)
        else:
            self.bg         = ( 15,  15,  30)
            self.bg_bar     = ( 25,  25,  45)
            self.star_b     = (255, 255, 255)
            self.star_d     = (160, 160, 190)
            self.const_line = ( 70,  70, 110)
            self.const_name = (100, 100, 160)
            self.horizon    = ( 50,  80, 120)
            self.crosshair  = (100, 180, 220)
            self.circle_tol = ( 60, 120, 160)
            self.target     = ( 50, 180, 255)
            self.arrow      = ( 80, 220,  80)
            self.text       = (220, 220, 240)
            self.text_dim   = (130, 130, 160)
            self.text_val   = (180, 220, 255)
            self.btn_bg     = ( 30,  30,  55)
            self.btn_hover  = ( 50,  50,  90)
            self.btn_border = ( 80,  80, 130)
            self.status_ok  = ( 60, 200,  80)
            self.status_warn= (220, 180,  50)
            self.status_err = (220,  70,  50)
            self.solve_flash= (255, 220, 100)


_night = _Palette(night=True)
_day   = _Palette(night=False)
_current: _Palette = _night


def set_night(on: bool) -> None:
    global _current
    _current = _night if on else _day


def p() -> _Palette:
    """Return the active palette."""
    return _current
