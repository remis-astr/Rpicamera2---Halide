"""Unified Target abstraction for the Finder mode.

A Target can come from three sources:
  - "stellarium" : GoTo received from Stellarium/SkySafari via LX200 server
  - "search"     : selected from the local catalog via the search modal
  - "manual"     : entered manually (RA/Dec direct input, rare)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Target:
    ra_deg:   float
    dec_deg:  float
    name:     str          # display name, e.g. "Andromeda Galaxy"
    code:     str          # catalog code, e.g. "M31" — empty for manual
    source:   str          # "stellarium" | "search" | "manual"
    mag:      float = 99.0
    obj_type: str   = ""


def target_from_catalog_entry(entry, source: str = "search") -> Target:
    """Build a Target from a CatalogEntry (data_loader.CatalogEntry)."""
    return Target(
        ra_deg   = entry.ra_deg,
        dec_deg  = entry.dec_deg,
        name     = entry.display_name,
        code     = entry.display_code,
        source   = source,
        mag      = entry.mag,
        obj_type = entry.type,
    )


def target_from_lx200(lx200_server) -> Optional[Target]:
    """Read latest GoTo target from a running LX200Server instance.

    Returns None if no target has been received yet.
    """
    result = lx200_server.get_target()
    if result is None:
        return None
    ra_deg, dec_deg = result
    return Target(
        ra_deg  = ra_deg,
        dec_deg = dec_deg,
        name    = f"RA {ra_deg:.3f}° Dec {dec_deg:+.3f}°",
        code    = "",
        source  = "stellarium",
    )
