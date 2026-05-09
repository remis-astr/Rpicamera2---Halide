"""Catalog loader + tolerant search for the Finder mode.

Loads Messier/NGC/IC objects, bright stars, and constellation line data
from the CSV/JSON files in catalogs/.

Search is tolerant of spacing, case, leading zeros, and common aliases:
  "m31", "M 31", "messier 31", "ngc 224", "NGC0224", "andromeda" → M31
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

_CATALOG_DIR = os.path.join(os.path.dirname(__file__), "catalogs")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    code:    str          # canonical OpenNGC code, e.g. "NGC0224"
    catalog: str          # "NGC", "IC", or "Other"
    messier: int          # 0 if not a Messier object
    name:    str          # common name, e.g. "Andromeda Galaxy"
    type:    str          # "Galaxy", "Open Cluster", …
    ra_deg:  float
    dec_deg: float
    mag:     float        # 99.0 if unknown
    const:   str          # constellation abbreviation
    _norm:   str = field(default="", repr=False)  # pre-computed normalised code

    @property
    def display_code(self) -> str:
        """Short human code: M31, NGC 224, IC 1."""
        if self.messier:
            return f"M{self.messier}"
        if self.catalog == "NGC":
            num = self.code[3:].lstrip("0") or "0"
            return f"NGC {num}"
        if self.catalog == "IC":
            num = self.code[2:].lstrip("0") or "0"
            return f"IC {num}"
        return self.code

    @property
    def display_name(self) -> str:
        return self.name or self.display_code


@dataclass
class StarEntry:
    hip:    str
    name:   str
    ra_deg: float
    dec_deg: float
    mag:    float


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_code(s: str) -> str:
    """Normalise a user query to a lookup key.

    'm31', 'M 31', 'messier31', 'Messier 31'  → 'm31'
    'ngc224', 'NGC 224', 'NGC0224'             → 'ngc224'
    'ic1', 'IC 0001', 'IC1'                   → 'ic1'
    """
    s = s.strip().lower()
    s = re.sub(r"messier\s*", "m", s)
    s = re.sub(r"\s+", "", s)                 # remove all spaces
    # strip leading zeros from trailing digits: ngc0224 → ngc224
    s = re.sub(r"([a-z]+)0*(\d+)", lambda m: m.group(1) + m.group(2), s)
    return s


def _entry_codes(e: CatalogEntry) -> list[str]:
    """Return all normalised lookup keys for a catalog entry."""
    keys = [_norm_code(e.code)]
    if e.messier:
        keys.append(f"m{e.messier}")
    # NGC entry also indexed as ICxxx if it has a cross-reference (not needed here —
    # the cross-references live in the same row; the canonical code is enough)
    return keys


# ---------------------------------------------------------------------------
# CatalogLoader
# ---------------------------------------------------------------------------

class CatalogLoader:
    """Load catalogs once, answer search queries fast."""

    def __init__(self) -> None:
        self.entries:      list[CatalogEntry] = []
        self.stars:        list[StarEntry]    = []
        self.constellations: dict             = {}   # abbr → {name, lines}
        self._code_index:  dict[str, CatalogEntry] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.monotonic()
        self._load_catalog()
        self._load_stars()
        self._load_constellations()
        self._loaded = True
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"[CatalogLoader] loaded {len(self.entries)} objects, "
              f"{len(self.stars)} stars, "
              f"{len(self.constellations)} constellations in {elapsed_ms:.0f} ms")

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        max_results: int = 20,
        above_horizon_alts: Optional[dict[str, float]] = None,
    ) -> list[CatalogEntry]:
        """Return up to max_results entries matching query, sorted by relevance.

        above_horizon_alts: if provided, dict of code → alt_deg; entries with
        alt_deg <= 0 are pushed to the bottom (still returned, not filtered out,
        so the user can see them greyed out).
        """
        if not self._loaded:
            self.load()
        q = query.strip()
        if not q:
            return []

        # 1 — exact code match
        key = _norm_code(q)
        if key in self._code_index:
            return [self._code_index[key]]

        # 2 — substring search across code + name
        q_low = q.lower()
        results: list[tuple[int, float, CatalogEntry]] = []
        for e in self.entries:
            score = _score(e, q_low, key)
            if score > 0:
                results.append((score, e.mag if e.mag < 90 else 99.0, e))

        results.sort(key=lambda x: (-x[0], x[1]))

        hits = [e for _, _, e in results[:max_results]]
        return hits

    # ------------------------------------------------------------------
    def get_by_code(self, code: str) -> Optional[CatalogEntry]:
        if not self._loaded:
            self.load()
        return self._code_index.get(_norm_code(code))

    # ------------------------------------------------------------------
    def _load_catalog(self) -> None:
        path = os.path.join(_CATALOG_DIR, "catalog.csv")
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mag_s = row["mag"].strip()
                try:
                    mag = float(mag_s)
                except ValueError:
                    mag = 99.0
                code = row["code"].strip()
                e = CatalogEntry(
                    code    = code,
                    catalog = row["catalog"].strip(),
                    messier = int(row["messier"]) if row["messier"].strip() else 0,
                    name    = row["name"].strip(),
                    type    = row["type"].strip(),
                    ra_deg  = float(row["ra_deg"]),
                    dec_deg = float(row["dec_deg"]),
                    mag     = mag,
                    const   = row["const"].strip(),
                    _norm   = _norm_code(code),
                )
                self.entries.append(e)
                for key in _entry_codes(e):
                    self._code_index[key] = e

    def _load_stars(self) -> None:
        path = os.path.join(_CATALOG_DIR, "bright_stars.csv")
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.stars.append(StarEntry(
                    hip     = row["hip"].strip(),
                    name    = row["name"].strip(),
                    ra_deg  = float(row["ra_deg"]),
                    dec_deg = float(row["dec_deg"]),
                    mag     = float(row["mag"]),
                ))

    def _load_constellations(self) -> None:
        path = os.path.join(_CATALOG_DIR, "constellations.json")
        with open(path, encoding="utf-8") as f:
            self.constellations = json.load(f)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(e: CatalogEntry, q_low: str, q_norm: str) -> int:
    """Return relevance score > 0 if entry matches query, else 0.

    5 — normalised code prefix match (ngc22 matches ngc224)
    4 — name exact match
    3 — name starts-with
    2 — name word starts-with
    1 — name contains
    """
    if e._norm.startswith(q_norm) and q_norm:
        return 5

    name_low = e.name.lower()
    if not name_low:
        return 0
    if name_low == q_low:
        return 4
    if name_low.startswith(q_low):
        return 3
    # word prefix: "gal" matches "galaxy", "androm" matches "andromeda galaxy"
    for word in name_low.split():
        if word.startswith(q_low):
            return 2
    if q_low in name_low:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_loader: Optional[CatalogLoader] = None


def get_loader() -> CatalogLoader:
    global _loader
    if _loader is None:
        _loader = CatalogLoader()
        _loader.load()
    return _loader
