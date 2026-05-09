"""One-time catalog build script — run once to produce the three data files.

Inputs (downloaded to /tmp beforehand):
  /tmp/NGC_raw.csv      — OpenNGC database_files/NGC.csv
  /tmp/hyg_v38.csv.gz   — HYG v38 star catalog
  /tmp/const_lines.json — d3-celestial constellation lines GeoJSON
  /tmp/const_names.json — d3-celestial constellation names GeoJSON

Outputs (written next to this script):
  catalog.csv           — all Messier + NGC + IC objects
  bright_stars.csv      — HYG stars with mag < MAG_LIMIT
  constellations.json   — constellation lines + names, RA in [0,360]°
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys

OUTDIR     = os.path.dirname(os.path.abspath(__file__))
MAG_LIMIT  = 5.0   # bright stars threshold

# ---------------------------------------------------------------------------
# Type mapping: OpenNGC codes → human label
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "G":    "Galaxy",
    "OC":   "Open Cluster",
    "GC":   "Globular Cluster",
    "BN":   "Bright Nebula",
    "PN":   "Planetary Nebula",
    "SNR":  "Supernova Remnant",
    "EN":   "Emission Nebula",
    "RN":   "Reflection Nebula",
    "HII":  "HII Region",
    "AGN":  "AGN",
    "Cl+N": "Cluster+Nebula",
    "NF":   "Not Found",
    "D*":   "Double Star",
    "*":    "Star",
    "**":   "Double Star",
    "***":  "Multiple Star",
    "*Ass": "Stellar Association",
    "OCl":  "Open Cluster",
    "GCl":  "Globular Cluster",
    "EmN":  "Emission Nebula",
    "RfN":  "Reflection Nebula",
    "Nova": "Nova",
    "NonEx":"Non-Existent",
    "Dup":  "Duplicate",
    "Other":"Other",
}


def _ra_hms_to_deg(s: str) -> float:
    """'HH:MM:SS.ss' → decimal degrees."""
    h, m, sec = s.split(":")
    return (float(h) + float(m) / 60 + float(sec) / 3600) * 15.0


def _dec_dms_to_deg(s: str) -> float:
    """'±DD:MM:SS.s' → decimal degrees."""
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    d, m, sec = s.split(":")
    return sign * (float(d) + float(m) / 60 + float(sec) / 3600)


# ---------------------------------------------------------------------------
# Messier objects absent from NGC/IC (no NGC/IC designation)
# ---------------------------------------------------------------------------
_MESSIER_ADDENDUM = [
    # code, catalog, messier, name, type, ra_deg, dec_deg, mag, const
    ("M045",  "M", 45,  "Pleiades",    "Open Cluster", 56.750,  24.117,  1.6,  "Tau"),
    ("M040",  "M", 40,  "Winnecke 4",  "Double Star",  185.553, 58.083,  8.4,  "UMa"),
    # M102 = NGC 5866 alias (Spindle Galaxy) — added as alias in loader
]
def build_catalog() -> None:
    print("Building catalog.csv …")
    rows = []
    with open("/tmp/NGC_raw.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row["Name"].strip()
            ra_s = row["RA"].strip()
            dec_s = row["Dec"].strip()
            if not ra_s or not dec_s:
                continue
            try:
                ra_deg  = _ra_hms_to_deg(ra_s)
                dec_deg = _dec_dms_to_deg(dec_s)
            except Exception:
                continue

            obj_type = _TYPE_MAP.get(row["Type"].strip(), row["Type"].strip() or "Unknown")
            vmag = row.get("V-Mag", "").strip()
            bmag = row.get("B-Mag", "").strip()
            mag  = vmag if vmag else (bmag if bmag else "")

            messier  = row.get("M", "").strip()
            common   = row.get("Common names", "").strip().split(",")[0].strip()

            # catalog tag: NGC or IC
            if name.startswith("NGC"):
                catalog = "NGC"
                num     = name[3:].lstrip("0") or "0"
            elif name.startswith("IC"):
                catalog = "IC"
                num     = name[2:].lstrip("0") or "0"
            else:
                catalog = "Other"
                num     = ""

            rows.append({
                "code":    name,
                "catalog": catalog,
                "messier": messier,
                "name":    common,
                "type":    obj_type,
                "ra_deg":  f"{ra_deg:.6f}",
                "dec_deg": f"{dec_deg:.6f}",
                "mag":     mag,
                "const":   row.get("Const", "").strip(),
            })

    # Add Messier objects that have no NGC/IC designation
    for code, cat, m, name, obj_type, ra, dec, mag, const in _MESSIER_ADDENDUM:
        rows.append({"code": code, "catalog": cat, "messier": m, "name": name,
                     "type": obj_type, "ra_deg": f"{ra:.6f}", "dec_deg": f"{dec:.6f}",
                     "mag": f"{mag:.1f}", "const": const})

    # Add M102 as alias to NGC5866 (already in catalog) via messier field update
    for r in rows:
        if r["code"] == "NGC5866":
            r["messier"] = 102
            break

    out = os.path.join(OUTDIR, "catalog.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code","catalog","messier","name","type",
                                          "ra_deg","dec_deg","mag","const"])
        w.writeheader()
        w.writerows(rows)
    messier_count = sum(1 for r in rows if r["messier"])
    print(f"  {len(rows)} objects  ({messier_count} Messier)  → {out}")


# ---------------------------------------------------------------------------
# 2. HYG → bright_stars.csv
# ---------------------------------------------------------------------------
def build_bright_stars() -> None:
    print(f"Building bright_stars.csv (mag < {MAG_LIMIT}) …")
    rows = []
    with gzip.open("/tmp/hyg_v38.csv.gz", "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mag_s = row.get("mag", "").strip()
            if not mag_s:
                continue
            try:
                mag = float(mag_s)
            except ValueError:
                continue
            if mag >= MAG_LIMIT:
                continue
            hip    = row.get("hip", "").strip()
            proper = row.get("proper", "").strip()
            bf     = row.get("bf", "").strip()
            ra_h   = row.get("ra", "").strip()
            dec_d  = row.get("dec", "").strip()
            if not ra_h or not dec_d:
                continue
            ra_deg = float(ra_h) * 15.0
            label  = proper or bf or (f"HIP{hip}" if hip else "")
            rows.append({
                "hip":    hip,
                "name":   label,
                "ra_deg": f"{ra_deg:.6f}",
                "dec_deg": dec_d,
                "mag":    f"{mag:.2f}",
            })

    out = os.path.join(OUTDIR, "bright_stars.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["hip","name","ra_deg","dec_deg","mag"])
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} stars → {out}")


# ---------------------------------------------------------------------------
# 3. d3-celestial → constellations.json
# ---------------------------------------------------------------------------
def build_constellations() -> None:
    print("Building constellations.json …")

    names_data = json.load(open("/tmp/const_names.json"))
    names: dict[str, str] = {}
    for feat in names_data["features"]:
        abbr = feat.get("id", "")
        en   = feat["properties"].get("en") or feat["properties"].get("name", abbr)
        names[abbr] = en

    lines_data = json.load(open("/tmp/const_lines.json"))
    result: dict[str, dict] = {}
    for feat in lines_data["features"]:
        abbr = feat.get("id", "")
        geom = feat.get("geometry", {})
        if geom.get("type") != "MultiLineString":
            continue
        # d3-celestial uses RA in [-180, 180]; convert to [0, 360]
        converted_lines = []
        for segment in geom["coordinates"]:
            seg = [[(ra + 360.0) % 360.0, dec] for ra, dec in segment]
            converted_lines.append(seg)
        result[abbr] = {
            "name":  names.get(abbr, abbr),
            "lines": converted_lines,
        }

    out = os.path.join(OUTDIR, "constellations.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  {len(result)} constellations → {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_catalog()
    build_bright_stars()
    build_constellations()
    print("Done.")
