"""Client HTTP pour le plugin Remote Control de Stellarium.

Le plugin doit être activé dans Stellarium :
  Configuration → Plugins → Remote Control → cocher "Activer au démarrage"
  Port par défaut : 8090

Usage::
    client = StellariumClient("192.168.1.100:8090")
    if client.ping():
        obj = client.get_selected_object()
        if obj:
            print(obj.name, obj.ra_deg, obj.dec_deg)
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class StelObject:
    name: str
    ra_deg: float   # J2000, degrés (0–360)
    dec_deg: float  # J2000, degrés (-90 à +90)


class StellariumClient:
    """Interroge le plugin Remote Control de Stellarium via HTTP."""

    def __init__(self, host: str = "localhost:8090", timeout: float = 3.0) -> None:
        self.host = host
        self.timeout = timeout

    def _get(self, path: str) -> Optional[dict]:
        url = f"http://{self.host}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.URLError as exc:
            log.debug("[Stellarium] %s → %s", url, exc)
            return None
        except Exception as exc:
            log.debug("[Stellarium] %s → %s", url, exc)
            return None

    def ping(self) -> bool:
        """Retourne True si Stellarium Remote Control est joignable."""
        return self._get("/api/main/status") is not None

    def get_selected_object(self) -> Optional[StelObject]:
        """Retourne l'objet actuellement sélectionné dans Stellarium.

        Gère les variations de format entre versions de Stellarium :
        - raJ2000 / decJ2000 (string HMS/DMS ou float heures/degrés)
        - ra / dec (string ou float)
        - vecteur J2000 (x, y, z)
        """
        data = self._get("/api/objects/info?format=json")
        if not data:
            log.warning("[Stellarium] Pas de réponse — plugin Remote Control actif?")
            return None
        if not data:
            return None

        name = (data.get("localized-name")
                or data.get("name")
                or data.get("designations", ["?"])[0]
                or "?")

        ra_deg: Optional[float] = None
        dec_deg: Optional[float] = None

        # Tentative 1 : champs raJ2000 / decJ2000
        if "raJ2000" in data:
            ra_deg = _parse_ra(data["raJ2000"])
            dec_deg = _parse_dec(data.get("decJ2000", 0))

        # Tentative 2 : champs ra / dec
        if ra_deg is None and "ra" in data:
            ra_deg = _parse_ra(data["ra"])
            dec_deg = _parse_dec(data.get("dec", 0))

        # Tentative 3 : vecteur J2000 unitaire [x, y, z]
        if ra_deg is None:
            j2000 = data.get("j2000")
            if isinstance(j2000, (list, tuple)) and len(j2000) >= 3:
                x, y, z = float(j2000[0]), float(j2000[1]), float(j2000[2])
                dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, z))))
                ra_deg = math.degrees(math.atan2(y, x)) % 360.0

        if ra_deg is None or dec_deg is None:
            log.warning("[Stellarium] Impossible de parser RA/Dec depuis : %s", data)
            return None

        return StelObject(name=str(name), ra_deg=float(ra_deg) % 360.0, dec_deg=float(dec_deg))


# ---------------------------------------------------------------------------
# Fonctions de parsing RA/Dec (tolérantes aux formats Stellarium)
# ---------------------------------------------------------------------------

def _parse_ra(v) -> Optional[float]:
    """RA en degrés J2000 à partir de float (heures ou degrés) ou string HMS."""
    if isinstance(v, (int, float)):
        # Stellarium envoie parfois des heures, parfois des degrés
        val = float(v)
        if val < 24.0:         # vraisemblablement en heures
            return val * 15.0
        return val             # déjà en degrés
    if isinstance(v, str):
        v = v.strip()
        if "h" in v.lower() or ("m" in v.lower() and "s" in v.lower()):
            return _hms_to_deg(v)
        if ":" in v:
            parts = v.split(":")
            try:
                h = float(parts[0])
                m = float(parts[1]) if len(parts) > 1 else 0.0
                s = float(parts[2]) if len(parts) > 2 else 0.0
                return (abs(h) + m / 60.0 + s / 3600.0) * 15.0
            except ValueError:
                pass
        try:
            val = float(v)
            return val * 15.0 if val < 24.0 else val
        except ValueError:
            return None
    return None


def _parse_dec(v) -> Optional[float]:
    """Dec en degrés J2000 à partir de float ou string DMS."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if "°" in v or ("d" in v.lower() and "m" in v.lower()):
            return _dms_to_deg(v)
        if ":" in v:
            parts = v.split(":")
            try:
                sign = -1 if parts[0].lstrip().startswith("-") else 1
                d = abs(float(parts[0]))
                m = float(parts[1]) if len(parts) > 1 else 0.0
                s = float(parts[2]) if len(parts) > 2 else 0.0
                return sign * (d + m / 60.0 + s / 3600.0)
            except ValueError:
                pass
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _hms_to_deg(s: str) -> Optional[float]:
    nums = re.findall(r"[\d.]+", s)
    if not nums:
        return None
    h = float(nums[0])
    m = float(nums[1]) if len(nums) > 1 else 0.0
    sc = float(nums[2]) if len(nums) > 2 else 0.0
    return (h + m / 60.0 + sc / 3600.0) * 15.0


def _dms_to_deg(s: str) -> Optional[float]:
    sign = -1 if s.strip().startswith("-") else 1
    nums = re.findall(r"[\d.]+", s)
    if not nums:
        return None
    d = float(nums[0])
    m = float(nums[1]) if len(nums) > 1 else 0.0
    sc = float(nums[2]) if len(nums) > 2 else 0.0
    return sign * (d + m / 60.0 + sc / 3600.0)
