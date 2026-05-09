"""Push-to guidance engine for the Finder mode.

Combines:
  - One plate-solve alignment point (RA/Dec + IMU quaternion at solve time)
  - Continuous IMU quaternion stream
  - Optional calibration quaternion (IMU frame → optical axis frame)

Pipeline per tick (FINDER.md §5):
  1. q_delta = q_ref⁻¹ ⊗ q_now          (rotation since last solve, IMU frame)
  2. q_opt   = r_calib ⊗ q_delta ⊗ r_calib⁻¹  (expressed in optical frame)
  3. v_now   = rotate(q_opt, v_ref)       (current pointing vector in alt/az ENU)
  4. (alt_now, az_now) from v_now
  5. (alt_tgt, az_tgt) from target RA/Dec at t_now  (accounts for sky rotation)
  6. Δalt, Δaz = signed differences

Coordinate convention (ENU — right-handed):
  v[0] = East  = cos(alt) · sin(az)
  v[1] = North = cos(alt) · cos(az)
  v[2] = Up    = sin(alt)
  az measured clockwise from North (N=0°, E=90°).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from libastrostack.finder.coordinates import CoordinateHelper

# Thresholds for status string
_DRIFT_S = 120   # seconds before ALIGNED → DRIFT
_STALE_S = 300   # seconds before DRIFT  → STALE


# ---------------------------------------------------------------------------
# Quaternion helpers (unit quaternions, [w, x, y, z])
# ---------------------------------------------------------------------------

def _q(v: list[float]) -> np.ndarray:
    return np.array(v, dtype=np.float64)


def _qinv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion = conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by unit quaternion q.  Uses Rodrigues' efficient form."""
    t = 2.0 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


# ---------------------------------------------------------------------------
# Alt/Az ↔ ENU unit vector
# ---------------------------------------------------------------------------

def _altaz_to_vec(alt_deg: float, az_deg: float) -> np.ndarray:
    alt = math.radians(alt_deg)
    az  = math.radians(az_deg)
    return np.array([
        math.cos(alt) * math.sin(az),   # East
        math.cos(alt) * math.cos(az),   # North
        math.sin(alt),                  # Up
    ])


def _vec_to_altaz(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)          # renormalise after float drift
    alt_deg = math.degrees(math.asin(float(np.clip(v[2], -1.0, 1.0))))
    az_deg  = math.degrees(math.atan2(float(v[0]), float(v[1]))) % 360.0
    return alt_deg, az_deg


def _angular_dist_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    """Great-circle separation in degrees between two unit vectors."""
    dot = float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class AlignPoint:
    """One plate-solve alignment record."""
    q_ref:     np.ndarray   # IMU quaternion at solve time [w,x,y,z]
    v_ref:     np.ndarray   # pointing unit vector (ENU) at solve time
    ra_deg:    float        # RA from solve (J2000)
    dec_deg:   float        # Dec from solve (J2000)
    t_solve:   float        # Unix timestamp of solve


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PushToEngine:
    """Maintain a solve reference and compute current pointing + delta to target."""

    _IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])

    def __init__(self, coord: CoordinateHelper) -> None:
        self._coord    = coord
        self._align:   Optional[AlignPoint] = None
        self._r_calib: np.ndarray = self._IDENTITY_Q.copy()  # quaternion form
        # Cache for the slow AltAz→ICRS conversion (astropy, ~50ms)
        self._radec_cache: tuple[float, float] = (0.0, 0.0)
        self._radec_cache_t: float = 0.0
        self._RADEC_TTL = 2.0   # seconds

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_r_calib(self, q_calib: list[float]) -> None:
        """Set the calibration quaternion [w,x,y,z] (IMU frame → optical axis).

        Use [1,0,0,0] (identity) until a real calibration is done.
        """
        self._r_calib = np.array(q_calib, dtype=np.float64)
        self._r_calib /= np.linalg.norm(self._r_calib)

    # ------------------------------------------------------------------
    # Alignment (called after each successful plate solve)
    # ------------------------------------------------------------------

    def align(
        self,
        q_imu:   list[float],
        ra_deg:  float,
        dec_deg: float,
        t:       Optional[float] = None,
    ) -> None:
        """Record a solve reference point.

        q_imu  : IMU quaternion [w,x,y,z] captured simultaneously with the solve
        ra_deg : field-centre RA from ASTAP (J2000)
        dec_deg: field-centre Dec from ASTAP (J2000)
        t      : Unix timestamp (defaults to now)
        """
        t = t or time.time()
        alt, az  = self._coord.altaz_at(ra_deg, dec_deg)
        v_ref    = _altaz_to_vec(alt, az)
        q        = np.array(q_imu, dtype=np.float64)
        q       /= np.linalg.norm(q)
        self._align = AlignPoint(
            q_ref   = q,
            v_ref   = v_ref,
            ra_deg  = ra_deg,
            dec_deg = dec_deg,
            t_solve = t,
        )

    # ------------------------------------------------------------------
    # Query (called every UI tick ~30 Hz)
    # ------------------------------------------------------------------

    def get_current_pointing(
        self, q_now: list[float]
    ) -> Optional[tuple[float, float, float, float]]:
        """Return (ra_deg, dec_deg, alt_deg, az_deg) for the current IMU quaternion.

        Returns None if not yet aligned.
        """
        if self._align is None:
            return None
        q     = np.array(q_now, dtype=np.float64)
        q    /= np.linalg.norm(q)
        v_now = self._apply_delta(q)
        alt_now, az_now = _vec_to_altaz(v_now)

        # RA/Dec conversion via astropy is slow (~50ms). Cache it at 2s TTL.
        # The compass view only needs alt/az; the info bar tolerates 2s latency.
        now = time.monotonic()
        if now - self._radec_cache_t > self._RADEC_TTL:
            try:
                from astropy.coordinates import AltAz, SkyCoord
                from astropy.time import Time
                import astropy.units as u
                frame = AltAz(obstime=Time.now(), location=self._coord._location)
                aa    = SkyCoord(alt=alt_now * u.deg, az=az_now * u.deg, frame=frame)
                icrs  = aa.icrs
                self._radec_cache   = (float(icrs.ra.deg), float(icrs.dec.deg))
                self._radec_cache_t = now
            except Exception:
                pass
        ra_deg, dec_deg = self._radec_cache
        return ra_deg, dec_deg, alt_now, az_now

    def get_delta_to_target(
        self,
        q_now:      list[float],
        target_ra:  float,
        target_dec: float,
    ) -> tuple[float, float, float]:
        """Return (separation_deg, dalt_deg, daz_deg) to the target.

        separation_deg : total angular distance (for arrow size)
        dalt_deg       : positive → tilt up
        daz_deg        : positive → rotate East (clockwise), wrapped to (-180, +180]
        """
        if self._align is None:
            return 0.0, 0.0, 0.0

        q     = np.array(q_now, dtype=np.float64)
        q    /= np.linalg.norm(q)
        v_now = self._apply_delta(q)

        alt_now, az_now = _vec_to_altaz(v_now)
        alt_tgt, az_tgt = self._coord.altaz_now(target_ra, target_dec)
        v_tgt = _altaz_to_vec(alt_tgt, az_tgt)

        sep  = _angular_dist_vec(v_now, v_tgt)
        dalt = alt_tgt - alt_now
        daz  = (az_tgt - az_now + 180.0) % 360.0 - 180.0
        return sep, dalt, daz

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_aligned(self) -> bool:
        return self._align is not None

    @property
    def age_s(self) -> float:
        """Seconds since last alignment (∞ if never aligned)."""
        if self._align is None:
            return float("inf")
        return time.time() - self._align.t_solve

    @property
    def status(self) -> str:
        """ALIGNED / DRIFT / STALE / NOT_ALIGNED — for the status bar."""
        if self._align is None:
            return "NOT_ALIGNED"
        age = self.age_s
        if age < _DRIFT_S:
            return "ALIGNED"
        if age < _STALE_S:
            return "DRIFT"
        return "STALE"

    @property
    def align_ra(self) -> Optional[float]:
        return self._align.ra_deg if self._align else None

    @property
    def align_dec(self) -> Optional[float]:
        return self._align.dec_deg if self._align else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_delta(self, q_now: np.ndarray) -> np.ndarray:
        """Compute current pointing vector from IMU delta.

        Uses world-frame rotation q_now ⊗ q_ref⁻¹ so that rotating a world-frame
        reference vector gives the correct new sky direction regardless of the
        IMU mounting orientation.  r_calib is not needed here: both q_ref and
        q_now share the same sensor-to-optical offset, which cancels in the product.
        """
        a = self._align
        q_world_rot = _qmul(q_now, _qinv(a.q_ref))
        return _qrotate(q_world_rot, a.v_ref)
