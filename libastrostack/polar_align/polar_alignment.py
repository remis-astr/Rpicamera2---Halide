"""
Polar alignment calculations for an equatorial table (3-foot design, ~15° travel).

Pure math module — no IO, no pygame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class Phase(Enum):
    IDLE      = auto()
    SETUP     = auto()
    CAPTURING = auto()
    RESULT    = auto()
    LIVE_ALT  = auto()
    LIVE_AZ   = auto()
    VERIFY    = auto()


@dataclass
class ImuState:
    roll_deg:  float       # gamma — W3C Y-axis tilt (±90°)
    pitch_deg: float       # beta  — W3C X-axis tilt (±180°)
    yaw_deg:   float       # alpha — W3C Z-axis (0–360°)
    quaternion: list       # [w, x, y, z]
    timestamp:  float      # time.time()


@dataclass
class Capture:
    ra_deg:        float
    dec_deg:       float
    timestamp_utc: datetime
    imu_state:     Optional[ImuState]
    image_path:    Optional[str] = None


@dataclass
class AlignmentResult:
    axis_az_deg:         float
    axis_alt_deg:        float
    delta_az_arcmin:     float   # positive = east of north
    delta_alt_arcmin:    float   # positive = axis too high
    total_error_arcmin:  float
    timestamp:           datetime
    n_captures:          int   = 3
    residual_rms_arcsec: Optional[float] = None
    outliers_indices:    list  = field(default_factory=list)


@dataclass
class LiveDeviation:
    delta_alt_arcmin: float
    delta_az_arcmin:  float


class PolarAlignmentState:
    def __init__(self) -> None:
        self.phase:         Phase = Phase.IDLE
        self.captures:      list[Capture] = []
        self.history:       list[AlignmentResult] = []
        self.imu_reference: Optional[ImuState] = None
        self.target_imu:    Optional[ImuState] = None
        self.result:        Optional[AlignmentResult] = None
        self.n_captures:    int = 3
        self.is_verify:     bool = False

    def reset_cycle(self) -> None:
        self.captures       = []
        self.result         = None
        self.imu_reference  = None
        self.target_imu     = None

    def add_capture(self, c: Capture) -> None:
        self.captures.append(c)

    def is_cycle_complete(self) -> bool:
        return len(self.captures) >= self.n_captures


# ---------------------------------------------------------------------------
# GMST and coordinate conversion
# ---------------------------------------------------------------------------

def _gmst_deg(t_utc: datetime) -> float:
    """Greenwich Mean Sidereal Time in degrees for given UTC datetime."""
    if t_utc.tzinfo is None:
        t_utc = t_utc.replace(tzinfo=timezone.utc)
    epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    days  = (t_utc - epoch).total_seconds() / 86400.0
    return (280.46061837 + 360.98564736629 * days) % 360.0


def equatorial_to_horizontal(ra_deg: float, dec_deg: float,
                              t_utc: datetime,
                              lat_deg: float, lon_deg: float
                              ) -> tuple[float, float]:
    """Convert equatorial (RA, Dec) to horizontal (Az, Alt) in degrees.

    Az convention: N=0°, E=90°, S=180°, W=270°.
    Alt: positive above horizon.
    """
    lst_deg = (_gmst_deg(t_utc) + lon_deg) % 360.0
    ha_rad  = math.radians((lst_deg - ra_deg) % 360.0)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)

    sin_alt = (math.sin(dec_rad) * math.sin(lat_rad)
               + math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad))
    alt_rad = math.asin(max(-1.0, min(1.0, sin_alt)))
    cos_alt = math.cos(alt_rad)

    if abs(cos_alt) > 1e-9:
        sin_az = -math.sin(ha_rad) * math.cos(dec_rad) / cos_alt
        cos_az = ((math.sin(dec_rad) - math.sin(alt_rad) * math.sin(lat_rad))
                  / (cos_alt * math.cos(lat_rad)))
        az_deg = math.degrees(math.atan2(sin_az, cos_az)) % 360.0
    else:
        az_deg = 0.0

    return az_deg, math.degrees(alt_rad)


def _altaz_to_vec(az_deg: float, alt_deg: float) -> np.ndarray:
    """(Az, Alt) degrees → unit 3-vector [cos·cos, cos·sin, sin].

    Coordinate frame: x=North, y=East, z=Zenith.
    """
    az  = math.radians(az_deg)
    alt = math.radians(alt_deg)
    return np.array([
        math.cos(alt) * math.cos(az),  # N component
        math.cos(alt) * math.sin(az),  # E component
        math.sin(alt),                 # Z component
    ])


# ---------------------------------------------------------------------------
# Axis computation
# ---------------------------------------------------------------------------

def compute_axis_three_points(captures: list[Capture],
                               lat_deg: float, lon_deg: float
                               ) -> AlignmentResult:
    """Closed-form 3-point solution using cross product of plane normal."""
    vecs = []
    for c in captures[:3]:
        az, alt = equatorial_to_horizontal(
            c.ra_deg, c.dec_deg, c.timestamp_utc, lat_deg, lon_deg)
        vecs.append(_altaz_to_vec(az, alt))

    v1, v2, v3 = vecs
    n  = np.cross(v2 - v1, v3 - v1)
    nm = float(np.linalg.norm(n))
    if nm < 1e-9:
        n = np.array([1.0, 0.0, 0.0])   # degenerate: assume northward
    else:
        n = n / nm
    if n[2] < 0:
        n = -n

    az_axis  = math.degrees(math.atan2(float(n[1]), float(n[0]))) % 360.0
    alt_axis = math.degrees(math.asin(max(-1.0, min(1.0, float(n[2])))))

    # Map az_axis to ±180° for ΔAz
    delta_az_deg = az_axis if az_axis <= 180.0 else az_axis - 360.0
    delta_az_arcmin  = delta_az_deg * 60.0
    delta_alt_arcmin = (alt_axis - lat_deg) * 60.0
    total = math.sqrt(delta_alt_arcmin**2
                      + (delta_az_arcmin * math.cos(math.radians(lat_deg)))**2)

    return AlignmentResult(
        axis_az_deg       = az_axis,
        axis_alt_deg      = alt_axis,
        delta_az_arcmin   = delta_az_arcmin,
        delta_alt_arcmin  = delta_alt_arcmin,
        total_error_arcmin= total,
        timestamp         = captures[-1].timestamp_utc,
        n_captures        = 3,
    )


def compute_axis_robust(captures: list[Capture],
                         lat_deg: float, lon_deg: float
                         ) -> AlignmentResult:
    """Least-squares fit for N≥4 captures with robust soft_l1 loss."""
    try:
        from scipy.optimize import least_squares as _lsq
    except ImportError:
        return compute_axis_three_points(captures, lat_deg, lon_deg)

    vecs = np.array([
        _altaz_to_vec(*equatorial_to_horizontal(
            c.ra_deg, c.dec_deg, c.timestamp_utc, lat_deg, lon_deg))
        for c in captures
    ])

    # Seed from first 3 captures
    seed  = compute_axis_three_points(captures[:3], lat_deg, lon_deg)
    az0   = math.radians(seed.axis_az_deg)
    alt0  = math.radians(seed.axis_alt_deg)
    n0    = np.array([math.cos(alt0)*math.cos(az0),
                      math.cos(alt0)*math.sin(az0),
                      math.sin(alt0)])
    theta0 = math.acos(max(-1.0, min(1.0, float(n0[2]))))
    phi0   = math.atan2(float(n0[1]), float(n0[0]))

    def residuals(params):
        th, ph = params
        n_v = np.array([math.sin(th)*math.cos(ph),
                        math.sin(th)*math.sin(ph),
                        math.cos(th)])
        d = vecs @ n_v
        return d - d.mean()

    res = _lsq(residuals, x0=[theta0, phi0], loss='soft_l1', f_scale=1e-4)
    th, ph = res.x
    n_final = np.array([math.sin(th)*math.cos(ph),
                        math.sin(th)*math.sin(ph),
                        math.cos(th)])
    if n_final[2] < 0:
        n_final = -n_final
        th = math.pi - th
        ph = ph + math.pi

    az_axis  = math.degrees(math.atan2(float(n_final[1]), float(n_final[0]))) % 360.0
    alt_axis = math.degrees(math.asin(max(-1.0, min(1.0, float(n_final[2])))))

    delta_az_deg = az_axis if az_axis <= 180.0 else az_axis - 360.0
    delta_az_arcmin  = delta_az_deg * 60.0
    delta_alt_arcmin = (alt_axis - lat_deg) * 60.0
    total = math.sqrt(delta_alt_arcmin**2
                      + (delta_az_arcmin * math.cos(math.radians(lat_deg)))**2)

    r_arr = residuals(res.x)
    rms_arcsec = float(np.std(r_arr) * 180 / math.pi * 3600)
    sigma  = float(np.std(r_arr))
    outliers = [i for i, r in enumerate(r_arr.tolist()) if abs(r) > 3 * sigma and sigma > 0]

    return AlignmentResult(
        axis_az_deg        = az_axis,
        axis_alt_deg       = alt_axis,
        delta_az_arcmin    = delta_az_arcmin,
        delta_alt_arcmin   = delta_alt_arcmin,
        total_error_arcmin = total,
        timestamp          = captures[-1].timestamp_utc,
        n_captures         = len(captures),
        residual_rms_arcsec= rms_arcsec,
        outliers_indices   = outliers,
    )


def compute_axis(captures: list[Capture],
                  lat_deg: float, lon_deg: float) -> AlignmentResult:
    """Dispatch to 3-point formula or robust LSQ depending on N."""
    n = len(captures)
    if n < 3:
        raise ValueError(f"Need at least 3 captures, got {n}")
    if n == 3:
        return compute_axis_three_points(captures, lat_deg, lon_deg)
    return compute_axis_robust(captures, lat_deg, lon_deg)


# ---------------------------------------------------------------------------
# IMU helpers
# ---------------------------------------------------------------------------

def imu_state_from_quaternion(q: list, ts: float) -> ImuState:
    """Convert [w, x, y, z] quaternion to ImuState (ZXY Euler, W3C convention)."""
    w, x, y, z = q
    beta  = math.degrees(math.asin(max(-1.0, min(1.0, 2.0*(w*x + y*z)))))
    alpha = math.degrees(math.atan2(2.0*(w*z - x*y), 1.0 - 2.0*(x*x + z*z))) % 360.0
    gamma = math.degrees(math.atan2(2.0*(w*y - x*z), 1.0 - 2.0*(x*x + y*y)))
    return ImuState(roll_deg=gamma, pitch_deg=beta, yaw_deg=alpha, quaternion=list(q), timestamp=ts)


def compute_target_imu_state(result: AlignmentResult, imu_ref: ImuState) -> ImuState:
    """Compute the target IMU state after applying the measured correction.

    Small-angle approximation: ΔAlt → delta pitch, ΔAz → delta yaw.
    """
    delta_alt_deg = result.delta_alt_arcmin / 60.0
    delta_az_deg  = result.delta_az_arcmin  / 60.0
    return ImuState(
        roll_deg  = imu_ref.roll_deg,
        pitch_deg = imu_ref.pitch_deg - delta_alt_deg,
        yaw_deg   = (imu_ref.yaw_deg  - delta_az_deg) % 360.0,
        quaternion= imu_ref.quaternion,
        timestamp = imu_ref.timestamp,
    )


def live_deviation(imu_current: ImuState, imu_target: ImuState) -> LiveDeviation:
    """Return current deviation from target IMU orientation."""
    delta_pitch = imu_current.pitch_deg - imu_target.pitch_deg
    delta_yaw   = imu_current.yaw_deg   - imu_target.yaw_deg
    if delta_yaw >  180: delta_yaw -= 360
    if delta_yaw < -180: delta_yaw += 360
    return LiveDeviation(
        delta_alt_arcmin = delta_pitch * 60.0,
        delta_az_arcmin  = delta_yaw   * 60.0,
    )


# ---------------------------------------------------------------------------
# Polaris position
# ---------------------------------------------------------------------------

def polaris_position(t_utc: datetime | None = None) -> tuple[float, float]:
    """Return (RA_deg, Dec_deg) of Polaris at given UTC date.

    Uses astropy with precession if available; otherwise a simplified linear model.
    J2000: RA=37.9546°, Dec=+89.2642°
    """
    if t_utc is None:
        t_utc = datetime.now(timezone.utc)

    try:
        from astropy.coordinates import SkyCoord, FK5
        from astropy.time import Time
        import astropy.units as u
        t  = Time(t_utc)
        p  = SkyCoord(ra=37.9546*u.deg, dec=89.2642*u.deg, frame='icrs')
        pn = p.transform_to(FK5(equinox=t))
        return float(pn.ra.deg), float(pn.dec.deg)
    except Exception:
        pass

    # Simplified linear precession (~+1.6 s/yr in RA, ~−15.5"/yr in Dec)
    if t_utc.tzinfo is None:
        t_utc = t_utc.replace(tzinfo=timezone.utc)
    epoch_j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    yrs = (t_utc - epoch_j2000).total_seconds() / 31_557_600.0
    ra  = (37.9546 + yrs * 0.400) % 360.0
    dec = 89.2642 - yrs * (15.5 / 3600.0)
    return ra, dec


def polaris_hour_angle_deg(t_utc: datetime, lon_deg: float) -> float:
    """Hour angle of Polaris at a given UTC time and observer longitude."""
    ra_p, _ = polaris_position(t_utc)
    lst_deg  = (_gmst_deg(t_utc) + lon_deg) % 360.0
    return (lst_deg - ra_p) % 360.0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_angle(deg: float) -> str:
    """Format angle in degrees → human-readable arcmin/arcsec string with sign."""
    am  = abs(deg) * 60.0
    sgn = "+" if deg >= 0 else "−"
    if am < 1.0:
        return f"{sgn}{am * 60:.0f}\""
    elif am < 60.0:
        return f"{sgn}{am:.1f}'"
    else:
        return f"{sgn}{abs(deg):.2f}°"


def direction_hint(delta_az_arcmin: float, delta_alt_arcmin: float) -> tuple[str, str]:
    """Return (alt_hint, az_hint) human-readable correction directions."""
    if delta_alt_arcmin > 0:
        alt_hint = "↓ Baisser côté Nord"
    else:
        alt_hint = "↑ Lever côté Nord"

    if delta_az_arcmin > 0:
        az_hint = "← Vers l'Ouest"
    else:
        az_hint = "→ Vers l'Est"

    return alt_hint, az_hint


def error_color(total_arcmin: float) -> tuple[int, int, int]:
    """Return RGB color depending on total alignment error."""
    if total_arcmin < 5.0:
        return (60, 200, 100)   # green
    elif total_arcmin < 15.0:
        return (220, 150, 50)   # orange
    else:
        return (220, 70, 70)    # red
