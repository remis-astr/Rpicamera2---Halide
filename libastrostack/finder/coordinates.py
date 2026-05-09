"""Astrometric helpers for the Finder mode.

Uses astropy (already installed) rather than skyfield — skyfield requires
downloading a ~17 MB ephemeris file; astropy ships with a built-in DE432s
sufficient for all Finder computations.

All positions stored and passed in J2000 equatorial degrees (ra, dec).
Alt/az conversion is done here for display only.

Observer location is passed at construction time; callers should read
obs_lat_deg / obs_lon_deg from the RPiCamera2 globals at startup.
"""
from __future__ import annotations

import math
from typing import Optional

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time


class CoordinateHelper:
    """Stateless helper bound to one observer location."""

    def __init__(
        self,
        lat_deg: float,
        lon_deg: float,
        elev_m: float = 0.0,
    ) -> None:
        self._location = EarthLocation(
            lat=lat_deg * u.deg,
            lon=lon_deg * u.deg,
            height=elev_m * u.m,
        )

    # ------------------------------------------------------------------
    # Core transformation

    def altaz_now(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        """Return (alt_deg, az_deg) for a J2000 RA/Dec at the current time."""
        t = Time.now()
        return self._radec_to_altaz(ra_deg, dec_deg, t)

    def altaz_at(
        self, ra_deg: float, dec_deg: float, t: Optional[Time] = None
    ) -> tuple[float, float]:
        """Same as altaz_now but with an explicit astropy Time."""
        return self._radec_to_altaz(ra_deg, dec_deg, t or Time.now())

    def altaz_to_radec(self, alt_deg: float, az_deg: float) -> tuple[float, float]:
        """Convert (alt, az) at the current time to (RA, Dec) J2000 degrees."""
        t     = Time.now()
        frame = AltAz(obstime=t, location=self._location)
        aa    = SkyCoord(alt=alt_deg * u.deg, az=az_deg * u.deg, frame=frame)
        icrs  = aa.icrs
        return float(icrs.ra.deg), float(icrs.dec.deg)

    def _radec_to_altaz(
        self, ra_deg: float, dec_deg: float, t: Time
    ) -> tuple[float, float]:
        frame = AltAz(obstime=t, location=self._location)
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        aa = coord.transform_to(frame)
        return float(aa.alt.deg), float(aa.az.deg)

    # ------------------------------------------------------------------
    # Body positions (Sun, Moon, planets)

    def body_altaz_now(self, body: str) -> tuple[float, float]:
        """Return (alt_deg, az_deg) for a solar system body by name.

        Accepted names: 'moon', 'sun', 'mars', 'venus', 'jupiter', 'saturn',
        'uranus', 'neptune'.
        """
        t = Time.now()
        frame = AltAz(obstime=t, location=self._location)
        coord = get_body(body, t, self._location)
        aa = coord.transform_to(frame)
        return float(aa.alt.deg), float(aa.az.deg)

    # ------------------------------------------------------------------
    # Angular math

    @staticmethod
    def angular_separation(
        ra1: float, dec1: float, ra2: float, dec2: float
    ) -> float:
        """Great-circle distance in degrees between two J2000 positions."""
        c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
        c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
        return float(c1.separation(c2).deg)

    def delta_altaz(
        self,
        target_ra: float,
        target_dec: float,
        current_ra: float,
        current_dec: float,
    ) -> tuple[float, float]:
        """Return signed (Δalt, Δaz) in degrees: target minus current.

        Positive Δalt → tilt up.  Positive Δaz → rotate East (clockwise N=0).
        Az delta is wrapped to (−180, +180].
        """
        t = Time.now()
        tgt_alt, tgt_az = self._radec_to_altaz(target_ra, target_dec, t)
        cur_alt, cur_az = self._radec_to_altaz(current_ra, current_dec, t)
        dalt = tgt_alt - cur_alt
        daz  = (tgt_az - cur_az + 180.0) % 360.0 - 180.0
        return dalt, daz

    # ------------------------------------------------------------------
    # Convenience

    @staticmethod
    def is_above_horizon(alt_deg: float) -> bool:
        return alt_deg > 0.0

    @staticmethod
    def ra_deg_to_hms(ra_deg: float) -> str:
        h = int(ra_deg / 15)
        m = int((ra_deg / 15 - h) * 60)
        s = ((ra_deg / 15 - h) * 60 - m) * 60
        return f"{h:02d}h{m:02d}m{s:04.1f}s"

    @staticmethod
    def dec_deg_to_dms(dec_deg: float) -> str:
        sign = "+" if dec_deg >= 0 else "-"
        d = int(abs(dec_deg))
        m = int((abs(dec_deg) - d) * 60)
        s = (abs(dec_deg) - d - m / 60) * 3600
        return f"{sign}{d:02d}°{m:02d}'{s:04.1f}\""
