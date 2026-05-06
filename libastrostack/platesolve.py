"""Plate solving via ASTAP (G05/D05 catalogues, /opt/astap).

Usage::
    from libastrostack.platesolve import PlateSolver
    solver = PlateSolver(pixel_size_um=2.9, max_stars=200, downsample=2)
    result = solver.solve(bgr_image, focal_mm=25.0)
    if result.success:
        print(result.ra_deg, result.dec_deg)
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

ASTAP_BIN = "astap"
ASTAP_CATALOG_DIR = "/opt/astap"
_DEFAULT_TIMEOUT = 30.0


@dataclass
class SolveResult:
    success: bool
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    field_angle_deg: float = 0.0
    pixel_scale_arcsec: float = 0.0
    field_w_deg: float = 0.0
    field_h_deg: float = 0.0
    focal_mm_measured: float = 0.0  # focale réelle = 206.265 × pixel_um / scale_arcsec
    cdelt1_sign: int = -1           # signe de CDELT1 : -1 standard (Est=gauche), +1 miroir (Est=droite)
    error: str = ""
    elapsed_s: float = 0.0

    def ra_hms(self) -> str:
        h = int(self.ra_deg / 15)
        m = int((self.ra_deg / 15 - h) * 60)
        s = ((self.ra_deg / 15 - h) * 60 - m) * 60
        return f"{h:02d}h{m:02d}m{s:04.1f}s"

    def dec_dms(self) -> str:
        sign = "+" if self.dec_deg >= 0 else "-"
        d = int(abs(self.dec_deg))
        m = int((abs(self.dec_deg) - d) * 60)
        return f"{sign}{d:02d}°{m:02d}'"


class PlateSolver:
    """Wraps ASTAP CLI for plate solving.

    G05 (Gaia) and/or D05 catalogues in /opt/astap (both installed).
    ASTAP auto-selects G05 for wide fields (FOV > ~5°), D05 for narrow.

    max_stars: top-N brightest stars used for matching (= implicit magnitude limit).
               200 is good for wide-angle (25mm), 500 for telephoto.
    downsample: 0=auto, 2=halve resolution (recommended for FOV > 7°),
                3=third resolution (for FOV > 15°, very wide angle).
    """

    def __init__(
        self,
        pixel_size_um: float = 2.9,
        timeout_s: float = _DEFAULT_TIMEOUT,
        catalog_dir: str = ASTAP_CATALOG_DIR,
        max_stars: int = 500,
        downsample: int = 0,
    ) -> None:
        self.pixel_size_um = pixel_size_um
        self.timeout_s = timeout_s
        self.catalog_dir = catalog_dir
        self.max_stars = max(10, max_stars)
        self.downsample = max(0, downsample)
        self._proc: Optional[subprocess.Popen] = None
        self._log_catalog_info()

    def plate_scale_arcsec(self, focal_mm: float) -> float:
        return 206.265 * self.pixel_size_um / focal_mm

    def solve(
        self,
        image_bgr: np.ndarray,
        focal_mm: float,
        hint_ra: Optional[float] = None,
        hint_dec: Optional[float] = None,
        hint_radius_deg: float = 180.0,
    ) -> SolveResult:
        """Run plate solving on *image_bgr* (uint8 or uint16 BGR numpy array).

        *hint_ra/dec* (degrees, J2000) and *hint_radius_deg* narrow the search.
        """
        t0 = time.time()
        scale_arcsec = self.plate_scale_arcsec(focal_mm)
        h, w = image_bgr.shape[:2]
        fov_h_deg = scale_arcsec * h / 3600.0

        print(f"[PlateSolver] ASTAP — focal={focal_mm:.0f}mm pixel={self.pixel_size_um}µm "
              f"→ scale={scale_arcsec:.2f}\"/px FOV_h={fov_h_deg:.2f}° "
              f"stars={self.max_stars} z={self.downsample}")

        gray = self._to_gray(image_bgr)

        with tempfile.TemporaryDirectory(prefix="astap_solve_") as tmp:
            img_path = os.path.join(tmp, "input.fits")
            self._write_fits(gray, img_path)

            cmd = [
                ASTAP_BIN,
                "-f", img_path,
                "-fov", f"{fov_h_deg:.4f}",
                "-r", f"{min(hint_radius_deg, 180.0):.1f}",
                "-d", self.catalog_dir,
                "-z", str(self.downsample),
                "-s", str(self.max_stars),
            ]

            if hint_ra is not None and hint_dec is not None and hint_radius_deg < 60.0:
                ra_hours = hint_ra / 15.0
                spd = 90.0 + hint_dec  # ASTAP: SPD = 90 + Dec (distance depuis pôle sud)
                cmd += ["-ra", f"{ra_hours:.6f}", "-spd", f"{spd:.4f}"]
                print(f"[PlateSolver] Hint RA={hint_ra:.3f}° ({ra_hours:.4f}h) "
                      f"Dec={hint_dec:.3f}° SPD={spd:.3f}° r={hint_radius_deg:.1f}°")

            print(f"[PlateSolver] Commande: {' '.join(cmd)}")

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    out, err = self._proc.communicate(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.communicate()
                    return SolveResult(
                        success=False, error="Timeout ASTAP",
                        elapsed_s=time.time() - t0,
                    )
                finally:
                    self._proc = None
            except FileNotFoundError:
                return SolveResult(success=False, error="astap introuvable")
            except Exception as exc:
                return SolveResult(success=False, error=str(exc))

            for line in out.decode(errors="replace").splitlines():
                if line.strip():
                    print(f"  [astap] {line}")
            for line in err.decode(errors="replace").splitlines():
                if line.strip() and "Gtk-WARNING" not in line and "fontconfig" not in line:
                    print(f"  [astap-err] {line}")

            ini_path = img_path.replace(".fits", ".ini")
            return self._parse_ini(ini_path, w, h, scale_arcsec, time.time() - t0)

    def cancel(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_catalog_info(self) -> None:
        try:
            files = os.listdir(self.catalog_dir)
        except OSError:
            print(f"[PlateSolver] Répertoire catalogue introuvable: {self.catalog_dir}")
            return
        g05 = sum(1 for f in files if f.startswith("g05"))
        d05 = sum(1 for f in files if f.startswith("d05"))
        catalogs = []
        if g05:
            catalogs.append(f"G05×{g05}")
        if d05:
            catalogs.append(f"D05×{d05}")
        print(f"[PlateSolver] Catalogues disponibles: {', '.join(catalogs) if catalogs else 'AUCUN'} "
              f"dans {self.catalog_dir}")
        print(f"[PlateSolver] Config: max_stars={self.max_stars} downsample={self.downsample} "
              f"timeout={self.timeout_s:.0f}s")

    @staticmethod
    def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim == 2:
            return image_bgr  # déjà en niveaux de gris
        if image_bgr.dtype == np.uint16:
            return image_bgr[:, :, 1]  # canal vert (moins de bruit que R ou B)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _write_fits(gray: np.ndarray, path: str) -> None:
        try:
            from astropy.io import fits as _fits
            _fits.PrimaryHDU(gray).writeto(path, overwrite=True)
        except ImportError:
            # Minimal FITS writer (2880-byte blocks, no compression)
            import struct
            data = gray.astype(">u2").tobytes()
            bitpix = 16
            naxis1, naxis2 = gray.shape[1], gray.shape[0]
            header = (
                f"SIMPLE  =                    T / file conforms to FITS standard".ljust(80) +
                f"BITPIX  = {bitpix:>20d} / number of bits per data pixel".ljust(80) +
                f"NAXIS   =                    2 / number of data axes".ljust(80) +
                f"NAXIS1  = {naxis1:>20d} / length of data axis 1".ljust(80) +
                f"NAXIS2  = {naxis2:>20d} / length of data axis 2".ljust(80) +
                "END".ljust(80)
            )
            header = header.encode("ascii")
            header += b" " * (2880 - len(header) % 2880)
            pad = (2880 - len(data) % 2880) % 2880
            with open(path, "wb") as f:
                f.write(header + data + b"\x00" * pad)

    def _parse_ini(
        self,
        ini_path: str,
        img_w: int,
        img_h: int,
        nominal_scale_arcsec: float,
        elapsed: float,
    ) -> SolveResult:
        if not os.path.exists(ini_path):
            return SolveResult(success=False, error="Pas de fichier .ini ASTAP", elapsed_s=elapsed)

        kv: dict[str, str] = {}
        with open(ini_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, _, v = line.partition("=")
                    kv[k.strip()] = v.strip()

        print(f"[PlateSolver] ASTAP .ini: {kv}")

        if kv.get("PLTSOLVD") != "T":
            err = kv.get("ERROR", kv.get("WARNING", "Pas de solution"))
            return SolveResult(success=False, error=err, elapsed_s=elapsed)

        try:
            ra_deg = float(kv["CRVAL1"]) % 360.0
            dec_deg = float(kv["CRVAL2"])
            # CDELT1 en degrés/pixel : négatif = Est à gauche (standard), positif = Est à droite (miroir)
            raw_cdelt1 = float(kv.get("CDELT1", -nominal_scale_arcsec / 3600.0))
            cdelt1_sign = 1 if raw_cdelt1 > 0 else -1
            cdelt = abs(raw_cdelt1)
            pix_scale = cdelt * 3600.0  # arcsec/px
            angle = float(kv.get("CROTA2", kv.get("CROTA1", 0.0)))
            field_w = pix_scale * img_w / 3600.0
            field_h = pix_scale * img_h / 3600.0
        except (KeyError, ValueError) as exc:
            return SolveResult(success=False, error=f"Parse .ini: {exc}", elapsed_s=elapsed)

        log.info("[PlateSolver] OK RA=%.4f Dec=%.4f scale=%.2f\"/px angle=%.1f° (%.1fs)",
                 ra_deg, dec_deg, pix_scale, angle, elapsed)

        focal_measured = 206.265 * self.pixel_size_um / pix_scale if pix_scale > 0 else 0.0

        return SolveResult(
            success=True,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            field_angle_deg=angle,
            pixel_scale_arcsec=pix_scale,
            field_w_deg=field_w,
            field_h_deg=field_h,
            focal_mm_measured=focal_measured,
            cdelt1_sign=cdelt1_sign,
            elapsed_s=elapsed,
        )
