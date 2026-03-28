"""
wavefront_psf.py — Calcul de PSF par optique de Fourier + polynômes de Zernike

Pipeline :
  1. Pupille circulaire (avec obstruction centrale optionnelle)
  2. Front d'onde W(rho,theta) = somme de polynômes de Zernike
  3. Fonction pupille complexe : P = aperture * exp(i * 2pi * W)
  4. PSF optique = |FFT(P)|²  (diffraction de Fraunhofer)
  5. Convolution avec le disque de seeing (turbulence atmosphérique)
  6. Rééchantillonnage vers l'échelle du capteur (arcsec/pixel)
  7. Découpage au support effectif (3x FWHM)

Paramètres exposés à l'UI :
  aperture_D_mm       : diamètre du télescope (mm)
  obstruction_ratio   : rapport obstruction/D (0=libre, 0.35=SCT typique)
  pixel_scale_arcsec  : échelle image en arcsec/pixel
  wavelength_nm       : longueur d'onde centrale (nm), défaut 550
  seeing_arcsec       : FWHM du seeing (arcsec), 0=désactivé
  z_defocus           : coefficient Zernike Z4  (defocus, en lambda RMS)
  z_astig_x           : coefficient Zernike Z5  (astigmatisme oblique, lambda)
  z_astig_y           : coefficient Zernike Z6  (astigmatisme droit, lambda)
  z_coma_x            : coefficient Zernike Z7  (coma X, lambda)
  z_coma_y            : coefficient Zernike Z8  (coma Y, lambda)
  z_spherical         : coefficient Zernike Z11 (aberration sphérique, lambda)

Usage typique :
  psf = compute_wavefront_psf(aperture_D_mm=200, obstruction_ratio=0.35,
                               pixel_scale_arcsec=0.2, seeing_arcsec=1.5,
                               z_defocus=0.0, z_astig_x=0.3)
  # psf : array float32 (K, K) normalisé (somme=1), K impair
"""

import math
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.ndimage import zoom


# ─────────────────────────────────────────────────────────────────────────────
# Polynômes de Zernike (indexation Noll, sur disque unitaire)
# ─────────────────────────────────────────────────────────────────────────────

def _zernike_radial(n, m, rho):
    """Partie radiale R_n^m(rho) du polynôme de Zernike."""
    R = np.zeros_like(rho)
    for s in range((n - abs(m)) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s) /
             (math.factorial(s) *
              math.factorial((n + abs(m)) // 2 - s) *
              math.factorial((n - abs(m)) // 2 - s)))
        R += c * rho ** (n - 2 * s)
    return R


def _zernike(j, rho, theta):
    """
    Polynôme de Zernike Z_j en indexation Noll.
    Modes courants :
      j=4  : defocus          n=2 m=0
      j=5  : astigmatisme ×   n=2 m=-2
      j=6  : astigmatisme +   n=2 m=+2
      j=7  : coma X           n=3 m=-1
      j=8  : coma Y           n=3 m=+1
      j=11 : sphérique        n=4 m=0
    """
    # Table Noll → (n, m)
    _NOLL = {
        1:  (0,  0),
        2:  (1,  1),  3: (1, -1),
        4:  (2,  0),
        5:  (2, -2),  6: (2,  2),
        7:  (3, -1),  8: (3,  1),
        9:  (3, -3), 10: (3,  3),
        11: (4,  0),
        12: (4, -2), 13: (4,  2),
        14: (4, -4), 15: (4,  4),
    }
    if j not in _NOLL:
        raise ValueError(f"Mode de Zernike j={j} non supporté (max j=15)")
    n, m = _NOLL[j]
    R = _zernike_radial(n, abs(m), rho)
    if m == 0:
        norm = np.sqrt(n + 1)
        return norm * R
    elif m > 0:
        norm = np.sqrt(2 * (n + 1))
        return norm * R * np.cos(abs(m) * theta)
    else:
        norm = np.sqrt(2 * (n + 1))
        return norm * R * np.sin(abs(m) * theta)


# ─────────────────────────────────────────────────────────────────────────────
# Calcul principal de la PSF
# ─────────────────────────────────────────────────────────────────────────────

def compute_wavefront_psf(
    aperture_D_mm     = 200.0,
    obstruction_ratio = 0.0,
    pixel_scale_arcsec= 0.2,
    wavelength_nm     = 550.0,
    seeing_arcsec     = 1.5,
    z_defocus         = 0.0,
    z_astig_x         = 0.0,
    z_astig_y         = 0.0,
    z_coma_x          = 0.0,
    z_coma_y          = 0.0,
    z_spherical       = 0.0,
    pupil_size        = 256,
    min_kernel        = 7,
    max_kernel        = 63,
):
    """
    Calcule la PSF par optique de Fourier depuis les aberrations du front d'onde.

    Retourne :
        psf : np.ndarray float32 (K, K) normalisé (sum=1), K impair
              Prêt à être passé au pipeline Halide RL.
        fwhm_px : float — FWHM estimé en pixels capteur
    """
    N = pupil_size   # taille de la grille pupille (oversampled)

    # ── Grille normalisée sur la pupille [-1, 1] ──────────────────────────
    coords = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(coords, coords)
    rho    = np.sqrt(xx**2 + yy**2)
    theta  = np.arctan2(yy, xx)

    # ── Masque d'ouverture (anneau : obstruction ≤ rho ≤ 1) ──────────────
    aperture = (rho <= 1.0) & (rho >= max(obstruction_ratio, 0.0))

    # ── Front d'onde W (en unités de longueur d'onde) ────────────────────
    W = np.zeros((N, N), dtype=np.float64)
    zernike_coeffs = {
        4:  z_defocus,
        5:  z_astig_x,
        6:  z_astig_y,
        7:  z_coma_x,
        8:  z_coma_y,
        11: z_spherical,
    }
    for j, coeff in zernike_coeffs.items():
        if coeff != 0.0:
            W += coeff * _zernike(j, rho, theta)

    # ── Fonction pupille complexe ─────────────────────────────────────────
    pupil = aperture.astype(np.complex128) * np.exp(1j * 2.0 * np.pi * W)

    # ── PSF optique = |TF(pupille)|² ─────────────────────────────────────
    asf        = fftshift(fft2(pupil))
    psf_optic  = np.abs(asf) ** 2

    # Échelle angulaire d'un pixel de la grille PSF (arcsec)
    lambda_m          = wavelength_nm * 1e-9
    D_m               = aperture_D_mm * 1e-3
    # Rayon d'Airy en arcsec
    airy_radius_arcsec = np.degrees(1.22 * lambda_m / D_m) * 3600.0
    # Résolution d'un pixel FFT = lambda/(D) / N * factor
    # (la taille totale de la grille FFT correspond à 2×lambda/(D/N)×N = 2×lambda/D×N)
    # En pratique : le premier zéro d'Airy se trouve à ~1.22 pixels dans la grille FFT
    # (car la pupille remplit exactement les N pixels de diamètre)
    psf_pixel_arcsec = airy_radius_arcsec / 1.22  # arcsec par pixel FFT

    # ── Seeing (turbulence atmosphérique, Gaussienne) ─────────────────────
    if seeing_arcsec > 0.0:
        sigma_see_px = (seeing_arcsec / psf_pixel_arcsec) / 2.355  # FWHM → sigma
        sigma_see_px = max(sigma_see_px, 0.3)
        x_g = np.arange(N) - N // 2
        xx_g, yy_g = np.meshgrid(x_g, x_g)
        seeing_kern = np.exp(-(xx_g**2 + yy_g**2) / (2.0 * sigma_see_px**2))
        seeing_kern /= seeing_kern.sum()
        # Convolution par transformée de Fourier (sur petite grille → rapide)
        from scipy.fft import ifft2 as _ifft2
        psf_fft = fft2(psf_optic)
        see_fft = fft2(seeing_kern)
        psf_conv = np.real(_ifft2(psf_fft * see_fft))
        # Recentrer (ifft2 place le centre en (0,0))
        psf_conv = np.roll(np.roll(psf_conv, N // 2, axis=0), N // 2, axis=1)
        psf_conv = np.clip(psf_conv, 0.0, None)
    else:
        psf_conv = psf_optic

    # ── Rééchantillonnage vers l'échelle du capteur ───────────────────────
    # Facteur de zoom : psf_pixel_arcsec / pixel_scale_arcsec
    # (on veut que 1 pixel du capteur = pixel_scale_arcsec)
    zoom_factor = psf_pixel_arcsec / max(pixel_scale_arcsec, 0.01)
    if abs(zoom_factor - 1.0) > 0.01:
        psf_resampled = zoom(psf_conv, zoom_factor, order=3, mode='constant', cval=0.0)
    else:
        psf_resampled = psf_conv

    psf_resampled = np.clip(psf_resampled, 0.0, None)
    total = psf_resampled.sum()
    if total < 1e-30:
        # PSF dégénérée — retourner une PSF Gaussienne de secours
        psf_resampled = _gaussian_psf_fallback(5, 1.5)
        return psf_resampled, 3.5

    psf_resampled /= total

    # ── Estimation du FWHM en pixels capteur ─────────────────────────────
    fwhm_px = _estimate_fwhm(psf_resampled)

    # ── Découpage au support effectif (centré) ────────────────────────────
    psf_cropped = _crop_to_support(psf_resampled, fwhm_px, min_kernel, max_kernel)

    # Renormaliser après découpage
    psf_cropped = np.clip(psf_cropped, 0.0, None)
    psf_cropped /= psf_cropped.sum()

    return psf_cropped.astype(np.float32), float(fwhm_px)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_fwhm(psf):
    """Estimation du FWHM radial en pixels (à partir du profil central)."""
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    peak   = psf[cy, cx]
    half   = peak / 2.0
    # Profil horizontal à travers le centre
    row = psf[cy, :]
    # Trouver les indices où le profil croise half-maximum
    above = row >= half
    if not above.any():
        return 3.0
    left  = np.argmax(above)
    right = len(above) - 1 - np.argmax(above[::-1])
    return float(right - left + 1)


def _crop_to_support(psf, fwhm_px, min_k, max_k):
    """Découpe la PSF à 3×FWHM centré, taille impaire bornée entre min_k et max_k."""
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    half_k = max(int(fwhm_px * 1.5), (min_k - 1) // 2)
    half_k = min(half_k, (max_k - 1) // 2)
    k = 2 * half_k + 1  # taille impaire

    h, w = psf.shape
    y0 = max(cy - half_k, 0)
    y1 = min(cy + half_k + 1, h)
    x0 = max(cx - half_k, 0)
    x1 = min(cx + half_k + 1, w)

    crop = psf[y0:y1, x0:x1]

    # Pad si le découpage dépasse les bords
    if crop.shape != (k, k):
        pad_y0 = max(half_k - cy, 0)
        pad_y1 = max(cy + half_k + 1 - h, 0)
        pad_x0 = max(half_k - cx, 0)
        pad_x1 = max(cx + half_k + 1 - w, 0)
        crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode='constant')

    return crop


def _gaussian_psf_fallback(radius, sigma):
    """PSF Gaussienne de secours (k = 2*radius+1)."""
    k = 2 * radius + 1
    x = np.arange(k) - radius
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    return psf.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires pour l'UI
# ─────────────────────────────────────────────────────────────────────────────

def psf_info(psf, fwhm_px, pixel_scale_arcsec):
    """Retourne un dict de métriques lisibles pour l'UI."""
    return {
        'kernel_size': psf.shape[0],
        'fwhm_px':     round(fwhm_px, 1),
        'fwhm_arcsec': round(fwhm_px * pixel_scale_arcsec, 2),
        'peak':        round(float(psf.max()), 5),
        'energy_center': round(float(psf[psf.shape[0]//2-2:psf.shape[0]//2+3,
                                         psf.shape[1]//2-2:psf.shape[1]//2+3].sum()), 3),
    }
