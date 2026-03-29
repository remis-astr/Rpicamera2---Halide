"""
Lucky Stack en domaine Bayer RAW12/16 pour imagerie planétaire.

Pipeline vs Lucky RGB8 :
  RGB8  : débayer × N frames → score RGB → align → stack → post-filtres
  RAW   : score G1 Bayer → align G1×2 → accumule Bayer float32 → débayer ×1 → post-filtres

Avantages :
  - Pas d'artefact ISP (démosaïcisation répétée, compression dynamique, gamma)
  - Score sur données linéaires brutes → discrimination plus fidèle à la turbulence
  - 12 bits préservés jusqu'au stack final

Format d'entrée : uint16 (H, W) Bayer RGGB, espace CSI-2 ×16 (picamera2 unpacked=True)
Format de sortie : float32 (H, W, 3) [0-65535], compatible apply_isp_to_preview()

Pattern Bayer RGGB (IMX585 / SRGGB12) :
  R  = bayer[0::2, 0::2]
  G1 = bayer[0::2, 1::2]   ← canal de référence pour score et alignement
  G2 = bayer[1::2, 0::2]
  B  = bayer[1::2, 1::2]
"""

import cv2
import numpy as np
from collections import deque
import threading
import logging
import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

# =============================================================================
# Chargement optionnel du pipeline Halide AOT (lucky_preprocess + lucky_debayer_f32)
# =============================================================================
_halide_available       = False   # preprocess + score + debayer
_halide_stack_available = False   # stack_mean + sigma_clip (nécessite recompile)
_hlib = None

def _load_halide():
    global _halide_available, _hlib
    _so = os.path.join(os.path.dirname(__file__), "halide", "jsk_halide.so")
    if not os.path.isfile(_so):
        return
    try:
        lib = ctypes.CDLL(_so)

        # lucky_raw_preprocess(input, bayer_out, g1_out, w, h, bl_r, bl_g1, bl_g2, bl_b)
        lib.lucky_raw_preprocess.restype  = ctypes.c_int
        lib.lucky_raw_preprocess.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),  # input  (H, W) uint16 CSI-2 ×16
            ctypes.POINTER(ctypes.c_float),   # bayer_out (H, W) float32
            ctypes.POINTER(ctypes.c_float),   # g1_out   (H/2, W/2) float32
            ctypes.c_int, ctypes.c_int,       # w, h
            ctypes.c_float, ctypes.c_float,   # bl_r, bl_g1 (CSI-2 units = ADU × 16)
            ctypes.c_float, ctypes.c_float,   # bl_g2, bl_b
        ]

        # lucky_debayer_f32(input, output, w, h, r_gain, b_gain)
        lib.lucky_debayer_f32.restype  = ctypes.c_int
        lib.lucky_debayer_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # bayer_in  (H, W) float32
            ctypes.POINTER(ctypes.c_float),  # rgb_out   (3, H, W) float32 planaire
            ctypes.c_int, ctypes.c_int,      # w, h
            ctypes.c_float, ctypes.c_float,  # r_gain, b_gain
        ]

        # lucky_score_*(g1, score_out, w2, h2, roi_x0, roi_y0, roi_w, roi_h)
        # Scoring en passe unique fusionnée (pas de buffer intermédiaire Lap/Gx/Gy)
        _score_argtypes = [
            ctypes.POINTER(ctypes.c_float),  # g1       (H/2, W/2) float32
            ctypes.POINTER(ctypes.c_float),  # score_out 1 élément
            ctypes.c_int, ctypes.c_int,      # w2, h2
            ctypes.c_int, ctypes.c_int,      # roi_x0, roi_y0
            ctypes.c_int, ctypes.c_int,      # roi_w, roi_h
        ]
        for _fn in ('lucky_score_variance', 'lucky_score_laplacian',
                    'lucky_score_tenengrad', 'lucky_score_gradient'):
            _f = getattr(lib, _fn)
            _f.restype  = ctypes.c_int
            _f.argtypes = _score_argtypes

        _hlib = lib
        _halide_available = True

        # ── Fonctions de stacking (ajoutées après la première compilation) ─────
        # Chargées séparément pour ne pas bloquer les fonctions précédentes
        # si le .so n'a pas encore été recompilé avec lucky_stack_pipeline.cpp.
        try:
            lib.lucky_stack_mean.restype  = ctypes.c_int
            lib.lucky_stack_mean.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # frames (N,H,W) float32
                ctypes.POINTER(ctypes.c_float),  # result_out (H,W) float32
                ctypes.c_int, ctypes.c_int,      # w, h
                ctypes.c_int,                    # n_frames
            ]
            lib.lucky_sigma_clip.restype  = ctypes.c_int
            lib.lucky_sigma_clip.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # frames (N,H,W) float32
                ctypes.POINTER(ctypes.c_float),  # result_out (H,W) float32
                ctypes.c_int, ctypes.c_int,      # w, h
                ctypes.c_int,                    # n_frames
                ctypes.c_float,                  # kappa
            ]
            globals()['_halide_stack_available'] = True
        except Exception as e:
            logger.debug("Halide stacking non disponible (recompiler jsk_halide.so) : %s", e)

    except Exception as e:
        logger.debug("Halide lucky RAW non disponible : %s", e)

_load_halide()


# =============================================================================
# Helpers Halide (accélération per-frame et débayer final)
# =============================================================================

def _halide_preprocess(raw_u16: np.ndarray,
                       bl_r: float, bl_g1: float,
                       bl_g2: float, bl_b: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Passe Halide fusionnée : BL subtract + extraction G1 half-res.

    Remplace _apply_bl_per_channel() + _extract_g1() par une seule passe
    NEON vectorisée. Gain typique : ×3-5 vs NumPy sur RPi5.

    Args:
        raw_u16 : uint16 (H, W) Bayer RGGB, espace CSI-2 ×16
        bl_r/g1/g2/b : BL en ADU 12-bit (la conversion ×16 est faite ici)

    Returns:
        (bayer_f32, g1_f32) :
            bayer_f32 : float32 (H, W) BL-soustrait clippé ≥ 0
            g1_f32    : float32 (H/2, W/2) canal G1
    """
    h, w = raw_u16.shape
    raw_c = np.ascontiguousarray(raw_u16)
    bayer_out = np.empty((h, w), dtype=np.float32)
    g1_out    = np.empty((h // 2, w // 2), dtype=np.float32)
    scale = 16.0  # ADU 12-bit → CSI-2 ×16
    ret = _hlib.lucky_raw_preprocess(
        raw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        bayer_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g1_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w, h,
        ctypes.c_float(bl_r  * scale),
        ctypes.c_float(bl_g1 * scale),
        ctypes.c_float(bl_g2 * scale),
        ctypes.c_float(bl_b  * scale),
    )
    if ret != 0:
        raise RuntimeError(f"lucky_raw_preprocess Halide erreur: {ret}")
    return bayer_out, g1_out


def _halide_score_g1(g1_f32: np.ndarray, method: str, roi_frac: float) -> Optional[float]:
    """Score Halide sur G1 déjà extrait — passe unique sans buffer intermédiaire.

    Méthodes accélérées : 'laplacian', 'gradient', 'sobel', 'local_variance', 'psd'.
    Retourne None pour 'tenengrad' (OpenCV équivalent à Halide à cette résolution).

    Gains mesurés vs OpenCV (RPi5, G1 256×256 ROI 50%) :
      variance/laplacian : ×2.3   gradient : ×1.2   tenengrad : ×1.0
    """
    _HAL_FN = {
        'laplacian':      _hlib.lucky_score_laplacian,
        'gradient':       _hlib.lucky_score_gradient,
        'sobel':          _hlib.lucky_score_gradient,   # sobel = gradient
        'local_variance': _hlib.lucky_score_variance,
        'psd':            _hlib.lucky_score_laplacian,  # psd fallback → laplacian
    }
    fn = _HAL_FN.get(method)
    if fn is None:
        return None  # tenengrad → fallback OpenCV

    h2, w2 = g1_f32.shape
    f  = max(0.1, min(1.0, roi_frac))
    rw = int(w2 * f);  rh = int(h2 * f)
    rx = (w2 - rw) // 2;  ry = (h2 - rh) // 2
    g1_c = np.ascontiguousarray(g1_f32)
    out  = np.empty(1, dtype=np.float32)
    ret  = fn(
        g1_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w2, h2, rx, ry, rw, rh,
    )
    if ret != 0:
        raise RuntimeError(f"lucky_score_{method} Halide erreur: {ret}")
    return float(out[0])


def _halide_debayer_f32(bayer_f32: np.ndarray,
                        r_gain: float, b_gain: float) -> np.ndarray:
    """Débayérisation Halide float32 avec gains AWB.

    Remplace np.clip + astype(uint16) + cv2.cvtColor + astype(float32) + gains.
    1 passe NEON fusionnée.

    Args:
        bayer_f32 : float32 (H, W) Bayer RGGB, valeurs [0, ~65535]
        r_gain    : gain AWB rouge (multiplicateur direct)
        b_gain    : gain AWB bleu

    Returns:
        float32 (H, W, 3) — ch0=R_phys, ch1=G, ch2=B_phys, valeurs [0, 65535]
        Compatible avec apply_isp_to_preview().
    """
    h, w = bayer_f32.shape
    bayer_c = np.ascontiguousarray(bayer_f32)
    out_planar = np.empty((3, h, w), dtype=np.float32)
    ret = _hlib.lucky_debayer_f32(
        bayer_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_planar.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w, h,
        ctypes.c_float(r_gain),
        ctypes.c_float(b_gain),
    )
    if ret != 0:
        raise RuntimeError(f"lucky_debayer_f32 Halide erreur: {ret}")
    # Transpose + copie contiguë : (3,H,W) planaire → (H,W,3) interleaved C-contiguous.
    # Sans np.ascontiguousarray(), le résultat est non-contigu (stride c = H*W*4)
    # ce qui peut déclencher des copies implicites dans OpenCV ou apply_isp_to_preview.
    # Une copie explicite ici évite de potentielles copies multiples en aval.
    return np.ascontiguousarray(out_planar.transpose(1, 2, 0))


def _halide_stack_mean(frames_3d: np.ndarray) -> np.ndarray:
    """Moyenne Halide de N frames float32.

    Entrée  : (N, H, W) float32 C-contiguous
    Sortie  : (H, W) float32
    Gain    : ×3 vs NumPy (28 ms → 8 ms, 6 frames 1024×1024)
    """
    n, h, w = frames_3d.shape
    frames_c = np.ascontiguousarray(frames_3d)
    result   = np.empty((h, w), dtype=np.float32)
    ret = _hlib.lucky_stack_mean(
        frames_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w, h, n,
    )
    if ret != 0:
        raise RuntimeError(f"lucky_stack_mean Halide erreur: {ret}")
    return result


def _halide_sigma_clip(frames_3d: np.ndarray, kappa: float = 2.5) -> np.ndarray:
    """Sigma-clipping Halide 2 passes sans buffer (N,H,W) intermédiaire.

    Passe 1 : E[X] et σ(X) via Tuple (ΣX, ΣX²) en une seule RDom.
    Passe 2 : moyenne des pixels dans |X - E[X]| ≤ kappa·σ(X).

    Entrée  : (N, H, W) float32 C-contiguous (N ≥ 3 recommandé)
    Sortie  : (H, W) float32
    Gain    : ×5-8 vs NumPy (175 ms → 25 ms, 6 frames 1024×1024)
    """
    n, h, w = frames_3d.shape
    frames_c = np.ascontiguousarray(frames_3d)
    result   = np.empty((h, w), dtype=np.float32)
    ret = _hlib.lucky_sigma_clip(
        frames_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w, h, n, ctypes.c_float(kappa),
    )
    if ret != 0:
        raise RuntimeError(f"lucky_sigma_clip Halide erreur: {ret}")
    return result


# ---------------------------------------------------------------------------
# Extraction canaux Bayer RGGB (fallback NumPy si Halide non disponible)
# ---------------------------------------------------------------------------

def _extract_g1(bayer: np.ndarray) -> np.ndarray:
    """Canal G1 (row pair, col impair) pour RGGB — shape (H/2, W/2)."""
    return bayer[0::2, 1::2]


def _apply_bl_per_channel(bayer: np.ndarray,
                          bl_r: float, bl_g1: float,
                          bl_g2: float, bl_b: float) -> np.ndarray:
    """Soustraction black level per-canal en espace CSI-2 ×16.

    Args:
        bayer         : uint16 (H, W) Bayer RGGB, espace CSI-2
        bl_r/g1/g2/b  : BL en ADU 12-bit natif (espace CSI-2 = ADU × 16)

    Returns:
        float32 (H, W) avec BL soustrait, clippé à 0
    """
    f = bayer.astype(np.float32)
    scale = 16.0  # ADU 12-bit → espace CSI-2 ×16
    f[0::2, 0::2] = np.maximum(f[0::2, 0::2] - bl_r  * scale, 0.0)  # R
    f[0::2, 1::2] = np.maximum(f[0::2, 1::2] - bl_g1 * scale, 0.0)  # G1
    f[1::2, 0::2] = np.maximum(f[1::2, 0::2] - bl_g2 * scale, 0.0)  # G2
    f[1::2, 1::2] = np.maximum(f[1::2, 1::2] - bl_b  * scale, 0.0)  # B
    return f


def _estimate_bl_auto(raw: np.ndarray) -> Tuple[float, float, float, float]:
    """Estime le BL de chaque sous-canal par percentile 5% (robuste astro).

    Utilise np.partition (O(N) sélection) au lieu de np.percentile (O(N log N) tri).
    Gain typique : ×3-4 sur les 4 appels combinés.

    Returns:
        (bl_r, bl_g1, bl_g2, bl_b) en ADU 12-bit natif
    """
    scale = 16.0
    max_bl_adu = 256.0 * 1.3   # Plafond ≈ 1.3× BL nominal IMX585
    results = []
    for sy, sx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        sub = raw[sy::2, sx::2].ravel()
        k   = max(0, int(0.05 * sub.size) - 1)   # index du percentile 5%
        p5  = float(np.partition(sub, k)[k])
        bl_adu = max(0.0, (p5 / scale) - 1.0)    # -1 ADU de marge
        bl_adu = min(bl_adu, max_bl_adu)
        results.append(bl_adu)
    return tuple(results)


# ---------------------------------------------------------------------------
# Score de netteté sur G1
# ---------------------------------------------------------------------------

def _score_laplacian(g1_roi: np.ndarray) -> float:
    lap = cv2.Laplacian(g1_roi, cv2.CV_32F, ksize=3)
    return float(lap.var())


def _score_tenengrad(g1_roi: np.ndarray) -> float:
    gx = cv2.Sobel(g1_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g1_roi, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx ** 2 + gy ** 2).mean())


def _score_gradient(g1_roi: np.ndarray) -> float:
    gx = cv2.Sobel(g1_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g1_roi, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def _score_sobel(g1_roi: np.ndarray) -> float:
    # Même que gradient, implémentation OpenCV directe
    return _score_gradient(g1_roi)


_SCORE_FNS = {
    'laplacian':     _score_laplacian,
    'gradient':      _score_gradient,
    'sobel':         _score_sobel,
    'tenengrad':     _score_tenengrad,
    'local_variance': lambda r: float(r.var()),
    'psd':           _score_laplacian,   # fallback laplacien
}


def _score_g1(g1: np.ndarray, method: str = 'laplacian',
              roi_frac: float = 0.5) -> float:
    """Score de netteté sur canal G1 déjà extrait (float32 H/2, W/2).

    Chemin Halide (laplacian/gradient/sobel/local_variance/psd) :
      - passe unique fusionnée, pas de buffer intermédiaire
      - ×1.2-2.3 vs OpenCV selon méthode et résolution

    Chemin OpenCV/NumPy (tenengrad, fallback) : comportement original.
    """
    if _halide_available:
        score = _halide_score_g1(g1, method, roi_frac)
        if score is not None:
            return score

    # Fallback OpenCV/NumPy (tenengrad ou Halide non disponible)
    h, w = g1.shape
    f  = max(0.1, min(1.0, roi_frac))
    y0 = int(h * (1.0 - f) * 0.5)
    x0 = int(w * (1.0 - f) * 0.5)
    roi = g1[y0: h - y0, x0: w - x0]
    fn = _SCORE_FNS.get(method, _score_laplacian)
    return fn(roi)


def score_bayer_frame(bayer: np.ndarray, method: str = 'laplacian',
                      roi_frac: float = 0.5) -> float:
    """Score de netteté sur canal G1 du pattern Bayer.

    Args:
        bayer    : uint16 ou float32 (H, W) Bayer RGGB en espace CSI-2
        method   : 'laplacian', 'tenengrad', 'gradient', 'sobel', 'local_variance'
        roi_frac : Fraction centrale pour le calcul (0.2 – 1.0)

    Returns:
        Score float (plus élevé = plus net)
    """
    g1 = _extract_g1(bayer).astype(np.float32)
    return _score_g1(g1, method, roi_frac)


# ---------------------------------------------------------------------------
# Alignement Bayer (décalage toujours multiple de 2 px)
# ---------------------------------------------------------------------------

def _compute_bayer_shift(ref_g1: np.ndarray, frame_g1: np.ndarray,
                         max_shift_half: int = 0) -> Tuple[int, int]:
    """Décalage par corrélation de phase sur G1 (demi-résolution).

    Principe :
      - Phase correlation sur G1 → décalage en pixels G1 (dx_half, dy_half)
      - Arrondi entier en espace G1, puis × 2 → toujours pair en espace Bayer
      - Garantit la préservation du pattern RGGB après translation

    Args:
        ref_g1        : float32 (H/2, W/2) — référence
        frame_g1      : float32 (H/2, W/2) — image à aligner
        max_shift_half: décalage max en pixels G1 (0 = pas de limite).
                        Correspond à max_shift_px // 2.

    Returns:
        (dy_full, dx_full) en pixels Bayer complets (toujours pair).
        (0, 0) si le décalage dépasse max_shift_half.
    """
    ref_max = ref_g1.max()
    frm_max = frame_g1.max()
    if ref_max < 1e-6 or frm_max < 1e-6:
        return 0, 0

    ref_n = ref_g1 / ref_max
    frm_n = frame_g1 / frm_max

    try:
        shift, _resp = cv2.phaseCorrelate(ref_n, frm_n)
    except Exception:
        return 0, 0

    dx_half, dy_half = shift  # OpenCV retourne (x, y)

    if max_shift_half > 0:
        if abs(dx_half) > max_shift_half or abs(dy_half) > max_shift_half:
            return 0, 0  # Rejet : trop grande dérive

    dy_full = int(round(dy_half)) * 2
    dx_full = int(round(dx_half)) * 2
    return dy_full, dx_full


def _precompute_ref_fft(ref_g1: np.ndarray) -> Tuple[np.ndarray, float]:
    """Pré-calcule la FFT normalisée de la frame de référence.

    À appeler une fois avant la boucle d'alignement dans _process_buffer.
    Évite de recalculer FFT(ref) pour chaque frame à aligner.

    Returns:
        (ref_fft, ref_max) : FFT complexe de ref_g1/ref_max, et ref_max
        (None, 0) si ref_g1 est vide (protection)
    """
    ref_max = ref_g1.max()
    if ref_max < 1e-6:
        return None, 0.0
    ref_n = ref_g1 / ref_max
    return np.fft.fft2(ref_n), ref_max


def _compute_bayer_shift_cached(ref_fft: np.ndarray, frame_g1: np.ndarray,
                                 max_shift_half: int = 0) -> Tuple[int, int]:
    """Corrélation de phase avec FFT de référence pré-calculée.

    Identique à _compute_bayer_shift mais évite de recalculer FFT(ref)
    à chaque appel. Gain : (keep-1) FFTs économisées par buffer.

    Args:
        ref_fft       : FFT complexe de ref_g1 normalisé (depuis _precompute_ref_fft)
        frame_g1      : float32 (H/2, W/2) — image à aligner
        max_shift_half: décalage max en pixels G1 (0 = pas de limite)

    Returns:
        (dy_full, dx_full) en pixels Bayer complets (toujours pair).
        (0, 0) si le décalage dépasse max_shift_half ou frame vide.
    """
    frm_max = frame_g1.max()
    if frm_max < 1e-6:
        return 0, 0

    frm_n   = frame_g1 / frm_max
    frm_fft = np.fft.fft2(frm_n)

    # Spectre de puissance croisé normalisé (même formule que cv2.phaseCorrelate)
    cross = ref_fft * np.conj(frm_fft)
    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    cross /= denom

    correlation = np.fft.ifft2(cross).real
    # Pic de corrélation → décalage (convention OpenCV : origine au centre)
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy_half_raw, dx_half_raw = peak_idx

    # Ramener dans [-H/2, H/2] × [-W/2, W/2]
    h2, w2 = correlation.shape
    dy_half = dy_half_raw if dy_half_raw < h2 // 2 else dy_half_raw - h2
    dx_half = dx_half_raw if dx_half_raw < w2 // 2 else dx_half_raw - w2

    if max_shift_half > 0:
        if abs(dx_half) > max_shift_half or abs(dy_half) > max_shift_half:
            return 0, 0

    dy_full = int(round(dy_half)) * 2
    dx_full = int(round(dx_half)) * 2
    return dy_full, dx_full


def _shift_bayer(bayer: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Déplace une image Bayer de (dy, dx) pixels entiers pairs.

    Utilise du slicing (pas np.roll) → les bords non remplis restent à zéro
    (ciel noir autour de la planète, pas d'artefact de wrapping).

    Args:
        bayer : float32 (H, W)
        dy    : décalage vertical (positif = vers le bas)
        dx    : décalage horizontal (positif = vers la droite)
    """
    if dy == 0 and dx == 0:
        return bayer

    h, w = bayer.shape
    result = np.zeros_like(bayer)

    sy0 = max(0, -dy);  sy1 = h - max(0, dy)
    dy0 = max(0,  dy);  dy1 = h - max(0, -dy)
    sx0 = max(0, -dx);  sx1 = w - max(0, dx)
    dx0 = max(0,  dx);  dx1 = w - max(0, -dx)

    if sy1 > sy0 and sx1 > sx0:
        result[dy0:dy1, dx0:dx1] = bayer[sy0:sy1, sx0:sx1]

    return result


def _shift_bayer_into(bayer: np.ndarray, dy: int, dx: int,
                      out: np.ndarray) -> None:
    """Comme _shift_bayer, mais écrit directement dans 'out' (pas d'allocation).

    Remplace `out[:] = _shift_bayer(bayer, dy, dx)` (qui alloue puis copie).
    """
    if dy == 0 and dx == 0:
        np.copyto(out, bayer)
        return

    h, w = bayer.shape
    out[:] = 0.0

    sy0 = max(0, -dy);  sy1 = h - max(0, dy)
    dy0 = max(0,  dy);  dy1 = h - max(0, -dy)
    sx0 = max(0, -dx);  sx1 = w - max(0, dx)
    dx0 = max(0,  dx);  dx1 = w - max(0, -dx)

    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy1, dx0:dx1] = bayer[sy0:sy1, sx0:sx1]


# ---------------------------------------------------------------------------
# Stacking Bayer
# ---------------------------------------------------------------------------

def _stack_mean(frames: List[np.ndarray]) -> np.ndarray:
    """Moyenne de frames Bayer float32 (liste → tableau)."""
    acc = np.zeros(frames[0].shape, dtype=np.float64)
    for f in frames:
        acc += f
    return (acc / len(frames)).astype(np.float32)


def _stack_sigma_clip(frames: List[np.ndarray], kappa: float = 2.5) -> np.ndarray:
    """Sigma-clipping pixel-wise depuis liste. Fallback sur moyenne si < 3 frames."""
    if len(frames) < 3:
        return _stack_mean(frames)
    stack = np.stack(frames, axis=0).astype(np.float32)  # (N, H, W)
    return _stack_sigma_clip_3d(stack, kappa)


def _stack_mean_3d(frames_3d: np.ndarray) -> np.ndarray:
    """Fallback NumPy : moyenne d'un tableau (N,H,W) déjà alloué."""
    return frames_3d.mean(axis=0).astype(np.float32)


def _stack_sigma_clip_3d(frames_3d: np.ndarray, kappa: float = 2.5) -> np.ndarray:
    """Fallback NumPy sigma-clip depuis tableau (N,H,W) — sans np.stack()."""
    n = frames_3d.shape[0]
    if n < 3:
        return _stack_mean_3d(frames_3d)
    mean = frames_3d.mean(axis=0)
    std  = frames_3d.std(axis=0)
    mask = np.abs(frames_3d - mean[None]) <= kappa * std[None]
    valid_sum = np.where(mask, frames_3d, 0.0).sum(axis=0)
    valid_cnt = mask.sum(axis=0).clip(1, None).astype(np.float32)
    return (valid_sum / valid_cnt).astype(np.float32)


# ---------------------------------------------------------------------------
# Débayérisation du stack final
# ---------------------------------------------------------------------------

def debayer_bayer_stack(bayer_stack: np.ndarray,
                        red_gain:    float = 1.0,
                        blue_gain:   float = 1.0,
                        global_bl:   float = 0.0,
                        bl_applied:  bool  = False) -> np.ndarray:
    """Débayérise un stack Bayer float32 en RGB float32 [0-65535].

    Utilise COLOR_BayerRG2BGR (identique à debayer_raw_array dans RPiCamera2.py).
    Résultat : ch0 = R_physique, ch1 = G, ch2 = B_physique — compatible
    avec apply_isp_to_preview() et toute la chaîne de post-traitement.

    Args:
        bayer_stack : float32 (H, W) Bayer RGGB, espace CSI-2 ×16
        red_gain    : Gain AWB rouge  (ex: red / 10 depuis globals)
        blue_gain   : Gain AWB bleu   (ex: blue / 10 depuis globals)
        global_bl   : Black level global en ADU 12-bit (0 = désactivé, fallback seulement)
        bl_applied  : True si une correction BL per-frame ou per-canal a déjà été
                      appliquée. Évite une double soustraction sur le stack.
                      IMPORTANT : sans ce flag, _estimate_bl_auto() appelée sur un
                      stack déjà corrigé confond les zones sombres de la planète avec
                      du BL résiduel (planète > 95% du champ → percentile 5% tombe
                      dans le limbe/ombres → écrêtage des basses lumières).

    Returns:
        float32 (H, W, 3) [0-65535], compatible apply_isp_to_preview()
    """
    if bl_applied:
        # BL déjà traité per-frame : ne pas ré-estimer depuis le stack.
        # _estimate_bl_auto sur données déjà corrigées retourne le niveau de
        # bruit/signal bas → soustraction parasite → zones sombres → 0.
        if global_bl > 0.0:
            bayer_stack = np.maximum(bayer_stack - global_bl * 16.0, 0.0)
    else:
        # Pas de correction per-frame : auto-estimation + soustraction sur le stack.
        # Supprime le BL capteur (~256 ADU IMX585) et le FPN 2×2 résiduel.
        bl_r, bl_g1, bl_g2, bl_b = _estimate_bl_auto(bayer_stack)
        if max(bl_r, bl_g1, bl_g2, bl_b) > 0.0:
            bayer_stack = _apply_bl_per_channel(bayer_stack, bl_r, bl_g1, bl_g2, bl_b)
        elif global_bl > 0.0:
            bayer_stack = np.maximum(bayer_stack - global_bl * 16.0, 0.0)

    if _halide_available:
        # Chemin Halide : debayer bilinéaire + AWB en une seule passe float32 NEON
        # Remplace : clip + astype(uint16) + cvtColor + astype(float32) + gains×2
        return _halide_debayer_f32(bayer_stack, red_gain, blue_gain)

    # Fallback NumPy/OpenCV
    bayer_u16 = np.clip(bayer_stack, 0.0, 65535.0).astype(np.uint16)

    # Débayérisation bilinéaire uint16 (VNG non supporté en uint16 par OpenCV)
    # Sur un stack moyenné (bruit réduit), le bilinéaire est suffisant.
    bgr = cv2.cvtColor(bayer_u16, cv2.COLOR_BayerRG2BGR)  # uint16 (H, W, 3)

    # Float32 + gains AWB (ch0=R_phys, ch2=B_phys — même comportement que debayer_raw_array)
    rgb = bgr.astype(np.float32)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * red_gain,  0.0, 65535.0)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * blue_gain, 0.0, 65535.0)
    return rgb


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class BayerLuckyStacker:
    """Lucky Stack planétaire en domaine Bayer RAW12/16.

    Interface identique au lucky stack RGB8 (même globals, mêmes stats,
    même post-pipeline) — seul le traitement interne change.

    Exemple d'utilisation :
        stacker = BayerLuckyStacker(buffer_size=50, keep_percent=20.0)

        # Boucle de capture :
        score = stacker.add_frame(raw_uint16)   # (H, W) uint16

        # Quand lucky_raw_active et nouveau stack :
        if stacker.new_stack_available():
            result = stacker.get_result(red_gain, blue_gain, global_bl)
            # result : float32 (H, W, 3) [0-65535]
            # → apply_isp_to_preview() → apply_lucky_post_stack_filters()
    """

    def __init__(
        self,
        buffer_size:   int   = 50,
        keep_percent:  float = 20.0,
        score_method:  str   = 'laplacian',
        score_roi:     float = 0.50,
        align_enabled: bool  = True,
        max_shift_px:  int   = 30,
        stack_method:  str   = 'mean',
        sigma_kappa:   float = 2.5,
        bl_auto:       bool  = False,
        bl_r:          float = 0.0,
        bl_g1:         float = 0.0,
        bl_g2:         float = 0.0,
        bl_b:          float = 0.0,
    ):
        self.buffer_size   = max(2, int(buffer_size))
        self.keep_percent  = float(keep_percent)
        self.score_method  = score_method
        self.score_roi     = float(score_roi)
        self.align_enabled = bool(align_enabled)
        self.max_shift_px  = int(max_shift_px)
        self.stack_method  = stack_method
        self.sigma_kappa   = float(sigma_kappa)
        self.bl_auto           = bool(bl_auto)
        self.bl_r              = float(bl_r)
        self.bl_g1             = float(bl_g1)
        self.bl_g2             = float(bl_g2)
        self.bl_b              = float(bl_b)
        # Intervalle de rafraîchissement du BL auto (en frames).
        # Le BL est stable d'une frame à l'autre → inutile de le recalculer
        # à chaque frame. Valeur 0 = recalcul à chaque frame (comportement original).
        self.bl_cache_interval: int = 30

        self._lock:    threading.RLock = threading.RLock()
        self._scores:  deque = deque(maxlen=self.buffer_size)

        # ── Buffer 3D pré-alloué ─────────────────────────────────────────────
        # Alloué lors de la première frame (dimensions inconnues avant).
        # Élimine np.stack() dans _process_buffer() → direct (N,H,W) view pour Halide.
        # _ring_buf     : (buffer_size, H, W) float32  — frames Bayer BL-soustraites
        # _g1_ring      : (buffer_size, H/2, W/2) float32 — G1 half-res
        # _aligned_buf  : (buffer_size, H, W) float32  — scratch pour frames alignées
        # _ring_head    : prochain slot à écrire (0..buffer_size-1)
        self._ring_buf:    Optional[np.ndarray] = None
        self._g1_ring:     Optional[np.ndarray] = None
        self._aligned_buf: Optional[np.ndarray] = None
        self._ring_head:   int = 0

        self.frame_count:  int = 0
        self.stacks_done:  int = 0
        self._prev_stacks: int = 0

        # Cache BL auto : évite de recalculer np.partition×4 à chaque frame
        self._bl_cached:     Optional[Tuple] = None   # dernière estimation (bl_r,g1,g2,b)
        self._bl_cache_age:  int = 0                  # frames depuis le dernier calcul

        self.last_bayer_stack: Optional[np.ndarray] = None
        self._align_ref_g1:    Optional[np.ndarray] = None

        # Stack cumulatif inter-buffer pondéré par score
        # Chaque buffer est pondéré par le score moyen de ses frames sélectionnées
        # → les buffers avec de meilleures frames contribuent davantage au résultat final
        self._cumul_stack:  Optional[np.ndarray] = None   # float64 (H, W) somme pondérée
        self._cumul_count:  int = 0                        # nombre de buffers accumulés
        self._cumul_weight: float = 0.0                    # somme des poids (scores moyens)

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def add_frame(self, raw: np.ndarray) -> float:
        """Ajoute une frame Bayer, déclenche le stack si buffer plein.

        Args:
            raw : uint8 (H, W*2) ou uint16 (H, W) Bayer RGGB, espace CSI-2 ×16
                  Picamera2 livre parfois le Bayer comme buffer uint8 (2 octets/pixel).
                  La conversion uint8→uint16 est faite ici, identique à debayer_raw_array().

        Returns:
            Score de netteté (float ≥ 0)
        """
        # Conversion uint8 → uint16 si picamera2 livre le buffer comme bytes bruts
        # Ex: (480, 1280) uint8 → (480, 640) uint16 pour un Bayer 640×480
        if raw.dtype == np.uint8:
            h = raw.shape[0]
            raw = raw.view(np.uint16).reshape(h, -1)[:, :raw.shape[1] // 2]

        # Black level per-canal (correction FPN 2×2)
        use_bl = self.bl_auto or (self.bl_r + self.bl_g1 + self.bl_g2 + self.bl_b) > 0.0
        if self.bl_auto:
            # Cache : recalcul seulement toutes les bl_cache_interval frames.
            # Le BL capteur est stable → pas besoin de recalculer à chaque frame.
            interval = self.bl_cache_interval
            if self._bl_cached is None or interval == 0 or self._bl_cache_age >= interval:
                self._bl_cached    = _estimate_bl_auto(raw)
                self._bl_cache_age = 0
            else:
                self._bl_cache_age += 1
            bl_vals = self._bl_cached
        else:
            bl_vals = (self.bl_r, self.bl_g1, self.bl_g2, self.bl_b)

        if _halide_available:
            # Chemin Halide : BL subtract + extraction G1 en une seule passe NEON
            bl_r_v, bl_g1_v, bl_g2_v, bl_b_v = bl_vals if use_bl else (0.0, 0.0, 0.0, 0.0)
            processed, g1_half = _halide_preprocess(raw, bl_r_v, bl_g1_v, bl_g2_v, bl_b_v)
        else:
            # Fallback NumPy
            if use_bl:
                processed = _apply_bl_per_channel(raw, *bl_vals)
            else:
                processed = raw.astype(np.float32)
            g1_half = _extract_g1(processed).astype(np.float32)

        # Score de netteté sur G1 déjà extrait (évite une deuxième extraction)
        score = _score_g1(g1_half, self.score_method, self.score_roi)

        with self._lock:
            # Allocation lazy du buffer 3D (dimensions connues après la première frame)
            if self._ring_buf is None:
                fh, fw = processed.shape
                self._ring_buf    = np.empty((self.buffer_size, fh, fw),
                                             dtype=np.float32)
                self._g1_ring     = np.empty((self.buffer_size, fh // 2, fw // 2),
                                             dtype=np.float32)

            # Écriture directe dans le ring buffer (pas d'allocation ni de np.stack plus tard)
            self._ring_buf [self._ring_head] = processed
            self._g1_ring  [self._ring_head] = g1_half
            self._scores.append(score)
            self._ring_head  += 1
            self.frame_count += 1

            if self._ring_head >= self.buffer_size:
                self._process_buffer()
                self._ring_head = 0
                self._scores.clear()

        return score

    def new_stack_available(self) -> bool:
        """True si un nouveau stack est prêt (consomme l'indicateur)."""
        result = self.stacks_done > self._prev_stacks
        if result:
            self._prev_stacks = self.stacks_done
        return result

    def get_result(self,
                   red_gain:  float = 1.0,
                   blue_gain: float = 1.0,
                   global_bl: float = 0.0) -> Optional[np.ndarray]:
        """Retourne le stack cumulatif débayérisé, ou None si pas encore de stack.

        Identique au lucky RGB8 : chaque buffer complété est accumulé dans une
        moyenne cumulative → l'image s'améliore progressivement d'un buffer à l'autre.

        Args:
            red_gain  : Gain AWB rouge  (red / 10 depuis globals RPiCamera2)
            blue_gain : Gain AWB bleu   (blue / 10 depuis globals RPiCamera2)
            global_bl : BL global en ADU 12-bit (isp_black_level si FPN désactivé)

        Returns:
            float32 (H, W, 3) [0-65535], ch0=R_phys, ch2=B_phys.
            Compatible avec apply_isp_to_preview().
        """
        if self._cumul_stack is None or self._cumul_count == 0:
            return None
        # Division par la somme des poids (score-weighted mean inter-buffer)
        divisor = self._cumul_weight if self._cumul_weight > 0 else float(self._cumul_count)
        bayer_mean = self._cumul_stack / np.float32(divisor)   # float32 ÷ float32 → float32
        # bl_applied=True si une correction BL a été appliquée per-frame → évite
        # la double soustraction qui écrase les basses lumières de la planète.
        bl_was_applied = self.bl_auto or (self.bl_r + self.bl_g1 + self.bl_g2 + self.bl_b) > 0.0
        return debayer_bayer_stack(bayer_mean, red_gain, blue_gain, global_bl,
                                   bl_applied=bl_was_applied)

    def get_stats(self) -> Dict:
        """Statistiques compatibles avec draw_lucky_stats_bar() de RPiCamera2.py."""
        with self._lock:
            scores = list(self._scores)
            n = self._ring_head
        return {
            # Clés requises par draw_lucky_stats_bar()
            'lucky_buffer_fill':  n,
            'lucky_buffer_size':  self.buffer_size,
            'lucky_stacks_done':  self.stacks_done,
            'total_frames':       self.frame_count,
            'lucky_avg_score':    float(np.mean(scores)) if scores else 0.0,
            # Infos supplémentaires
            'lucky_max_score':    float(max(scores)) if scores else 0.0,
            'keep_percent':       self.keep_percent,
            'buffer_mode':        'ring',  # Toujours ring pour RAW lucky
        }

    def reset(self):
        """Réinitialise complètement le stacker (buffer + cumulatif)."""
        with self._lock:
            self._scores.clear()
            # Le ring buffer est conservé (pré-alloué) mais le pointeur d'écriture
            # revient à 0 — les anciennes données seront écrasées.
            self._ring_head    = 0
            self.frame_count   = 0
            self.stacks_done   = 0
            self._prev_stacks  = 0
            self.last_bayer_stack = None
            self._align_ref_g1    = None
            self._cumul_stack     = None
            self._cumul_count     = 0
            self._cumul_weight    = 0.0
            self._bl_cached       = None
            self._bl_cache_age    = 0

    def update_config(self, **kwargs):
        """Met à jour la configuration à chaud (sans reset du buffer).

        Paramètres supportés : buffer_size, keep_percent, score_method,
        score_roi, align_enabled, max_shift_px, stack_method, sigma_kappa,
        bl_auto, bl_r, bl_g1, bl_g2, bl_b, bl_cache_interval.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'buffer_size':
                    v = max(2, int(v))
                    if v != self.buffer_size:
                        # Re-allouer les buffers 3D (perte du buffer courant acceptable)
                        self._ring_buf    = None   # Réalloué à la prochaine frame
                        self._g1_ring     = None
                        self._aligned_buf = None
                        self._ring_head   = 0
                        self._scores      = deque(maxlen=v)
                elif k == 'bl_auto' and v != self.bl_auto:
                    # Changement de mode BL → invalider le cache
                    self._bl_cached    = None
                    self._bl_cache_age = 0
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _process_buffer(self):
        """Sélectionne, aligne et stacke les meilleures frames du buffer.

        Utilise les ring buffers 3D pré-alloués → pas de np.stack().
        Stacking via Halide AOT si disponible.
        """
        n = self._ring_head   # frames valides dans le ring buffer
        if n == 0:
            return

        scores = list(self._scores)

        # ── Sélection top N% ──────────────────────────────────────────────────
        keep = max(1, int(n * self.keep_percent / 100.0))
        idx_sorted = sorted(range(n), key=lambda i: scores[i], reverse=True)
        best_idx   = idx_sorted[:keep]

        # ── Référence = meilleure frame (G1 depuis le cache) ──────────────────
        ref_idx = best_idx[0]
        ref_g1  = self._g1_ring[ref_idx]

        max_half = (self.max_shift_px // 2) if self.max_shift_px > 0 else 0

        # ── Allocation lazy du scratch aligné ─────────────────────────────────
        fh, fw = self._ring_buf.shape[1:]
        if self._aligned_buf is None or self._aligned_buf.shape[0] < self.buffer_size:
            self._aligned_buf = np.empty((self.buffer_size, fh, fw), dtype=np.float32)

        # ── Alignement → écriture directe dans _aligned_buf ───────────────────
        # Chaque frame est écrite dans son slot sans allocation intermédiaire :
        #   dy=dx=0  → np.copyto (1 copie mémoire)
        #   sinon    → _shift_bayer_into() remplit le slot directement (1 copie)
        #
        # Opt. 1 — FFT(ref) pré-calculée une fois : (keep-1) FFTs économisées.
        # Opt. 3 — Parallélisme (keep ≥ 3) : chaque frame est indépendante
        #   (lecture de slots distincts, écriture dans des slots distincts de
        #   _aligned_buf) → ThreadPoolExecutor libère le GIL sur les ops NumPy.
        #   Seuil keep ≥ 3 : en dessous l'overhead thread dépasse le gain.
        if self.align_enabled and keep > 1:
            ref_fft, _ = _precompute_ref_fft(ref_g1)
            use_cached = ref_fft is not None
        else:
            use_cached = False

        if self.align_enabled and keep >= 3 and use_cached:
            # ── Chemin parallèle ──────────────────────────────────────────────
            # Capture des tableaux locaux pour les closures de thread.
            _ring_buf    = self._ring_buf
            _g1_ring     = self._g1_ring
            _aligned_buf = self._aligned_buf

            def _align_one(args):
                rank, orig_idx = args
                fg1 = _g1_ring[orig_idx]
                dy, dx = _compute_bayer_shift_cached(ref_fft, fg1, max_half)
                _shift_bayer_into(_ring_buf[orig_idx], dy, dx, _aligned_buf[rank])

            n_workers = min(4, keep)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(_align_one, enumerate(best_idx)))
        else:
            # ── Chemin séquentiel (keep < 3, align désactivé, ou ref_fft nul) ─
            for rank, orig_idx in enumerate(best_idx):
                frame = self._ring_buf[orig_idx]
                dst   = self._aligned_buf[rank]
                if self.align_enabled and keep > 1:
                    fg1 = self._g1_ring[orig_idx]
                    if use_cached:
                        dy, dx = _compute_bayer_shift_cached(ref_fft, fg1, max_half)
                    else:
                        dy, dx = _compute_bayer_shift(ref_g1, fg1, max_half)
                    _shift_bayer_into(frame, dy, dx, dst)
                else:
                    np.copyto(dst, frame)

        # Vue (keep, H, W) C-contiguous — pas de copie (slice de dim 0)
        aligned_3d = self._aligned_buf[:keep]

        # ── Stacking (Halide si disponible, sinon NumPy 3D sans np.stack()) ───
        use_sigma = self.stack_method == 'sigma_clip' and keep >= 3
        if _halide_stack_available:
            if use_sigma:
                result = _halide_sigma_clip(aligned_3d, self.sigma_kappa)
            else:
                result = _halide_stack_mean(aligned_3d)
            if self.stacks_done == 0:
                print(f"[Lucky RAW] stack #1 : Halide "
                      f"{'sigma_clip' if use_sigma else 'mean'}  "
                      f"keep={keep}/{n}")
            else:
                logger.debug("[Lucky RAW] stack #%d : Halide %s  keep=%d/%d",
                             self.stacks_done + 1,
                             "sigma_clip" if use_sigma else "mean", keep, n)
        else:
            if use_sigma:
                result = _stack_sigma_clip_3d(aligned_3d, self.sigma_kappa)
            else:
                result = _stack_mean_3d(aligned_3d)
            if self.stacks_done == 0:
                print(f"[Lucky RAW] stack #1 : NumPy "
                      f"{'sigma_clip' if use_sigma else 'mean'}  "
                      f"keep={keep}/{n}  "
                      f"(recompiler jsk_halide.so pour Halide)")

        self.last_bayer_stack = result
        self._align_ref_g1    = ref_g1
        self.stacks_done     += 1

        # ── Accumulation inter-buffer pondérée par score moyen ────────────────
        buffer_score = float(np.mean([scores[i] for i in best_idx]))
        weight = max(buffer_score, 1e-9)

        # float32 suffit pour ~100 buffers (erreur d'arrondi < 0.001 ADU sur le
        # résultat final). float64 doublait la RAM du stack cumulatif sans apport
        # visible sur l'image.
        weighted = result * np.float32(weight)   # float32 × float32 → float32
        if self._cumul_stack is None:
            self._cumul_stack = weighted
        else:
            self._cumul_stack += weighted
        self._cumul_weight += weight
        self._cumul_count  += 1
        # Note : _ring_head et _scores sont remis à zéro dans add_frame() après retour


# ---------------------------------------------------------------------------
# Fabrique
# ---------------------------------------------------------------------------

def create_bayer_lucky_stacker(
    buffer_size:   int   = 50,
    keep_percent:  float = 20.0,
    score_method:  str   = 'laplacian',
    score_roi:     float = 0.50,
    align_enabled: bool  = True,
    max_shift_px:  int   = 30,
    stack_method:  str   = 'mean',
    sigma_kappa:   float = 2.5,
    bl_auto:       bool  = False,
    bl_r:          float = 0.0,
    bl_g1:         float = 0.0,
    bl_g2:         float = 0.0,
    bl_b:          float = 0.0,
) -> BayerLuckyStacker:
    """Crée un BayerLuckyStacker depuis les globals RPiCamera2."""
    return BayerLuckyStacker(
        buffer_size   = buffer_size,
        keep_percent  = keep_percent,
        score_method  = score_method,
        score_roi     = score_roi,
        align_enabled = align_enabled,
        max_shift_px  = max_shift_px,
        stack_method  = stack_method,
        sigma_kappa   = sigma_kappa,
        bl_auto       = bl_auto,
        bl_r          = bl_r,
        bl_g1         = bl_g1,
        bl_g2         = bl_g2,
        bl_b          = bl_b,
    )
