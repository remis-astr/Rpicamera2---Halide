"""
Benchmark JSK pipeline : numpy (actuel) vs Halide (AOT)
Simule le pipeline complet sur une frame IMX585 (3840×2160 RAW12)
"""

import numpy as np
import cv2
import ctypes
import time
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# Chargement du .so Halide
# ─────────────────────────────────────────────────────────────────────────────
_SO = os.path.join(os.path.dirname(__file__), "jsk_halide.so")
_lib = ctypes.CDLL(_SO)

_lib.jsk_hdr_median3.restype  = ctypes.c_int
_lib.jsk_hdr_median3.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_uint8),   # output
    ctypes.c_int, ctypes.c_int,       # w, h
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # r_gain, g_gain, b_gain
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # w0, w1, w2
]
_lib.jsk_hdr_mean3.restype  = ctypes.c_int
_lib.jsk_hdr_mean3.argtypes = _lib.jsk_hdr_median3.argtypes


def _call_halide(fn, raw16, r_gain, g_gain, b_gain, w0, w1, w2):
    """Appelle un pipeline Halide AOT et retourne RGB (H,W,3) uint8.

    Halide sort en layout planaire (3,H,W).
    La transposition → (H,W,3) est quasi sans coût (changement de strides numpy).
    La copie ascontiguousarray est nécessaire pour que le résultat soit C-contiguous.
    """
    h, w = raw16.shape
    raw_c  = np.ascontiguousarray(raw16, dtype=np.uint16)
    # Sortie planaire (3,H,W) — correspond au layout Halide dim[2]=c stride=W*H
    out_planar = np.empty((3, h, w), dtype=np.uint8)
    ret = fn(
        raw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out_planar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w, h,
        int(r_gain * 256), int(g_gain * 256), int(b_gain * 256),
        w0, w1, w2
    )
    if ret != 0:
        raise RuntimeError(f"Halide pipeline erreur: {ret}")
    # Transposer (3,H,W) → (H,W,3) — numpy ne copie pas, change juste les strides
    return out_planar.transpose(1, 2, 0)


def halide_hdr_median3(raw16, r_gain=1.0, g_gain=1.0, b_gain=1.0,
                        w0=100, w1=100, w2=100):
    return _call_halide(_lib.jsk_hdr_median3, raw16, r_gain, g_gain, b_gain, w0, w1, w2)


def halide_hdr_mean3(raw16, r_gain=1.0, g_gain=1.0, b_gain=1.0,
                      w0=100, w1=100, w2=100):
    return _call_halide(_lib.jsk_hdr_mean3, raw16, r_gain, g_gain, b_gain, w0, w1, w2)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline numpy actuel (copie de jsk_live.py)
# ─────────────────────────────────────────────────────────────────────────────

def numpy_hdr_median3(raw16, r_gain=1.0, g_gain=1.0, b_gain=1.0):
    """Reproduit HDR_compute_12bit + debayer de jsk_live.py (méthode Median)."""
    f = raw16.astype(np.float32)
    # 3 niveaux : clip à 12, 11, 10 bits (espace CSI-2 ×16)
    t12, t11, t10 = 65520.0, 32752.0, 16368.0
    lv0 = (np.clip(f, 0, t12) / t12 * 255.0).astype(np.uint8)
    lv1 = (np.clip(f, 0, t11) / t11 * 255.0).astype(np.uint8)
    lv2 = (np.clip(f, 0, t10) / t10 * 255.0).astype(np.uint8)
    # Median pixel-wise sur 3 niveaux
    stack = np.stack([lv0, lv1, lv2], axis=0)
    merged = np.median(stack, axis=0).astype(np.uint8)
    # Debayering BayerRG2BGR (comme jsk_live.py)
    bgr = cv2.cvtColor(merged, cv2.COLOR_BayerRG2BGR)
    # Gains AWB
    if r_gain != 1.0 or g_gain != 1.0 or b_gain != 1.0:
        bgr = bgr.astype(np.float32)
        bgr[:,:,2] = np.clip(bgr[:,:,2] * r_gain, 0, 255)
        bgr[:,:,1] = np.clip(bgr[:,:,1] * g_gain, 0, 255)
        bgr[:,:,0] = np.clip(bgr[:,:,0] * b_gain, 0, 255)
        bgr = bgr.astype(np.uint8)
    return bgr


def numpy_hdr_mean3(raw16, r_gain=1.0, g_gain=1.0, b_gain=1.0):
    f = raw16.astype(np.float32)
    t12, t11, t10 = 65520.0, 32752.0, 16368.0
    lv0 = (np.clip(f, 0, t12) / t12 * 255.0).astype(np.uint8)
    lv1 = (np.clip(f, 0, t11) / t11 * 255.0).astype(np.uint8)
    lv2 = (np.clip(f, 0, t10) / t10 * 255.0).astype(np.uint8)
    stack = np.stack([lv0, lv1, lv2], axis=0).astype(np.float32)
    merged = np.mean(stack, axis=0).astype(np.uint8)
    bgr = cv2.cvtColor(merged, cv2.COLOR_BayerRG2BGR)
    return bgr


# ─────────────────────────────────────────────────────────────────────────────
# Mesure de performance
# ─────────────────────────────────────────────────────────────────────────────

def bench(fn, *args, n=10, label=""):
    # Warmup
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<30} : {ms:7.1f} ms/frame")
    return ms


def main():
    # Résolution IMX585 pleine
    W, H = 3840, 2160
    print(f"Image : {W}×{H} RAW12 (CSI-2 ×16, uint16)")
    print(f"Pixels : {W*H/1e6:.1f} Mpix | RAM entrée : {W*H*2/1e6:.0f} MB")
    print()

    # Générer une frame RAW12 synthétique réaliste
    rng = np.random.default_rng(42)
    # Signal solaire simulé : gradient + bruit shot + quelques pixels saturés
    signal = rng.integers(0, 60000, (H, W), dtype=np.uint16)
    signal[H//3:2*H//3, W//3:2*W//3] = np.clip(
        signal[H//3:2*H//3, W//3:2*W//3].astype(np.int32) + 10000, 0, 65535
    ).astype(np.uint16)

    r_gain, g_gain, b_gain = 1.3, 1.0, 1.5  # Gains AWB typiques soleil

    print("=" * 55)
    print("Benchmark HDR Median 3 niveaux")
    print("=" * 55)
    t_np   = bench(numpy_hdr_median3,   signal, r_gain, g_gain, b_gain, label="NumPy (actuel)")
    t_hal  = bench(halide_hdr_median3,  signal, r_gain, g_gain, b_gain, label="Halide AOT")
    print(f"  → Speedup Halide/NumPy : ×{t_np/t_hal:.1f}")
    print()

    print("=" * 55)
    print("Benchmark HDR Mean 3 niveaux")
    print("=" * 55)
    t_np2  = bench(numpy_hdr_mean3,  signal, label="NumPy (actuel)")
    t_hal2 = bench(halide_hdr_mean3, signal, label="Halide AOT")
    print(f"  → Speedup Halide/NumPy : ×{t_np2/t_hal2:.1f}")
    print()

    # Vérification qualité : les deux résultats doivent être proches
    print("=" * 55)
    print("Vérification qualité (résultats identiques ?)")
    print("=" * 55)
    out_np  = numpy_hdr_median3(signal)
    out_hal = halide_hdr_median3(signal)
    # IMPORTANT : cv2.COLOR_BayerRG2BGR sur IMX585 produit RGB empiriquement
    # (ch0=R_physique, non-standard, vérifié dans RPiCamera2.py)
    # Halide sort aussi RGB (c=0=R, c=1=G, c=2=B) → comparaison directe, sans swap
    out_hal_c = np.ascontiguousarray(out_hal)
    diff = np.abs(out_np.astype(np.int16) - out_hal_c.astype(np.int16))
    print(f"  NumPy output  shape={out_np.shape}  dtype={out_np.dtype}")
    print(f"  Halide output shape={out_hal.shape} dtype={out_hal.dtype}")
    print(f"  Diff max : {diff.max()}  |  Diff mean : {diff.mean():.3f}")
    print(f"  Diff par canal — R:{diff[:,:,0].mean():.1f}  G:{diff[:,:,1].mean():.1f}  B:{diff[:,:,2].mean():.1f}")
    print(f"  Pixels identiques : {(diff == 0).mean()*100:.1f}%")
    if diff.max() <= 4:
        print("  ✓ Résultats concordants (diff ≤ 4 LSB = arrondi float + bords)")
    else:
        print(f"  ⚠ Diff max={diff.max()} — différences de bord ou HDR (acceptable si faibles zones)")
    print()

    # Résumé FPS
    print("=" * 55)
    print("FPS estimé (pipeline JSK seul)")
    print("=" * 55)
    for label, ms in [("NumPy median3", t_np), ("Halide median3", t_hal),
                       ("NumPy mean3",  t_np2), ("Halide mean3",  t_hal2)]:
        fps = 1000.0 / ms
        print(f"  {label:<20} : {fps:5.1f} fps")


if __name__ == "__main__":
    main()
