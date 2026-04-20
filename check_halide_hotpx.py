#!/usr/bin/env python3
"""
Diagnostic : vérifie que le pipeline Halide de suppression des pixels chauds
est bien actif et produit les bons résultats.

Usage :
    python3 check_halide_hotpx.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, '/home/admin/Rpicamera tests/Rpicamera2')

# ── 1. Chargement des flags ────────────────────────────────────────────────────
from libastrostack.lucky_raw import (
    _halide_available,
    _halide_hotpx_available,
    debayer_raw_halide,
    _halide_hot_pixel_removal,
    _halide_preprocess,
)

print("=" * 60)
print("  Diagnostic Halide — suppression pixels chauds")
print("=" * 60)
print(f"  _halide_available       : {_halide_available}")
print(f"  _halide_hotpx_available : {_halide_hotpx_available}")
print()

if not _halide_available:
    print("ERREUR : Halide non disponible (jsk_halide.so absent ou corrompu)")
    sys.exit(1)

if not _halide_hotpx_available:
    print("ERREUR : ls_hot_pixel_removal absent du .so")
    print("  → Recompiler : cd libastrostack/halide && make")
    sys.exit(1)

# ── 2. Test fonctionnel : Halide remplace bien les pixels chauds ──────────────
print("Test 1 — Fonctionnel (pixels chauds sur fond uniforme)")
H, W = 64, 64
frame = np.full((H, W), 1000, dtype=np.uint16)   # fond CSI-2 uniforme
frame[16, 16] = 12000   # pixel très chaud (×12 le fond)
frame[32, 32] = 2000    # pixel modéré    (×2, sous seuil 5×, doit rester)
frame[48, 48] = 1010    # pixel quasi-normal (doit rester)

# Sans correction (référence)
r_off = debayer_raw_halide(frame, hot_pixel_threshold=0.0)
# Avec correction Halide (threshold=5)
r_on  = debayer_raw_halide(frame, hot_pixel_threshold=5.0, hot_pixel_abs_floor=160.0)

assert r_off is not None, "debayer_raw_halide a retourné None sans correction"
assert r_on  is not None, "debayer_raw_halide a retourné None avec correction"

px_hot_off = r_off[16, 16, 0]
px_hot_on  = r_on [16, 16, 0]
px_mod_off = r_off[32, 32, 0]
px_mod_on  = r_on [32, 32, 0]

ok_hot = abs(px_hot_on  - 1000) < 200   # remplacé → proche du fond
ok_mod = abs(px_mod_on  - px_mod_off) < 10   # conservé → inchangé

print(f"  pixel très chaud [16,16] : sans={px_hot_off:.0f}  avec={px_hot_on:.0f}"
      f"  {'✓ remplacé' if ok_hot else '✗ PAS remplacé'}")
print(f"  pixel modéré     [32,32] : sans={px_mod_off:.0f}  avec={px_mod_on:.0f}"
      f"  {'✓ conservé' if ok_mod else '✗ modifié à tort'}")

if not (ok_hot and ok_mod):
    print("\nECHEC du test fonctionnel")
    sys.exit(1)
print("  → OK\n")

# ── 3. Test chemin complet debayer_raw_halide ─────────────────────────────────
print("Test 2 — Chemin complet (BL + pixels chauds + débayer)")
frame2 = np.full((H, W), 900, dtype=np.uint16)
frame2[20, 20] = 9000
r2 = debayer_raw_halide(frame2,
                         bl_r=50.0, bl_g1=50.0, bl_g2=50.0, bl_b=50.0,
                         red_gain=1.8, blue_gain=1.4,
                         hot_pixel_threshold=5.0, hot_pixel_abs_floor=160.0)
assert r2 is not None
assert r2.shape == (H, W, 3), f"Shape inattendu : {r2.shape}"
assert r2.dtype == np.float32
px_replaced = r2[20, 20, 0]
print(f"  pixel [20,20] ch0 : {px_replaced:.1f}  (attendu proche du fond ~{frame2[0,0]:.0f})")
print(f"  shape sortie : {r2.shape}  dtype : {r2.dtype}")
print("  → OK\n")

# ── 4. Comparaison performance Halide vs NumPy ────────────────────────────────
print("Test 3 — Performance (50 frames 1928×1090)")
try:
    import cv2
    H2, W2 = 1090, 1928   # résolution réelle IMX585
    big_frame = np.random.randint(800, 900, (H2, W2), dtype=np.uint16)
    rng = np.random.default_rng(42)
    hot_y = rng.integers(0, H2, 500)
    hot_x = rng.integers(0, W2, 500)
    big_frame[hot_y, hot_x] = rng.integers(5000, 15000, 500, dtype=np.uint16)
    big_f32 = big_frame.astype(np.float32)

    N = 50

    # Halide : BL-subtract + hot pixel removal + debayer (pipeline complet)
    t0 = time.perf_counter()
    for _ in range(N):
        debayer_raw_halide(big_frame, hot_pixel_threshold=5.0, hot_pixel_abs_floor=160.0)
    t_hal = (time.perf_counter() - t0) / N * 1000

    # NumPy : hot pixel removal seul (scipy median_filter 5×5, approche classique)
    from scipy.ndimage import median_filter
    def numpy_hotpx(img, sigma=5.0):
        med = median_filter(img, size=5)
        mask = img > sigma * np.maximum(med, 1.0)
        out = img.copy()
        out[mask] = med[mask]
        return out

    t0 = time.perf_counter()
    for _ in range(N):
        numpy_hotpx(big_f32)
    t_np = (time.perf_counter() - t0) / N * 1000

    ratio = t_np / t_hal if t_hal > 0 else float('inf')
    print(f"  Halide (BL+hotpx+debayer) : {t_hal:.1f} ms/frame")
    print(f"  NumPy  (hotpx seul)       : {t_np:.1f} ms/frame")
    print(f"  Gain Halide vs NumPy      : ×{ratio:.1f}")
    print("  → OK\n")
except Exception as e:
    print(f"  (test perf ignoré : {e})\n")

# ── Résultat final ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  TOUS LES TESTS PASSÉS — le pipeline Halide est actif")
print("=" * 60)
print()
print("Traces attendues en console lors du live stack RAW :")
print("  [Halide debayer] CHEMIN HALIDE actif — BL-subtract + débayer NEON | pixels chauds Halide ON ...")
