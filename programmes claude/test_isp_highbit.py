#!/usr/bin/env python3
"""
Test ISP en haute résolution (float32/16-bit)
Compare le traitement uint8 vs float32 pour voir la différence d'histogramme
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from libastrostack.isp import ISP, ISPCalibrator


def analyze_histogram_quality(img, title):
    """Analyse la qualité de l'histogramme"""
    if len(img.shape) == 3:
        # Pour RGB, analyser le canal vert (le plus représentatif)
        channel = img[:, :, 1]
    else:
        channel = img

    unique_vals = np.unique(channel)
    n_unique = len(unique_vals)
    n_possible = 256 if img.dtype == np.uint8 else 65536

    gaps = []
    for i in range(int(np.min(channel)), int(np.max(channel))):
        if i not in unique_vals:
            gaps.append(i)

    print(f"\n{title}:")
    print(f"  Type: {img.dtype}")
    print(f"  Range: [{np.min(channel):.1f}, {np.max(channel):.1f}]")
    print(f"  Valeurs uniques: {n_unique} / {n_possible} possibles")
    print(f"  Gaps dans l'histogramme: {len(gaps)}")
    print(f"  Continuité: {n_unique / (np.max(channel) - np.min(channel) + 1) * 100:.1f}%")

    return n_unique, len(gaps)


def plot_histogram_comparison(img_uint8_early, img_float32, img_uint8_late):
    """Compare les histogrammes des 3 méthodes"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    titles = [
        'Méthode 1: Conversion uint8 AVANT ISP\n(MAUVAIS - gaps)',
        'Méthode 2: Traitement en float32\n(INTERMÉDIAIRE)',
        'Méthode 3: Conversion uint8 APRÈS ISP\n(BON - lisse)'
    ]

    images = [img_uint8_early, (img_float32 * 255).astype(np.uint8), img_uint8_late]

    for col, (img, title) in enumerate(zip(images, titles)):
        # Image
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')
        axes[0, col].axis('off')

        # Histogramme du canal vert
        if len(img.shape) == 3:
            channel = img[:, :, 1]
        else:
            channel = img

        axes[1, col].hist(channel.ravel(), bins=256, range=(0, 255),
                         color='green', alpha=0.7)
        axes[1, col].set_xlabel('Valeur')
        axes[1, col].set_ylabel('Fréquence')
        axes[1, col].set_title('Histogramme canal vert', fontsize=10)
        axes[1, col].grid(True, alpha=0.3)

        # Analyser qualité
        unique_vals = len(np.unique(channel))
        axes[1, col].text(0.02, 0.98, f'{unique_vals} valeurs uniques',
                         transform=axes[1, col].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                         fontsize=9)

    plt.suptitle('Comparaison: Impact de la conversion uint8 sur l\'histogramme',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('histogram_uint8_vs_float32.png', dpi=150)
    print("\n✓ Graphique sauvegardé: histogram_uint8_vs_float32.png")


def main():
    print("="*70)
    print("TEST ISP HAUTE PRÉCISION (float32 vs uint8)")
    print("="*70)

    # Charger images
    raw_path = Path("/media/admin/THKAILAR/Stacks/stack_SRGGB12_mean_20251212_100655.png")
    yuv_path = Path("/media/admin/THKAILAR/Stacks/stack_YUV420_mean_20251212_100812.png")

    print(f"\nChargement des images...")
    raw_img = cv2.imread(str(raw_path))
    yuv_img = cv2.imread(str(yuv_path))

    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    yuv_img = cv2.cvtColor(yuv_img, cv2.COLOR_BGR2RGB)

    print(f"  RAW: {raw_img.shape}, dtype: {raw_img.dtype}")
    print(f"  YUV: {yuv_img.shape}, dtype: {yuv_img.dtype}")

    # ========================================
    # MÉTHODE 1: Conversion uint8 AVANT ISP (MAUVAIS)
    # ========================================
    print("\n" + "="*70)
    print("MÉTHODE 1: Image uint8 → ISP → uint8 (GÉNÈRE DES GAPS)")
    print("="*70)

    config = ISPCalibrator.calibrate_from_images(raw_img, yuv_img)
    isp = ISP(config)

    # L'image est déjà en uint8, traiter directement
    result_uint8_early = isp.process(raw_img, return_uint8=True)

    analyze_histogram_quality(result_uint8_early,
                              "Méthode 1 (uint8 avant ISP)")

    # ========================================
    # MÉTHODE 2: Traitement complet en float32 (BON)
    # ========================================
    print("\n" + "="*70)
    print("MÉTHODE 2: Image uint8 → float32 → ISP → float32 (HAUTE PRÉCISION)")
    print("="*70)

    # Traiter en float32
    result_float32 = isp.process(raw_img, return_uint8=False)

    print(f"\nRésultat float32:")
    print(f"  Type: {result_float32.dtype}")
    print(f"  Range: [{result_float32.min():.3f}, {result_float32.max():.3f}]")

    # ========================================
    # MÉTHODE 3: Conversion uint8 APRÈS ISP (BON)
    # ========================================
    print("\n" + "="*70)
    print("MÉTHODE 3: Image uint8 → float32 → ISP → float32 → uint8 (OPTIMAL)")
    print("="*70)

    # Conversion finale en uint8 (simulant l'export PNG)
    result_uint8_late = (np.clip(result_float32, 0, 1) * 255).astype(np.uint8)

    analyze_histogram_quality(result_uint8_late,
                              "Méthode 3 (uint8 après ISP)")

    # ========================================
    # BONUS: Test avec uint16 (MEILLEUR)
    # ========================================
    print("\n" + "="*70)
    print("BONUS: Traitement en uint16 (PRECISION MAXIMALE)")
    print("="*70)

    result_uint16 = isp.process(raw_img, return_uint16=True)
    analyze_histogram_quality(result_uint16,
                              "Bonus (uint16)")

    # ========================================
    # Comparaison visuelle
    # ========================================
    print("\n" + "="*70)
    print("GÉNÉRATION DES COMPARAISONS")
    print("="*70)

    plot_histogram_comparison(result_uint8_early, result_float32, result_uint8_late)

    # Sauvegarder les résultats
    cv2.imwrite("result_uint8_early_GAPS.png",
                cv2.cvtColor(result_uint8_early, cv2.COLOR_RGB2BGR))
    cv2.imwrite("result_float32_then_uint8_SMOOTH.png",
                cv2.cvtColor(result_uint8_late, cv2.COLOR_RGB2BGR))

    print("✓ Résultats sauvegardés:")
    print("    - result_uint8_early_GAPS.png (avec gaps)")
    print("    - result_float32_then_uint8_SMOOTH.png (lisse)")

    # ========================================
    # Recommandations
    # ========================================
    print("\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70)
    print("""
Pour intégrer l'ISP dans le pipeline de stacking:

1. Le stack doit rester en float32/float64 (haute précision)
2. Appliquer l'ISP sur le résultat du stack (AVANT stretch)
3. Appliquer le stretch sur le résultat ISP (toujours en float)
4. Convertir en uint8 uniquement pour la sauvegarde PNG

Pipeline recommandé:
  Stack (float32) → ISP (float32) → Stretch (float32) → uint8 → PNG

Code dans session.py get_preview_png():
  result = self.stacker.get_result()  # float32

  # NOUVEAU: Appliquer ISP si configuré
  if self.isp is not None:
      result = self.isp.process(result)  # Reste en float32

  # Stretch (existant)
  stretched = apply_stretch(result, ...)  # float32

  # Conversion finale en uint8 (existant)
  preview = (stretched * 255).astype(np.uint8)

Cela éliminera les gaps dans l'histogramme!
    """)


if __name__ == "__main__":
    main()
