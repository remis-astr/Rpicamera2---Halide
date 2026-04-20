#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de calibration et application ISP automatique
Compare RAW12 vs YUV420 et applique les paramètres ISP calibrés
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ajouter le chemin du module
sys.path.insert(0, str(Path(__file__).parent))

from libastrostack.isp import ISP, ISPCalibrator, ISPConfig, quick_calibrate_and_process


def plot_comparison(raw_img, yuv_img, isp_img, config: ISPConfig, output_path: str):
    """
    Crée une visualisation comparative des 3 images avec histogrammes
    """
    # Redimensionner les images si nécessaire pour la comparaison
    if raw_img.shape != yuv_img.shape or isp_img.shape != yuv_img.shape:
        h = min(raw_img.shape[0], yuv_img.shape[0], isp_img.shape[0])
        w = min(raw_img.shape[1], yuv_img.shape[1], isp_img.shape[1])
        raw_img = raw_img[:h, :w]
        yuv_img = yuv_img[:h, :w]
        isp_img = isp_img[:h, :w]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # === ROW 1: Images ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(raw_img)
    ax1.set_title('RAW12 Original', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(yuv_img)
    ax2.set_title('YUV420 Référence', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(isp_img)
    ax3.set_title('RAW12 + ISP Calibré', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Différence YUV vs ISP
    ax4 = fig.add_subplot(gs[0, 3])
    diff = np.abs(yuv_img.astype(np.float32) - isp_img.astype(np.float32))
    diff_display = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    ax4.imshow(diff_display)
    ax4.set_title(f'Différence (MAE={np.mean(diff):.2f})', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # === ROW 2: Histogrammes RGB ===
    colors = ['red', 'green', 'blue']
    channel_names = ['Rouge', 'Vert', 'Bleu']

    for i, (color, name) in enumerate(zip(colors, channel_names)):
        ax = fig.add_subplot(gs[1, i])

        # Histogrammes des 3 images
        ax.hist(raw_img[:, :, i].ravel(), bins=256, range=(0, 255),
               color='blue', alpha=0.3, label='RAW12', density=True)
        ax.hist(yuv_img[:, :, i].ravel(), bins=256, range=(0, 255),
               color='red', alpha=0.3, label='YUV420', density=True)
        ax.hist(isp_img[:, :, i].ravel(), bins=256, range=(0, 255),
               color='green', alpha=0.3, label='ISP', density=True)

        ax.set_title(f'Canal {name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Valeur')
        ax.set_ylabel('Densité')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # === ROW 2, Col 4: Paramètres ISP ===
    ax_params = fig.add_subplot(gs[1, 3])
    ax_params.axis('off')

    params_text = f"""
    PARAMÈTRES ISP CALIBRÉS

    Balance des blancs:
      • Rouge:  {config.wb_red_gain:.3f}
      • Vert:   {config.wb_green_gain:.3f}
      • Bleu:   {config.wb_blue_gain:.3f}

    Gamma: {config.gamma:.2f}
    Contraste: {config.contrast:.2f}
    Saturation: {config.saturation:.2f}

    Black Level: {config.black_level}
    """

    if 'residual_error' in config.calibration_info:
        params_text += f"\n    Erreur résiduelle: {config.calibration_info['residual_error']:.4f}"

    ax_params.text(0.1, 0.5, params_text, fontsize=10,
                  verticalalignment='center', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # === ROW 3: Histogrammes cumulés ===
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        ax = fig.add_subplot(gs[2, i])

        # Histogrammes cumulés
        raw_hist, _ = np.histogram(raw_img[:, :, i].ravel(), bins=256, range=(0, 255))
        yuv_hist, _ = np.histogram(yuv_img[:, :, i].ravel(), bins=256, range=(0, 255))
        isp_hist, _ = np.histogram(isp_img[:, :, i].ravel(), bins=256, range=(0, 255))

        raw_cumul = np.cumsum(raw_hist) / np.sum(raw_hist)
        yuv_cumul = np.cumsum(yuv_hist) / np.sum(yuv_hist)
        isp_cumul = np.cumsum(isp_hist) / np.sum(isp_hist)

        ax.plot(raw_cumul, color='blue', alpha=0.7, label='RAW12', linewidth=2)
        ax.plot(yuv_cumul, color='red', alpha=0.7, label='YUV420', linewidth=2)
        ax.plot(isp_cumul, color='green', alpha=0.7, label='ISP', linewidth=2)

        ax.set_title(f'Cumul {name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Valeur')
        ax.set_ylabel('Cumul')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # === ROW 3, Col 4: Statistiques ===
    ax_stats = fig.add_subplot(gs[2, 3])
    ax_stats.axis('off')

    raw_mean = np.mean(raw_img, axis=(0, 1))
    yuv_mean = np.mean(yuv_img, axis=(0, 1))
    isp_mean = np.mean(isp_img, axis=(0, 1))

    stats_text = f"""
    STATISTIQUES

    Moyennes RAW12:
      R={raw_mean[0]:.1f}  G={raw_mean[1]:.1f}  B={raw_mean[2]:.1f}

    Moyennes YUV420:
      R={yuv_mean[0]:.1f}  G={yuv_mean[1]:.1f}  B={yuv_mean[2]:.1f}

    Moyennes ISP:
      R={isp_mean[0]:.1f}  G={isp_mean[1]:.1f}  B={isp_mean[2]:.1f}

    Écarts ISP vs YUV:
      R={abs(isp_mean[0]-yuv_mean[0]):.1f}
      G={abs(isp_mean[1]-yuv_mean[1]):.1f}
      B={abs(isp_mean[2]-yuv_mean[2]):.1f}
    """

    ax_stats.text(0.1, 0.5, stats_text, fontsize=9,
                 verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Calibration et Application ISP Automatique', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Graphique de comparaison sauvegardé: {output_path}")


def main():
    print("="*70)
    print("TEST DE CALIBRATION ISP AUTOMATIQUE")
    print("="*70)

    # Chemins des images
    raw_path = Path("/media/admin/THKAILAR/Stacks/stack_SRGGB12_mean_20251212_100655.png")
    yuv_path = Path("/media/admin/THKAILAR/Stacks/stack_YUV420_mean_20251212_100812.png")

    if not raw_path.exists():
        print(f"❌ Erreur: Image RAW non trouvée: {raw_path}")
        return
    if not yuv_path.exists():
        print(f"❌ Erreur: Image YUV non trouvée: {yuv_path}")
        return

    print(f"\nChargement des images...")
    print(f"  RAW:  {raw_path.name}")
    print(f"  YUV:  {yuv_path.name}")

    # Charger les images
    raw_img = cv2.imread(str(raw_path))
    yuv_img = cv2.imread(str(yuv_path))

    # Convertir BGR -> RGB
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    yuv_img = cv2.cvtColor(yuv_img, cv2.COLOR_BGR2RGB)

    print(f"  Dimensions RAW: {raw_img.shape}")
    print(f"  Dimensions YUV: {yuv_img.shape}")

    # ========================================
    # MÉTHODE 1: Calibration automatique
    # ========================================
    print("\n" + "="*70)
    print("MÉTHODE 1: Calibration automatique depuis les images")
    print("="*70)

    config_auto = ISPCalibrator.calibrate_from_images(raw_img, yuv_img)

    # Appliquer l'ISP avec la config calibrée
    print("\n→ Application de l'ISP avec paramètres calibrés...")
    isp_auto = ISP(config_auto)
    result_auto = isp_auto.process(raw_img)

    # Sauvegarder le résultat
    result_auto_bgr = cv2.cvtColor(result_auto, cv2.COLOR_RGB2BGR)
    output_auto_path = "test_isp_auto_result.png"
    cv2.imwrite(output_auto_path, result_auto_bgr)
    print(f"✓ Résultat sauvegardé: {output_auto_path}")

    # Sauvegarder la configuration
    config_auto_path = Path("isp_config_auto.json")
    isp_auto.save_config(config_auto_path)

    # ========================================
    # MÉTHODE 2: Fonction helper rapide
    # ========================================
    print("\n" + "="*70)
    print("MÉTHODE 2: Fonction helper quick_calibrate_and_process")
    print("="*70)

    result_quick, config_quick = quick_calibrate_and_process(raw_img, yuv_img)

    # Sauvegarder
    result_quick_bgr = cv2.cvtColor(result_quick, cv2.COLOR_RGB2BGR)
    output_quick_path = "test_isp_quick_result.png"
    cv2.imwrite(output_quick_path, result_quick_bgr)
    print(f"✓ Résultat sauvegardé: {output_quick_path}")

    # ========================================
    # Comparaison visuelle et statistiques
    # ========================================
    print("\n" + "="*70)
    print("GÉNÉRATION DES COMPARAISONS")
    print("="*70)

    # Créer la visualisation comparative
    plot_comparison(raw_img, yuv_img, result_auto, config_auto,
                   "isp_calibration_comparison.png")

    # ========================================
    # TEST: Appliquer la config sur une autre image RAW
    # ========================================
    print("\n" + "="*70)
    print("RÉUTILISATION DE LA CONFIG CALIBRÉE")
    print("="*70)
    print("\nPour appliquer cette config à d'autres images RAW de la même session:")
    print("""
    # Charger la config sauvegardée
    isp = ISP.load_config('isp_config_auto.json')

    # Appliquer à de nouvelles images RAW
    raw_new = cv2.imread('nouvelle_image_raw.png')
    raw_new = cv2.cvtColor(raw_new, cv2.COLOR_BGR2RGB)
    result = isp.process(raw_new)
    """)

    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    print(f"\n✓ Calibration ISP réussie!")
    print(f"✓ Configuration sauvegardée: {config_auto_path}")
    print(f"✓ Résultats:")
    print(f"    - {output_auto_path}")
    print(f"    - {output_quick_path}")
    print(f"    - isp_calibration_comparison.png")

    print(f"\n📊 Paramètres calibrés:")
    print(f"    • Balance blancs: R={config_auto.wb_red_gain:.3f}, "
          f"G={config_auto.wb_green_gain:.3f}, B={config_auto.wb_blue_gain:.3f}")
    print(f"    • Gamma: {config_auto.gamma:.2f}")
    print(f"    • Contraste: {config_auto.contrast:.2f}")
    print(f"    • Saturation: {config_auto.saturation:.2f}")

    # Calculer la similarité (ajuster les dimensions si nécessaire)
    yuv_compare = yuv_img
    result_compare = result_auto
    if yuv_compare.shape != result_compare.shape:
        h = min(yuv_compare.shape[0], result_compare.shape[0])
        w = min(yuv_compare.shape[1], result_compare.shape[1])
        yuv_compare = yuv_compare[:h, :w]
        result_compare = result_compare[:h, :w]

    mae = np.mean(np.abs(yuv_compare.astype(np.float32) - result_compare.astype(np.float32)))
    mse = np.mean((yuv_compare.astype(np.float32) - result_compare.astype(np.float32))**2)
    print(f"\n📈 Métriques de similarité (ISP vs YUV420):")
    print(f"    • MAE (erreur absolue moyenne): {mae:.2f}")
    print(f"    • MSE (erreur quadratique moyenne): {mse:.2f}")
    print(f"    • RMSE: {np.sqrt(mse):.2f}")


if __name__ == "__main__":
    main()
