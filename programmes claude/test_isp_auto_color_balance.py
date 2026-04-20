#!/usr/bin/env python3
"""
Test de la calibration automatique de l'ISP basée sur les pics d'histogrammes
Démontre l'utilisation de la nouvelle fonctionnalité d'alignement des couleurs
"""

import numpy as np
import cv2
from pathlib import Path
from libastrostack.isp import ISP, ISPCalibrator, ISPConfig
import matplotlib.pyplot as plt

def plot_histograms(image: np.ndarray, title: str = "Histogrammes RGB"):
    """
    Affiche les histogrammes des 3 canaux RGB avec les pics marqués
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['red', 'green', 'blue']
    channel_names = ['Rouge', 'Vert', 'Bleu']

    # Convertir en float si nécessaire
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        img_float = image.astype(np.float32) / 65535.0
    else:
        img_float = image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / img_float.max()

    for i, (ax, color, name) in enumerate(zip(axes, colors, channel_names)):
        channel = img_float[:, :, i].flatten()

        # Calculer l'histogramme
        hist, bins = np.histogram(channel, bins=256, range=(0.0, 1.0))
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Tracer l'histogramme en mode escalier (step) pour un alignement correct
        ax.fill_between(bin_centers, hist, alpha=0.5, color=color, step='mid')
        ax.step(bin_centers, hist, color=color, linewidth=2, where='mid')

        # Marquer le pic
        peak_idx = np.argmax(hist)
        peak_value = bin_centers[peak_idx]
        ax.axvline(peak_value, color='black', linestyle='--', linewidth=2,
                  label=f'Pic: {peak_value:.3f}')

        ax.set_title(f'{name} ({img_float[:,:,i].mean():.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Intensité')
        ax.set_ylabel('Nombre de pixels')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def test_calibration_from_stacked_image(image_path: Path):
    """
    Test la calibration automatique depuis une image stackée
    """
    print(f"\n{'='*70}")
    print(f"TEST: Calibration automatique par pics d'histogrammes")
    print(f"{'='*70}")

    # Charger l'image
    if not image_path.exists():
        print(f"❌ Fichier introuvable: {image_path}")
        return

    print(f"\n1. Chargement de l'image: {image_path.name}")

    # Lire l'image (supposer qu'elle est RGB ou BGR)
    if image_path.suffix.lower() in ['.fits', '.fit']:
        from libastrostack.io import read_fits
        img = read_fits(image_path)
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Erreur lors du chargement de {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"   Dimensions: {img.shape}")
    print(f"   Type: {img.dtype}")

    # Afficher les histogrammes AVANT calibration
    print("\n2. Analyse des histogrammes AVANT calibration")
    fig_before = plot_histograms(img, "Histogrammes AVANT calibration ISP")
    fig_before.savefig('histogram_before_calibration.png', dpi=150, bbox_inches='tight')
    print("   ✓ Histogrammes sauvegardés: histogram_before_calibration.png")

    # Calibrer l'ISP automatiquement
    print("\n3. Calibration automatique de l'ISP par pics d'histogrammes")
    isp_config = ISPCalibrator.calibrate_from_stacked_image(
        img,
        method='histogram_peaks'
    )

    # Afficher la configuration
    print(f"\n4. Configuration ISP calibrée:")
    print(f"   Balance des blancs:")
    print(f"     - Rouge:  {isp_config.wb_red_gain:.4f}")
    print(f"     - Vert:   {isp_config.wb_green_gain:.4f}")
    print(f"     - Bleu:   {isp_config.wb_blue_gain:.4f}")
    print(f"   Gamma:      {isp_config.gamma:.2f}")
    print(f"   Contraste:  {isp_config.contrast:.2f}")
    print(f"   Saturation: {isp_config.saturation:.2f}")

    # Appliquer l'ISP
    print("\n5. Application du pipeline ISP")
    isp = ISP(isp_config)
    img_processed = isp.process(img, return_uint8=False)

    # Afficher les histogrammes APRÈS calibration
    print("\n6. Analyse des histogrammes APRÈS calibration")
    fig_after = plot_histograms(img_processed, "Histogrammes APRÈS calibration ISP")
    fig_after.savefig('histogram_after_calibration.png', dpi=150, bbox_inches='tight')
    print("   ✓ Histogrammes sauvegardés: histogram_after_calibration.png")

    # Sauvegarder les images
    print("\n7. Sauvegarde des résultats")

    # Image avant (convertir pour sauvegarde)
    if img.dtype == np.uint8:
        img_save = img
    elif img.dtype == np.uint16:
        img_save = (img / 256).astype(np.uint8)
    else:
        img_save = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    cv2.imwrite('image_before_isp.png', cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
    print("   ✓ Image avant ISP: image_before_isp.png")

    # Image après
    img_processed_uint8 = (np.clip(img_processed, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite('image_after_isp.png', cv2.cvtColor(img_processed_uint8, cv2.COLOR_RGB2BGR))
    print("   ✓ Image après ISP: image_after_isp.png")

    # Sauvegarder la configuration
    isp.save_config(Path('isp_config_auto_histogram.json'))

    print(f"\n{'='*70}")
    print("✓ Test terminé avec succès!")
    print(f"{'='*70}\n")


def demo_usage_in_session():
    """
    Démontre comment intégrer la calibration automatique dans une session de live stacking
    """
    print("\n" + "="*70)
    print("EXEMPLE D'UTILISATION DANS UNE SESSION DE LIVE STACKING")
    print("="*70 + "\n")

    code_example = '''
# Exemple d'intégration dans RPiCamera2.py ou session.py

from libastrostack.session import LiveStackSession
from libastrostack.config import StackingConfig
from libastrostack.isp import ISP, ISPCalibrator

# 1. Créer la session de stacking
config = StackingConfig()
config.video_format = 'raw12'  # Format RAW
config.isp_enable = True       # Activer l'ISP
session = LiveStackSession(config)

# 2. Stacker quelques frames (au moins 10-20 pour avoir un bon signal)
for i in range(20):
    frame = capture_frame()  # Votre méthode de capture
    session.process_image_data(frame)

# 3. Après quelques frames, calibrer automatiquement l'ISP
if session.stacker.num_stacked > 10:
    print("\\nRecalibration automatique de l'ISP...")

    # Récupérer l'image stackée actuelle
    stacked_image = session.stacker.get_stacked_image()

    # Calibrer l'ISP basé sur les pics d'histogrammes
    isp_config = ISPCalibrator.calibrate_from_stacked_image(
        stacked_image,
        method='histogram_peaks'  # Recommandé pour l'astrophotographie
    )

    # Appliquer la nouvelle configuration
    session.isp = ISP(isp_config)

    # Optionnel: sauvegarder la config pour réutilisation
    session.isp.save_config('session_isp_config.json')

    print("✓ ISP recalibré automatiquement!")

# 4. Continuer le stacking avec l'ISP calibré
for i in range(100):
    frame = capture_frame()
    session.process_image_data(frame)

    # Re-calibrer périodiquement si nécessaire (tous les 50 frames par exemple)
    if (i + 1) % 50 == 0 and session.stacker.num_stacked > 0:
        stacked = session.stacker.get_stacked_image()
        isp_config = ISPCalibrator.calibrate_from_stacked_image(stacked)
        session.isp = ISP(isp_config)
        print(f"✓ ISP recalibré après {session.stacker.num_stacked} frames")
'''

    print(code_example)
    print("="*70 + "\n")


if __name__ == '__main__':
    import sys

    print("\n" + "="*70)
    print("TEST DE CALIBRATION AUTOMATIQUE ISP - ALIGNEMENT PAR PICS D'HISTOGRAMMES")
    print("="*70)

    # Afficher l'exemple d'utilisation
    demo_usage_in_session()

    # Si un fichier est fourni en argument, le tester
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        test_calibration_from_stacked_image(image_path)
    else:
        print("\nUsage:")
        print("  python test_isp_auto_color_balance.py <chemin_image_stackée>")
        print("\nExemple:")
        print("  python test_isp_auto_color_balance.py stacked_image.png")
        print("  python test_isp_auto_color_balance.py stacked_image.fits")
        print("\nLe script va:")
        print("  1. Analyser les histogrammes RGB de l'image")
        print("  2. Calibrer automatiquement l'ISP pour aligner les pics")
        print("  3. Appliquer l'ISP et comparer les résultats")
        print("  4. Sauvegarder les images et histogrammes avant/après")
        print("  5. Sauvegarder la configuration ISP calibrée")
