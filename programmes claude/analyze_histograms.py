#!/usr/bin/env python3
"""
Analyse les histogrammes des images RAW12 et YUV420 pour identifier
les traitements ISP à appliquer
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_image(img_path):
    """Analyse une image et retourne ses statistiques"""
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    stats = {
        'mean': np.mean(img_rgb, axis=(0, 1)),
        'median': np.median(img_rgb, axis=(0, 1)),
        'std': np.std(img_rgb, axis=(0, 1)),
        'min': np.min(img_rgb, axis=(0, 1)),
        'max': np.max(img_rgb, axis=(0, 1)),
        'percentile_1': np.percentile(img_rgb, 1, axis=(0, 1)),
        'percentile_99': np.percentile(img_rgb, 99, axis=(0, 1)),
        'shape': img_rgb.shape
    }

    return img_rgb, stats

def plot_histograms(img_raw, img_yuv, stats_raw, stats_yuv):
    """Compare les histogrammes des deux images"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    colors = ['red', 'green', 'blue']
    channel_names = ['Rouge', 'Vert', 'Bleu']

    # Histogrammes par canal
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        # Histogramme RAW
        axes[i, 0].hist(img_raw[:, :, i].ravel(), bins=256, range=(0, 256),
                       color=color, alpha=0.7)
        axes[i, 0].set_title(f'{name} - RAW12')
        axes[i, 0].set_xlim(0, 255)
        axes[i, 0].axvline(stats_raw['mean'][i], color='black',
                          linestyle='--', label=f'Mean: {stats_raw["mean"][i]:.1f}')
        axes[i, 0].legend()

        # Histogramme YUV
        axes[i, 1].hist(img_yuv[:, :, i].ravel(), bins=256, range=(0, 256),
                       color=color, alpha=0.7)
        axes[i, 1].set_title(f'{name} - YUV420')
        axes[i, 1].set_xlim(0, 255)
        axes[i, 1].axvline(stats_yuv['mean'][i], color='black',
                          linestyle='--', label=f'Mean: {stats_yuv["mean"][i]:.1f}')
        axes[i, 1].legend()

        # Histogrammes superposés (échelle log)
        axes[i, 2].hist(img_raw[:, :, i].ravel(), bins=256, range=(0, 256),
                       color='blue', alpha=0.5, label='RAW12')
        axes[i, 2].hist(img_yuv[:, :, i].ravel(), bins=256, range=(0, 256),
                       color='red', alpha=0.5, label='YUV420')
        axes[i, 2].set_title(f'{name} - Comparaison')
        axes[i, 2].set_xlim(0, 255)
        axes[i, 2].set_yscale('log')
        axes[i, 2].legend()

    plt.tight_layout()
    plt.savefig('histogram_comparison.png', dpi=150)
    print("\nGraphiques sauvegardés dans: histogram_comparison.png")

def estimate_gamma(img_raw, img_yuv):
    """Estime la valeur de gamma appliquée"""
    # Utilise les valeurs moyennes pour estimer gamma
    raw_mean = np.mean(img_raw) / 255.0
    yuv_mean = np.mean(img_yuv) / 255.0

    if raw_mean > 0:
        # yuv = raw^(1/gamma)
        gamma = np.log(yuv_mean) / np.log(raw_mean)
        return gamma
    return None

def estimate_color_gains(stats_raw, stats_yuv):
    """Estime les gains de balance des blancs"""
    gains = stats_yuv['mean'] / (stats_raw['mean'] + 1e-6)
    return gains

def apply_isp_pipeline(img_raw, gamma=2.2, wb_gains=None, saturation=1.3, contrast=1.1):
    """
    Applique une pipeline ISP simplifiée à une image RAW

    Args:
        img_raw: Image RAW (0-255)
        gamma: Valeur de gamma à appliquer
        wb_gains: Gains RGB pour la balance des blancs [R, G, B]
        saturation: Facteur de saturation (1.0 = pas de changement)
        contrast: Facteur de contraste (1.0 = pas de changement)
    """
    img = img_raw.astype(np.float32) / 255.0

    # 1. Balance des blancs
    if wb_gains is not None:
        for i in range(3):
            img[:, :, i] *= wb_gains[i]

    # Clip après balance des blancs
    img = np.clip(img, 0, 1)

    # 2. Correction gamma
    img = np.power(img, 1.0 / gamma)

    # 3. Contraste
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5
        img = np.clip(img, 0, 1)

    # 4. Saturation
    if saturation != 1.0:
        # Conversion en HSV pour ajuster la saturation
        img_uint8 = (img * 255).astype(np.uint8)
        img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:, :, 1] *= saturation
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return (img * 255).astype(np.uint8)

def main():
    # Chemins des images
    raw_path = Path("/media/admin/THKAILAR/Stacks/stack_SRGGB12_mean_20251212_100655.png")
    yuv_path = Path("/media/admin/THKAILAR/Stacks/stack_YUV420_mean_20251212_100812.png")

    print("=== Analyse des images RAW12 vs YUV420 ===\n")

    # Chargement et analyse
    print("Chargement des images...")
    img_raw, stats_raw = analyze_image(raw_path)
    img_yuv, stats_yuv = analyze_image(yuv_path)

    print("\n--- Statistiques RAW12 ---")
    print(f"Dimensions: {stats_raw['shape']}")
    print(f"Moyenne RGB: {stats_raw['mean']}")
    print(f"Médiane RGB: {stats_raw['median']}")
    print(f"Écart-type RGB: {stats_raw['std']}")
    print(f"Min RGB: {stats_raw['min']}")
    print(f"Max RGB: {stats_raw['max']}")
    print(f"1er percentile: {stats_raw['percentile_1']}")
    print(f"99e percentile: {stats_raw['percentile_99']}")

    print("\n--- Statistiques YUV420 ---")
    print(f"Dimensions: {stats_yuv['shape']}")
    print(f"Moyenne RGB: {stats_yuv['mean']}")
    print(f"Médiane RGB: {stats_yuv['median']}")
    print(f"Écart-type RGB: {stats_yuv['std']}")
    print(f"Min RGB: {stats_yuv['min']}")
    print(f"Max RGB: {stats_yuv['max']}")
    print(f"1er percentile: {stats_yuv['percentile_1']}")
    print(f"99e percentile: {stats_yuv['percentile_99']}")

    # Estimation des paramètres ISP
    print("\n=== Paramètres ISP estimés ===\n")

    # Gamma
    gamma_estimated = estimate_gamma(img_raw, img_yuv)
    print(f"Gamma estimé: {gamma_estimated:.2f}")

    # Balance des blancs
    wb_gains = estimate_color_gains(stats_raw, stats_yuv)
    print(f"Gains balance des blancs (R, G, B): {wb_gains}")

    # Ratio des moyennes
    mean_ratio = stats_yuv['mean'] / (stats_raw['mean'] + 1e-6)
    print(f"Ratio moyennes (YUV/RAW): {mean_ratio}")

    # Analyse de la saturation
    # Calculer la saturation moyenne de chaque image
    hsv_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2HSV)
    hsv_yuv = cv2.cvtColor(img_yuv, cv2.COLOR_RGB2HSV)
    sat_ratio = np.mean(hsv_yuv[:, :, 1]) / (np.mean(hsv_raw[:, :, 1]) + 1e-6)
    print(f"Ratio de saturation (YUV/RAW): {sat_ratio:.2f}")

    # Ratio d'écart-type (proxy pour le contraste)
    contrast_ratio = np.mean(stats_yuv['std']) / (np.mean(stats_raw['std']) + 1e-6)
    print(f"Ratio d'écart-type (contraste proxy): {contrast_ratio:.2f}")

    # Génération des histogrammes
    print("\nGénération des histogrammes...")
    plot_histograms(img_raw, img_yuv, stats_raw, stats_yuv)

    # Test de la pipeline ISP avec paramètres estimés
    print("\n=== Test de la pipeline ISP ===\n")

    # Normaliser les gains WB (utiliser le canal vert comme référence)
    wb_gains_normalized = wb_gains / wb_gains[1]
    print(f"Gains WB normalisés (G=1.0): {wb_gains_normalized}")

    # Tester différentes combinaisons de paramètres
    test_configs = [
        {"name": "Estimation auto", "gamma": gamma_estimated, "wb_gains": None,
         "saturation": sat_ratio, "contrast": contrast_ratio},
        {"name": "Pipeline typique 1", "gamma": 2.2, "wb_gains": None,
         "saturation": 1.3, "contrast": 1.1},
        {"name": "Pipeline typique 2", "gamma": 2.0, "wb_gains": None,
         "saturation": 1.2, "contrast": 1.15},
        {"name": "Pipeline avec WB", "gamma": 2.2, "wb_gains": wb_gains_normalized,
         "saturation": 1.3, "contrast": 1.1},
    ]

    for config in test_configs:
        print(f"\nTest: {config['name']}")
        print(f"  Gamma: {config['gamma']:.2f}")
        print(f"  WB gains: {config['wb_gains']}")
        print(f"  Saturation: {config['saturation']:.2f}")
        print(f"  Contrast: {config['contrast']:.2f}")

        img_processed = apply_isp_pipeline(
            img_raw,
            gamma=config['gamma'],
            wb_gains=config['wb_gains'],
            saturation=config['saturation'],
            contrast=config['contrast']
        )

        # Sauvegarder le résultat
        filename = f"test_isp_{config['name'].replace(' ', '_').lower()}.png"
        cv2.imwrite(filename, cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
        print(f"  Résultat sauvegardé: {filename}")

    print("\n=== Recommandations ===\n")
    print("Pour transformer l'image RAW12 en YUV420, appliquer dans l'ordre:")
    print(f"1. Balance des blancs avec gains: R={wb_gains_normalized[0]:.3f}, G=1.0, B={wb_gains_normalized[2]:.3f}")
    print(f"2. Correction gamma: {gamma_estimated:.2f} (ou utiliser 2.2 pour un résultat standard)")
    print(f"3. Ajustement du contraste: facteur ~{contrast_ratio:.2f}")
    print(f"4. Augmentation de la saturation: facteur ~{sat_ratio:.2f}")
    print("\nNOTE: Ces valeurs sont estimées. Des ajustements manuels peuvent être nécessaires.")

if __name__ == "__main__":
    main()
