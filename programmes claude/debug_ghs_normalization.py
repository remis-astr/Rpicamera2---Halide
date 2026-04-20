#!/usr/bin/env python3
"""
Debug: Pourquoi GHS produit des images sombres?
Analyse de la normalisation et de l'histogramme
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from libastrostack.io import load_image

def analyze_image_statistics(image_path):
    """Analyse les statistiques de l'image DNG brute"""
    print("="*80)
    print("ANALYSE DE L'IMAGE BRUTE")
    print("="*80)

    data = load_image(image_path)
    if data is None:
        return None

    print(f"\nShape: {data.shape}")
    print(f"Dtype: {data.dtype}")

    # Statistiques brutes
    print(f"\n--- Statistiques brutes ---")
    print(f"Min: {data.min():.2f}")
    print(f"Max: {data.max():.2f}")
    print(f"Mean: {data.mean():.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Std: {data.std():.2f}")

    # Percentiles
    print(f"\n--- Percentiles ---")
    for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.98]:
        val = np.percentile(data, p)
        print(f"P{p:5.2f}: {val:10.2f}")

    # Histogramme
    print(f"\n--- Distribution ---")
    counts, bins = np.histogram(data.flatten(), bins=100)
    total = counts.sum()
    cumsum = 0
    for i in range(0, 100, 10):
        cumsum += counts[i:i+10].sum()
        pct = cumsum / total * 100
        print(f"Bins {i:2d}-{i+10:2d} ({bins[i]:8.1f} - {bins[i+10]:8.1f}): {pct:5.1f}% cumulé")

    return data


def compare_normalization_methods(data):
    """Compare différentes méthodes de normalisation"""
    print("\n" + "="*80)
    print("COMPARAISON DES MÉTHODES DE NORMALISATION")
    print("="*80)

    # Convertir en mono pour simplifier
    if len(data.shape) == 3:
        data_mono = np.mean(data, axis=2)
    else:
        data_mono = data

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Normalisation par max (méthode actuelle dans test_ghs_full.py)
    print("\n1. Normalisation par MAX")
    norm_max = data_mono / data_mono.max()
    print(f"   Range après norm: [{norm_max.min():.6f}, {norm_max.max():.6f}]")
    print(f"   Mean: {norm_max.mean():.6f}")
    print(f"   Pixels > 0.5: {(norm_max > 0.5).sum() / norm_max.size * 100:.2f}%")

    axes[0, 0].imshow(norm_max, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Normalisation par MAX\n(utilisée dans test GHS)')
    axes[0, 0].axis('off')

    axes[1, 0].hist(norm_max.flatten(), bins=100, alpha=0.7, color='blue')
    axes[1, 0].set_title('Histogramme (norm par MAX)')
    axes[1, 0].set_xlabel('Valeur normalisée')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_yscale('log')

    # 2. Normalisation par percentiles 1-99.5 (méthode ARCSINH)
    print("\n2. Normalisation par PERCENTILES (1, 99.5)")
    vmin = np.percentile(data_mono, 1.0)
    vmax = np.percentile(data_mono, 99.5)
    norm_percentile = np.clip((data_mono - vmin) / (vmax - vmin), 0, 1)
    print(f"   vmin (P1): {vmin:.2f}")
    print(f"   vmax (P99.5): {vmax:.2f}")
    print(f"   Range après norm: [{norm_percentile.min():.6f}, {norm_percentile.max():.6f}]")
    print(f"   Mean: {norm_percentile.mean():.6f}")
    print(f"   Pixels > 0.5: {(norm_percentile > 0.5).sum() / norm_percentile.size * 100:.2f}%")

    axes[0, 1].imshow(norm_percentile, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Normalisation par PERCENTILES (1, 99.5)\n(utilisée dans ARCSINH)')
    axes[0, 1].axis('off')

    axes[1, 1].hist(norm_percentile.flatten(), bins=100, alpha=0.7, color='green')
    axes[1, 1].set_title('Histogramme (norm par percentiles)')
    axes[1, 1].set_xlabel('Valeur normalisée')
    axes[1, 1].set_ylabel('Fréquence')

    # 3. Normalisation par percentiles 0.01-99.98 (GHS par défaut)
    print("\n3. Normalisation par PERCENTILES (0.01, 99.98)")
    vmin_ghs = np.percentile(data_mono, 0.01)
    vmax_ghs = np.percentile(data_mono, 99.98)
    norm_ghs = np.clip((data_mono - vmin_ghs) / (vmax_ghs - vmin_ghs), 0, 1)
    print(f"   vmin (P0.01): {vmin_ghs:.2f}")
    print(f"   vmax (P99.98): {vmax_ghs:.2f}")
    print(f"   Range après norm: [{norm_ghs.min():.6f}, {norm_ghs.max():.6f}]")
    print(f"   Mean: {norm_ghs.mean():.6f}")
    print(f"   Pixels > 0.5: {(norm_ghs > 0.5).sum() / norm_ghs.size * 100:.2f}%")

    axes[0, 2].imshow(norm_ghs, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Normalisation par PERCENTILES (0.01, 99.98)\n(GHS par défaut)')
    axes[0, 2].axis('off')

    axes[1, 2].hist(norm_ghs.flatten(), bins=100, alpha=0.7, color='red')
    axes[1, 2].set_title('Histogramme (norm GHS)')
    axes[1, 2].set_xlabel('Valeur normalisée')
    axes[1, 2].set_ylabel('Fréquence')

    plt.tight_layout()
    plt.savefig('debug_normalization_comparison.png', dpi=150)
    print(f"\n✓ Sauvegardé: debug_normalization_comparison.png")
    plt.close()

    return norm_max, norm_percentile, norm_ghs


def test_ghs_with_different_normalizations(data):
    """Teste GHS avec différentes normalisations"""
    print("\n" + "="*80)
    print("TEST GHS AVEC DIFFÉRENTES NORMALISATIONS")
    print("="*80)

    # Importer la fonction GHS complète
    sys.path.insert(0, '/home/admin/Rpicamera tests/Rpicamera2')

    # Copier la fonction ghs_stretch (version simplifiée pour test)
    def simple_ghs_transform(x_norm, D, b, SP):
        """Transformation GHS simplifiée pour debugging"""
        epsilon = 1e-10

        if abs(D) < epsilon:
            return x_norm

        # Transformation hyperbolique de base (b > 0)
        if b > epsilon:
            # Zone basse: x < SP
            mask_low = x_norm < SP
            mask_high = x_norm >= SP

            result = np.zeros_like(x_norm)

            if np.any(mask_low):
                x_norm_low = x_norm[mask_low] / SP
                base = 1.0 + b * D * x_norm_low
                T_x = 1.0 - np.power(base, -1.0 / b)
                T_1 = 1.0 - np.power(1.0 + b * D, -1.0 / b)
                result[mask_low] = SP * (T_x / T_1)

            if np.any(mask_high):
                x_mirror = 1.0 - x_norm[mask_high]
                x_norm_mirror = x_mirror / (1.0 - SP)
                base = 1.0 + b * D * x_norm_mirror
                T_x = 1.0 - np.power(base, -1.0 / b)
                T_1 = 1.0 - np.power(1.0 + b * D, -1.0 / b)
                result[mask_high] = 1.0 - (1.0 - SP) * (T_x / T_1)

            return np.clip(result, 0, 1)
        else:
            return x_norm

    # Convertir en mono
    if len(data.shape) == 3:
        data_mono = np.mean(data, axis=2)
    else:
        data_mono = data

    # Paramètres GHS "Galaxies"
    D, b, SP = 2.0, 6.0, 0.15

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'GHS (D={D}, b={b}, SP={SP}) avec différentes normalisations', fontsize=16)

    # 1. GHS après normalisation par MAX
    norm_max = data_mono / data_mono.max()
    ghs_max = simple_ghs_transform(norm_max, D, b, SP)

    axes[0, 0].imshow(ghs_max, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('GHS après norm MAX\n(méthode test_ghs_full.py)')
    axes[0, 0].axis('off')

    axes[1, 0].hist(ghs_max.flatten(), bins=100, alpha=0.7, color='blue')
    axes[1, 0].set_title(f'Histogramme - Mean: {ghs_max.mean():.4f}')
    axes[1, 0].set_xlabel('Valeur')

    # 2. GHS après normalisation par percentiles ARCSINH
    vmin = np.percentile(data_mono, 1.0)
    vmax = np.percentile(data_mono, 99.5)
    norm_percentile = np.clip((data_mono - vmin) / (vmax - vmin), 0, 1)
    ghs_percentile = simple_ghs_transform(norm_percentile, D, b, SP)

    axes[0, 1].imshow(ghs_percentile, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('GHS après norm PERCENTILES (1, 99.5)\n(comme ARCSINH)')
    axes[0, 1].axis('off')

    axes[1, 1].hist(ghs_percentile.flatten(), bins=100, alpha=0.7, color='green')
    axes[1, 1].set_title(f'Histogramme - Mean: {ghs_percentile.mean():.4f}')
    axes[1, 1].set_xlabel('Valeur')

    # 3. GHS après normalisation GHS par défaut
    vmin_ghs = np.percentile(data_mono, 0.01)
    vmax_ghs = np.percentile(data_mono, 99.98)
    norm_ghs = np.clip((data_mono - vmin_ghs) / (vmax_ghs - vmin_ghs), 0, 1)
    ghs_ghs = simple_ghs_transform(norm_ghs, D, b, SP)

    axes[0, 2].imshow(ghs_ghs, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('GHS après norm PERCENTILES (0.01, 99.98)\n(GHS par défaut)')
    axes[0, 2].axis('off')

    axes[1, 2].hist(ghs_ghs.flatten(), bins=100, alpha=0.7, color='red')
    axes[1, 2].set_title(f'Histogramme - Mean: {ghs_ghs.mean():.4f}')
    axes[1, 2].set_xlabel('Valeur')

    plt.tight_layout()
    plt.savefig('debug_ghs_normalization_impact.png', dpi=150)
    print(f"\n✓ Sauvegardé: debug_ghs_normalization_impact.png")
    plt.close()

    print(f"\nRésultats GHS:")
    print(f"  Avec norm MAX:         Mean = {ghs_max.mean():.6f}")
    print(f"  Avec norm P(1, 99.5):  Mean = {ghs_percentile.mean():.6f}")
    print(f"  Avec norm P(0.01, 99.98): Mean = {ghs_ghs.mean():.6f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("DEBUG: POURQUOI GHS PRODUIT DES IMAGES SOMBRES?")
    print("="*80)
    print(f"Image: {test_image}\n")

    # Étape 1: Analyser l'image brute
    data = analyze_image_statistics(test_image)

    if data is not None:
        # Étape 2: Comparer les normalisations
        compare_normalization_methods(data)

        # Étape 3: Tester GHS avec différentes normalisations
        test_ghs_with_different_normalizations(data)

    print("\n" + "="*80)
    print("✓ ANALYSE TERMINÉE")
    print("="*80)
    print("\nFichiers générés:")
    print("  - debug_normalization_comparison.png (comparaison normalisations)")
    print("  - debug_ghs_normalization_impact.png (impact sur GHS)")
