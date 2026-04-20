#!/usr/bin/env python3
"""
Test GHS avec les percentiles corrigés (1.0, 99.5) vs ARCSINH
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from libastrostack.io import load_image
from libastrostack.stretch import stretch_asinh, stretch_ghs

def compare_arcsinh_vs_ghs_fixed(image_path, output_dir="ghs_fixed_results"):
    """Compare ARCSINH vs GHS avec percentiles corrigés"""
    print("="*80)
    print("COMPARAISON ARCSINH vs GHS (percentiles corrigés)")
    print("="*80)
    print(f"Image: {image_path}\n")

    data = load_image(image_path)
    if data is None:
        return

    print(f"Image chargée: shape={data.shape}, dtype={data.dtype}")
    print(f"Range: [{data.min():.1f}, {data.max():.1f}]")

    # Normaliser
    if data.max() > 1.0:
        data = data / data.max()

    # Convertir en mono
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    # Créer répertoire
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Tests
    tests = [
        {
            'name': 'Original\n(clipé 0-1)',
            'data': np.clip(data, 0, 1),
            'color': 'gray'
        },
        {
            'name': 'ARCSINH\n(factor=10, clip=(1, 99.5))',
            'data': stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5),
            'color': 'green'
        },
        {
            'name': 'GHS ANCIEN\n(D=0, B=0, SP=0.5, clip=(0.01, 99.98))',
            'data': stretch_ghs(data, D=0.0, B=0.0, SP=0.5, clip_low=0.01, clip_high=99.98),
            'color': 'red'
        },
        {
            'name': 'GHS CORRIGÉ\n(D=0, B=0, SP=0.5, clip=(1, 99.5))',
            'data': stretch_ghs(data, D=0.0, B=0.0, SP=0.5, clip_low=1.0, clip_high=99.5),
            'color': 'blue'
        }
    ]

    # Afficher les statistiques
    print("\n" + "="*80)
    print("STATISTIQUES DES RÉSULTATS")
    print("="*80)
    for test in tests:
        result = test['data']
        print(f"\n{test['name'].replace(chr(10), ' ')}:")
        print(f"  Mean:   {result.mean():.6f}")
        print(f"  Median: {np.median(result):.6f}")
        print(f"  Std:    {result.std():.6f}")
        print(f"  Range:  [{result.min():.6f}, {result.max():.6f}]")
        print(f"  >0.5:   {(result > 0.5).sum() / result.size * 100:.2f}%")

    # Visualisation comparative
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Comparaison ARCSINH vs GHS (ancien vs corrigé)', fontsize=16, fontweight='bold')

    for idx, test in enumerate(tests):
        # Images
        axes[0, idx].imshow(test['data'], cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(test['name'], fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')

        # Histogrammes
        axes[1, idx].hist(test['data'].flatten(), bins=100, alpha=0.7, color=test['color'])
        axes[1, idx].set_title(f"Mean: {test['data'].mean():.4f}", fontsize=10)
        axes[1, idx].set_xlabel('Valeur pixel')
        axes[1, idx].set_ylabel('Fréquence')
        axes[1, idx].set_xlim(0, 1)

    plt.tight_layout()
    output_file = output_path / "comparison_arcsinh_vs_ghs_fixed.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sauvegardé: {output_file}")
    plt.close()


def test_ghs_presets_with_fixed_percentiles(image_path, output_dir="ghs_fixed_results"):
    """Teste les presets GHS avec les percentiles corrigés"""
    print("\n" + "="*80)
    print("TEST DES PRESETS GHS AVEC PERCENTILES CORRIGÉS")
    print("="*80)

    data = load_image(image_path)
    if data is None:
        return

    if data.max() > 1.0:
        data = data / data.max()

    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    output_path = Path(output_dir)

    # Presets à tester
    presets = [
        {
            'name': 'Original',
            'params': None
        },
        {
            'name': 'GHS D=0, B=0, SP=0.5\n(défaut libastrostack)',
            'params': {'D': 0.0, 'B': 0.0, 'SP': 0.5}
        },
        {
            'name': 'GHS D=2, B=6, SP=0.15\n(preset Galaxies RPiCamera2)',
            'params': {'D': 2.0, 'B': 6.0, 'SP': 0.15}
        },
        {
            'name': 'ARCSINH factor=10',
            'params': 'arcsinh'
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('Presets GHS avec percentiles corrigés (1.0, 99.5)', fontsize=16, fontweight='bold')

    for idx, preset in enumerate(presets):
        row = idx // 2
        col = idx % 2

        if preset['params'] is None:
            result = np.clip(data, 0, 1)
            print(f"\n{preset['name']}: Original")
        elif preset['params'] == 'arcsinh':
            result = stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5)
            print(f"\n{preset['name']}: Mean={result.mean():.4f}")
        else:
            result = stretch_ghs(data, **preset['params'], clip_low=1.0, clip_high=99.5)
            print(f"\n{preset['name']}: Mean={result.mean():.4f}")

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f"{preset['name']}\nMean: {result.mean():.4f}", fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_presets_fixed_percentiles.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sauvegardé: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("TEST GHS AVEC PERCENTILES CORRIGÉS")
    print("="*80)
    print(f"Nouvelle configuration:")
    print(f"  clip_low:  1.0   (avant: 0.01)")
    print(f"  clip_high: 99.5  (avant: 99.98)")
    print("="*80 + "\n")

    # Test 1: Comparaison directe
    compare_arcsinh_vs_ghs_fixed(test_image)

    # Test 2: Presets
    test_ghs_presets_with_fixed_percentiles(test_image)

    print("\n" + "="*80)
    print("✓ TESTS TERMINÉS")
    print("="*80)
    print("\nFichiers générés dans ghs_fixed_results/:")
    print("  - comparison_arcsinh_vs_ghs_fixed.png")
    print("  - ghs_presets_fixed_percentiles.png")
