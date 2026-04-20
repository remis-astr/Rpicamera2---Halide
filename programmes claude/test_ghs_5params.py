#!/usr/bin/env python3
"""
Test GHS 5 paramètres (D, b, SP, LP, HP) - Version complète RPiCamera2
Analyse de l'impact des percentiles sur le signal
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from libastrostack.io import load_image
from libastrostack.stretch import stretch_asinh

# Copier la fonction GHS complète de RPiCamera2
def ghs_stretch_5params(array, D, b, SP, LP, HP):
    """
    GHS complet avec 5 paramètres (version RPiCamera2)

    Args:
        array: Image uint8 (0-255)
        D: Stretch factor (0.0 à 5.0)
        b: Local intensity (-5.0 à 15.0)
        SP: Symmetry point (0.0 à 1.0)
        LP: Protect shadows (0.0 à SP)
        HP: Protect highlights (SP à 1.0)
    """
    # Normaliser l'image entre 0 et 1
    img_float = array.astype(np.float64) / 255.0

    epsilon = 1e-10
    img_float = np.clip(img_float, epsilon, 1.0 - epsilon)

    if abs(D) < epsilon:
        return array

    # Assurer les contraintes
    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))

    def T_base(x, D, b):
        """Transformation de base"""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            result = np.log1p(D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            exponent = (b + 1.0) / b
            result = (1.0 - np.power(base, exponent)) / (D * (b + 1.0))
        elif abs(b) < epsilon:
            result = 1.0 - np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = 1.0 - 1.0 / (1.0 + D * x)
        else:  # b > 0, b != 1
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = 1.0 - np.power(base, -1.0 / b)

        return result

    def T_prime(x, D, b):
        """Dérivée première"""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            result = D / (1.0 + D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            result = np.power(base, 1.0 / b)
        elif abs(b) < epsilon:
            result = D * np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = D / np.power(1.0 + D * x, 2)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = D * np.power(base, -(1.0 / b + 1.0))

        return result

    # Normalisation
    T_1 = T_base(1.0, D, b)

    img_stretched = np.zeros_like(img_float)

    # Zone 1: x < LP (linéaire - shadows)
    if LP > epsilon:
        mask_low = img_float < LP
        if np.any(mask_low):
            slope_SP = T_prime(SP, D, b) / T_1
            slope_LP = slope_SP * (LP / SP)
            img_stretched[mask_low] = slope_LP * img_float[mask_low]

    # Zone 2: LP <= x < SP (transformation hyperbolic - partie basse)
    mask_mid_low = (img_float >= LP) & (img_float < SP)
    if np.any(mask_mid_low):
        x_norm = img_float[mask_mid_low] / SP
        T_x = T_base(x_norm, D, b) / T_1
        img_stretched[mask_mid_low] = SP * T_x

    # Zone 3: x >= SP (symétrie miroir)
    mask_high = img_float >= SP
    if np.any(mask_high):
        if HP < 1.0 - epsilon:
            # Sous-zone 3a: SP <= x < HP
            mask_mid_high = (img_float >= SP) & (img_float < HP)
            if np.any(mask_mid_high):
                x_mirror = 1.0 - img_float[mask_mid_high]
                x_norm_mirror = x_mirror / (1.0 - SP)
                T_mirror = T_base(x_norm_mirror, D, b) / T_1
                img_stretched[mask_mid_high] = 1.0 - (1.0 - SP) * T_mirror

            # Sous-zone 3b: x >= HP (linéaire - highlights)
            mask_very_high = img_float >= HP
            if np.any(mask_very_high):
                slope_SP_high = T_prime(SP, D, b) / T_1
                slope_HP = slope_SP_high * ((1.0 - HP) / (1.0 - SP))

                T_HP_val = T_base((1.0 - HP) / (1.0 - SP), D, b) / T_1
                y_HP = 1.0 - (1.0 - SP) * T_HP_val

                img_stretched[mask_very_high] = y_HP + slope_HP * (img_float[mask_very_high] - HP)
        else:
            # Pas de protection highlights
            x_mirror = 1.0 - img_float[mask_high]
            x_norm_mirror = x_mirror / (1.0 - SP)
            T_mirror = T_base(x_norm_mirror, D, b) / T_1
            img_stretched[mask_high] = 1.0 - (1.0 - SP) * T_mirror

    img_stretched = np.clip(img_stretched, 0.0, 1.0)
    return (img_stretched * 255.0).astype(np.uint8)


def apply_ghs_5params_with_percentiles(data, D, b, SP, LP, HP, clip_low, clip_high):
    """Applique GHS 5 params avec normalisation par percentiles"""
    # Normaliser avec percentiles
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)

    if vmax == vmin:
        return np.zeros_like(data)

    img_normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)

    # Convertir en uint8
    img_uint8 = (img_normalized * 255).astype(np.uint8)

    # Appliquer GHS
    result_uint8 = ghs_stretch_5params(img_uint8, D, b, SP, LP, HP)

    return result_uint8.astype(np.float32) / 255.0


def test_percentiles_impact(image_path, output_dir="ghs_5params_results"):
    """Teste l'impact des percentiles sur le signal"""
    print("="*80)
    print("TEST IMPACT DES PERCENTILES SUR LE SIGNAL")
    print("="*80)

    data = load_image(image_path)
    if data is None:
        return

    print(f"Image: shape={data.shape}, dtype={data.dtype}")
    print(f"Range brute: [{data.min():.1f}, {data.max():.1f}]")

    # Normaliser
    if data.max() > 1.0:
        data = data / data.max()

    # Convertir en mono
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Paramètres GHS Galaxies
    D, b, SP, LP, HP = 2.0, 6.0, 0.15, 0.0, 0.85

    # Différents percentiles à tester
    percentile_configs = [
        {'name': 'Très réduites\n(0.01, 99.98)', 'low': 0.01, 'high': 99.98},
        {'name': 'Réduites\n(0.1, 99.9)', 'low': 0.1, 'high': 99.9},
        {'name': 'Modérées\n(0.5, 99.5)', 'low': 0.5, 'high': 99.5},
        {'name': 'Standard\n(1.0, 99.5)', 'low': 1.0, 'high': 99.5},
        {'name': 'Agressives\n(2.0, 98.0)', 'low': 2.0, 'high': 98.0},
        {'name': 'Très agressives\n(5.0, 95.0)', 'low': 5.0, 'high': 95.0},
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f'Impact des percentiles sur GHS 5 params\n(D={D}, b={b}, SP={SP}, LP={LP}, HP={HP})',
                 fontsize=14, fontweight='bold')

    print("\n" + "="*80)
    print("RÉSULTATS:")
    print("="*80)

    for idx, config in enumerate(percentile_configs):
        row = idx // 2
        col = idx % 2

        result = apply_ghs_5params_with_percentiles(
            data, D, b, SP, LP, HP, config['low'], config['high']
        )

        # Calculer vmin/vmax utilisés
        vmin = np.percentile(data, config['low'])
        vmax = np.percentile(data, config['high'])

        print(f"\n{config['name'].replace(chr(10), ' ')}:")
        print(f"  vmin (P{config['low']:.2f}): {vmin:.6f}")
        print(f"  vmax (P{config['high']:.2f}): {vmax:.6f}")
        print(f"  Plage normalisée: {vmax - vmin:.6f}")
        print(f"  Mean résultat: {result.mean():.6f}")
        print(f"  Pixels > 0.5: {(result > 0.5).sum() / result.size * 100:.2f}%")

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f"{config['name']}\nMean: {result.mean():.4f}", fontsize=11)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_5params_percentiles_impact.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Sauvegardé: {output_file}")
    plt.close()


def compare_ghs_presets_5params(image_path, output_dir="ghs_5params_results"):
    """Compare les presets GHS 5 paramètres avec percentiles très réduites"""
    print("\n" + "="*80)
    print("COMPARAISON PRESETS GHS 5 PARAMÈTRES")
    print("="*80)

    data = load_image(image_path)
    if data is None:
        return

    if data.max() > 1.0:
        data = data / data.max()

    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    output_path = Path(output_dir)

    # Presets RPiCamera2
    presets = [
        {
            'name': 'Original',
            'params': None
        },
        {
            'name': 'Galaxies\n(D=2, b=6, SP=0.15, LP=0, HP=0.85)',
            'params': {'D': 2.0, 'b': 6.0, 'SP': 0.15, 'LP': 0.0, 'HP': 0.85}
        },
        {
            'name': 'Nébuleuses\n(D=2.5, b=4, SP=0.1, LP=0.02, HP=0.95)',
            'params': {'D': 2.5, 'b': 4.0, 'SP': 0.10, 'LP': 0.02, 'HP': 0.95}
        },
        {
            'name': 'Étirement initial\n(D=3.5, b=12, SP=0.08, LP=0, HP=1.0)',
            'params': {'D': 3.5, 'b': 12.0, 'SP': 0.08, 'LP': 0.0, 'HP': 1.0}
        },
        {
            'name': 'ARCSINH\n(factor=10, clip=(1, 99.5))',
            'params': 'arcsinh'
        }
    ]

    # Utiliser percentiles très réduites (0.01, 99.98)
    clip_low, clip_high = 0.01, 99.98

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Presets GHS 5 params avec percentiles très réduites ({clip_low}, {clip_high})',
                 fontsize=14, fontweight='bold')

    print(f"\nUtilisation des percentiles: ({clip_low}, {clip_high})")
    print("="*80)

    for idx, preset in enumerate(presets):
        if idx >= 6:
            break

        row = idx // 3
        col = idx % 3

        if preset['params'] is None:
            result = np.clip(data, 0, 1)
            print(f"\n{preset['name'].replace(chr(10), ' ')}: Original")
        elif preset['params'] == 'arcsinh':
            result = stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5)
            print(f"\n{preset['name'].replace(chr(10), ' ')}: Mean={result.mean():.4f}")
        else:
            result = apply_ghs_5params_with_percentiles(
                data, clip_low=clip_low, clip_high=clip_high, **preset['params']
            )
            print(f"\n{preset['name'].replace(chr(10), ' ')}: Mean={result.mean():.4f}")

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f"{preset['name']}\nMean: {result.mean():.4f}", fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_5params_presets_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Sauvegardé: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("TEST GHS 5 PARAMÈTRES (Version complète RPiCamera2)")
    print("="*80)
    print(f"Image: {test_image}\n")

    # Test 1: Impact des percentiles
    test_percentiles_impact(test_image)

    # Test 2: Presets avec percentiles très réduites
    compare_ghs_presets_5params(test_image)

    print("\n" + "="*80)
    print("✓ TESTS TERMINÉS")
    print("="*80)
    print("\nFichiers dans ghs_5params_results/:")
    print("  - ghs_5params_percentiles_impact.png (impact des percentiles)")
    print("  - ghs_5params_presets_comparison.png (presets comparés)")
