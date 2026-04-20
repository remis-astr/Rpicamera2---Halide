#!/usr/bin/env python3
"""
GHS 5 paramètres UNIVERSEL - Supporte uint8, uint16, et float
Compatible YUV420→RGB888, RAW12, RAW16
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from libastrostack.io import load_image
from libastrostack.stretch import stretch_asinh


def ghs_stretch_universal(data, D, b, SP, LP=0.0, HP=1.0, output_dtype=None):
    """
    GHS 5 paramètres UNIVERSEL - Détecte automatiquement le format

    Args:
        data: Image en uint8, uint16, ou float32/64
        D: Stretch factor (0.0 à 5.0)
        b: Local intensity (-5.0 à 15.0)
        SP: Symmetry point (0.0 à 1.0)
        LP: Protect shadows (0.0 à SP)
        HP: Protect highlights (SP à 1.0)
        output_dtype: Type de sortie (None = même que l'entrée)

    Returns:
        Image étirée dans le même dtype que l'entrée (ou output_dtype)
    """
    original_dtype = data.dtype
    original_shape = data.shape

    # Détection automatique de la plage des données
    if data.dtype == np.uint8:
        max_val = 255.0
        print(f"[GHS] Détecté: uint8 (RGB888/YUV420) - max_val={max_val}")
    elif data.dtype == np.uint16:
        # Détecter si RAW12 (max ~4095) ou RAW16 (max ~65535)
        data_max = data.max()
        if data_max <= 4095:
            max_val = 4095.0
            print(f"[GHS] Détecté: uint16 RAW12 - max_val={max_val}")
        else:
            max_val = 65535.0
            print(f"[GHS] Détecté: uint16 RAW16 - max_val={max_val}")
    elif data.dtype in [np.float32, np.float64]:
        data_max = data.max()
        if data_max <= 1.0:
            max_val = 1.0
            print(f"[GHS] Détecté: float normalisé (0-1) - max_val={max_val}")
        else:
            max_val = data_max
            print(f"[GHS] Détecté: float non-normalisé - max_val={max_val}")
    else:
        raise ValueError(f"Type de données non supporté: {data.dtype}")

    # Normaliser en float64 (0-1)
    img_float = data.astype(np.float64) / max_val

    epsilon = 1e-10
    img_float = np.clip(img_float, epsilon, 1.0 - epsilon)

    # Si D = 0, pas de transformation
    if abs(D) < epsilon:
        if output_dtype:
            return (img_float * max_val).astype(output_dtype)
        return data

    # Assurer les contraintes
    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))

    # =========================================================================
    # TRANSFORMATION GHS (identique à RPiCamera2)
    # =========================================================================

    def T_base(x, D, b):
        """Transformation de base selon b"""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            # Logarithmic
            result = np.log1p(D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            # Integral
            base = np.maximum(1.0 - b * D * x, epsilon)
            exponent = (b + 1.0) / b
            result = (1.0 - np.power(base, exponent)) / (D * (b + 1.0))
        elif abs(b) < epsilon:
            # Exponential
            result = 1.0 - np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            # Harmonic
            result = 1.0 - 1.0 / (1.0 + D * x)
        else:  # b > 0, b != 1
            # Hyperbolic
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

    # Normalisation T(1) = 1
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

    # Clipper
    img_stretched = np.clip(img_stretched, 0.0, 1.0)

    # Reconvertir dans le format de sortie
    if output_dtype:
        target_dtype = output_dtype
    else:
        target_dtype = original_dtype

    if target_dtype == np.uint8:
        return (img_stretched * 255.0).astype(np.uint8)
    elif target_dtype == np.uint16:
        return (img_stretched * max_val).astype(np.uint16)
    elif target_dtype in [np.float32, np.float64]:
        return img_stretched.astype(target_dtype)
    else:
        return (img_stretched * max_val).astype(target_dtype)


def test_ghs_on_different_formats(image_path, output_dir="ghs_universal_results"):
    """Teste GHS sur différents formats de données"""
    print("="*80)
    print("TEST GHS UNIVERSEL SUR DIFFÉRENTS FORMATS")
    print("="*80)

    # Charger l'image DNG (float32)
    data_float = load_image(image_path)
    if data_float is None:
        return

    print(f"\nImage originale: shape={data_float.shape}, dtype={data_float.dtype}")
    print(f"Range: [{data_float.min():.1f}, {data_float.max():.1f}]")

    # Convertir en mono
    if len(data_float.shape) == 3:
        data_float = np.mean(data_float, axis=2)

    # Créer différentes versions
    print("\n--- Création des versions de test ---")

    # 1. Float normalisé (0-1)
    data_float_norm = data_float / data_float.max()
    print(f"Float normalisé (0-1): dtype={data_float_norm.dtype}, range=[{data_float_norm.min():.6f}, {data_float_norm.max():.6f}]")

    # 2. Simuler RAW12 (uint16, valeurs 0-4095)
    data_raw12 = (data_float_norm * 4095).astype(np.uint16)
    print(f"Simulé RAW12: dtype={data_raw12.dtype}, range=[{data_raw12.min()}, {data_raw12.max()}]")

    # 3. Simuler RAW16 (uint16, valeurs 0-65535)
    data_raw16 = (data_float_norm * 65535).astype(np.uint16)
    print(f"Simulé RAW16: dtype={data_raw16.dtype}, range=[{data_raw16.min()}, {data_raw16.max()}]")

    # 4. Simuler RGB888 (uint8, valeurs 0-255)
    data_rgb888 = (data_float_norm * 255).astype(np.uint8)
    print(f"Simulé RGB888: dtype={data_rgb888.dtype}, range=[{data_rgb888.min()}, {data_rgb888.max()}]")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Paramètres GHS Galaxies
    D, b, SP, LP, HP = 2.0, 6.0, 0.15, 0.0, 0.85

    # Tester sur chaque format
    print(f"\n--- Application GHS (D={D}, b={b}, SP={SP}, LP={LP}, HP={HP}) ---")

    formats = [
        ("Float (0-1)", data_float_norm, np.float32),
        ("RAW12 (uint16)", data_raw12, None),
        ("RAW16 (uint16)", data_raw16, None),
        ("RGB888 (uint8)", data_rgb888, None),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'GHS Universel - Test sur différents formats\n(D={D}, b={b}, SP={SP}, LP={LP}, HP={HP})',
                 fontsize=14, fontweight='bold')

    for idx, (name, data, out_dtype) in enumerate(formats):
        print(f"\n{name}:")
        result = ghs_stretch_universal(data, D, b, SP, LP, HP, output_dtype=out_dtype)

        # Normaliser pour affichage
        if result.dtype == np.uint8:
            result_display = result.astype(np.float32) / 255.0
        elif result.dtype == np.uint16:
            result_display = result.astype(np.float32) / result.max()
        else:
            result_display = result

        print(f"  Résultat: dtype={result.dtype}, range=[{result.min()}, {result.max()}]")
        print(f"  Mean (normalisé): {result_display.mean():.6f}")

        # Afficher original
        if data.dtype == np.uint8:
            data_display = data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            data_display = data.astype(np.float32) / data.max()
        else:
            data_display = data

        axes[0, idx].imshow(data_display, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f'{name}\nOriginal', fontsize=10)
        axes[0, idx].axis('off')

        axes[1, idx].imshow(result_display, cmap='gray', vmin=0, vmax=1)
        axes[1, idx].set_title(f'GHS\nMean: {result_display.mean():.4f}', fontsize=10)
        axes[1, idx].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_universal_formats_test.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Sauvegardé: {output_file}")
    plt.close()


def compare_ghs_universal_vs_arcsinh(image_path, output_dir="ghs_universal_results"):
    """Compare GHS universel vs ARCSINH sur données haute profondeur"""
    print("\n" + "="*80)
    print("COMPARAISON GHS UNIVERSEL vs ARCSINH (haute profondeur)")
    print("="*80)

    data = load_image(image_path)
    if data is None:
        return

    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    # Garder en float pour préserver la dynamique
    data_float = data.astype(np.float32)
    data_max = data_float.max()

    print(f"\nDonnées: dtype={data_float.dtype}, max={data_max:.1f}")

    # Normaliser (0-1)
    data_norm = data_float / data_max

    output_path = Path(output_dir)

    # Tests
    tests = [
        {
            'name': 'Original',
            'data': data_norm,
            'method': None
        },
        {
            'name': 'ARCSINH\n(factor=10)',
            'data': stretch_asinh(data_norm, factor=10.0, clip_low=1.0, clip_high=99.5),
            'method': 'arcsinh'
        },
        {
            'name': 'GHS Universel\nGalaxies (D=2, b=6, SP=0.15)',
            'data': ghs_stretch_universal(data_norm, D=2.0, b=6.0, SP=0.15, LP=0.0, HP=0.85),
            'method': 'ghs'
        },
        {
            'name': 'GHS Universel\nNébuleuses (D=2.5, b=4, SP=0.1)',
            'data': ghs_stretch_universal(data_norm, D=2.5, b=4.0, SP=0.10, LP=0.02, HP=0.95),
            'method': 'ghs'
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('Comparaison GHS Universel vs ARCSINH (données float haute profondeur)',
                 fontsize=14, fontweight='bold')

    for idx, test in enumerate(tests):
        row = idx // 2
        col = idx % 2

        axes[row, col].imshow(test['data'], cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f"{test['name']}\nMean: {test['data'].mean():.4f}", fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_universal_vs_arcsinh.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Sauvegardé: {output_file}")
    plt.close()

    # Statistiques
    print("\n" + "="*80)
    print("STATISTIQUES")
    print("="*80)
    for test in tests:
        print(f"\n{test['name'].replace(chr(10), ' ')}:")
        print(f"  Mean: {test['data'].mean():.6f}")
        print(f"  Std:  {test['data'].std():.6f}")
        print(f"  Range: [{test['data'].min():.6f}, {test['data'].max():.6f}]")
        print(f"  >0.5: {(test['data'] > 0.5).sum() / test['data'].size * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("TEST GHS UNIVERSEL - Support uint8/uint16/float")
    print("="*80)
    print(f"Image: {test_image}\n")

    # Test 1: Différents formats
    test_ghs_on_different_formats(test_image)

    # Test 2: Comparaison avec ARCSINH
    compare_ghs_universal_vs_arcsinh(test_image)

    print("\n" + "="*80)
    print("✓ TESTS TERMINÉS")
    print("="*80)
    print("\nFichiers dans ghs_universal_results/:")
    print("  - ghs_universal_formats_test.png (test formats)")
    print("  - ghs_universal_vs_arcsinh.png (comparaison)")
