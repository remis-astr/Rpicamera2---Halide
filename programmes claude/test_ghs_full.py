#!/usr/bin/env python3
"""
Test de l'implémentation COMPLÈTE de GHS (RPiCamera2) avec tous les paramètres
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importer la fonction GHS complète depuis RPiCamera2
sys.path.insert(0, '/home/admin/Rpicamera tests/Rpicamera2')

# Importer load_image
from libastrostack.io import load_image

# Copier la fonction ghs_stretch complète de RPiCamera2
def ghs_stretch(array, D, b, SP, LP, HP):
    """
    Generalized Hyperbolic Stretch (GHS) - Algorithme conforme Siril/PixInsight
    """
    # Normaliser l'image entre 0 et 1
    img_float = array.astype(np.float64) / 255.0

    # Constante pour éviter divisions par zéro
    epsilon = 1e-10
    img_float = np.clip(img_float, epsilon, 1.0 - epsilon)

    # Si D = 0, pas de transformation (identité)
    if abs(D) < epsilon:
        return array

    # Assurer les contraintes : 0 <= LP <= SP <= HP <= 1
    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))

    def T_base(x, D, b):
        """Transformation de base T(x) selon le type déterminé par b"""
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
        """Dérivée première T'(x)"""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            result = D / (1.0 + D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            exponent = (1.0 + b) / b
            result = np.power(base, 1.0 / b)
        elif abs(b) < epsilon:
            result = D * np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = D / np.power(1.0 + D * x, 2)
        else:  # b > 0, b != 1
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = D * np.power(base, -(1.0 / b + 1.0))

        return result

    # Normalisation pour T(1) = 1
    T_1 = T_base(1.0, D, b)

    # Transformation symétrique autour de SP
    img_stretched = np.zeros_like(img_float)

    # Zone 1: x < LP (linéaire protégée - shadows)
    if LP > epsilon:
        mask_low = img_float < LP
        if np.any(mask_low):
            slope_SP = T_prime(SP, D, b) / T_1
            T_LP = T_base(LP, D, b) / T_1
            T_prime_SP_val = slope_SP

            # Pente à LP pour continuité
            slope_LP = T_prime_SP_val * (LP / SP)

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
            # Sous-zone 3a: SP <= x < HP (transformation hyperbolic - partie haute)
            mask_mid_high = (img_float >= SP) & (img_float < HP)
            if np.any(mask_mid_high):
                x_mirror = 1.0 - img_float[mask_mid_high]
                x_norm_mirror = x_mirror / (1.0 - SP)
                T_mirror = T_base(x_norm_mirror, D, b) / T_1
                img_stretched[mask_mid_high] = 1.0 - (1.0 - SP) * T_mirror

            # Sous-zone 3b: x >= HP (linéaire protégée - highlights)
            mask_very_high = img_float >= HP
            if np.any(mask_very_high):
                slope_SP_high = T_prime(SP, D, b) / T_1
                slope_HP = slope_SP_high * ((1.0 - HP) / (1.0 - SP))

                T_HP_val = T_base((1.0 - HP) / (1.0 - SP), D, b) / T_1
                y_HP = 1.0 - (1.0 - SP) * T_HP_val

                img_stretched[mask_very_high] = y_HP + slope_HP * (img_float[mask_very_high] - HP)
        else:
            # Pas de protection highlights (HP = 1.0)
            x_mirror = 1.0 - img_float[mask_high]
            x_norm_mirror = x_mirror / (1.0 - SP)
            T_mirror = T_base(x_norm_mirror, D, b) / T_1
            img_stretched[mask_high] = 1.0 - (1.0 - SP) * T_mirror

    # Clipper et reconvertir en uint8
    img_stretched = np.clip(img_stretched, 0.0, 1.0)
    return (img_stretched * 255.0).astype(np.uint8)


def test_ghs_presets(image_path, output_dir="ghs_full_test_results"):
    """Teste les 3 presets GHS de RPiCamera2"""
    print(f"Chargement de l'image: {image_path}")
    data = load_image(image_path)

    if data is None:
        print("Erreur de chargement")
        return

    print(f"Image chargée: shape={data.shape}, dtype={data.dtype}")

    # Normaliser si nécessaire
    if data.max() > 255:
        data = (data / data.max() * 255).astype(np.uint8)
    else:
        data = data.astype(np.uint8)

    # Convertir en niveaux de gris pour simplifier
    if len(data.shape) == 3:
        data = np.mean(data, axis=2).astype(np.uint8)

    # Créer répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Presets GHS
    presets = {
        'Original': None,
        'Galaxies\n(défaut)': {'D': 2.0, 'b': 6.0, 'SP': 0.15, 'LP': 0.0, 'HP': 0.85},
        'Nébuleuses': {'D': 2.5, 'b': 4.0, 'SP': 0.10, 'LP': 0.02, 'HP': 0.95},
        'Étirement initial': {'D': 3.5, 'b': 12.0, 'SP': 0.08, 'LP': 0.0, 'HP': 1.0},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('Presets GHS (implémentation complète RPiCamera2)', fontsize=16, fontweight='bold')

    for idx, (name, params) in enumerate(presets.items()):
        row = idx // 2
        col = idx % 2

        if params is None:
            # Image originale
            result = data
            print(f"Original")
        else:
            print(f"Test preset '{name}': {params}")
            result = ghs_stretch(data, **params)

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(name, fontsize=14, fontweight='bold')
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_presets_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


def test_ghs_parameters(image_path, output_dir="ghs_full_test_results"):
    """Teste l'influence de chaque paramètre GHS"""
    print(f"\nChargement de l'image pour tests paramétriques")
    data = load_image(image_path)

    if data is None:
        return

    # Normaliser
    if data.max() > 255:
        data = (data / data.max() * 255).astype(np.uint8)
    else:
        data = data.astype(np.uint8)

    if len(data.shape) == 3:
        data = np.mean(data, axis=2).astype(np.uint8)

    output_path = Path(output_dir)

    # Test 1: Variation de D (Stretch factor)
    print("\n--- Test du paramètre D (Stretch factor) ---")
    D_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Influence du paramètre D (b=6.0, SP=0.15, LP=0.0, HP=0.85)', fontsize=14)

    for idx, D in enumerate(D_values):
        row = idx // 3
        col = idx % 3
        print(f"  D={D}")
        result = ghs_stretch(data, D=D, b=6.0, SP=0.15, LP=0.0, HP=0.85)
        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'D={D}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / "ghs_param_D.png", dpi=150)
    print(f"✓ Sauvegardé: {output_path / 'ghs_param_D.png'}")
    plt.close()

    # Test 2: Variation de b (Local intensity)
    print("\n--- Test du paramètre b (Local intensity) ---")
    b_values = [0.0, 2.0, 4.0, 6.0, 10.0, 15.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Influence du paramètre b (D=2.0, SP=0.15, LP=0.0, HP=0.85)', fontsize=14)

    for idx, b in enumerate(b_values):
        row = idx // 3
        col = idx % 3
        print(f"  b={b}")
        result = ghs_stretch(data, D=2.0, b=b, SP=0.15, LP=0.0, HP=0.85)
        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'b={b}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / "ghs_param_b.png", dpi=150)
    print(f"✓ Sauvegardé: {output_path / 'ghs_param_b.png'}")
    plt.close()

    # Test 3: Variation de SP (Symmetry point)
    print("\n--- Test du paramètre SP (Symmetry point) ---")
    SP_values = [0.05, 0.10, 0.15, 0.25, 0.50, 0.75]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Influence du paramètre SP (D=2.0, b=6.0, LP=0.0, HP=0.85)', fontsize=14)

    for idx, SP in enumerate(SP_values):
        row = idx // 3
        col = idx % 3
        print(f"  SP={SP}")
        result = ghs_stretch(data, D=2.0, b=6.0, SP=SP, LP=0.0, HP=0.85)
        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'SP={SP}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / "ghs_param_SP.png", dpi=150)
    print(f"✓ Sauvegardé: {output_path / 'ghs_param_SP.png'}")
    plt.close()

    # Test 4: Variation de HP (Protect highlights)
    print("\n--- Test du paramètre HP (Protect highlights) ---")
    HP_values = [0.70, 0.80, 0.85, 0.90, 0.95, 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Influence du paramètre HP (D=2.0, b=6.0, SP=0.15, LP=0.0)', fontsize=14)

    for idx, HP in enumerate(HP_values):
        row = idx // 3
        col = idx % 3
        print(f"  HP={HP}")
        result = ghs_stretch(data, D=2.0, b=6.0, SP=0.15, LP=0.0, HP=HP)
        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'HP={HP}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / "ghs_param_HP.png", dpi=150)
    print(f"✓ Sauvegardé: {output_path / 'ghs_param_HP.png'}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("TEST GHS COMPLET (Implémentation RPiCamera2)")
    print("="*80)
    print(f"Image de test: {test_image}\n")

    # Test des presets
    print("="*80)
    print("TEST DES PRESETS GHS")
    print("="*80)
    test_ghs_presets(test_image)

    # Test des paramètres
    print("\n" + "="*80)
    print("TEST DES PARAMÈTRES GHS")
    print("="*80)
    test_ghs_parameters(test_image)

    print("\n" + "="*80)
    print("✓ TESTS TERMINÉS")
    print("="*80)
    print("Résultats dans: ghs_full_test_results/")
    print("  - ghs_presets_comparison.png (3 presets)")
    print("  - ghs_param_D.png (influence de D)")
    print("  - ghs_param_b.png (influence de b)")
    print("  - ghs_param_SP.png (influence de SP)")
    print("  - ghs_param_HP.png (influence de HP)")
