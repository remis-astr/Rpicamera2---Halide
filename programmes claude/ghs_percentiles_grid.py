#!/usr/bin/env python3
"""
Grille de comparaison: GHS avec différents percentiles
Pour trouver la meilleure combinaison percentiles + paramètres GHS
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libastrostack.io import load_image
from libastrostack.stretch import stretch_asinh


def ghs_stretch_simple(data, D, b, SP, LP=0.0, HP=1.0):
    """GHS simplifié"""
    epsilon = 1e-10

    img_float = data.astype(np.float64)
    if img_float.max() > 1.0:
        img_float = img_float / img_float.max()

    img_float = np.clip(img_float, epsilon, 1.0 - epsilon)

    if abs(D) < epsilon:
        return img_float

    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))

    def T_base(x, D, b):
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b) < epsilon:
            result = 1.0 - np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = 1.0 - 1.0 / (1.0 + D * x)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = 1.0 - np.power(base, -1.0 / b)

        return result

    def T_prime(x, D, b):
        x = np.asarray(x, dtype=np.float64)
        if abs(b) < epsilon:
            return D * np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            return D / np.power(1.0 + D * x, 2)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            return D * np.power(base, -(1.0 / b + 1.0))

    T_1 = T_base(1.0, D, b)
    img_stretched = np.zeros_like(img_float)

    if LP > epsilon:
        mask_low = img_float < LP
        if np.any(mask_low):
            slope_SP = T_prime(SP, D, b) / T_1
            slope_LP = slope_SP * (LP / SP)
            img_stretched[mask_low] = slope_LP * img_float[mask_low]

    mask_mid_low = (img_float >= LP) & (img_float < SP)
    if np.any(mask_mid_low):
        x_norm = img_float[mask_mid_low] / SP
        T_x = T_base(x_norm, D, b) / T_1
        img_stretched[mask_mid_low] = SP * T_x

    mask_high = img_float >= SP
    if np.any(mask_high):
        if HP < 1.0 - epsilon:
            mask_mid_high = (img_float >= SP) & (img_float < HP)
            if np.any(mask_mid_high):
                x_mirror = 1.0 - img_float[mask_mid_high]
                x_norm_mirror = x_mirror / (1.0 - SP)
                T_mirror = T_base(x_norm_mirror, D, b) / T_1
                img_stretched[mask_mid_high] = 1.0 - (1.0 - SP) * T_mirror

            mask_very_high = img_float >= HP
            if np.any(mask_very_high):
                slope_SP_high = T_prime(SP, D, b) / T_1
                slope_HP = slope_SP_high * ((1.0 - HP) / (1.0 - SP))
                T_HP_val = T_base((1.0 - HP) / (1.0 - SP), D, b) / T_1
                y_HP = 1.0 - (1.0 - SP) * T_HP_val
                img_stretched[mask_very_high] = y_HP + slope_HP * (img_float[mask_very_high] - HP)
        else:
            x_mirror = 1.0 - img_float[mask_high]
            x_norm_mirror = x_mirror / (1.0 - SP)
            T_mirror = T_base(x_norm_mirror, D, b) / T_1
            img_stretched[mask_high] = 1.0 - (1.0 - SP) * T_mirror

    return np.clip(img_stretched, 0.0, 1.0)


def test_percentiles_combinations(image_path):
    """Teste différentes combinaisons de percentiles avec GHS"""
    print("="*80)
    print("GRILLE PERCENTILES × GHS")
    print("="*80)

    data = load_image(image_path)
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    # Différents percentiles à tester
    percentiles_configs = [
        {'name': 'Aucun', 'low': None, 'high': None},
        {'name': '0.01-99.99', 'low': 0.01, 'high': 99.99},
        {'name': '0.1-99.9', 'low': 0.1, 'high': 99.9},
        {'name': '1.0-99.5\n(ARCSINH)', 'low': 1.0, 'high': 99.5},
        {'name': '2.0-98.0', 'low': 2.0, 'high': 98.0},
        {'name': '5.0-95.0', 'low': 5.0, 'high': 95.0},
    ]

    # Paramètres GHS à tester
    ghs_configs = [
        {'name': 'Galaxies\nD=2,b=6,SP=0.15', 'D': 2.0, 'b': 6.0, 'SP': 0.15, 'LP': 0, 'HP': 0.85},
        {'name': 'Personnalisé\nD=3,b=0.13,SP=0.19', 'D': 3.06, 'b': 0.131, 'SP': 0.194, 'LP': 0, 'HP': 0},
        {'name': 'Fort\nD=5,b=15,SP=0.5', 'D': 5.0, 'b': 15.0, 'SP': 0.5, 'LP': 0, 'HP': 0.9},
    ]

    # Créer une grande grille
    fig, axes = plt.subplots(len(ghs_configs), len(percentiles_configs), figsize=(24, 12))
    fig.suptitle('Impact des Percentiles sur GHS\n(Lignes=Paramètres GHS, Colonnes=Percentiles)',
                 fontsize=14, fontweight='bold')

    for row_idx, ghs_config in enumerate(ghs_configs):
        for col_idx, perc_config in enumerate(percentiles_configs):
            # Préparer les données
            if perc_config['low'] is None:
                # Pas de normalisation
                data_norm = data / data.max()
            else:
                # Normalisation par percentiles
                vmin = np.percentile(data, perc_config['low'])
                vmax = np.percentile(data, perc_config['high'])
                data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)

            # Appliquer GHS
            result = ghs_stretch_simple(
                data_norm,
                ghs_config['D'],
                ghs_config['b'],
                ghs_config['SP'],
                ghs_config['LP'],
                ghs_config['HP']
            )

            # Afficher
            axes[row_idx, col_idx].imshow(result, cmap='gray', vmin=0, vmax=1)

            # Titre avec stats
            title = f"{perc_config['name']}\n"
            if row_idx == 0:  # Seulement en haut
                title = f"{perc_config['name']}\n"
            if col_idx == 0:  # Seulement à gauche
                title = f"{ghs_config['name']}\n"
            title += f"Mean:{result.mean():.3f}"

            axes[row_idx, col_idx].set_title(title, fontsize=9)
            axes[row_idx, col_idx].axis('off')

            print(f"GHS[{ghs_config['name'].split()[0]}] + Perc[{perc_config['name']}] → Mean={result.mean():.4f}")

    plt.tight_layout()
    plt.savefig('ghs_percentiles_grid.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Sauvegardé: ghs_percentiles_grid.png")

    # Comparaison avec ARCSINH
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Meilleure combinaison GHS+Percentiles vs ARCSINH', fontsize=14, fontweight='bold')

    # ARCSINH référence
    result_arcsinh = stretch_asinh(data / data.max(), factor=10.0, clip_low=1.0, clip_high=99.5)
    axes2[0].imshow(result_arcsinh, cmap='gray', vmin=0, vmax=1)
    axes2[0].set_title(f'ARCSINH (référence)\nMean: {result_arcsinh.mean():.3f}', fontsize=12, fontweight='bold')
    axes2[0].axis('off')

    # GHS avec percentiles (1%, 99.5%)
    vmin = np.percentile(data, 1.0)
    vmax = np.percentile(data, 99.5)
    data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    result_ghs = ghs_stretch_simple(data_norm, D=3.06, b=0.131, SP=0.194, LP=0, HP=0)
    axes2[1].imshow(result_ghs, cmap='gray', vmin=0, vmax=1)
    axes2[1].set_title(f'GHS (D=3, b=0.13, SP=0.19)\n+ Percentiles (1%, 99.5%)\nMean: {result_ghs.mean():.3f}',
                      fontsize=12, fontweight='bold', color='green')
    axes2[1].axis('off')

    # GHS fort avec percentiles agressifs
    vmin = np.percentile(data, 2.0)
    vmax = np.percentile(data, 98.0)
    data_norm2 = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    result_ghs2 = ghs_stretch_simple(data_norm2, D=5.0, b=15.0, SP=0.5, LP=0, HP=0.9)
    axes2[2].imshow(result_ghs2, cmap='gray', vmin=0, vmax=1)
    axes2[2].set_title(f'GHS Fort (D=5, b=15, SP=0.5)\n+ Percentiles (2%, 98%)\nMean: {result_ghs2.mean():.3f}',
                      fontsize=12, fontweight='bold', color='blue')
    axes2[2].axis('off')

    plt.tight_layout()
    plt.savefig('ghs_best_combinations.png', dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: ghs_best_combinations.png\n")


if __name__ == "__main__":
    image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"
    test_percentiles_combinations(image_path)

    print("="*80)
    print("✓ TERMINÉ")
    print("="*80)
    print("\nConsultez:")
    print("  - ghs_percentiles_grid.png (grille complète)")
    print("  - ghs_best_combinations.png (meilleures combinaisons)")
