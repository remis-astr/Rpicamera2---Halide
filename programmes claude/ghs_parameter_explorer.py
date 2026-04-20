#!/usr/bin/env python3
"""
Explorateur de paramètres GHS - Génère une grille de tests
Permet de trouver les meilleurs paramètres visuellement
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans GUI
import matplotlib.pyplot as plt
from pathlib import Path

from libastrostack.io import load_image
from libastrostack.stretch import stretch_asinh


def ghs_stretch_universal(data, D, b, SP, LP=0.0, HP=1.0):
    """GHS 5 paramètres"""
    epsilon = 1e-10

    # Normaliser en float (0-1)
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
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = 1.0 - np.power(base, -1.0 / b)

        return result

    def T_prime(x, D, b):
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


def explore_D_parameter(data, output_dir="ghs_exploration"):
    """Explorer différentes valeurs de D"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    D_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Exploration paramètre D (b=6, SP=0.15, LP=0, HP=0.85)', fontsize=14, fontweight='bold')

    for idx, D in enumerate(D_values):
        row = idx // 5
        col = idx % 5

        result = ghs_stretch_universal(data, D=D, b=6.0, SP=0.15, LP=0.0, HP=0.85)

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'D={D:.1f}\nMean: {result.mean():.3f}', fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "explore_D.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


def explore_b_parameter(data, output_dir="ghs_exploration"):
    """Explorer différentes valeurs de b"""
    output_path = Path(output_dir)

    b_values = [0, 2, 4, 6, 8, 10, 12, 14]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Exploration paramètre b (D=2, SP=0.15, LP=0, HP=0.85)', fontsize=14, fontweight='bold')

    for idx, b in enumerate(b_values):
        row = idx // 4
        col = idx % 4

        result = ghs_stretch_universal(data, D=2.0, b=b, SP=0.15, LP=0.0, HP=0.85)

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'b={b}\nMean: {result.mean():.3f}', fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "explore_b.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


def explore_SP_parameter(data, output_dir="ghs_exploration"):
    """Explorer différentes valeurs de SP"""
    output_path = Path(output_dir)

    SP_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Exploration paramètre SP (D=2, b=6, LP=0, HP=0.85)', fontsize=14, fontweight='bold')

    for idx, SP in enumerate(SP_values):
        row = idx // 4
        col = idx % 4

        result = ghs_stretch_universal(data, D=2.0, b=6.0, SP=SP, LP=0.0, HP=0.85)

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'SP={SP:.2f}\nMean: {result.mean():.3f}', fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "explore_SP.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


def explore_HP_parameter(data, output_dir="ghs_exploration"):
    """Explorer différentes valeurs de HP"""
    output_path = Path(output_dir)

    HP_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Exploration paramètre HP (D=2, b=6, SP=0.15, LP=0)', fontsize=14, fontweight='bold')

    for idx, HP in enumerate(HP_values):
        row = idx // 4
        col = idx % 4

        result = ghs_stretch_universal(data, D=2.0, b=6.0, SP=0.15, LP=0.0, HP=HP)

        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'HP={HP:.2f}\nMean: {result.mean():.3f}', fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "explore_HP.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


def compare_with_arcsinh(data, output_dir="ghs_exploration"):
    """Comparer GHS avec ARCSINH"""
    output_path = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Comparaison: GHS vs ARCSINH', fontsize=14, fontweight='bold')

    # ARCSINH
    result_arcsinh = stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5)
    axes[0].imshow(result_arcsinh, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'ARCSINH (factor=10)\nMean: {result_arcsinh.mean():.3f}', fontsize=12)
    axes[0].axis('off')

    # GHS preset Galaxies
    result_ghs = ghs_stretch_universal(data, D=2.0, b=6.0, SP=0.15, LP=0.0, HP=0.85)
    axes[1].imshow(result_ghs, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'GHS Galaxies (D=2, b=6, SP=0.15)\nMean: {result_ghs.mean():.3f}', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    output_file = output_path / "compare_arcsinh_vs_ghs.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("EXPLORATEUR DE PARAMÈTRES GHS")
    print("="*80)
    print(f"Image: {image_path}\n")

    # Charger l'image
    data = load_image(image_path)
    if data is None:
        print("Erreur de chargement")
        sys.exit(1)

    # Convertir en mono
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    # Normaliser
    if data.max() > 1.0:
        data = data / data.max()

    print(f"Données: shape={data.shape}, range=[{data.min():.6f}, {data.max():.6f}]\n")

    # Explorer chaque paramètre
    print("Exploration du paramètre D...")
    explore_D_parameter(data)

    print("Exploration du paramètre b...")
    explore_b_parameter(data)

    print("Exploration du paramètre SP...")
    explore_SP_parameter(data)

    print("Exploration du paramètre HP...")
    explore_HP_parameter(data)

    print("Comparaison avec ARCSINH...")
    compare_with_arcsinh(data)

    print("\n" + "="*80)
    print("✓ EXPLORATION TERMINÉE")
    print("="*80)
    print("\nFichiers dans ghs_exploration/:")
    print("  - explore_D.png (variation de D)")
    print("  - explore_b.png (variation de b)")
    print("  - explore_SP.png (variation de SP)")
    print("  - explore_HP.png (variation de HP)")
    print("  - compare_arcsinh_vs_ghs.png (comparaison)")
    print("\nAnalysez ces images pour trouver les meilleurs paramètres!")
