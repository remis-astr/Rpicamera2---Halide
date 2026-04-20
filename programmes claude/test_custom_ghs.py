#!/usr/bin/env python3
"""
Test des paramètres GHS personnalisés trouvés via les sliders
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libastrostack.io import load_image
from libastrostack.stretch import stretch_asinh


def ghs_stretch_universal(data, D, b, SP, LP=0.0, HP=1.0):
    """GHS 5 paramètres"""
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


if __name__ == "__main__":
    image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("TEST PARAMÈTRES GHS PERSONNALISÉS")
    print("="*80)

    # Charger
    data = load_image(image_path)
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)
    if data.max() > 1.0:
        data = data / data.max()

    # Paramètres trouvés par l'utilisateur
    D_custom = 3.06
    b_custom = 0.131
    SP_custom = 0.194
    LP_custom = 0.0
    HP_custom = 0.0

    print(f"\nParamètres personnalisés:")
    print(f"  D  = {D_custom}")
    print(f"  b  = {b_custom}")
    print(f"  SP = {SP_custom}")
    print(f"  LP = {LP_custom}")
    print(f"  HP = {HP_custom}")

    # Tests
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparaison: GHS personnalisé vs Presets vs ARCSINH', fontsize=14, fontweight='bold')

    # 1. Original
    axes[0, 0].imshow(data, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Original\nMean: {data.mean():.3f}', fontsize=12)
    axes[0, 0].axis('off')

    # 2. GHS personnalisé
    result_custom = ghs_stretch_universal(data, D_custom, b_custom, SP_custom, LP_custom, HP_custom)
    axes[0, 1].imshow(result_custom, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'GHS PERSONNALISÉ\n(D={D_custom}, b={b_custom}, SP={SP_custom})\nMean: {result_custom.mean():.3f}',
                        fontsize=12, fontweight='bold', color='green')
    axes[0, 1].axis('off')

    # 3. ARCSINH
    result_arcsinh = stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5)
    axes[0, 2].imshow(result_arcsinh, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'ARCSINH\n(factor=10)\nMean: {result_arcsinh.mean():.3f}',
                        fontsize=12, fontweight='bold', color='blue')
    axes[0, 2].axis('off')

    # 4. GHS preset Galaxies
    result_galaxies = ghs_stretch_universal(data, 2.0, 6.0, 0.15, 0.0, 0.85)
    axes[1, 0].imshow(result_galaxies, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'GHS Preset Galaxies\n(D=2, b=6, SP=0.15, HP=0.85)\nMean: {result_galaxies.mean():.3f}',
                        fontsize=10)
    axes[1, 0].axis('off')

    # 5. Histogrammes
    axes[1, 1].hist([data.flatten(), result_custom.flatten(), result_arcsinh.flatten()],
                    bins=100, alpha=0.6, label=['Original', 'GHS custom', 'ARCSINH'],
                    color=['gray', 'green', 'blue'])
    axes[1, 1].set_xlabel('Valeur pixel')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].set_title('Histogrammes comparés', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Stats
    stats_text = f"""
STATISTIQUES COMPARÉES

Original:
  Mean: {data.mean():.6f}
  >0.5: {(data > 0.5).sum() / data.size * 100:.2f}%

GHS PERSONNALISÉ:
  Mean: {result_custom.mean():.6f}
  >0.5: {(result_custom > 0.5).sum() / result_custom.size * 100:.2f}%
  Range: [{result_custom.min():.3f}, {result_custom.max():.3f}]

ARCSINH:
  Mean: {result_arcsinh.mean():.6f}
  >0.5: {(result_arcsinh > 0.5).sum() / result_arcsinh.size * 100:.2f}%
  Range: [{result_arcsinh.min():.3f}, {result_arcsinh.max():.3f}]

GHS Galaxies (défaut):
  Mean: {result_galaxies.mean():.6f}
  >0.5: {(result_galaxies > 0.5).sum() / result_galaxies.size * 100:.2f}%
"""

    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('ghs_custom_vs_arcsinh.png', dpi=150)
    print("\n✓ Sauvegardé: ghs_custom_vs_arcsinh.png")

    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"GHS PERSONNALISÉ: Mean = {result_custom.mean():.6f}")
    print(f"ARCSINH:          Mean = {result_arcsinh.mean():.6f}")
    print(f"Différence:       {abs(result_custom.mean() - result_arcsinh.mean()):.6f}")

    if result_custom.mean() > 0.4:
        print("\n✅ SUCCÈS! GHS personnalisé donne de bons résultats!")
    else:
        print("\n⚠️  GHS personnalisé encore un peu sombre")
