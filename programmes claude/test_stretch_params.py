#!/usr/bin/env python3
"""
Script de test pour évaluer les paramètres de stretch ARCSINH et GHS
sur des images astronomiques DNG
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importer les fonctions de libastrostack
from libastrostack.io import load_image
from libastrostack.stretch import stretch_asinh, stretch_ghs

def test_arcsinh_params(image_path, output_dir="stretch_test_results"):
    """
    Teste différents paramètres ARCSINH sur une image DNG
    """
    print(f"Chargement de l'image: {image_path}")
    data = load_image(image_path)

    if data is None:
        print("Erreur de chargement de l'image")
        return

    print(f"Image chargée: shape={data.shape}, dtype={data.dtype}")
    print(f"Plage des valeurs: min={data.min():.1f}, max={data.max():.1f}")

    # Normaliser si nécessaire
    if data.max() > 1.0:
        print("Normalisation des données...")
        data = data / data.max()

    # Convertir en niveaux de gris si couleur (pour simplifier)
    if len(data.shape) == 3:
        print("Conversion en niveaux de gris pour les tests...")
        data = np.mean(data, axis=2)

    # Créer répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Test de différents facteurs ARCSINH
    factors = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison des facteurs ARCSINH', fontsize=16)

    for idx, factor in enumerate(factors):
        row = idx // 3
        col = idx % 3

        print(f"Test ARCSINH factor={factor}")
        stretched = stretch_asinh(data, factor=factor, clip_low=1.0, clip_high=99.5)

        axes[row, col].imshow(stretched, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'factor={factor}')
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "arcsinh_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()

    # Test de différents percentiles de clipping
    clips = [
        (0.1, 99.9),
        (0.5, 99.5),
        (1.0, 99.5),  # défaut actuel
        (1.0, 99.0),
        (2.0, 98.0),
        (5.0, 95.0)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison des percentiles de clipping ARCSINH (factor=10)', fontsize=16)

    for idx, (clip_low, clip_high) in enumerate(clips):
        row = idx // 3
        col = idx % 3

        print(f"Test ARCSINH clip_low={clip_low}, clip_high={clip_high}")
        stretched = stretch_asinh(data, factor=10.0, clip_low=clip_low, clip_high=clip_high)

        axes[row, col].imshow(stretched, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'clip=({clip_low}, {clip_high})')
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "arcsinh_clipping_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


def test_ghs_params(image_path, output_dir="stretch_test_results"):
    """
    Teste différents paramètres GHS sur une image DNG
    """
    print(f"\nChargement de l'image pour GHS: {image_path}")
    data = load_image(image_path)

    if data is None:
        print("Erreur de chargement de l'image")
        return

    # Normaliser si nécessaire
    if data.max() > 1.0:
        data = data / data.max()

    # Convertir en niveaux de gris si couleur
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    # Créer répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Test de différents paramètres D (linked stretching)
    D_values = [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison du paramètre D (GHS) - B=0, SP=0.5', fontsize=16)

    for idx, D in enumerate(D_values):
        row = idx // 3
        col = idx % 3

        print(f"Test GHS D={D}")
        stretched = stretch_ghs(data, D=D, B=0.0, SP=0.5)

        axes[row, col].imshow(stretched, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'D={D}')
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_D_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()

    # Test de différents paramètres B (blackpoint)
    B_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison du paramètre B (GHS) - D=0, SP=0.5', fontsize=16)

    for idx, B in enumerate(B_values):
        row = idx // 3
        col = idx % 3

        print(f"Test GHS B={B}")
        stretched = stretch_ghs(data, D=0.0, B=B, SP=0.5)

        axes[row, col].imshow(stretched, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'B={B}')
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_B_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()

    # Test de différents paramètres SP (symmetry point)
    SP_values = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison du paramètre SP (GHS) - D=0, B=0', fontsize=16)

    for idx, SP in enumerate(SP_values):
        row = idx // 3
        col = idx % 3

        print(f"Test GHS SP={SP}")
        stretched = stretch_ghs(data, D=0.0, B=0.0, SP=SP)

        axes[row, col].imshow(stretched, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'SP={SP}')
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_path / "ghs_SP_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


def compare_arcsinh_vs_ghs(image_path, output_dir="stretch_test_results"):
    """
    Compare ARCSINH et GHS avec leurs paramètres par défaut
    """
    print(f"\nComparaison ARCSINH vs GHS")
    data = load_image(image_path)

    if data is None:
        return

    # Normaliser si nécessaire
    if data.max() > 1.0:
        data = data / data.max()

    # Convertir en niveaux de gris si couleur
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Comparer les méthodes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparaison ARCSINH vs GHS (paramètres par défaut)', fontsize=16)

    # Image originale (avec clip simple)
    axes[0].imshow(np.clip(data, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original (clipé)')
    axes[0].axis('off')

    # ARCSINH avec paramètres par défaut
    print("Test ARCSINH (défaut: factor=10, clip=(1, 99.5))")
    stretched_asinh = stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5)
    axes[1].imshow(stretched_asinh, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('ARCSINH (factor=10)')
    axes[1].axis('off')

    # GHS avec paramètres par défaut
    print("Test GHS (défaut: D=0, B=0, SP=0.5)")
    stretched_ghs = stretch_ghs(data, D=0.0, B=0.0, SP=0.5)
    axes[2].imshow(stretched_ghs, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('GHS (D=0, B=0, SP=0.5)')
    axes[2].axis('off')

    plt.tight_layout()
    output_file = output_path / "arcsinh_vs_ghs_default.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()

    # Afficher les histogrammes
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle('Histogrammes des différentes méthodes', fontsize=16)

    axes[0].hist(data.flatten(), bins=100, color='blue', alpha=0.7)
    axes[0].set_title('Original')
    axes[0].set_xlabel('Valeur de pixel')
    axes[0].set_ylabel('Fréquence')

    axes[1].hist(stretched_asinh.flatten(), bins=100, color='green', alpha=0.7)
    axes[1].set_title('ARCSINH')
    axes[1].set_xlabel('Valeur de pixel')
    axes[1].set_ylabel('Fréquence')

    axes[2].hist(stretched_ghs.flatten(), bins=100, color='red', alpha=0.7)
    axes[2].set_title('GHS')
    axes[2].set_xlabel('Valeur de pixel')
    axes[2].set_ylabel('Fréquence')

    plt.tight_layout()
    output_file = output_path / "histograms_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé: {output_file}")
    plt.close()


if __name__ == "__main__":
    # Utiliser une image de test
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*70)
    print("TEST DES PARAMÈTRES DE STRETCH")
    print("="*70)
    print(f"Image de test: {test_image}")
    print()

    # Tester ARCSINH
    print("\n" + "="*70)
    print("TEST ARCSINH")
    print("="*70)
    test_arcsinh_params(test_image)

    # Tester GHS
    print("\n" + "="*70)
    print("TEST GHS")
    print("="*70)
    test_ghs_params(test_image)

    # Comparer les deux méthodes
    print("\n" + "="*70)
    print("COMPARAISON ARCSINH vs GHS")
    print("="*70)
    compare_arcsinh_vs_ghs(test_image)

    print("\n" + "="*70)
    print("✓ TESTS TERMINÉS")
    print("="*70)
    print(f"Les résultats sont dans le répertoire: stretch_test_results/")
    print()
    print("Fichiers générés:")
    print("  - arcsinh_comparison.png (différents facteurs)")
    print("  - arcsinh_clipping_comparison.png (différents percentiles)")
    print("  - ghs_D_comparison.png (paramètre D)")
    print("  - ghs_B_comparison.png (paramètre B)")
    print("  - ghs_SP_comparison.png (paramètre SP)")
    print("  - arcsinh_vs_ghs_default.png (comparaison directe)")
    print("  - histograms_comparison.png (histogrammes)")
