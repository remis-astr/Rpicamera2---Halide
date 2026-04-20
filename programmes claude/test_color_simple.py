#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic SIMPLIFIÉ pour identifier l'inversion RGB/BGR
Utilise les captures existantes pour analyser le pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# Debug output directory
DEBUG_DIR = Path("/home/admin/Rpicamera/debug_output")
DEBUG_DIR.mkdir(exist_ok=True)

def analyze_image(image_data, stage_name):
    """Analyse une image et affiche ses caractéristiques"""

    print(f"\n{'='*60}")
    print(f"[{stage_name}]")
    print(f"{'='*60}")
    print(f"  Shape: {image_data.shape}")
    print(f"  DType: {image_data.dtype}")
    print(f"  Range: [{np.min(image_data):.2f}, {np.max(image_data):.2f}]")
    print(f"  Mean: {np.mean(image_data):.2f}")

    # Analyser les canaux si RGB
    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
        r_mean = np.mean(image_data[:,:,0])
        g_mean = np.mean(image_data[:,:,1])
        b_mean = np.mean(image_data[:,:,2])

        print(f"\n  Moyennes par canal:")
        print(f"    Canal 0: {r_mean:.2f}")
        print(f"    Canal 1: {g_mean:.2f}")
        print(f"    Canal 2: {b_mean:.2f}")

        # Déterminer la dominante
        channels = {'R': r_mean, 'G': g_mean, 'B': b_mean}
        dominant = max(channels, key=channels.get)

        print(f"\n  Canal dominant: {dominant}")

        if r_mean > b_mean * 1.3:
            print(f"  ✓ Dominante ROUGE (cerises rouges = OK)")
        elif b_mean > r_mean * 1.3:
            print(f"  ✗ Dominante BLEUE (cerises bleues = PROBLÈME)")
        else:
            print(f"  ? Balance neutre")

        # Ratio R/B
        rb_ratio = r_mean / b_mean if b_mean > 0 else 0
        print(f"  Ratio R/B: {rb_ratio:.2f}")


def test_opencv_behavior():
    """Test le comportement de OpenCV avec les couleurs"""

    print("\n" + "="*60)
    print("TEST: Comportement OpenCV - imread/imwrite")
    print("="*60)

    # Créer une image de test simple
    test_img = np.zeros((100, 300, 3), dtype=np.uint8)

    # 3 bandes de couleur: Rouge, Vert, Bleu
    test_img[:, 0:100, 0] = 255    # Rouge pur
    test_img[:, 100:200, 1] = 255  # Vert pur
    test_img[:, 200:300, 2] = 255  # Bleu pur

    print("\n[TEST IMAGE] Image RGB créée:")
    print(f"  Rouge: canal 0 = 255")
    print(f"  Vert: canal 1 = 255")
    print(f"  Bleu: canal 2 = 255")

    # Sauvegarder avec cv2.imwrite (qui attend BGR!)
    test_path_raw = DEBUG_DIR / "test_rgb_saved_as_is.png"
    cv2.imwrite(str(test_path_raw), test_img)
    print(f"\n[SAVE 1] Sauvegardé directement: {test_path_raw}")
    print(f"  Note: cv2.imwrite attend BGR, mais on lui donne RGB")

    # Sauvegarder après conversion RGB->BGR
    test_img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_path_converted = DEBUG_DIR / "test_rgb_converted_to_bgr.png"
    cv2.imwrite(str(test_path_converted), test_img_bgr)
    print(f"\n[SAVE 2] Sauvegardé après RGB->BGR: {test_path_converted}")
    print(f"  Note: conversion RGB->BGR avant cv2.imwrite")

    # Relire les deux
    print("\n[RELOAD] Relecture avec cv2.imread:")
    reloaded_raw = cv2.imread(str(test_path_raw))
    reloaded_converted = cv2.imread(str(test_path_converted))

    print(f"\n  Image 1 (raw) - canal 0 moyen: {np.mean(reloaded_raw[:,:,0]):.1f}")
    print(f"  Image 1 (raw) - canal 2 moyen: {np.mean(reloaded_raw[:,:,2]):.1f}")

    print(f"\n  Image 2 (converted) - canal 0 moyen: {np.mean(reloaded_converted[:,:,0]):.1f}")
    print(f"  Image 2 (converted) - canal 2 moyen: {np.mean(reloaded_converted[:,:,2]):.1f}")

    print("\n" + "="*60)
    print("CONCLUSION OpenCV:")
    print("  - cv2.imread() retourne BGR")
    print("  - cv2.imwrite() attend BGR")
    print("  - Si on donne RGB à imwrite, les couleurs sont inversées!")
    print("="*60)


def test_fits_behavior():
    """Test le comportement FITS"""

    try:
        from astropy.io import fits

        print("\n" + "="*60)
        print("TEST: Comportement FITS")
        print("="*60)

        # Créer image test RGB
        test_img = np.zeros((100, 300, 3), dtype=np.uint16)
        test_img[:, 0:100, 0] = 65535    # Rouge
        test_img[:, 100:200, 1] = 65535  # Vert
        test_img[:, 200:300, 2] = 65535  # Bleu

        print("\n[TEST IMAGE] Image RGB (H, W, 3):")
        print(f"  Shape: {test_img.shape}")
        print(f"  Rouge dans canal 0")
        print(f"  Vert dans canal 1")
        print(f"  Bleu dans canal 2")

        # Transposer pour FITS (3, H, W)
        test_fits = test_img.transpose(2, 0, 1)
        print(f"\n[TRANSPOSE] Pour FITS (3, H, W):")
        print(f"  Shape: {test_fits.shape}")

        # Sauvegarder
        fits_path = DEBUG_DIR / "test_fits_colors.fits"
        hdu = fits.PrimaryHDU(test_fits)
        hdu.header['COLORTYP'] = 'RGB'
        hdu.writeto(fits_path, overwrite=True)
        print(f"\n[SAVE] FITS sauvegardé: {fits_path}")

        # Relire
        with fits.open(fits_path) as hdul:
            reloaded = hdul[0].data
            print(f"\n[RELOAD] FITS relu:")
            print(f"  Shape: {reloaded.shape}")

            # Reconvertir
            if len(reloaded.shape) == 3 and reloaded.shape[0] == 3:
                reloaded_hwc = reloaded.transpose(1, 2, 0)
                print(f"  Reconverti en (H,W,3): {reloaded_hwc.shape}")

                # Vérifier les couleurs
                print(f"\n  Canal 0 moyen: {np.mean(reloaded_hwc[:,:,0]):.0f}")
                print(f"  Canal 1 moyen: {np.mean(reloaded_hwc[:,:,1]):.0f}")
                print(f"  Canal 2 moyen: {np.mean(reloaded_hwc[:,:,2]):.0f}")

                if np.mean(reloaded_hwc[:, 0:100, 0]) > 60000:
                    print(f"\n  ✓ Rouge préservé dans canal 0")
                else:
                    print(f"\n  ✗ Rouge perdu!")

        print("\n" + "="*60)

    except ImportError:
        print("\n[SKIP] Astropy non disponible")


def analyze_existing_stack():
    """Analyse un stack PNG existant s'il y en a"""

    print("\n" + "="*60)
    print("ANALYSE: Stack PNG existant")
    print("="*60)

    # Chercher des PNG récents dans le répertoire de sortie
    stack_dir = Path("/media/admin/THKAILAR/Stacks")

    if not stack_dir.exists():
        print(f"Répertoire {stack_dir} non trouvé")
        return

    png_files = sorted(stack_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not png_files:
        print("Aucun fichier PNG trouvé")
        return

    # Analyser le plus récent
    latest_png = png_files[0]
    print(f"\nFichier le plus récent: {latest_png.name}")

    # Lire avec OpenCV (retourne BGR)
    img_bgr = cv2.imread(str(latest_png))
    if img_bgr is None:
        print("Erreur lecture")
        return

    print(f"\n[CV2.IMREAD] Lu avec OpenCV (BGR):")
    analyze_image(img_bgr, "Stack PNG - format BGR d'OpenCV")

    # Convertir en RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"\n[CONVERT] Après BGR->RGB:")
    analyze_image(img_rgb, "Stack PNG - converti en RGB")


def main():
    """Fonction principale"""

    print("\n" + "="*60)
    print("DIAGNOSTIC COULEURS - Version simplifiée")
    print("="*60)
    print(f"Répertoire debug: {DEBUG_DIR}")

    # Test 1: Comportement OpenCV
    test_opencv_behavior()

    # Test 2: Comportement FITS
    test_fits_behavior()

    # Test 3: Analyser stack existant
    analyze_existing_stack()

    print("\n" + "="*60)
    print("DIAGNOSTIC TERMINÉ")
    print("="*60)
    print(f"\nFichiers de test créés dans: {DEBUG_DIR}")
    print("\nVérifiez visuellement:")
    print("  - test_rgb_saved_as_is.png (devrait être inversé)")
    print("  - test_rgb_converted_to_bgr.png (devrait être correct)")


if __name__ == "__main__":
    main()
