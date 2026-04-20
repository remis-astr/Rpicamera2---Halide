#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic pour identifier l'inversion RGB/BGR (cerises bleues)
Capture et sauvegarde les données à différentes étapes du pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from picamera2 import Picamera2
import time

# Debug output directory
DEBUG_DIR = Path("/home/admin/Rpicamera/debug_output")
DEBUG_DIR.mkdir(exist_ok=True)

def save_debug_image(data, stage_name, timestamp):
    """
    Sauvegarde une image de debug avec informations sur le format

    Args:
        data: Image numpy array
        stage_name: Nom de l'étape (ex: "01_camera_capture")
        timestamp: Timestamp pour le nom du fichier
    """
    # Informations sur l'image
    info = {
        'shape': data.shape,
        'dtype': data.dtype,
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data)
    }

    print(f"\n[{stage_name}]")
    print(f"  Shape: {info['shape']}")
    print(f"  DType: {info['dtype']}")
    print(f"  Range: [{info['min']:.2f}, {info['max']:.2f}]")
    print(f"  Mean: {info['mean']:.2f}")

    # Analyser les canaux si RGB
    if len(data.shape) == 3 and data.shape[2] == 3:
        r_mean = np.mean(data[:,:,0])
        g_mean = np.mean(data[:,:,1])
        b_mean = np.mean(data[:,:,2])
        print(f"  Channel means: R={r_mean:.2f}, G={g_mean:.2f}, B={b_mean:.2f}")

        # Vérifier si cerises rouges
        if r_mean > b_mean * 1.5:
            print(f"  ✓ Dominante ROUGE (OK pour cerises rouges)")
        elif b_mean > r_mean * 1.5:
            print(f"  ✗ Dominante BLEUE (PROBLÈME - inversion!)")
        else:
            print(f"  ? Dominante neutre")

    # Normaliser pour sauvegarde PNG
    if data.dtype == np.float32 or data.dtype == np.float64:
        # Float: normaliser 0-1 puis convertir uint8
        data_normalized = np.clip(data / np.max(data), 0, 1)
        data_8bit = (data_normalized * 255).astype(np.uint8)
    elif data.dtype == np.uint16:
        # uint16: convertir en uint8
        data_8bit = (data / 256).astype(np.uint8)
    else:
        # Déjà uint8
        data_8bit = data

    # Sauvegarder 3 versions
    base_path = DEBUG_DIR / f"{timestamp}_{stage_name}"

    # 1. Version brute (pas de conversion BGR)
    cv2.imwrite(str(base_path) + "_raw.png", data_8bit)
    print(f"  Saved: {base_path}_raw.png")

    # 2. Version avec conversion BGR->RGB (ce qu'OpenCV fait normalement)
    if len(data_8bit.shape) == 3:
        data_rgb = cv2.cvtColor(data_8bit, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(base_path) + "_bgr2rgb.png", data_rgb)
        print(f"  Saved: {base_path}_bgr2rgb.png")

        # 3. Version avec inversion manuelle des canaux
        data_inverted = data_8bit[:,:,::-1].copy()
        cv2.imwrite(str(base_path) + "_inverted.png", data_inverted)
        print(f"  Saved: {base_path}_inverted.png")

    # Sauvegarder les métadonnées
    info_path = base_path.with_suffix('.txt')
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    return info


def test_color_pipeline():
    """
    Teste le pipeline complet de traitement des couleurs
    """
    print("="*60)
    print("TEST DIAGNOSTIC - Pipeline couleurs")
    print("="*60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialiser caméra
    print("\n[INIT] Initialisation caméra...")
    picam2 = Picamera2()

    # Configuration pour capture RAW
    config = picam2.create_still_configuration(
        main={"size": (4608, 2592), "format": "RGB888"},
        raw={"size": (4608, 2592)},
    )
    picam2.configure(config)

    # Réglages manuels (cerises rouges)
    picam2.set_controls({
        "ExposureTime": 100000,  # 100ms
        "AnalogueGain": 8.0,
        "ColourGains": (1.5, 1.2)  # Rouge, Bleu
    })

    picam2.start()
    time.sleep(2)  # Temps de stabilisation

    print("\n[CAPTURE] Capture d'une frame...")

    # === ÉTAPE 1: Capture caméra brute ===
    # Obtenir l'array numpy directement de la caméra
    camera_array = picam2.capture_array("main")

    print(f"\nArray caméra reçu:")
    print(f"  Type: {type(camera_array)}")
    print(f"  Shape: {camera_array.shape}")
    print(f"  DType: {camera_array.dtype}")

    save_debug_image(camera_array, "01_camera_capture", timestamp)

    # === ÉTAPE 2: Conversion en float32 (comme dans le code) ===
    if camera_array.dtype == np.uint8:
        image_float = camera_array.astype(np.float32) * 256.0
    else:
        image_float = camera_array.astype(np.float32)

    save_debug_image(image_float, "02_after_float_conversion", timestamp)

    # === ÉTAPE 3: Simulation stretch (normalisation) ===
    # Simuler un stretch simple
    vmin = np.percentile(image_float, 1.0)
    vmax = np.percentile(image_float, 99.0)
    stretched = np.clip((image_float - vmin) / (vmax - vmin), 0, 1)

    save_debug_image(stretched, "03_after_stretch", timestamp)

    # === ÉTAPE 4: Conversion uint8 pour PNG ===
    png_data = (stretched * 255).astype(np.uint8)

    save_debug_image(png_data, "04_ready_for_png", timestamp)

    # === ÉTAPE 5: Sauvegarde PNG avec OpenCV (comme dans le code) ===
    # Simuler ce que fait le code dans session.py
    print("\n[PNG SAVE] Simulation sauvegarde PNG...")

    # Version 1: Sauvegarde directe (comme actuellement)
    png_path_direct = DEBUG_DIR / f"{timestamp}_05_png_direct.png"
    cv2.imwrite(str(png_path_direct), png_data)
    print(f"  Direct: {png_path_direct}")

    # Version 2: Avec conversion BGR2RGB avant sauvegarde
    png_path_converted = DEBUG_DIR / f"{timestamp}_05_png_bgr2rgb.png"
    if len(png_data.shape) == 3:
        png_rgb = cv2.cvtColor(png_data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(png_path_converted), png_rgb)
        print(f"  BGR2RGB: {png_path_converted}")

    # === ÉTAPE 6: Test FITS ===
    print("\n[FITS SAVE] Test sauvegarde FITS...")

    try:
        from astropy.io import fits

        # Préparation données FITS (comme dans io.py)
        stretched_fits = np.zeros_like(stretched)
        for i in range(3):
            vmin = np.percentile(stretched[:,:,i], 1)
            vmax = np.percentile(stretched[:,:,i], 99)
            stretched_fits[:,:,i] = np.clip((stretched[:,:,i] - vmin) / (vmax - vmin), 0, 1)

        result_16bit = (stretched_fits * 65535).astype(np.uint16)

        # Analyse avant transposition
        save_debug_image(result_16bit, "06a_fits_before_transpose", timestamp)

        # Transposer pour FITS (3, H, W)
        result_transposed = result_16bit.transpose(2, 0, 1)
        print(f"\nFITS transposé: shape {result_transposed.shape}")

        # Sauvegarder FITS
        fits_path = DEBUG_DIR / f"{timestamp}_06b_output.fits"
        hdu = fits.PrimaryHDU(result_transposed)
        hdu.header['COLORTYP'] = 'RGB'
        hdu.header['NOTE'] = 'Test diagnostic couleurs'
        hdu.writeto(fits_path, overwrite=True)
        print(f"  FITS saved: {fits_path}")

        # Relire le FITS pour vérifier
        with fits.open(fits_path) as hdul:
            fits_data = hdul[0].data
            print(f"  FITS relu: shape {fits_data.shape}")

            # Reconvertir (3, H, W) -> (H, W, 3)
            if len(fits_data.shape) == 3 and fits_data.shape[0] == 3:
                fits_data = fits_data.transpose(1, 2, 0)

            save_debug_image(fits_data, "06c_fits_reloaded", timestamp)

    except ImportError:
        print("  Astropy non disponible, skip FITS test")

    # Cleanup
    picam2.stop()
    picam2.close()

    print("\n" + "="*60)
    print("DIAGNOSTIC TERMINÉ")
    print("="*60)
    print(f"\nFichiers sauvegardés dans: {DEBUG_DIR}")
    print("\nCOMPAREZ les fichiers:")
    print("  - *_raw.png: données brutes sans conversion")
    print("  - *_bgr2rgb.png: avec conversion BGR->RGB")
    print("  - *_inverted.png: canaux inversés manuellement")
    print("\nPour identifier le problème, regardez quelle version")
    print("montre les CERISES ROUGES correctement!")


if __name__ == "__main__":
    test_color_pipeline()
