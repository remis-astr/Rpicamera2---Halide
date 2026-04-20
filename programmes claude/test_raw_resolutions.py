#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test des résolutions RAW pour l'IMX585
Vérifie quelles résolutions fonctionnent correctement en mode RAW12
"""

import sys
import os

# Ajouter le path pour importer RPiCamera2
sys.path.insert(0, '/home/admin/Rpicamera tests/Rpicamera2')

from picamera2 import Picamera2
import time
import numpy as np

# Résolutions à tester (correspondant aux modes zoom)
test_resolutions = {
    'Full Native': (3856, 2180),
    'Full Binned': (1928, 1090),
    'Zoom 1 (2.8K)': (2880, 2160),
    'Zoom 2 (FHD)': (1920, 1080),
    'Zoom 3 (HD)': (1280, 720),
    'Zoom 4 (SVGA)': (800, 600),
}

print("=" * 70)
print("TEST DES RÉSOLUTIONS RAW POUR IMX585")
print("=" * 70)
print()

results = []

for name, (width, height) in test_resolutions.items():
    print(f"\n{'='*70}")
    print(f"Test: {name} - {width}x{height}")
    print(f"{'='*70}")

    picam2 = None
    try:
        # Créer instance
        picam2 = Picamera2()

        # Configuration avec RAW12
        print(f"  Configuration RAW12...")
        config = picam2.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"},
            raw={"size": (width, height), "format": "SRGGB12", "unpacked": True}
        )

        print(f"  Config demandée:")
        print(f"    main: {config['main']}")
        print(f"    raw: {config['raw']}")

        # Configurer
        picam2.configure(config)

        # Vérifier config réelle
        actual_config = picam2.camera_configuration()
        print(f"  Config réelle:")
        print(f"    main: size={actual_config['main']['size']}, format={actual_config['main']['format']}")
        print(f"    raw: size={actual_config['raw']['size']}, format={actual_config['raw']['format']}")

        # Démarrer
        print(f"  Démarrage...")
        picam2.start()
        time.sleep(0.5)

        # Capturer RAW
        print(f"  Capture RAW...")
        raw_array = picam2.capture_array("raw")

        print(f"  ✅ SUCCÈS!")
        print(f"    Array shape: {raw_array.shape}")
        print(f"    Array dtype: {raw_array.dtype}")
        print(f"    Array range: [{raw_array.min()}, {raw_array.max()}]")

        # Vérifier que la résolution correspond
        expected_height = height
        expected_stride = width * 2  # 2 bytes per pixel for SRGGB12

        if raw_array.shape[0] == expected_height:
            print(f"    ✅ Hauteur correcte: {raw_array.shape[0]}")
        else:
            print(f"    ⚠️  Hauteur incorrecte: {raw_array.shape[0]} (attendu: {expected_height})")

        results.append((name, width, height, "✅ OK", raw_array.shape))

    except Exception as e:
        print(f"  ❌ ÉCHEC: {e}")
        results.append((name, width, height, f"❌ ERREUR: {str(e)[:50]}", None))

    finally:
        if picam2:
            try:
                picam2.stop()
                picam2.close()
            except:
                pass
        time.sleep(1)

# Résumé
print(f"\n\n{'='*70}")
print("RÉSUMÉ DES TESTS")
print(f"{'='*70}\n")

print(f"{'Mode':<20} {'Résolution':<15} {'Statut':<20} {'Shape réel':<20}")
print("-" * 70)

for name, w, h, status, shape in results:
    shape_str = str(shape) if shape else "N/A"
    print(f"{name:<20} {w}x{h:<12} {status:<20} {shape_str:<20}")

print()
print("=" * 70)
print("FIN DES TESTS")
print("=" * 70)
