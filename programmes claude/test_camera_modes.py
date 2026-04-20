#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script 1 : Lister tous les modes disponibles de la caméra
============================================================

Ce script affiche :
- Les configurations disponibles (preview, video, still)
- Les formats supportés (RGB, YUV, RAW Bayer)
- Les résolutions disponibles
- Les limites de frame rate
- Les informations sur le capteur

Usage:
    python3 test_camera_modes.py
"""

import sys
from picamera2 import Picamera2
import time

def print_separator(title):
    """Affiche un séparateur avec titre"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    print("\n🎥 ANALYSE DES MODES CAMÉRA DISPONIBLES")
    print("="*70)

    try:
        # Initialiser la caméra
        print("\n📷 Initialisation de la caméra...")
        picam2 = Picamera2()

        # Informations sur le capteur
        print_separator("INFORMATIONS CAPTEUR")
        camera_properties = picam2.camera_properties

        print(f"Modèle : {camera_properties.get('Model', 'N/A')}")
        print(f"Pixel Array Size : {camera_properties.get('PixelArraySize', 'N/A')}")
        print(f"Unit Cell Size : {camera_properties.get('UnitCellSize', 'N/A')}")

        if 'ColorFilterArrangement' in camera_properties:
            cfa = camera_properties['ColorFilterArrangement']
            bayer_patterns = {0: 'RGGB', 1: 'GRBG', 2: 'GBRG', 3: 'BGGR', 4: 'MONO'}
            print(f"Matrice Bayer : {bayer_patterns.get(cfa, f'Unknown ({cfa})')}")

        # Lister tous les modes du capteur
        print_separator("MODES CAPTEUR NATIFS")
        sensor_modes = picam2.sensor_modes

        for i, mode in enumerate(sensor_modes):
            print(f"\n🔹 Mode {i}:")
            for key, value in mode.items():
                if key == 'format':
                    # Décoder le format Bayer
                    print(f"   {key:20s} : {value}")
                else:
                    print(f"   {key:20s} : {value}")

        # Tester les configurations disponibles
        print_separator("CONFIGURATIONS DISPONIBLES")

        # 1. Preview Configuration
        print("\n1️⃣  PREVIEW CONFIGURATION")
        print("-" * 70)
        try:
            preview_config = picam2.create_preview_configuration()
            print("✅ Preview config disponible")
            print(f"   Main stream : {preview_config['main']}")
            if 'lores' in preview_config:
                print(f"   Lores stream : {preview_config['lores']}")
            if 'raw' in preview_config:
                print(f"   Raw stream : {preview_config['raw']}")
        except Exception as e:
            print(f"❌ Erreur : {e}")

        # 2. Video Configuration
        print("\n2️⃣  VIDEO CONFIGURATION")
        print("-" * 70)
        try:
            video_config = picam2.create_video_configuration()
            print("✅ Video config disponible")
            print(f"   Main stream : {video_config['main']}")
            if 'lores' in video_config:
                print(f"   Lores stream : {video_config['lores']}")
            if 'raw' in video_config:
                print(f"   Raw stream : {video_config['raw']}")
        except Exception as e:
            print(f"❌ Erreur : {e}")

        # 3. Still Configuration
        print("\n3️⃣  STILL CONFIGURATION")
        print("-" * 70)
        try:
            still_config = picam2.create_still_configuration()
            print("✅ Still config disponible")
            print(f"   Main stream : {still_config['main']}")
            if 'lores' in still_config:
                print(f"   Lores stream : {still_config['lores']}")
            if 'raw' in still_config:
                print(f"   Raw stream : {still_config['raw']}")
        except Exception as e:
            print(f"❌ Erreur : {e}")

        # Tester les formats disponibles
        print_separator("FORMATS DISPONIBLES POUR MAIN STREAM")

        formats_to_test = [
            ("RGB888", "RGB 8-bit standard"),
            ("RGB161616", "RGB 16-bit (si supporté)"),
            ("YUV420", "YUV420 (non compressé)"),
            ("XRGB8888", "XRGB 8-bit avec alpha"),
            ("BGR888", "BGR 8-bit"),
            ("SRGGB10", "RAW Bayer 10-bit packed"),
            ("SRGGB10_CSI2P", "RAW Bayer 10-bit CSI2 packed"),
            ("SRGGB12", "RAW Bayer 12-bit"),
            ("SRGGB16", "RAW Bayer 16-bit unpacked"),
            ("SGRBG10", "RAW Bayer GRBG 10-bit"),
            ("SGBRG10", "RAW Bayer GBRG 10-bit"),
            ("SBGGR10", "RAW Bayer BGGR 10-bit"),
        ]

        available_formats = []

        for format_name, description in formats_to_test:
            try:
                test_config = picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": format_name}
                )
                print(f"✅ {format_name:20s} - {description}")
                available_formats.append(format_name)
            except Exception as e:
                print(f"❌ {format_name:20s} - Non supporté ({str(e)[:50]}...)")

        # Résumé
        print_separator("RÉSUMÉ")
        print(f"\n✅ Formats disponibles : {len(available_formats)}/{len(formats_to_test)}")
        print(f"   Liste : {', '.join(available_formats)}")

        # Tester capture RAW
        print_separator("TEST CAPTURE RAW")

        print("\n🔬 Test de capture du flux RAW...")
        try:
            # Configuration avec RAW
            raw_config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                raw={"size": picam2.sensor_resolution}
            )

            picam2.configure(raw_config)
            picam2.start()
            time.sleep(1)  # Laisser la caméra se stabiliser

            # Tester la capture
            print("   Capture du flux 'main' (RGB)...")
            main_array = picam2.capture_array("main")
            print(f"   ✅ Main array : shape={main_array.shape}, dtype={main_array.dtype}")

            print("   Capture du flux 'raw' (Bayer)...")
            raw_array = picam2.capture_array("raw")
            print(f"   ✅ Raw array : shape={raw_array.shape}, dtype={raw_array.dtype}")

            # Analyser le flux RAW
            print(f"\n   📊 Analyse du flux RAW :")
            print(f"      Résolution : {raw_array.shape[1]}×{raw_array.shape[0]}")
            print(f"      Profondeur : {raw_array.dtype}")
            print(f"      Taille mémoire : {raw_array.nbytes / 1024 / 1024:.2f} MB")
            print(f"      Min/Max : {raw_array.min()} / {raw_array.max()}")

            picam2.stop()

        except Exception as e:
            print(f"   ❌ Erreur capture RAW : {e}")
            import traceback
            traceback.print_exc()

        # Fermer la caméra
        print_separator("FIN DES TESTS")
        print("\n✅ Tests terminés avec succès !")
        print("\n💡 Prochaines étapes :")
        print("   1. Exécuter test_dng_capture.py pour tester la capture DNG")
        print("   2. Exécuter test_raw_video.py pour tester l'enregistrement vidéo RAW")

        picam2.close()

    except Exception as e:
        print(f"\n❌ ERREUR FATALE : {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
