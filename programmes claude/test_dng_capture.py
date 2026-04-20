#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script 2 : Test de capture DNG pour le stacking
====================================================

Ce script teste :
- Capture de séquence DNG (RAW natif)
- Capture de séquence RAW array
- Performance (fps, latence)
- Comparaison DNG vs RAW array pour le stacking

Les fichiers de test sont sauvegardés dans ./test_output/

Usage:
    python3 test_dng_capture.py [nombre_frames]

    Exemple : python3 test_dng_capture.py 50
"""

import sys
import os
import time
import numpy as np
from picamera2 import Picamera2
from pathlib import Path
import cv2

# Créer le répertoire de sortie
OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def print_separator(title):
    """Affiche un séparateur avec titre"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_dng_capture(num_frames=50):
    """
    Test de capture DNG (format RAW natif)

    Args:
        num_frames: Nombre de frames à capturer
    """
    print_separator("TEST 1 : CAPTURE DNG (RAW NATIF)")

    print(f"\n📸 Capture de {num_frames} images DNG...")

    try:
        picam2 = Picamera2()

        # Configuration pour capture DNG
        # DNG nécessite still_configuration avec raw stream
        config = picam2.create_still_configuration(
            main={"size": picam2.sensor_resolution},
            raw={"size": picam2.sensor_resolution}
        )

        print(f"   Configuration : {config}")
        picam2.configure(config)
        picam2.start()

        # Warm-up
        time.sleep(1)

        # Capturer la séquence
        dng_dir = OUTPUT_DIR / "dng_sequence"
        dng_dir.mkdir(exist_ok=True)

        print(f"   📁 Sauvegarde dans : {dng_dir}")
        print(f"   ⏱️  Début capture...")

        start_time = time.time()
        frame_times = []
        file_sizes = []

        for i in range(num_frames):
            frame_start = time.time()

            # Capturer en DNG
            dng_path = dng_dir / f"frame_{i:04d}.dng"

            # Méthode 1 : capture_file (capture directe en DNG)
            metadata = picam2.capture_file(str(dng_path))

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # Vérifier la taille du fichier
            if dng_path.exists():
                file_sizes.append(dng_path.stat().st_size)

            # Afficher la progression
            if (i + 1) % 10 == 0:
                avg_fps = (i + 1) / (time.time() - start_time)
                print(f"      {i+1}/{num_frames} frames - {avg_fps:.1f} fps")

        total_time = time.time() - start_time

        # Statistiques
        print(f"\n   ✅ Capture DNG terminée !")
        print(f"   📊 Statistiques :")
        print(f"      Frames capturées : {num_frames}")
        print(f"      Temps total : {total_time:.2f}s")
        print(f"      FPS moyen : {num_frames / total_time:.2f}")
        print(f"      Latence par frame : {np.mean(frame_times)*1000:.1f}ms (±{np.std(frame_times)*1000:.1f}ms)")

        if file_sizes:
            print(f"      Taille moyenne : {np.mean(file_sizes)/1024/1024:.2f} MB/frame")
            print(f"      Taille totale : {sum(file_sizes)/1024/1024:.2f} MB")

        picam2.stop()
        picam2.close()

        return True, {
            'fps': num_frames / total_time,
            'latency_ms': np.mean(frame_times) * 1000,
            'size_mb': np.mean(file_sizes) / 1024 / 1024 if file_sizes else 0
        }

    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_raw_array_capture(num_frames=50):
    """
    Test de capture RAW array (pour stacking direct)

    Args:
        num_frames: Nombre de frames à capturer
    """
    print_separator("TEST 2 : CAPTURE RAW ARRAY (POUR STACKING)")

    print(f"\n📸 Capture de {num_frames} arrays RAW en mémoire...")

    try:
        picam2 = Picamera2()

        # Configuration avec flux RAW
        config = picam2.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"},
            raw={"size": picam2.sensor_resolution}
        )

        print(f"   Configuration : {config}")
        picam2.configure(config)
        picam2.start()

        # Warm-up
        time.sleep(1)

        # Capturer la séquence en mémoire
        print(f"   ⏱️  Début capture...")

        raw_arrays = []
        start_time = time.time()
        frame_times = []

        for i in range(num_frames):
            frame_start = time.time()

            # Capturer le flux RAW
            raw_array = picam2.capture_array("raw")
            raw_arrays.append(raw_array.copy())  # Copie pour éviter l'écrasement

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # Afficher la progression
            if (i + 1) % 10 == 0:
                avg_fps = (i + 1) / (time.time() - start_time)
                print(f"      {i+1}/{num_frames} frames - {avg_fps:.1f} fps")

        total_time = time.time() - start_time

        # Statistiques
        print(f"\n   ✅ Capture RAW array terminée !")
        print(f"   📊 Statistiques :")
        print(f"      Frames capturées : {num_frames}")
        print(f"      Temps total : {total_time:.2f}s")
        print(f"      FPS moyen : {num_frames / total_time:.2f}")
        print(f"      Latence par frame : {np.mean(frame_times)*1000:.1f}ms (±{np.std(frame_times)*1000:.1f}ms)")

        if raw_arrays:
            array_shape = raw_arrays[0].shape
            array_dtype = raw_arrays[0].dtype
            array_size = raw_arrays[0].nbytes

            print(f"      Format array : {array_shape}, dtype={array_dtype}")
            print(f"      Taille mémoire : {array_size/1024/1024:.2f} MB/frame")
            print(f"      Total en RAM : {num_frames * array_size/1024/1024:.2f} MB")

        # Test de debayerisation
        print(f"\n   🔬 Test de debayerisation...")

        if raw_arrays:
            test_array = raw_arrays[0]

            # Déterminer le pattern Bayer
            print(f"      Array shape : {test_array.shape}")
            print(f"      Array dtype : {test_array.dtype}")
            print(f"      Min/Max : {test_array.min()} / {test_array.max()}")

            try:
                # Test des différents patterns Bayer
                bayer_patterns = [
                    (cv2.COLOR_BayerRG2RGB, "RGGB"),
                    (cv2.COLOR_BayerGR2RGB, "GRBG"),
                    (cv2.COLOR_BayerGB2RGB, "GBRG"),
                    (cv2.COLOR_BayerBG2RGB, "BGGR"),
                ]

                debayer_dir = OUTPUT_DIR / "debayer_test"
                debayer_dir.mkdir(exist_ok=True)

                print(f"      Test des 4 patterns Bayer...")

                for pattern_code, pattern_name in bayer_patterns:
                    try:
                        # Convertir en uint8 si nécessaire
                        if test_array.dtype == np.uint16:
                            test_array_8bit = (test_array / 256).astype(np.uint8)
                        else:
                            test_array_8bit = test_array.astype(np.uint8)

                        # Debayeriser
                        rgb = cv2.cvtColor(test_array_8bit, pattern_code)

                        # Sauvegarder pour vérification visuelle
                        output_path = debayer_dir / f"debayer_{pattern_name}.png"
                        cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                        print(f"         ✅ {pattern_name:4s} : {output_path}")

                    except Exception as e:
                        print(f"         ❌ {pattern_name:4s} : {e}")

                print(f"\n      💡 Vérifiez les images dans {debayer_dir}/")
                print(f"         pour identifier le bon pattern Bayer")

            except Exception as e:
                print(f"      ❌ Erreur debayerisation : {e}")

        picam2.stop()
        picam2.close()

        return True, {
            'fps': num_frames / total_time,
            'latency_ms': np.mean(frame_times) * 1000,
            'size_mb': array_size / 1024 / 1024 if raw_arrays else 0
        }

    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_rgb_comparison(num_frames=50):
    """
    Test de capture RGB classique pour comparaison

    Args:
        num_frames: Nombre de frames à capturer
    """
    print_separator("TEST 3 : CAPTURE RGB (COMPARAISON)")

    print(f"\n📸 Capture de {num_frames} arrays RGB...")

    try:
        picam2 = Picamera2()

        # Configuration RGB standard
        config = picam2.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )

        picam2.configure(config)
        picam2.start()

        # Warm-up
        time.sleep(1)

        # Capturer la séquence
        print(f"   ⏱️  Début capture...")

        start_time = time.time()
        frame_times = []

        for i in range(num_frames):
            frame_start = time.time()

            # Capturer RGB
            rgb_array = picam2.capture_array("main")

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # Afficher la progression
            if (i + 1) % 10 == 0:
                avg_fps = (i + 1) / (time.time() - start_time)
                print(f"      {i+1}/{num_frames} frames - {avg_fps:.1f} fps")

        total_time = time.time() - start_time

        # Statistiques
        print(f"\n   ✅ Capture RGB terminée !")
        print(f"   📊 Statistiques :")
        print(f"      Frames capturées : {num_frames}")
        print(f"      Temps total : {total_time:.2f}s")
        print(f"      FPS moyen : {num_frames / total_time:.2f}")
        print(f"      Latence par frame : {np.mean(frame_times)*1000:.1f}ms (±{np.std(frame_times)*1000:.1f}ms)")

        picam2.stop()
        picam2.close()

        return True, {
            'fps': num_frames / total_time,
            'latency_ms': np.mean(frame_times) * 1000,
        }

    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Fonction principale"""

    print("\n🎥 TEST DE CAPTURE DNG/RAW POUR STACKING")
    print("="*70)

    # Nombre de frames à tester
    num_frames = 50
    if len(sys.argv) > 1:
        try:
            num_frames = int(sys.argv[1])
        except:
            print("⚠️  Argument invalide, utilisation de 50 frames par défaut")

    print(f"\n📋 Configuration du test : {num_frames} frames par méthode")
    print(f"📁 Répertoire de sortie : {OUTPUT_DIR.absolute()}")

    # Résultats
    results = {}

    # Test 1 : DNG
    success_dng, stats_dng = test_dng_capture(num_frames)
    if success_dng:
        results['DNG'] = stats_dng

    time.sleep(2)  # Pause entre les tests

    # Test 2 : RAW Array
    success_raw, stats_raw = test_raw_array_capture(num_frames)
    if success_raw:
        results['RAW Array'] = stats_raw

    time.sleep(2)  # Pause entre les tests

    # Test 3 : RGB (comparaison)
    success_rgb, stats_rgb = test_rgb_comparison(num_frames)
    if success_rgb:
        results['RGB'] = stats_rgb

    # Comparaison finale
    print_separator("COMPARAISON DES MÉTHODES")

    if results:
        print("\n📊 Tableau comparatif :\n")
        print(f"{'Méthode':<15} {'FPS':>8} {'Latence (ms)':>15} {'Taille (MB)':>15}")
        print("-" * 70)

        for method, stats in results.items():
            fps = stats.get('fps', 0)
            latency = stats.get('latency_ms', 0)
            size = stats.get('size_mb', 0)

            print(f"{method:<15} {fps:>8.1f} {latency:>15.1f} {size:>15.2f}")

    # Recommandations
    print_separator("RECOMMANDATIONS")

    print("\n💡 Pour le stacking :")

    if 'RAW Array' in results and 'RGB' in results:
        raw_fps = results['RAW Array']['fps']
        rgb_fps = results['RGB']['fps']

        if raw_fps > rgb_fps * 0.8:  # Si RAW >= 80% de RGB
            print("   ✅ RAW Array recommandé :")
            print("      - Performance acceptable")
            print("      - Qualité maximale (pas de compression)")
            print("      - Pas de fichiers intermédiaires")
            print("      - Idéal pour Lucky Imaging haute cadence")
        else:
            print("   ⚠️  RAW Array plus lent que RGB :")
            print(f"      - Perte de {(1 - raw_fps/rgb_fps)*100:.1f}% de performance")
            print("      - Considérer DNG pour stacking offline")

    if 'DNG' in results:
        print("\n   📁 DNG (fichiers) recommandé pour :")
        print("      - Traitement ultérieur avec logiciels externes")
        print("      - Archivage long terme (format standard)")
        print("      - Calibration précise (dark, flat, bias)")

    print("\n✅ Tests terminés !")
    print(f"📁 Fichiers de test dans : {OUTPUT_DIR.absolute()}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
