#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script 3 : Test d'enregistrement vidéo RAW
===============================================

Ce script teste :
- Enregistrement vidéo en YUV420 (non compressé)
- Enregistrement vidéo en RAW Bayer
- Performance et débit disque
- Méthodes d'enregistrement selon le forum Raspberry Pi

Les fichiers de test sont sauvegardés dans ./test_output/

Usage:
    python3 test_raw_video.py [durée_secondes]

    Exemple : python3 test_raw_video.py 10
"""

import sys
import os
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import Encoder
from pathlib import Path

# Créer le répertoire de sortie
OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def print_separator(title):
    """Affiche un séparateur avec titre"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def get_disk_write_speed(file_path, size_mb=100):
    """
    Test de vitesse d'écriture disque

    Args:
        file_path: Chemin du fichier de test
        size_mb: Taille du test en MB
    """
    print("\n💾 Test de vitesse d'écriture disque...")

    try:
        test_data = np.random.randint(0, 255, size_mb * 1024 * 1024, dtype=np.uint8)

        start_time = time.time()
        with open(file_path, 'wb') as f:
            f.write(test_data.tobytes())
        write_time = time.time() - start_time

        # Nettoyer
        os.remove(file_path)

        write_speed = size_mb / write_time

        print(f"   ✅ Vitesse d'écriture : {write_speed:.1f} MB/s")

        return write_speed

    except Exception as e:
        print(f"   ⚠️  Impossible de tester : {e}")
        return None

def test_yuv420_recording(duration_sec=10):
    """
    Test d'enregistrement vidéo YUV420 (non compressé)

    Args:
        duration_sec: Durée de l'enregistrement en secondes
    """
    print_separator("TEST 1 : ENREGISTREMENT YUV420 (NON COMPRESSÉ)")

    print(f"\n🎬 Enregistrement de {duration_sec}s en YUV420...")

    try:
        picam2 = Picamera2()

        # Test de vitesse disque
        test_file = OUTPUT_DIR / "disk_test.bin"
        disk_speed = get_disk_write_speed(test_file)

        # Configuration pour YUV420
        # YUV420 : Y plane + U/2 + V/2 = 1.5 bytes per pixel
        width, height = 1920, 1080
        bytes_per_frame = width * height * 1.5
        fps = 30

        print(f"\n   📊 Estimations :")
        print(f"      Résolution : {width}×{height}")
        print(f"      Format : YUV420")
        print(f"      FPS cible : {fps}")
        print(f"      Bytes/frame : {bytes_per_frame/1024/1024:.2f} MB")
        print(f"      Débit : {bytes_per_frame * fps / 1024 / 1024:.1f} MB/s")

        if disk_speed:
            if bytes_per_frame * fps / 1024 / 1024 > disk_speed * 0.8:
                print(f"      ⚠️  ATTENTION : Débit proche de la limite disque !")

        # Créer la configuration
        config = picam2.create_video_configuration(
            main={"size": (width, height), "format": "YUV420"},
            controls={"FrameRate": fps}
        )

        print(f"\n   Configuration : {config}")
        picam2.configure(config)

        # Préparer l'encodeur (passthrough, pas de compression)
        output_path = OUTPUT_DIR / f"test_yuv420_{width}x{height}.yuv"

        print(f"   📁 Fichier de sortie : {output_path}")

        # Méthode 1 : Enregistrement avec encodeur
        print(f"\n   🎥 Méthode 1 : Encodeur non compressé")

        try:
            # Créer un "encodeur" qui ne fait que passer les données
            encoder = Encoder()

            picam2.start_recording(encoder, str(output_path))

            print(f"   ⏺️  Enregistrement en cours...")

            # Enregistrer pendant la durée spécifiée
            for i in range(duration_sec):
                time.sleep(1)
                print(f"      {i+1}/{duration_sec}s", end='\r')

            picam2.stop_recording()
            print()

            # Statistiques
            if output_path.exists():
                file_size = output_path.stat().st_size
                expected_size = bytes_per_frame * fps * duration_sec

                print(f"\n   ✅ Enregistrement terminé !")
                print(f"   📊 Statistiques :")
                print(f"      Taille fichier : {file_size/1024/1024:.2f} MB")
                print(f"      Taille attendue : {expected_size/1024/1024:.2f} MB")
                print(f"      Débit moyen : {file_size/duration_sec/1024/1024:.1f} MB/s")

                # Vérifier que le fichier est correct
                actual_frames = file_size / bytes_per_frame
                print(f"      Frames estimées : {actual_frames:.0f} (~{actual_frames/duration_sec:.1f} fps)")

                result = {
                    'success': True,
                    'file_size_mb': file_size / 1024 / 1024,
                    'bitrate_mbps': file_size / duration_sec / 1024 / 1024,
                    'fps_actual': actual_frames / duration_sec
                }

            else:
                print(f"   ❌ Fichier non créé !")
                result = {'success': False}

        except Exception as e:
            print(f"   ❌ Erreur encodeur : {e}")
            import traceback
            traceback.print_exc()
            result = {'success': False}

        finally:
            try:
                picam2.stop_recording()
            except:
                pass

        picam2.close()

        return result

    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return {'success': False}

def test_raw_bayer_recording(duration_sec=5):
    """
    Test d'enregistrement vidéo RAW Bayer

    Args:
        duration_sec: Durée de l'enregistrement en secondes
    """
    print_separator("TEST 2 : ENREGISTREMENT RAW BAYER")

    print(f"\n🎬 Enregistrement de {duration_sec}s en RAW Bayer...")

    try:
        picam2 = Picamera2()

        # Configuration pour RAW
        # Tester différents formats RAW
        raw_formats = [
            ("SRGGB10", "RAW 10-bit packed"),
            ("SRGGB12", "RAW 12-bit"),
            ("SRGGB16", "RAW 16-bit unpacked"),
        ]

        sensor_res = picam2.sensor_resolution
        print(f"   Résolution capteur : {sensor_res}")

        results = {}

        for format_name, description in raw_formats:
            print(f"\n   🔬 Test format : {format_name} ({description})")

            try:
                # Configuration
                config = picam2.create_video_configuration(
                    main={"size": sensor_res, "format": format_name}
                )

                picam2.configure(config)

                # Fichier de sortie
                output_path = OUTPUT_DIR / f"test_raw_{format_name}.raw"

                # Méthode : Capture manuelle frame par frame
                print(f"      📁 Sortie : {output_path}")
                print(f"      ⏺️  Capture...")

                picam2.start()
                time.sleep(0.5)  # Stabilisation

                start_time = time.time()
                frames_captured = 0

                with open(output_path, 'wb') as f:
                    while time.time() - start_time < duration_sec:
                        # Capturer une frame
                        array = picam2.capture_array("main")

                        # Écrire dans le fichier
                        f.write(array.tobytes())

                        frames_captured += 1

                        if frames_captured % 10 == 0:
                            elapsed = time.time() - start_time
                            fps = frames_captured / elapsed
                            print(f"         {frames_captured} frames ({fps:.1f} fps)", end='\r')

                elapsed_time = time.time() - start_time
                print()

                picam2.stop()

                # Statistiques
                if output_path.exists():
                    file_size = output_path.stat().st_size

                    print(f"      ✅ Capture terminée !")
                    print(f"         Frames : {frames_captured}")
                    print(f"         Durée : {elapsed_time:.2f}s")
                    print(f"         FPS moyen : {frames_captured/elapsed_time:.1f}")
                    print(f"         Taille : {file_size/1024/1024:.2f} MB")
                    print(f"         Débit : {file_size/elapsed_time/1024/1024:.1f} MB/s")

                    results[format_name] = {
                        'success': True,
                        'frames': frames_captured,
                        'fps': frames_captured / elapsed_time,
                        'size_mb': file_size / 1024 / 1024,
                        'bitrate_mbps': file_size / elapsed_time / 1024 / 1024
                    }

                else:
                    print(f"      ❌ Fichier non créé !")
                    results[format_name] = {'success': False}

            except Exception as e:
                print(f"      ❌ Erreur : {e}")
                results[format_name] = {'success': False, 'error': str(e)}

        picam2.close()

        return results

    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_circular_buffer_method(duration_sec=10):
    """
    Test de la méthode CircularOutput pour enregistrement continu

    Args:
        duration_sec: Durée de l'enregistrement en secondes
    """
    print_separator("TEST 3 : MÉTHODE CIRCULAR BUFFER")

    print(f"\n🎬 Test CircularOutput (méthode du forum)...")

    try:
        from picamera2.outputs import CircularOutput

        picam2 = Picamera2()

        # Configuration
        config = picam2.create_video_configuration(
            main={"size": (1920, 1080), "format": "YUV420"}
        )

        picam2.configure(config)

        # Créer un buffer circulaire
        buffer_size_frames = 300  # 10s à 30fps
        output_path = OUTPUT_DIR / "test_circular.yuv"

        print(f"   Buffer : {buffer_size_frames} frames")
        print(f"   📁 Sortie : {output_path}")

        try:
            encoder = Encoder()
            circular = CircularOutput(buffersize=buffer_size_frames)

            picam2.start_recording(encoder, circular)

            print(f"   ⏺️  Enregistrement...")

            # Enregistrer
            for i in range(duration_sec):
                time.sleep(1)
                print(f"      {i+1}/{duration_sec}s", end='\r')

            print()

            # Sauvegarder le buffer
            print(f"   💾 Sauvegarde du buffer...")
            circular.fileoutput = str(output_path)
            circular.stop()

            picam2.stop_recording()

            # Statistiques
            if output_path.exists():
                file_size = output_path.stat().st_size

                print(f"   ✅ Sauvegarde terminée !")
                print(f"      Taille : {file_size/1024/1024:.2f} MB")

                return {
                    'success': True,
                    'size_mb': file_size / 1024 / 1024
                }
            else:
                print(f"   ❌ Fichier non créé !")
                return {'success': False}

        except Exception as e:
            print(f"   ❌ Erreur : {e}")
            import traceback
            traceback.print_exc()
            return {'success': False}

        finally:
            try:
                picam2.stop_recording()
            except:
                pass

        picam2.close()

    except ImportError:
        print(f"   ⚠️  CircularOutput non disponible dans cette version de picamera2")
        return {'success': False, 'error': 'CircularOutput not available'}

    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return {'success': False}

def main():
    """Fonction principale"""

    print("\n🎥 TEST D'ENREGISTREMENT VIDÉO RAW")
    print("="*70)

    # Durée d'enregistrement
    duration_sec = 10
    if len(sys.argv) > 1:
        try:
            duration_sec = int(sys.argv[1])
        except:
            print("⚠️  Argument invalide, utilisation de 10s par défaut")

    print(f"\n📋 Configuration : {duration_sec}s d'enregistrement par test")
    print(f"📁 Répertoire de sortie : {OUTPUT_DIR.absolute()}")

    # Avertissement sur l'espace disque
    print(f"\n⚠️  ATTENTION :")
    print(f"   Les vidéos RAW sont TRÈS volumineuses !")
    print(f"   Estimation : 30-50 MB/s pour YUV420 1080p")
    print(f"   Assurez-vous d'avoir suffisamment d'espace disque.")

    input("\n   Appuyez sur ENTRÉE pour continuer...")

    # Résultats
    all_results = {}

    # Test 1 : YUV420
    print()
    result_yuv = test_yuv420_recording(duration_sec)
    all_results['YUV420'] = result_yuv

    time.sleep(2)

    # Test 2 : RAW Bayer
    print()
    result_raw = test_raw_bayer_recording(min(duration_sec, 5))  # Limiter à 5s pour RAW
    all_results['RAW_Bayer'] = result_raw

    time.sleep(2)

    # Test 3 : CircularOutput
    print()
    result_circular = test_circular_buffer_method(duration_sec)
    all_results['Circular'] = result_circular

    # Comparaison finale
    print_separator("RÉSUMÉ DES TESTS")

    print("\n📊 Résultats :\n")

    for method, result in all_results.items():
        print(f"🔹 {method}")

        if isinstance(result, dict):
            if result.get('success'):
                if 'bitrate_mbps' in result:
                    print(f"   ✅ Succès - Débit : {result['bitrate_mbps']:.1f} MB/s")
                else:
                    print(f"   ✅ Succès")

                if 'fps_actual' in result:
                    print(f"      FPS réel : {result['fps_actual']:.1f}")
            else:
                error = result.get('error', 'Échec')
                print(f"   ❌ {error}")
        else:
            # Résultats multiples (RAW Bayer)
            for fmt, res in result.items():
                if res.get('success'):
                    print(f"   ✅ {fmt:12s} : {res['fps']:.1f} fps, {res['bitrate_mbps']:.1f} MB/s")
                else:
                    print(f"   ❌ {fmt:12s} : {res.get('error', 'Échec')}")

    # Recommandations
    print_separator("RECOMMANDATIONS")

    print("\n💡 Pour l'enregistrement vidéo RAW :")

    if all_results.get('YUV420', {}).get('success'):
        print("\n   ✅ YUV420 (recommandé) :")
        print("      - Format non compressé")
        print("      - Compatible avec outils standards")
        print("      - Débit gérable (~30-50 MB/s)")
        print("      - Conversion facile vers SER/AVI")

    if any(r.get('success') for r in all_results.get('RAW_Bayer', {}).values()):
        print("\n   🔬 RAW Bayer (avancé) :")
        print("      - Qualité maximale (données capteur brutes)")
        print("      - Nécessite debayerisation")
        print("      - Débit très élevé (50-100+ MB/s)")
        print("      - Pour utilisateurs experts")

    print("\n   📝 Format recommandé selon usage :")
    print("      - Planétaire : YUV420 → conversion SER")
    print("      - Ciel profond : RAW Bayer → FITS")
    print("      - Tests/debug : YUV420 (plus facile)")

    print("\n✅ Tests terminés !")
    print(f"📁 Fichiers vidéo dans : {OUTPUT_DIR.absolute()}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
