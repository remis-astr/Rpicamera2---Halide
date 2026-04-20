#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier le mode Clear HDR 16-bit de l'IMX585
"""

import numpy as np
from pathlib import Path
import sys
import time

# Vérifier si une image FITS existe
stack_dir = Path("/media/admin/THKAILAR/Stacks")

print("="*70)
print("🔬 TEST MODE CLEAR HDR 16-BIT - IMX585")
print("="*70)

# Chercher les fichiers récents
if stack_dir.exists():
    print(f"\n📁 Recherche dans: {stack_dir}")

    # Chercher les fichiers FITS récents
    fits_files = sorted(stack_dir.glob("*.fit"), key=lambda x: x.stat().st_mtime, reverse=True)

    if fits_files:
        print(f"\n✓ {len(fits_files)} fichiers FITS trouvés")
        print("\nFichiers récents (5 derniers):")
        for i, f in enumerate(fits_files[:5], 1):
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {i}. {f.name} ({size_mb:.1f} MB, {mtime})")

        # Analyser le fichier le plus récent
        latest_file = fits_files[0]
        print(f"\n🔬 Analyse de: {latest_file.name}")

        try:
            from astropy.io import fits

            with fits.open(latest_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header

                print(f"\n📊 Informations FITS:")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  Min: {data.min():.1f}")
                print(f"  Max: {data.max():.1f}")
                print(f"  Mean: {data.mean():.1f}")
                print(f"  Std: {data.std():.1f}")

                # Vérifier le header
                print(f"\n📝 Header FITS:")
                if 'STACKED' in header:
                    print(f"  Frames empilées: {header['STACKED']}")
                if 'REJECTED' in header:
                    print(f"  Frames rejetées: {header['REJECTED']}")
                if 'BITPIX' in header:
                    print(f"  BITPIX: {header['BITPIX']}")

                # Analyser la dynamique
                print(f"\n🎯 Analyse de la dynamique:")

                # Calculer l'histogramme
                if len(data.shape) == 3:
                    # RGB
                    for i, channel in enumerate(['Rouge', 'Vert', 'Bleu']):
                        ch_data = data[:, :, i]
                        p1 = np.percentile(ch_data, 1)
                        p99 = np.percentile(ch_data, 99)
                        dynamic_range = p99 - p1
                        print(f"  {channel}: P1={p1:.1f}, P99={p99:.1f}, DR={dynamic_range:.1f}")
                else:
                    # Mono
                    p1 = np.percentile(data, 1)
                    p99 = np.percentile(data, 99)
                    dynamic_range = p99 - p1
                    print(f"  P1={p1:.1f}, P99={p99:.1f}, DR={dynamic_range:.1f}")

                # Vérifier la profondeur de bit effective
                unique_values = len(np.unique(data))
                print(f"\n🔢 Profondeur de bit:")
                print(f"  Valeurs uniques: {unique_values}")

                if data.max() > 4096:
                    print(f"  ✓ Données > 12-bit détectées (max={data.max():.0f})")
                    print(f"  ✓ Mode 16-bit confirmé!")
                    bit_depth_estimate = np.log2(data.max())
                    print(f"  Profondeur estimée: {bit_depth_estimate:.1f} bits")
                else:
                    print(f"  ⚠ Données dans la plage 12-bit (max={data.max():.0f})")
                    print(f"  Vérifiez que le mode Clear HDR est bien activé")

                # Test de plage dynamique
                print(f"\n💡 Interprétation:")
                if data.max() > 10000:
                    print(f"  ✓ Excellente plage dynamique détectée")
                    print(f"  ✓ Le Clear HDR semble fonctionner correctement")
                elif data.max() > 4096:
                    print(f"  ✓ Plage dynamique étendue (> 12-bit)")
                    print(f"  ⚠ Vérifiez l'exposition et les gains")
                else:
                    print(f"  ⚠ Plage dynamique limitée")
                    print(f"  Recommandation: Augmentez l'exposition ou le gain")

        except ImportError:
            print("\n⚠ astropy non installé. Installation:")
            print("  pip install astropy")
            print("\nAnalyse manuelle:")
            print(f"  Taille fichier: {latest_file.stat().st_size / 1024 / 1024:.1f} MB")

            # Estimation basée sur la taille
            expected_size_16bit = 1928 * 1090 * 3 * 4  # RGB float32
            if latest_file.stat().st_size > expected_size_16bit * 0.8:
                print(f"  ✓ Taille cohérente avec données 16-bit")
            else:
                print(f"  ⚠ Taille plus petite qu'attendu")

        except Exception as e:
            print(f"\n❌ Erreur lors de l'analyse: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"\n⚠ Aucun fichier FITS trouvé dans {stack_dir}")
        print("  Capturez d'abord une image en mode Clear HDR")

else:
    print(f"\n❌ Répertoire non trouvé: {stack_dir}")

print("\n" + "="*70)
print("📸 GUIDE DE TEST COMPARATIF")
print("="*70)
print("""
Pour vérifier le Clear HDR, capturez la MÊME scène avec:

1. Mode RAW12 (baseline):
   - OTHER SETTINGS → RAW Format → RAW12 Bayer
   - Activez LiveStack ou LuckyStack
   - Capturez 10-20 frames
   - Sauvegardez (note le nom du fichier)

2. Mode RAW16 Clear HDR:
   - OTHER SETTINGS → RAW Format → RAW16 Clear HDR
   - Activez LiveStack ou LuckyStack
   - Capturez 10-20 frames
   - Sauvegardez

3. Comparez les deux fichiers FITS:
   - Valeur max: RAW16 doit être > 10000 (vs ~4095 pour RAW12)
   - Plage dynamique: RAW16 doit avoir plus de détails
   - Zones sombres: mieux préservées en RAW16
   - Zones claires: moins de saturation en RAW16

4. Scène de test idéale:
   - Ciel nocturne avec étoiles brillantes ET faibles
   - Ou scène avec fort contraste (ombres + zones lumineuses)
   - Exposition: 5-10s, Gain: 100-200

ASTUCE: Utilisez ce script après chaque capture:
  python3 test_clear_hdr.py
""")
print("="*70)
