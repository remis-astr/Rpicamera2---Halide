#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de l'effet ISP sur les images RAW
Compare images avec/sans ISP pour vérifier l'application
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

stack_dir = Path("/media/admin/THKAILAR/Stacks")

print("="*70)
print("🔬 TEST EFFET ISP - Vérification application balance des blancs")
print("="*70)

# Instructions
print("""
PROTOCOLE DE TEST:

1. Avec ISP DÉSACTIVÉ (isp_enable = 0):
   - Capturez une image RAW12/16 en LiveStack
   - Notez le nom du fichier PNG/FITS

2. Avec ISP ACTIVÉ (isp_enable = 1):
   - Capturez la MÊME scène en LiveStack
   - Notez le nom du fichier PNG/FITS

3. Effet attendu avec isp_config_neutral.json:
   - Le BLEU devrait être réduit de 50%
   - L'image devrait paraître plus JAUNE/ORANGE
   - Si aucune différence visible → ISP non appliqué

4. Vérification automatique:
   Ce script analyse les deux derniers fichiers FITS
""")

print("\n" + "="*70)
print("📁 Analyse des fichiers récents")
print("="*70)

fits_files = sorted(stack_dir.glob("*.fit"), key=lambda x: x.stat().st_mtime, reverse=True)

if len(fits_files) >= 2:
    file1 = fits_files[0]
    file2 = fits_files[1]

    print(f"\nFichier 1 (récent): {file1.name}")
    print(f"Fichier 2 (ancien):  {file2.name}")

    with fits.open(file1) as hdul:
        data1 = hdul[0].data
    with fits.open(file2) as hdul:
        data2 = hdul[0].data

    if len(data1.shape) == 3 and len(data2.shape) == 3:
        print("\n📊 Analyse des canaux RGB:")
        print("-"*70)
        print(f"{'Canal':<10} {'Fichier 1':<20} {'Fichier 2':<20} {'Différence':<15}")
        print("-"*70)

        channels = ['Rouge', 'Vert', 'Bleu']
        diffs = []

        for i, channel in enumerate(channels):
            mean1 = data1[i].mean()
            mean2 = data2[i].mean()
            diff = ((mean1 - mean2) / mean2) * 100 if mean2 > 0 else 0
            diffs.append(abs(diff))

            sign = "+" if diff > 0 else ""
            print(f"{channel:<10} {mean1:<20.1f} {mean2:<20.1f} {sign}{diff:>6.1f}%")

        print("\n💡 INTERPRÉTATION:")
        print("-"*70)

        # Si ISP appliqué, le bleu devrait diminuer de ~50%
        if diffs[2] > 30:  # Bleu a changé de plus de 30%
            print("✅ ISP semble APPLIQUÉ!")
            print(f"   Le canal BLEU a changé de {diffs[2]:.1f}%")
            if data1[2].mean() < data2[2].mean():
                print("   → Bleu RÉDUIT (conforme à wb_red_gain=0.5 avec swap_rb)")
            else:
                print("   → Bleu AUGMENTÉ (vérifier config ISP)")
        elif max(diffs) < 5:
            print("❌ ISP semble NON APPLIQUÉ ou NEUTRE")
            print(f"   Différences minimes: R={diffs[0]:.1f}%, V={diffs[1]:.1f}%, B={diffs[2]:.1f}%")
            print("\n   Causes possibles:")
            print("   1. isp_enable = 0 (ISP désactivé)")
            print("   2. Même config ISP utilisée pour les deux captures")
            print("   3. Bug dans l'application de l'ISP")
        else:
            print("⚠️ Changement détecté mais pas sur le canal attendu")
            print(f"   Différences: R={diffs[0]:.1f}%, V={diffs[1]:.1f}%, B={diffs[2]:.1f}%")
    else:
        print("⚠️ Les fichiers ne sont pas en format RGB")
else:
    print("\n❌ Pas assez de fichiers FITS pour comparer")
    print("   Capturez d'abord deux images (avec/sans ISP)")

print("\n" + "="*70)
