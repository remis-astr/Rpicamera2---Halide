#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de la profondeur de bit réelle - RAW12 vs RAW16
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

stack_dir = Path("/media/admin/THKAILAR/Stacks")

# Trouver les fichiers
fits_files = sorted(stack_dir.glob("*.fit"), key=lambda x: x.stat().st_mtime, reverse=True)

if len(fits_files) >= 2:
    raw16_file = fits_files[0]
    raw12_file = fits_files[1]

    print("="*70)
    print("🔬 ANALYSE PROFONDEUR DE BIT RÉELLE - RAW12 vs RAW16")
    print("="*70)

    # Charger
    with fits.open(raw12_file) as hdul:
        data12 = hdul[0].data
    with fits.open(raw16_file) as hdul:
        data16 = hdul[0].data

    # Aplatir les données (tous les pixels)
    flat12 = data12.flatten()
    flat16 = data16.flatten()

    print(f"\n📊 STATISTIQUES GLOBALES")
    print("-" * 70)
    print(f"{'Métrique':<25} {'RAW12':<20} {'RAW16 Clear HDR':<20}")
    print("-" * 70)
    print(f"{'Min':<25} {flat12.min():<20.1f} {flat16.min():<20.1f}")
    print(f"{'Max':<25} {flat12.max():<20.1f} {flat16.max():<20.1f}")
    print(f"{'Plage (Max-Min)':<25} {flat12.max()-flat12.min():<20.1f} {flat16.max()-flat16.min():<20.1f}")
    print(f"{'Valeurs uniques totales':<25} {len(np.unique(flat12)):<20} {len(np.unique(flat16)):<20}")

    # Analyse des bits utilisés
    print(f"\n🔢 ANALYSE DES BITS UTILISÉS")
    print("-" * 70)

    # Vérifier si les valeurs sont des multiples de puissances de 2
    # RAW12 shifté de 4 bits devrait avoir des valeurs multiples de 16 (2^4)
    def analyze_lsb(data, name):
        """Analyse les bits de poids faible pour détecter le shift"""
        # Compter combien de valeurs sont divisibles par différentes puissances de 2
        nonzero = data[data > 0]

        print(f"\n{name}:")
        for shift in range(5):
            divisor = 2 ** shift
            divisible_count = np.sum(nonzero % divisor == 0)
            percent = 100 * divisible_count / len(nonzero)
            print(f"  Divisibles par {divisor:>3} (shift {shift}): {percent:>6.2f}% ({divisible_count}/{len(nonzero)})")

        # Calculer la granularité (plus petit écart entre valeurs consécutives)
        unique_sorted = np.sort(np.unique(nonzero))
        if len(unique_sorted) > 1:
            diffs = np.diff(unique_sorted)
            min_diff = diffs[diffs > 0].min() if np.any(diffs > 0) else 0
            median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0
            print(f"  Plus petit écart: {min_diff:.1f}")
            print(f"  Écart médian: {median_diff:.1f}")

            # Vérifier si c'est un multiple de 16 (shift de 4 bits pour RAW12)
            if min_diff >= 16:
                probable_shift = int(np.log2(min_diff))
                print(f"  → Shift probable: {probable_shift} bits (granularité {2**probable_shift})")
            else:
                print(f"  → Pas de shift détecté (granularité fine)")

    analyze_lsb(flat12, "RAW12")
    analyze_lsb(flat16, "RAW16 Clear HDR")

    # Calculer la résolution effective (bits réellement utilisés)
    print(f"\n📐 RÉSOLUTION EFFECTIVE")
    print("-" * 70)

    def effective_bits(data):
        """Calcule le nombre de bits effectivement utilisés"""
        # Basé sur la plage de valeurs
        value_range = data.max() - data.min()
        if value_range > 0:
            bits_from_range = np.log2(value_range)
        else:
            bits_from_range = 0

        # Basé sur le nombre de valeurs uniques
        unique_count = len(np.unique(data))
        bits_from_unique = np.log2(unique_count)

        return bits_from_range, bits_from_unique

    bits_range_12, bits_unique_12 = effective_bits(flat12)
    bits_range_16, bits_unique_16 = effective_bits(flat16)

    print(f"{'Métrique':<25} {'RAW12':<20} {'RAW16 Clear HDR':<20}")
    print("-" * 70)
    print(f"{'Bits (plage valeurs)':<25} {bits_range_12:<20.2f} {bits_range_16:<20.2f}")
    print(f"{'Bits (valeurs uniques)':<25} {bits_unique_12:<20.2f} {bits_unique_16:<20.2f}")

    # Analyse de la distribution
    print(f"\n📈 DISTRIBUTION DES VALEURS")
    print("-" * 70)

    # Diviser en bins pour voir la distribution
    # Pour RAW12 vrai: devrait avoir ~4096 valeurs distinctes max
    # Pour RAW16: devrait avoir ~65536 valeurs distinctes max

    # Vérifier si les valeurs forment des "paliers" (quantization)
    def check_quantization(data, name, expected_levels):
        """Vérifie si les données sont quantifiées"""
        unique_vals = np.unique(data[data > 0])

        # Si RAW12 shifté de 4 bits, on devrait avoir ~4096 valeurs séparées de 16
        # Vérifier la densité de valeurs
        value_range = data.max() - data.min()
        unique_count = len(unique_vals)

        density = unique_count / value_range if value_range > 0 else 0

        print(f"\n{name}:")
        print(f"  Plage: {data.min():.0f} - {data.max():.0f} (span: {value_range:.0f})")
        print(f"  Valeurs uniques: {unique_count}")
        print(f"  Densité: {density:.4f} valeurs/niveau")

        if density < 0.1:
            print(f"  → Quantification forte détectée (RAW12 shifté probable)")
            print(f"  → Niveaux théoriques: ~{value_range/16:.0f} (si shift 4 bits)")
        elif density > 0.5:
            print(f"  → Distribution dense (proche de 16-bit natif)")
        else:
            print(f"  → Distribution intermédiaire")

        # Vérifier le ratio avec les niveaux attendus
        if expected_levels > 0:
            ratio = unique_count / expected_levels
            print(f"  → Ratio vs attendu ({expected_levels}): {ratio:.2f}x")
            if ratio > 8:
                print(f"     Beaucoup de bruit/suréchantillonnage")

    check_quantization(flat12, "RAW12", 4096)
    check_quantization(flat16, "RAW16 Clear HDR", 65536)

    # Conclusion
    print(f"\n" + "="*70)
    print("💡 CONCLUSION")
    print("="*70)

    if bits_unique_12 < 13:
        print("\n✓ RAW12 est bien limité à ~12 bits de résolution")
        print(f"  Résolution effective: {bits_unique_12:.1f} bits")
    else:
        print("\n⚠ RAW12 semble avoir plus de résolution que 12 bits!")
        print(f"  Résolution effective: {bits_unique_12:.1f} bits")
        print("  Possible causes: bruit, suréchantillonnage, ou mauvaise config")

    if bits_unique_16 > 14:
        print("\n✓ RAW16 Clear HDR utilise bien la plage 16-bit")
        print(f"  Résolution effective: {bits_unique_16:.1f} bits")
    else:
        print("\n⚠ RAW16 Clear HDR n'utilise pas toute la plage 16-bit")
        print(f"  Résolution effective: {bits_unique_16:.1f} bits")

    # Vérifier si Clear HDR apporte un gain
    if bits_unique_16 > bits_unique_12 + 1:
        print(f"\n🎯 Clear HDR apporte un gain de résolution:")
        print(f"   +{bits_unique_16 - bits_unique_12:.1f} bits effectifs")
    else:
        print(f"\n⚠ Clear HDR n'apporte pas de gain significatif sur cette scène")
        print(f"   Scène trop faible en contraste pour bénéficier du mode 16-bit")

    print("="*70)

else:
    print("❌ Pas assez de fichiers")
