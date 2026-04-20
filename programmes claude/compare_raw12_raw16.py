#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison RAW12 vs RAW16 Clear HDR
"""

import numpy as np
from pathlib import Path
from astropy.io import fits

stack_dir = Path("/media/admin/THKAILAR/Stacks")

# Trouver les deux derniers fichiers
fits_files = sorted(stack_dir.glob("*.fit"), key=lambda x: x.stat().st_mtime, reverse=True)

if len(fits_files) >= 2:
    raw16_file = fits_files[0]  # Plus récent
    raw12_file = fits_files[1]  # Précédent

    print("="*70)
    print("📊 COMPARAISON RAW12 vs RAW16 CLEAR HDR")
    print("="*70)

    print(f"\n📁 Fichiers analysés:")
    print(f"  RAW12:  {raw12_file.name}")
    print(f"  RAW16:  {raw16_file.name}")

    # Charger les données
    with fits.open(raw12_file) as hdul:
        data12 = hdul[0].data
        header12 = hdul[0].header

    with fits.open(raw16_file) as hdul:
        data16 = hdul[0].data
        header16 = hdul[0].header

    print(f"\n📊 Statistiques comparatives:")
    print(f"{'Métrique':<20} {'RAW12':<15} {'RAW16 Clear HDR':<15} {'Gain':<10}")
    print("-" * 70)

    # Min/Max
    print(f"{'Min':<20} {data12.min():<15.1f} {data16.min():<15.1f} {'-':<10}")
    print(f"{'Max':<20} {data12.max():<15.1f} {data16.max():<15.1f} {f'{data16.max()/data12.max():.2f}x':<10}")

    # Mean/Std
    print(f"{'Mean':<20} {data12.mean():<15.1f} {data16.mean():<15.1f} {f'{data16.mean()/data12.mean():.2f}x':<10}")
    print(f"{'Std':<20} {data12.std():<15.1f} {data16.std():<15.1f} {f'{data16.std()/data12.std():.2f}x':<10}")

    # Valeurs uniques
    unique12 = len(np.unique(data12))
    unique16 = len(np.unique(data16))
    print(f"{'Valeurs uniques':<20} {unique12:<15} {unique16:<15} {f'{unique16/unique12:.1f}x':<10}")

    # Plage dynamique (P1-P99)
    p1_12 = np.percentile(data12, 1)
    p99_12 = np.percentile(data12, 99)
    dr12 = p99_12 - p1_12

    p1_16 = np.percentile(data16, 1)
    p99_16 = np.percentile(data16, 99)
    dr16 = p99_16 - p1_16

    print(f"{'P1 (1%)':<20} {p1_12:<15.1f} {p1_16:<15.1f} {f'{p1_16/p1_12:.2f}x':<10}")
    print(f"{'P99 (99%)':<20} {p99_12:<15.1f} {p99_16:<15.1f} {f'{p99_16/p99_12:.2f}x':<10}")
    print(f"{'DR (P99-P1)':<20} {dr12:<15.1f} {dr16:<15.1f} {f'{dr16/dr12:.2f}x':<10}")

    # Profondeur de bit estimée
    bit_depth_12 = np.log2(data12.max()) if data12.max() > 0 else 0
    bit_depth_16 = np.log2(data16.max()) if data16.max() > 0 else 0

    print(f"{'Profondeur (bits)':<20} {bit_depth_12:<15.1f} {bit_depth_16:<15.1f} {f'+{bit_depth_16-bit_depth_12:.1f}bit':<10}")

    # Header
    if 'STACKED' in header12 and 'STACKED' in header16:
        print(f"{'Frames empilées':<20} {header12['STACKED']:<15} {header16['STACKED']:<15} {'-':<10}")

    # Analyse par canal
    print(f"\n📈 Analyse par canal (plage dynamique P1-P99):")
    print(f"{'Canal':<15} {'RAW12':<15} {'RAW16 Clear HDR':<15} {'Amélioration':<15}")
    print("-" * 70)

    for i, channel in enumerate(['Rouge', 'Vert', 'Bleu']):
        ch_data12 = data12[i]
        ch_data16 = data16[i]

        p1_ch12 = np.percentile(ch_data12, 1)
        p99_ch12 = np.percentile(ch_data12, 99)
        dr_ch12 = p99_ch12 - p1_ch12

        p1_ch16 = np.percentile(ch_data16, 1)
        p99_ch16 = np.percentile(ch_data16, 99)
        dr_ch16 = p99_ch16 - p1_ch16

        print(f"{channel:<15} {dr_ch12:<15.1f} {dr_ch16:<15.1f} {f'{dr_ch16/dr_ch12:.2f}x':<15}")

    # Conclusion
    print("\n" + "="*70)
    print("💡 CONCLUSION")
    print("="*70)

    if data16.max() > data12.max() * 2:
        print("✅ Clear HDR 16-bit apporte une amélioration significative!")
        print(f"   • Plage de valeurs: {data16.max()/data12.max():.1f}x plus large")
        print(f"   • Valeurs uniques: {unique16/unique12:.1f}x plus nombreuses")
        print(f"   • Plage dynamique: {dr16/dr12:.1f}x supérieure")
        print("\n🎯 Bénéfices:")
        print("   • Meilleure préservation des détails dans les ombres")
        print("   • Moins de saturation dans les hautes lumières")
        print("   • Post-traitement plus flexible (stretch, etc.)")
    else:
        print("⚠ Les deux modes donnent des résultats similaires")
        print("  Vérifiez que la scène a assez de contraste pour bénéficier du Clear HDR")

    print("="*70)

else:
    print("❌ Pas assez de fichiers FITS pour comparer")
    print("   Capturez une image en RAW12 puis une en RAW16")
