#!/usr/bin/env python3
"""
Test pour vérifier la profondeur réelle du format SRGGB16.
Détermine si c'est du 12-bit codé sur 16-bit ou du vrai 16-bit.
"""

import numpy as np
from picamera2 import Picamera2
import time

def analyze_bit_depth(raw_array, format_name):
    """Analyse la profondeur de bits réelle d'un array brut."""

    print(f"\n{'='*60}")
    print(f"ANALYSE DE PROFONDEUR : {format_name}")
    print(f"{'='*60}\n")

    # Stats de base
    print(f"Shape: {raw_array.shape}")
    print(f"Dtype: {raw_array.dtype}")
    print(f"Min value: {raw_array.min()}")
    print(f"Max value: {raw_array.max()}")
    print(f"Mean value: {raw_array.mean():.2f}")
    print(f"Std dev: {raw_array.std():.2f}")

    # Analyse des bits utilisés
    print(f"\n--- ANALYSE DES BITS ---")

    # Vérifier si les valeurs sont multiples de 16 (12-bit shifté de 4 bits)
    multiples_of_16 = np.sum(raw_array % 16 == 0)
    total_pixels = raw_array.size
    percentage_mult_16 = (multiples_of_16 / total_pixels) * 100

    print(f"Valeurs multiples de 16: {multiples_of_16}/{total_pixels} ({percentage_mult_16:.2f}%)")

    # Vérifier si les valeurs sont multiples de autres puissances de 2
    for shift in [1, 2, 3, 4, 5, 6, 7, 8]:
        divisor = 2 ** shift
        multiples = np.sum(raw_array % divisor == 0)
        percentage = (multiples / total_pixels) * 100
        print(f"Valeurs multiples de {divisor:3d}: {multiples}/{total_pixels} ({percentage:.2f}%)")

    # Analyse des 4 bits de poids faible
    print(f"\n--- BITS DE POIDS FAIBLE ---")
    low_4_bits = raw_array & 0x0F  # Masque les 4 bits de poids faible
    unique_low_bits = np.unique(low_4_bits)
    print(f"Valeurs uniques dans les 4 bits de poids faible: {len(unique_low_bits)}")
    print(f"Valeurs: {unique_low_bits[:20]}...")  # Affiche les 20 premières

    if len(unique_low_bits) == 1 and unique_low_bits[0] == 0:
        print("⚠️  Les 4 bits de poids faible sont toujours à 0 → 12-bit codé sur 16-bit")
    else:
        print("✓ Les 4 bits de poids faible varient → Probablement du vrai 16-bit")

    # Analyse des 8 bits de poids faible
    print(f"\n--- BITS DE POIDS FAIBLE (8 bits) ---")
    low_8_bits = raw_array & 0xFF
    unique_low_8_bits = np.unique(low_8_bits)
    print(f"Valeurs uniques dans les 8 bits de poids faible: {len(unique_low_8_bits)}")

    # Histogramme des valeurs
    print(f"\n--- DISTRIBUTION DES VALEURS ---")
    hist, bin_edges = np.histogram(raw_array, bins=16)
    for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
        next_edge = bin_edges[i+1]
        print(f"  [{edge:6.0f} - {next_edge:6.0f}): {count:8d} pixels ({count/total_pixels*100:5.2f}%)")

    # Échantillon de valeurs brutes
    print(f"\n--- ÉCHANTILLON DE VALEURS BRUTES (50 pixels au centre) ---")
    center_y = raw_array.shape[0] // 2
    center_x = raw_array.shape[1] // 2
    sample = raw_array[center_y:center_y+10, center_x:center_x+5].flatten()

    print("Valeurs décimales:")
    print(sample)
    print("\nValeurs hexadécimales:")
    print([hex(v) for v in sample])
    print("\nValeurs binaires (4 bits de poids faible):")
    print([bin(v & 0x0F) for v in sample])

    # Conclusion
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print(f"{'='*60}")

    max_possible_12bit = 4095 << 4  # 12-bit shifté de 4 bits = 65520

    if percentage_mult_16 > 99.0:
        print("✓ Format: 12-bit codé sur 16-bit (shifté de 4 bits)")
        print(f"  Plage réelle: 0 - {max_possible_12bit} (0x0000 - 0xFFF0)")
        print(f"  Bits utilisés: [15:4], bits [3:0] toujours à 0")
        bit_depth = 12
    elif raw_array.max() > max_possible_12bit:
        print("✓ Format: Vrai 16-bit")
        print(f"  Plage réelle: 0 - 65535 (0x0000 - 0xFFFF)")
        print(f"  Tous les bits sont utilisés")
        bit_depth = 16
    elif len(unique_low_bits) > 1:
        print("✓ Format: Probablement vrai 16-bit (bits de poids faible variables)")
        bit_depth = 16
    else:
        print("? Format incertain, analyse manuelle recommandée")
        bit_depth = None

    print(f"{'='*60}\n")

    return bit_depth


def main():
    print("Test de profondeur de bits pour formats RAW de Picamera2")
    print("="*60)

    picam2 = Picamera2()

    # Test 1: SRGGB16
    print("\n\n### TEST 1: Format SRGGB16 ###")

    try:
        config = picam2.create_video_configuration(
            raw={"format": "SRGGB16", "size": (3856, 2180)},
            buffer_count=2
        )
        picam2.configure(config)

        print(f"Configuration appliquée: {config['raw']}")

        picam2.start()
        time.sleep(2)  # Laisser l'AE/AWB se stabiliser

        # Capturer
        print("Capture en cours...")
        start = time.time()
        raw_array = picam2.capture_array("raw")
        capture_time = (time.time() - start) * 1000

        print(f"Capture: {capture_time:.1f} ms")

        # Analyser
        bit_depth_16 = analyze_bit_depth(raw_array, "SRGGB16")

        picam2.stop()

    except Exception as e:
        print(f"❌ Erreur avec SRGGB16: {e}")
        bit_depth_16 = None

    # Test 2: SRGGB12 pour comparaison
    print("\n\n### TEST 2: Format SRGGB12 (pour comparaison) ###")

    try:
        config = picam2.create_video_configuration(
            raw={"format": "SRGGB12", "size": (3856, 2180)},
            buffer_count=2
        )
        picam2.configure(config)

        print(f"Configuration appliquée: {config['raw']}")

        picam2.start()
        time.sleep(2)

        print("Capture en cours...")
        start = time.time()
        raw_array = picam2.capture_array("raw")
        capture_time = (time.time() - start) * 1000

        print(f"Capture: {capture_time:.1f} ms")

        # Analyser
        bit_depth_12 = analyze_bit_depth(raw_array, "SRGGB12")

        picam2.stop()

    except Exception as e:
        print(f"❌ Erreur avec SRGGB12: {e}")
        bit_depth_12 = None

    picam2.close()

    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)

    if bit_depth_16:
        print(f"SRGGB16: {bit_depth_16}-bit effectifs")
    else:
        print("SRGGB16: Non testé ou erreur")

    if bit_depth_12:
        print(f"SRGGB12: {bit_depth_12}-bit effectifs")
    else:
        print("SRGGB12: Non testé ou erreur")

    print("\nRECOMMANDATION:")
    if bit_depth_16 == 16 and bit_depth_12 == 12:
        print("✓ SRGGB16 offre plus de profondeur que SRGGB12")
        print("  → Utiliser SRGGB16 pour DSO/Lunaire (meilleure qualité)")
        print("  → Tester les performances pour voir si acceptable")
    elif bit_depth_16 == 12 and bit_depth_12 == 12:
        print("⚠️  SRGGB16 et SRGGB12 ont la même profondeur effective (12-bit)")
        print("  → Utiliser SRGGB12 (probablement plus rapide)")
    else:
        print("? Résultats inattendus, analyse manuelle recommandée")

    print("="*60)


if __name__ == "__main__":
    main()
