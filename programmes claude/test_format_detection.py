#!/usr/bin/env python3
"""
Test de détection automatique du format et sauvegarde adaptative
"""

import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from libastrostack.io import detect_source_format, save_png_auto, save_fits


def test_format_detection():
    """Test la détection de format sur différents types de données"""

    print("="*70)
    print("TEST DE DÉTECTION AUTOMATIQUE DU FORMAT")
    print("="*70)

    # Test 1: Données YUV420 (8-bit)
    print("\n--- Test 1: YUV420 (uint8) ---")
    data_yuv = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    result = detect_source_format(data_yuv)
    print(f"Format détecté: {result['format']}")
    print(f"Bit depth: {result['bit_depth']}")
    print(f"PNG recommandé: {result['png_bit_depth']}-bit")
    print(f"Raison: {result['reason']}")

    # Test 2: Données RAW12 (valeurs 0-4095)
    print("\n--- Test 2: RAW12 (valeurs 0-4095) ---")
    data_raw12 = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
    result = detect_source_format(data_raw12)
    print(f"Format détecté: {result['format']}")
    print(f"Bit depth: {result['bit_depth']}")
    print(f"PNG recommandé: {result['png_bit_depth']}-bit")
    print(f"Raison: {result['reason']}")

    # Test 3: Données RAW16 (valeurs 0-65535)
    print("\n--- Test 3: RAW16 (valeurs 0-65535) ---")
    data_raw16 = np.random.randint(0, 65536, (100, 100, 3), dtype=np.uint16)
    result = detect_source_format(data_raw16)
    print(f"Format détecté: {result['format']}")
    print(f"Bit depth: {result['bit_depth']}")
    print(f"PNG recommandé: {result['png_bit_depth']}-bit")
    print(f"Raison: {result['reason']}")

    # Test 4: Float normalisé (0-1), peu de valeurs = 8-bit
    print("\n--- Test 4: Float normalisé, peu de valeurs (8-bit) ---")
    data_float_8bit = np.random.choice(np.linspace(0, 1, 256), (100, 100, 3))
    result = detect_source_format(data_float_8bit)
    print(f"Format détecté: {result['format']}")
    print(f"Bit depth: {result['bit_depth']}")
    print(f"PNG recommandé: {result['png_bit_depth']}-bit")
    print(f"Raison: {result['reason']}")

    # Test 5: Float normalisé (0-1), beaucoup de valeurs = 12/16-bit
    print("\n--- Test 5: Float normalisé, nombreuses valeurs (12/16-bit) ---")
    data_float_16bit = np.random.random((100, 100, 3)).astype(np.float32)
    result = detect_source_format(data_float_16bit)
    print(f"Format détecté: {result['format']}")
    print(f"Bit depth: {result['bit_depth']}")
    print(f"PNG recommandé: {result['png_bit_depth']}-bit")
    print(f"Raison: {result['reason']}")

    # Test 6: Avec format_hint
    print("\n--- Test 6: Avec format_hint='RAW12' ---")
    result = detect_source_format(data_yuv, format_hint='SRGGB12')
    print(f"Format détecté: {result['format']}")
    print(f"Bit depth: {result['bit_depth']}")
    print(f"PNG recommandé: {result['png_bit_depth']}-bit")
    print(f"Raison: {result['reason']}")


def test_png_auto_save():
    """Test la sauvegarde PNG automatique"""

    print("\n" + "="*70)
    print("TEST DE SAUVEGARDE PNG AUTOMATIQUE")
    print("="*70)

    # Charger les vraies images de test
    raw_path = Path("/media/admin/THKAILAR/Stacks/stack_SRGGB12_mean_20251212_100655.png")
    yuv_path = Path("/media/admin/THKAILAR/Stacks/stack_YUV420_mean_20251212_100812.png")

    if not raw_path.exists() or not yuv_path.exists():
        print("❌ Images de test non trouvées")
        return

    raw_img = cv2.imread(str(raw_path))
    yuv_img = cv2.imread(str(yuv_path))

    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    yuv_img = cv2.cvtColor(yuv_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Test 1: RAW12 avec hint
    print("\n--- Sauvegarde RAW12 avec format_hint ---")
    info = save_png_auto(raw_img, "test_raw12_auto.png", format_hint="SRGGB12")
    print(f"✓ Sauvegardé: {info['path']}")
    print(f"  Format détecté: {info['format_detected']}")
    print(f"  Bit depth utilisé: {info['bit_depth']}-bit")
    print(f"  Taille: {info['file_size_kb']:.1f} KB")
    print(f"  Raison: {info['reason']}")

    # Test 2: YUV420 avec hint
    print("\n--- Sauvegarde YUV420 avec format_hint ---")
    info = save_png_auto(yuv_img, "test_yuv420_auto.png", format_hint="YUV420")
    print(f"✓ Sauvegardé: {info['path']}")
    print(f"  Format détecté: {info['format_detected']}")
    print(f"  Bit depth utilisé: {info['bit_depth']}-bit")
    print(f"  Taille: {info['file_size_kb']:.1f} KB")
    print(f"  Raison: {info['reason']}")

    # Test 3: Sans hint (détection auto)
    print("\n--- Sauvegarde avec détection automatique ---")
    info = save_png_auto(raw_img, "test_auto_detect.png")
    print(f"✓ Sauvegardé: {info['path']}")
    print(f"  Format détecté: {info['format_detected']}")
    print(f"  Bit depth utilisé: {info['bit_depth']}-bit")
    print(f"  Taille: {info['file_size_kb']:.1f} KB")
    print(f"  Raison: {info['reason']}")

    # Test 4: Forcer 16-bit sur YUV420
    print("\n--- Forcer 16-bit sur YUV420 ---")
    info = save_png_auto(yuv_img, "test_yuv420_forced_16bit.png",
                        format_hint="YUV420", force_bit_depth=16)
    print(f"✓ Sauvegardé: {info['path']}")
    print(f"  Format détecté: {info['format_detected']}")
    print(f"  Bit depth utilisé: {info['bit_depth']}-bit (forcé)")
    print(f"  Taille: {info['file_size_kb']:.1f} KB")
    print(f"  Raison: {info['reason']}")


def test_fits_linear():
    """Test la sauvegarde FITS linéaire vs stretched"""

    print("\n" + "="*70)
    print("TEST FITS LINÉAIRE vs STRETCHED")
    print("="*70)

    # Créer données de test (simule un stack RAW)
    np.random.seed(42)
    # Données linéaires (typique d'un stack RAW non traité)
    data_linear = np.random.gamma(2.0, 0.3, (100, 100, 3)).astype(np.float32)
    data_linear = np.clip(data_linear, 0, 1)

    print(f"\nDonnées de test:")
    print(f"  Shape: {data_linear.shape}")
    print(f"  Type: {data_linear.dtype}")
    print(f"  Range: [{data_linear.min():.3f}, {data_linear.max():.3f}]")
    print(f"  Moyenne: {data_linear.mean():.3f}")

    # Sauvegarder en mode linéaire (NOUVEAU - vrai RAW)
    print("\n--- FITS Linéaire (vrai RAW) ---")
    success = save_fits(data_linear, "test_stack_linear.fit",
                       header_data={'NFRAMES': 100, 'EXPTIME': 1.0},
                       linear=True)
    if success:
        size = Path("test_stack_linear.fit").stat().st_size
        print(f"✓ Sauvegardé: test_stack_linear.fit")
        print(f"  Taille: {size / 1024:.1f} KB")
        print(f"  Type: LINEAR (données brutes pour post-traitement)")
        print(f"  ✅ C'est un VRAI RAW - données linéaires non étirées")

    # Sauvegarder en mode stretched (ANCIEN)
    print("\n--- FITS Stretched (legacy) ---")
    success = save_fits(data_linear, "test_stack_stretched.fit",
                       header_data={'NFRAMES': 100, 'EXPTIME': 1.0},
                       linear=False)
    if success:
        size = Path("test_stack_stretched.fit").stat().st_size
        print(f"✓ Sauvegardé: test_stack_stretched.fit")
        print(f"  Taille: {size / 1024:.1f} KB")
        print(f"  Type: STRETCHED (étiré 1-99 percentile)")
        print(f"  ⚠️  Pas optimal pour post-traitement")


def main():
    test_format_detection()
    test_png_auto_save()
    test_fits_linear()

    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    print("""
✅ Nouvelles fonctionnalités implémentées:

1. DÉTECTION AUTOMATIQUE DU FORMAT
   - detect_source_format(data, format_hint)
   - Détecte YUV420 (8-bit) vs RAW12 (12-bit) vs RAW16 (16-bit)
   - Recommande automatiquement PNG 8-bit ou 16-bit

2. SAUVEGARDE PNG ADAPTATIVE
   - save_png_auto(data, path, format_hint, force_bit_depth)
   - YUV420 → PNG 8-bit (léger, suffisant)
   - RAW12/16 → PNG 16-bit (préserve l'information)
   - Option force_bit_depth pour overrider

3. FITS LINÉAIRE (VRAI RAW)
   - save_fits(data, path, linear=True) ← NOUVEAU
   - Sauvegarde données linéaires non étirées
   - Optimal pour post-traitement (PixInsight, Siril, etc.)
   - Header DATATYPE='LINEAR' pour identification

UTILISATION RECOMMANDÉE:

# Pour YUV420
save_png_auto(stack, "output.png", format_hint="YUV420")  # → 8-bit

# Pour RAW12/RAW16
save_png_auto(stack, "output.png", format_hint="RAW12")   # → 16-bit
save_fits(stack, "output.fit", linear=True)                # → Vrai RAW

# Auto-détection (si format inconnu)
save_png_auto(stack, "output.png")  # Détecte automatiquement
    """)


if __name__ == "__main__":
    main()
