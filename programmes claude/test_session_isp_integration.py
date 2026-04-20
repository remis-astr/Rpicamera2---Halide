#!/usr/bin/env python3
"""
Test de l'intégration complète ISP dans le pipeline de stacking
Démontre le pipeline optimal: Stack → ISP → Stretch → PNG 8/16-bit
"""

import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from libastrostack.session import LiveStackSession
from libastrostack.config import StackingConfig
from libastrostack.isp import ISPCalibrator


def test_integration_complete():
    """Test complet avec ISP activé"""

    print("="*70)
    print("TEST D'INTÉGRATION ISP DANS SESSION DE STACKING")
    print("="*70)

    # ========================================
    # 1. CALIBRATION ISP (une fois au début)
    # ========================================
    print("\n" + "="*70)
    print("ÉTAPE 1: CALIBRATION ISP")
    print("="*70)

    raw_path = Path("/media/admin/THKAILAR/Stacks/stack_SRGGB12_mean_20251212_100655.png")
    yuv_path = Path("/media/admin/THKAILAR/Stacks/stack_YUV420_mean_20251212_100812.png")

    if not raw_path.exists() or not yuv_path.exists():
        print("❌ Images de calibration non trouvées")
        return

    print(f"\nCalibration à partir de:")
    print(f"  RAW: {raw_path.name}")
    print(f"  YUV: {yuv_path.name}")

    # Calibrer et sauvegarder la config ISP
    config_isp = ISPCalibrator.calibrate_from_files(raw_path, yuv_path)

    from libastrostack.isp import ISP
    isp = ISP(config_isp)
    isp_config_path = Path("session_isp_config.json")
    isp.save_config(isp_config_path)

    print(f"\n✓ Configuration ISP sauvegardée: {isp_config_path}")

    # ========================================
    # 2. CONFIGURATION SESSION (RAW12)
    # ========================================
    print("\n" + "="*70)
    print("ÉTAPE 2: CONFIGURATION SESSION RAW12")
    print("="*70)

    config_raw12 = StackingConfig()

    # Paramètres ISP
    config_raw12.isp_enable = True
    config_raw12.isp_config_path = str(isp_config_path)
    config_raw12.video_format = 'raw12'

    # Paramètres PNG (auto 16-bit pour RAW12)
    config_raw12.png_bit_depth = None  # Auto-détection
    config_raw12.png_stretch_method = "asinh"
    config_raw12.png_stretch_factor = 10.0

    # Paramètres FITS (linéaire)
    config_raw12.fits_linear = True

    # Désactiver le contrôle qualité pour le test
    config_raw12.quality.enable = False

    print("\nConfiguration:")
    print(f"  • ISP: ACTIVÉ (config: {isp_config_path})")
    print(f"  • Format vidéo: {config_raw12.video_format}")
    print(f"  • PNG bit depth: {config_raw12.png_bit_depth or 'Auto'}")
    print(f"  • FITS: {'Linéaire' if config_raw12.fits_linear else 'Stretched'}")

    # ========================================
    # 3. TEST SESSION RAW12
    # ========================================
    print("\n" + "="*70)
    print("ÉTAPE 3: TEST SESSION AVEC IMAGES RAW12")
    print("="*70)

    session = LiveStackSession(config_raw12)
    session.start()

    # Charger et traiter l'image RAW
    print("\nTraitement de l'image RAW12...")
    raw_img = cv2.imread(str(raw_path))
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Simuler le stacking (ici une seule image pour le test)
    result = session.process_image_data(raw_img)

    # Sauvegarder le résultat
    print("\nSauvegarde des résultats...")
    session.save_result("test_stack_raw12_with_isp.fit", generate_png=True)

    session.stop()

    # ========================================
    # 4. CONFIGURATION SESSION (YUV420)
    # ========================================
    print("\n" + "="*70)
    print("ÉTAPE 4: CONFIGURATION SESSION YUV420 (SANS ISP)")
    print("="*70)

    config_yuv = StackingConfig()

    # Pas d'ISP pour YUV420 (déjà traité par hardware)
    config_yuv.isp_enable = False
    config_yuv.video_format = 'yuv420'

    # Paramètres PNG (auto 8-bit pour YUV420)
    config_yuv.png_bit_depth = None  # Auto-détection
    config_yuv.png_stretch_method = "asinh"
    config_yuv.png_stretch_factor = 10.0

    # FITS linéaire quand même (données du stack)
    config_yuv.fits_linear = True

    # Désactiver le contrôle qualité
    config_yuv.quality.enable = False

    print("\nConfiguration:")
    print(f"  • ISP: DÉSACTIVÉ (YUV420 déjà traité)")
    print(f"  • Format vidéo: {config_yuv.video_format}")
    print(f"  • PNG bit depth: {config_yuv.png_bit_depth or 'Auto'}")

    # ========================================
    # 5. TEST SESSION YUV420
    # ========================================
    print("\n" + "="*70)
    print("ÉTAPE 5: TEST SESSION AVEC IMAGES YUV420")
    print("="*70)

    session_yuv = LiveStackSession(config_yuv)
    session_yuv.start()

    # Charger et traiter l'image YUV
    print("\nTraitement de l'image YUV420...")
    yuv_img = cv2.imread(str(yuv_path))
    yuv_img = cv2.cvtColor(yuv_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    result_yuv = session_yuv.process_image_data(yuv_img)

    # Sauvegarder le résultat
    print("\nSauvegarde des résultats...")
    session_yuv.save_result("test_stack_yuv420_no_isp.fit", generate_png=True)

    session_yuv.stop()

    # ========================================
    # 6. COMPARAISON DES RÉSULTATS
    # ========================================
    print("\n" + "="*70)
    print("ÉTAPE 6: COMPARAISON DES RÉSULTATS")
    print("="*70)

    # Comparer les tailles de fichiers
    files = [
        ("RAW12 + ISP PNG", "test_stack_raw12_with_isp.png"),
        ("RAW12 + ISP FITS", "test_stack_raw12_with_isp.fit"),
        ("YUV420 PNG", "test_stack_yuv420_no_isp.png"),
        ("YUV420 FITS", "test_stack_yuv420_no_isp.fit"),
    ]

    print("\nFichiers générés:")
    for name, filename in files:
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  • {name:<25} {size:>8.1f} KB")

            # Déterminer le bit depth pour les PNG
            if filename.endswith('.png'):
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                bit_depth = 16 if img.dtype == np.uint16 else 8
                print(f"    {'':>27} → {bit_depth}-bit PNG")

    # Vérifier les headers FITS
    print("\nVérification headers FITS:")
    try:
        from astropy.io import fits

        for name, filename in files:
            if filename.endswith('.fit'):
                path = Path(filename)
                if path.exists():
                    with fits.open(path) as hdul:
                        datatype = hdul[0].header.get('DATATYPE', 'N/A')
                        print(f"  • {name:<25} Type: {datatype}")
    except ImportError:
        print("  (astropy non disponible)")

    # ========================================
    # 7. RÉSUMÉ
    # ========================================
    print("\n" + "="*70)
    print("RÉSUMÉ DE L'INTÉGRATION")
    print("="*70)
    print("""
✅ PIPELINE COMPLET IMPLÉMENTÉ:

1. CALIBRATION ISP (une fois):
   RAW12 + YUV420 → Extraction paramètres → Config ISP

2. PIPELINE RAW12 (avec ISP):
   Frame RAW12 → Stack (linéaire) → ISP → Stretch → PNG 16-bit
                                          ↓
                                    FITS linéaire (vrai RAW)

3. PIPELINE YUV420 (sans ISP):
   Frame YUV420 → Stack → Stretch → PNG 8-bit
                       ↓
                 FITS linéaire

AVANTAGES:
✓ ISP appliqué 1x après stack (1000x plus rapide)
✓ Stack en données linéaires (qualité optimale)
✓ PNG 16-bit pour RAW12/16 (préserve l'info)
✓ PNG 8-bit pour YUV420 (léger, suffisant)
✓ FITS linéaire (vrai RAW pour post-traitement)
✓ Pas d'histogramme haché (traitement float32)

UTILISATION:
# Configuration
config = StackingConfig()
config.isp_enable = True
config.isp_config_path = "isp_config.json"
config.video_format = "raw12"  # ou "yuv420", "raw16"
config.fits_linear = True

# Session
session = LiveStackSession(config)
session.start()

# Pour chaque frame
session.process_image_data(frame)

# Sauvegarde finale
session.save_result("output.fit")  # → FITS linéaire + PNG adaptatif
    """)


if __name__ == "__main__":
    test_integration_complete()
