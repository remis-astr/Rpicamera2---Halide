#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la configuration ISP IMX585 convertie depuis libcamera
"""

import numpy as np
from libastrostack.isp import ISP

print("=" * 60)
print("TEST CONFIGURATION ISP IMX585")
print("=" * 60)

# Charger la config
print("\n1. Chargement de la configuration...")
isp = ISP.load_config('isp_config_imx585.json')

print("\n2. Paramètres ISP chargés:")
print(f"   • Black level: {isp.config.black_level}")
print(f"   • White balance: R={isp.config.wb_red_gain:.3f}, "
      f"G={isp.config.wb_green_gain:.3f}, B={isp.config.wb_blue_gain:.3f}")
print(f"   • Gamma: {isp.config.gamma:.2f}")
print(f"   • Contraste: {isp.config.contrast:.2f}")
print(f"   • Saturation: {isp.config.saturation:.2f}")
print(f"   • CCM présente: {isp.config.ccm is not None and not np.allclose(isp.config.ccm, np.eye(3))}")

if isp.config.ccm is not None:
    print(f"\n   Color Correction Matrix:")
    for row in isp.config.ccm:
        print(f"   [{row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f}]")

print(f"\n3. Métadonnées de calibration:")
for key, value in isp.config.calibration_info.items():
    print(f"   • {key}: {value}")

print("\n4. Test de traitement sur image synthétique...")
# Créer une image RAW synthétique (12-bit range)
test_raw = np.random.randint(3200, 4096, (100, 100, 3), dtype=np.uint16)
print(f"   Image RAW: shape={test_raw.shape}, dtype={test_raw.dtype}, "
      f"range=[{test_raw.min()}, {test_raw.max()}]")

# Traiter avec ISP
processed = isp.process(test_raw, return_uint16=True)
print(f"   Image ISP: shape={processed.shape}, dtype={processed.dtype}, "
      f"range=[{processed.min()}, {processed.max()}]")

print("\n✅ Configuration ISP IMX585 validée !")
print("\n" + "=" * 60)
print("UTILISATION DANS RPICAMERA2")
print("=" * 60)
print("\nDans PiLCConfig104.txt, ajoutez:")
print("   isp_enable=True")
print("   isp_config_path=isp_config_imx585.json")
print("   video_format=raw12")
print("\nOu pour tester différentes températures de couleur:")
print("   python3 convert_libcamera_isp.py imx585_lowlight.json \\")
print("       -o isp_daylight.json -t 6500  # Lumière du jour")
print("   python3 convert_libcamera_isp.py imx585_lowlight.json \\")
print("       -o isp_tungsten.json -t 3000  # Tungstène")
print()
