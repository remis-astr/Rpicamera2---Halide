#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic complet du pipeline ISP
Trace l'application des paramètres ISP étape par étape
"""

import sys
sys.path.insert(0, '/home/admin/Rpicamera tests/Rpicamera2')

import numpy as np
import json
from pathlib import Path
from libastrostack.isp import ISP, ISPConfig

print("="*70)
print("🔬 DIAGNOSTIC PIPELINE ISP - Test application balance des blancs")
print("="*70)

# 1. Charger config ISP
config_path = Path("/home/admin/Rpicamera tests/Rpicamera2/isp_config_neutral.json")
print(f"\n📁 Chargement config: {config_path.name}")

with open(config_path) as f:
    config_data = json.load(f)

print("\n📊 Paramètres ISP:")
print(f"  wb_red_gain   = {config_data.get('wb_red_gain', 1.0)}")
print(f"  wb_green_gain = {config_data.get('wb_green_gain', 1.0)}")
print(f"  wb_blue_gain  = {config_data.get('wb_blue_gain', 1.0)}")
print(f"  black_level   = {config_data.get('black_level', 0)}")
print(f"  gamma         = {config_data.get('gamma', 1.0)}")
print(f"  contrast      = {config_data.get('contrast', 1.0)}")
print(f"  saturation    = {config_data.get('saturation', 1.0)}")

# 2. Créer ISP et image de test
print("\n🔧 Initialisation ISP...")
isp = ISP.load_config(str(config_path))

# Image de test : blanc pur RGB (toutes valeurs égales)
print("\n🖼️  Création image de test...")
test_image = np.ones((100, 100, 3), dtype=np.float32) * 0.5  # Gris moyen [0.5, 0.5, 0.5]

print(f"  Input:  R={test_image[:,:,0].mean():.3f}, "
      f"G={test_image[:,:,1].mean():.3f}, "
      f"B={test_image[:,:,2].mean():.3f}")

# 3. Test SANS swap_rb
print("\n🔄 Test 1: SANS swap_rb (normal)")
result_no_swap = isp.process(test_image, swap_rb=False)

print(f"  Output: R={result_no_swap[:,:,0].mean():.3f}, "
      f"G={result_no_swap[:,:,1].mean():.3f}, "
      f"B={result_no_swap[:,:,2].mean():.3f}")

print(f"\n  Changements:")
print(f"    Rouge:  {test_image[:,:,0].mean():.3f} → {result_no_swap[:,:,0].mean():.3f} "
      f"({(result_no_swap[:,:,0].mean()/test_image[:,:,0].mean()):.2f}x)")
print(f"    Vert:   {test_image[:,:,1].mean():.3f} → {result_no_swap[:,:,1].mean():.3f} "
      f"({(result_no_swap[:,:,1].mean()/test_image[:,:,1].mean()):.2f}x)")
print(f"    Bleu:   {test_image[:,:,2].mean():.3f} → {result_no_swap[:,:,2].mean():.3f} "
      f"({(result_no_swap[:,:,2].mean()/test_image[:,:,2].mean()):.2f}x)")

# 4. Test AVEC swap_rb (mode utilisé pour RAW)
print("\n🔄 Test 2: AVEC swap_rb=True (mode RAW actuel)")
result_swap = isp.process(test_image, swap_rb=True)

print(f"  Output: R={result_swap[:,:,0].mean():.3f}, "
      f"G={result_swap[:,:,1].mean():.3f}, "
      f"B={result_swap[:,:,2].mean():.3f}")

print(f"\n  Changements:")
print(f"    Rouge:  {test_image[:,:,0].mean():.3f} → {result_swap[:,:,0].mean():.3f} "
      f"({(result_swap[:,:,0].mean()/test_image[:,:,0].mean()):.2f}x)")
print(f"    Vert:   {test_image[:,:,1].mean():.3f} → {result_swap[:,:,1].mean():.3f} "
      f"({(result_swap[:,:,1].mean()/test_image[:,:,1].mean()):.2f}x)")
print(f"    Bleu:   {test_image[:,:,2].mean():.3f} → {result_swap[:,:,2].mean():.3f} "
      f"({(result_swap[:,:,2].mean()/test_image[:,:,2].mean()):.2f}x)")

# 5. Explication swap_rb
print("\n💡 EXPLICATION swap_rb:")
print("-"*70)
print("Sans swap_rb:")
print(f"  Canal Rouge ← wb_red_gain   = {config_data.get('wb_red_gain', 1.0)}")
print(f"  Canal Vert  ← wb_green_gain = {config_data.get('wb_green_gain', 1.0)}")
print(f"  Canal Bleu  ← wb_blue_gain  = {config_data.get('wb_blue_gain', 1.0)}")

print("\nAvec swap_rb=True (RAW):")
print(f"  Canal Rouge ← wb_BLUE_gain  = {config_data.get('wb_blue_gain', 1.0)}")
print(f"  Canal Vert  ← wb_green_gain = {config_data.get('wb_green_gain', 1.0)}")
print(f"  Canal Bleu  ← wb_RED_gain   = {config_data.get('wb_red_gain', 1.0)}")

# 6. Conclusion
print("\n" + "="*70)
print("📋 CONCLUSION")
print("="*70)

if abs(result_swap[:,:,2].mean() - 0.25) < 0.05:  # Bleu devrait être ~0.25 (0.5 * 0.5)
    print("✅ ISP fonctionne correctement!")
    print(f"   Le canal BLEU a bien été multiplié par {config_data.get('wb_red_gain', 1.0)}")
    print(f"   Résultat: 0.5 × {config_data.get('wb_red_gain', 1.0)} = {result_swap[:,:,2].mean():.3f}")
else:
    print("❌ ISP ne semble pas appliquer les gains correctement")
    print(f"   Attendu: ~0.25, Obtenu: {result_swap[:,:,2].mean():.3f}")

print("\n🎨 EFFET VISUEL ATTENDU sur une image réelle:")
print("   Avec wb_red_gain=0.5 et swap_rb=True:")
print("   → Canal BLEU réduit de 50%")
print("   → Image plus JAUNE/ORANGE (moins de bleu)")
print("   → Ciel bleu → Ciel vert/cyan")
print("   → Étoiles bleues → Étoiles jaunâtres")

print("\n⚠️  Si vous ne voyez AUCUN effet sur vos PNG:")
print("   1. Vérifiez que isp_enable = 1 dans PiLCConfig104.txt")
print("   2. Lancez ce test pour confirmer que l'ISP fonctionne")
print("   3. Capturez une nouvelle image en LiveStack/LuckyStack")
print("   4. Si toujours pas d'effet → possible bug dans le pipeline")

print("="*70)
