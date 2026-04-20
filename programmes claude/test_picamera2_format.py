#!/usr/bin/env python3
"""
Test rapide pour vérifier le format des données Picamera2
"""

import numpy as np
import cv2
from pathlib import Path

# Chercher un fichier stack PNG existant
stack_dir = Path("/media/admin/THKAILAR/Stacks")
png_files = sorted(stack_dir.glob("stack_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)

if not png_files:
    print("Aucun stack PNG trouvé")
    exit(1)

latest_png = png_files[0]
print(f"Analyse: {latest_png.name}\n")

# Lire le PNG (cv2.imread retourne BGR)
img = cv2.imread(str(latest_png))

print("="*60)
print("ANALYSE DU STACK PNG")
print("="*60)

print(f"\nShape: {img.shape}")
print(f"DType: {img.dtype}")

# Moyennes par canal (ordre BGR d'OpenCV)
b_mean = np.mean(img[:,:,0])
g_mean = np.mean(img[:,:,1])
r_mean = np.mean(img[:,:,2])

print(f"\nMoyennes par canal (après cv2.imread = BGR):")
print(f"  Canal 0 (B): {b_mean:.2f}")
print(f"  Canal 1 (G): {g_mean:.2f}")
print(f"  Canal 2 (R): {r_mean:.2f}")

# Diagnostic
print(f"\nDIAGNOSTIC:")
if r_mean > b_mean * 1.3:
    print("  ✓ Les ROUGES dominent → Cerises ROUGES = CORRECT!")
    print("  → Les données sont CORRECTEMENT en BGR pour OpenCV")
elif b_mean > r_mean * 1.3:
    print("  ✗ Les BLEUS dominent → Cerises BLEUES = PROBLÈME!")
    print("  → Inversion RGB/BGR détectée")
    print("\n  CAUSE PROBABLE:")
    print("  1. Picamera2 retourne RGB")
    print("  2. Le code fait cv2.cvtColor(RGB2BGR)")
    print("  3. cv2.imwrite() attend BGR mais reçoit BGR déjà converti")
    print("  4. Résultat: double inversion = couleurs inversées")
    print("\n  SOLUTION:")
    print("  → RETIRER la conversion RGB2BGR dans session.py:319")
    print("  → OU vérifier que Picamera2 retourne bien RGB")
else:
    print("  ? Balance neutre")

ratio = r_mean / b_mean if b_mean > 0 else 0
print(f"\nRatio R/B: {ratio:.2f}")
print(f"  (Pour cerises rouges, devrait être > 1.5)")
