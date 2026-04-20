#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convertisseur de configuration ISP libcamera → libastrostack
Extrait les paramètres ISP d'un fichier JSON libcamera (imx585_lowlight.json)
et les convertit au format ISPConfig de libastrostack
"""

import json
import numpy as np
from pathlib import Path
import sys

# Importer notre classe ISPConfig
sys.path.insert(0, str(Path(__file__).parent))
from libastrostack.isp import ISPConfig, ISP


def extract_black_level(libcam_config):
    """Extrait le black level"""
    for algo in libcam_config.get('algorithms', []):
        if 'rpi.black_level' in algo:
            return algo['rpi.black_level']['black_level']
    return 64  # Valeur par défaut


def extract_ccm_matrices(libcam_config):
    """Extrait les Color Correction Matrices"""
    for algo in libcam_config.get('algorithms', []):
        if 'rpi.ccm' in algo:
            ccms = algo['rpi.ccm'].get('ccms', [])
            return ccms
    return []


def extract_gamma_curve(libcam_config):
    """Extrait et analyse la courbe gamma"""
    for algo in libcam_config.get('algorithms', []):
        if 'rpi.contrast' in algo:
            gamma_curve = algo['rpi.contrast'].get('gamma_curve', [])
            return gamma_curve
    return []


def extract_wb_curve(libcam_config):
    """Extrait la courbe de balance des blancs"""
    for algo in libcam_config.get('algorithms', []):
        if 'rpi.awb' in algo:
            ct_curve = algo['rpi.awb'].get('ct_curve', [])
            return ct_curve
    return []


def estimate_gamma_from_curve(gamma_curve):
    """
    Estime une valeur de gamma à partir de la courbe libcamera
    La courbe est : [x0, y0, x1, y1, ...]
    où x est l'entrée (0-65535) et y la sortie (0-65535)
    """
    if not gamma_curve or len(gamma_curve) < 4:
        return 2.2

    # Convertir en paires (x, y)
    points = [(gamma_curve[i], gamma_curve[i+1]) for i in range(0, len(gamma_curve), 2)]

    # Normaliser (0-1)
    points = [(x/65535, y/65535) for x, y in points]

    # Trouver le gamma qui approxime le mieux
    # On teste autour du point milieu (0.5)
    mid_points = [p for p in points if 0.3 < p[0] < 0.7]
    if not mid_points:
        return 2.2

    # Prendre le point le plus proche de 0.5 en entrée
    mid_point = min(mid_points, key=lambda p: abs(p[0] - 0.5))
    x_mid, y_mid = mid_point

    # Calculer gamma: y = x^(1/gamma)
    # donc gamma = log(x) / log(y)
    if y_mid > 0 and x_mid > 0:
        gamma = np.log(x_mid) / np.log(y_mid)
        # Limiter à une plage raisonnable
        gamma = max(1.0, min(3.0, gamma))
        return gamma

    return 2.2


def select_best_ccm(ccm_list, target_ct=5500):
    """
    Sélectionne la meilleure CCM pour une température de couleur cible
    target_ct: Température de couleur cible en Kelvin (5500K = lumière du jour)
    """
    if not ccm_list:
        return np.eye(3, dtype=np.float32)

    # Trouver la CCM la plus proche de la température cible
    best_ccm = min(ccm_list, key=lambda c: abs(c['ct'] - target_ct))

    # Convertir la liste en matrice 3x3
    ccm_values = best_ccm['ccm']
    ccm_matrix = np.array(ccm_values, dtype=np.float32).reshape(3, 3)

    return ccm_matrix, best_ccm['ct']


def estimate_wb_from_ct_curve(ct_curve, target_ct=5500):
    """
    Estime les gains de balance des blancs à partir de la courbe ct
    ct_curve format: [ct0, r0, b0, ct1, r1, b1, ...]
    """
    if not ct_curve or len(ct_curve) < 6:
        return 1.0, 1.0, 1.0

    # Convertir en triplets (ct, r, b)
    points = [(ct_curve[i], ct_curve[i+1], ct_curve[i+2])
              for i in range(0, len(ct_curve), 3)]

    # Trouver les deux points encadrant notre température cible
    points_sorted = sorted(points, key=lambda p: p[0])

    # Si target_ct est hors plage, prendre le point le plus proche
    if target_ct <= points_sorted[0][0]:
        ct, r_gain, b_gain = points_sorted[0]
    elif target_ct >= points_sorted[-1][0]:
        ct, r_gain, b_gain = points_sorted[-1]
    else:
        # Interpolation linéaire
        for i in range(len(points_sorted) - 1):
            ct1, r1, b1 = points_sorted[i]
            ct2, r2, b2 = points_sorted[i+1]
            if ct1 <= target_ct <= ct2:
                # Interpoler
                t = (target_ct - ct1) / (ct2 - ct1)
                r_gain = r1 + t * (r2 - r1)
                b_gain = b1 + t * (b2 - b1)
                break

    # Inverser les gains (libcamera utilise des gains inverses)
    # et normaliser par rapport au vert (=1.0)
    return r_gain, 1.0, b_gain


def convert_libcamera_to_isp(libcam_json_path, target_ct=5500):
    """
    Convertit un fichier JSON libcamera en ISPConfig

    Args:
        libcam_json_path: Chemin vers le fichier JSON libcamera
        target_ct: Température de couleur cible (5500K = lumière du jour)

    Returns:
        ISPConfig configuré
    """
    print(f"\n{'='*60}")
    print(f"CONVERSION ISP LIBCAMERA → LIBASTROSTACK")
    print(f"{'='*60}\n")

    # Lire le fichier JSON
    with open(libcam_json_path, 'r') as f:
        libcam_config = json.load(f)

    print(f"📄 Fichier: {Path(libcam_json_path).name}")
    print(f"🌡️  Température cible: {target_ct}K\n")

    # Créer la config ISP
    isp_config = ISPConfig()

    # 1. Black level
    black_level = extract_black_level(libcam_config)
    isp_config.black_level = black_level
    print(f"⚫ Black level: {black_level}")

    # 2. Color Correction Matrix
    ccm_list = extract_ccm_matrices(libcam_config)
    if ccm_list:
        ccm_matrix, selected_ct = select_best_ccm(ccm_list, target_ct)
        isp_config.ccm = ccm_matrix
        print(f"🎨 CCM sélectionnée pour {selected_ct}K:")
        print(f"   {ccm_matrix[0]}")
        print(f"   {ccm_matrix[1]}")
        print(f"   {ccm_matrix[2]}")
    else:
        print(f"⚠️  Aucune CCM trouvée, utilisation de l'identité")

    # 3. Gamma
    gamma_curve = extract_gamma_curve(libcam_config)
    if gamma_curve:
        gamma = estimate_gamma_from_curve(gamma_curve)
        isp_config.gamma = gamma
        print(f"📈 Gamma estimé: {gamma:.2f}")
    else:
        print(f"⚠️  Courbe gamma non trouvée, utilisation de 2.2")

    # 4. White Balance
    wb_curve = extract_wb_curve(libcam_config)
    if wb_curve:
        r_gain, g_gain, b_gain = estimate_wb_from_ct_curve(wb_curve, target_ct)
        isp_config.wb_red_gain = r_gain
        isp_config.wb_green_gain = g_gain
        isp_config.wb_blue_gain = b_gain
        print(f"⚖️  Balance des blancs ({target_ct}K):")
        print(f"   R: {r_gain:.3f}, G: {g_gain:.3f}, B: {b_gain:.3f}")
    else:
        print(f"⚠️  Courbe WB non trouvée, utilisation de gains neutres")

    # 5. Contraste et saturation (valeurs par défaut raisonnables)
    isp_config.contrast = 1.1
    isp_config.saturation = 1.2
    print(f"🎭 Contraste: {isp_config.contrast}")
    print(f"🌈 Saturation: {isp_config.saturation}")

    # Métadonnées
    isp_config.calibration_info = {
        'source': str(Path(libcam_json_path).name),
        'target_ct': target_ct,
        'method': 'libcamera_conversion',
        'version': libcam_config.get('version', 'unknown'),
        'target': libcam_config.get('target', 'unknown')
    }

    return isp_config


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Convertit un fichier ISP libcamera en format libastrostack'
    )
    parser.add_argument('input', type=str,
                       help='Fichier JSON libcamera (ex: imx585_lowlight.json)')
    parser.add_argument('-o', '--output', type=str, default='isp_config_converted.json',
                       help='Fichier de sortie (défaut: isp_config_converted.json)')
    parser.add_argument('-t', '--temperature', type=int, default=5500,
                       help='Température de couleur cible en Kelvin (défaut: 5500)')

    args = parser.parse_args()

    # Conversion
    isp_config = convert_libcamera_to_isp(args.input, args.temperature)

    # Sauvegarder
    isp = ISP(isp_config)
    isp.save_config(args.output)

    print(f"\n✅ Configuration ISP sauvegardée: {args.output}")
    print(f"\n{'='*60}")
    print(f"UTILISATION")
    print(f"{'='*60}\n")
    print(f"Dans PiLCConfig104.txt:")
    print(f"  isp_enable=True")
    print(f"  isp_config_path={args.output}")
    print(f"\nOu dans le code Python:")
    print(f"  from libastrostack.isp import ISP")
    print(f"  isp = ISP.load_config('{args.output}')")
    print(f"  processed = isp.process(raw_image)")
    print()


if __name__ == '__main__':
    main()
