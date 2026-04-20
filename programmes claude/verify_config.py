#!/usr/bin/env python3
"""
Script de vérification de la configuration RPiCamera2
Vérifie que toutes les valeurs du fichier config sont dans les limites correctes
"""

import sys
import os

# Définir les limites pour chaque paramètre
LIMITS = {
    # Live Stack parameters
    'ls_preview_refresh': (1, 10),
    'ls_alignment_mode': (0, 2),
    'ls_enable_qc': (0, 1),
    'ls_max_fwhm': (0, 300),
    'ls_min_sharpness': (0, 200),
    'ls_max_drift': (0, 5000),
    'ls_min_stars': (0, 20),

    # Stacking parameters
    'ls_stack_method': (0, 4),
    'ls_stack_kappa': (10, 40),
    'ls_stack_iterations': (1, 10),

    # Planetary parameters
    'ls_planetary_enable': (0, 1),
    'ls_planetary_mode': (0, 2),
    'ls_planetary_disk_min': (10, 200),
    'ls_planetary_disk_max': (50, 1000),
    'ls_planetary_threshold': (10, 100),
    'ls_planetary_margin': (5, 50),
    'ls_planetary_ellipse': (0, 1),
    'ls_planetary_window': (0, 2),
    'ls_planetary_upsample': (5, 20),
    'ls_planetary_highpass': (0, 1),
    'ls_planetary_roi_center': (0, 1),
    'ls_planetary_corr': (10, 100),
    'ls_planetary_max_shift': (10, 500),

    # Lucky Imaging parameters
    'ls_lucky_buffer': (10, 200),
    'ls_lucky_keep': (1, 50),
    'ls_lucky_score': (0, 3),
    'ls_lucky_stack': (0, 2),
    'ls_lucky_align': (0, 1),
    'ls_lucky_roi': (20, 100),

    # Metrics parameters
    'focus_method': (0, 4),
    'star_metric': (0, 2),
    'snr_display': (0, 1),
    'metrics_interval': (1, 10),

    # Sensor mode
    'use_native_sensor_mode': (0, 1),
}

def verify_config_file(config_file):
    """Vérifie que toutes les valeurs sont dans les limites correctes"""
    if not os.path.exists(config_file):
        print(f"❌ ERREUR: Fichier {config_file} introuvable")
        return False

    errors = []
    warnings = []

    with open(config_file, 'r') as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line or ':' not in line:
            continue

        parts = line.split(' : ')
        if len(parts) != 2:
            continue

        param_name = parts[0].strip()
        try:
            value = float(parts[1].strip())
            value_int = int(value)
        except ValueError:
            warnings.append(f"Ligne {line_no}: {param_name} = '{parts[1]}' (valeur non numérique)")
            continue

        if param_name in LIMITS:
            min_val, max_val = LIMITS[param_name]
            if value_int < min_val or value_int > max_val:
                errors.append(
                    f"Ligne {line_no}: {param_name} = {value_int} "
                    f"(HORS LIMITES ! Valide: {min_val}-{max_val})"
                )

    # Afficher les résultats
    print(f"\n{'='*70}")
    print(f"VÉRIFICATION DU FICHIER: {config_file}")
    print(f"{'='*70}\n")

    if errors:
        print(f"❌ {len(errors)} ERREUR(S) TROUVÉE(S):\n")
        for error in errors:
            print(f"  • {error}")
        print()
    else:
        print("✅ Aucune erreur trouvée\n")

    if warnings:
        print(f"⚠️  {len(warnings)} AVERTISSEMENT(S):\n")
        for warning in warnings:
            print(f"  • {warning}")
        print()

    print(f"{'='*70}\n")

    return len(errors) == 0

if __name__ == '__main__':
    # Déterminer le chemin du fichier config
    home = os.path.expanduser('~')
    config_file = os.path.join(home, 'PiLCConfig104.txt')

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    success = verify_config_file(config_file)
    sys.exit(0 if success else 1)
