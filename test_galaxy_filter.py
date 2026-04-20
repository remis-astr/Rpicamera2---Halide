#!/usr/bin/env python3
"""
test_galaxy_filter.py — Test standalone du module galaxy_filter.py
Usage : python3 test_galaxy_filter.py [image.png]
"""

import sys
import os
import time

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

import cv2
import numpy as np
from libastrostack.galaxy_filter import GalaxyEnhancer, GALAXY_PRESETS

OUTPUT_DIR  = '/home/admin/Stack/galaxy_filter_tests'
DEFAULT_IMG = '/home/admin/Stack/stack_image-2026-04-01-09-32-32-765351.png'


def save(path, img, label=''):
    cv2.imwrite(path, img)
    print(f'  {"[" + label + "]":20s} → {os.path.basename(path)}')


def side_by_side(a, b, la='Original', lb='Filtre'):
    h, w = a.shape[:2]
    out = np.zeros((h, w * 2, 3), dtype=np.uint8)
    out[:, :w]  = a
    out[:, w:]  = b
    out[:, w-1:w+1] = (0, 200, 200)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, la, (20, 50),     f, 1.4, (220, 220, 100), 2)
    cv2.putText(out, lb, (w+20, 50),   f, 1.4, (220, 220, 100), 2)
    return out


def timed(ge, img):
    t0 = time.perf_counter()
    r  = ge.process(img)
    ms = (time.perf_counter() - t0) * 1000
    return r, ms


def run_tests(input_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        print(f'ERREUR : impossible de lire {input_path}')
        sys.exit(1)
    h, w = img.shape[:2]
    print(f'\nImage : {w}×{h} px   Sorties : {OUTPUT_DIR}\n')

    ge = GalaxyEnhancer()
    ge.enabled = True

    # ------------------------------------------------------------------
    # 1. Presets de base
    # ------------------------------------------------------------------
    print('=== 1. Presets (Frangi+gamma + USM) ===')
    for i, p in enumerate(GALAXY_PRESETS):
        ge.apply_preset(i)
        r, ms = timed(ge, img)
        fname = os.path.join(OUTPUT_DIR, f'01_preset_{i}_{p["name"]}.png')
        save(fname, r, f'preset{i} {ms:.0f}ms')
        print(f'     sigma[{p["sigma_min"]:.0f}-{p["sigma_max"]:.0f}]'
              f'  enh={p["enhancement"]}  usm={p["usm_strength"]}  '
              f'star_k={p["star_kernel"]}  star_r={p["star_reduction"]}')

    # ------------------------------------------------------------------
    # 2. Star reduction seule (enhancement=0, usm=0)
    # ------------------------------------------------------------------
    print('\n=== 2. Star reduction seule ===')
    ge.apply_preset(1)
    for sk in [3, 5, 7, 9, 13]:
        ge.configure(enhancement=0.0, usm_strength=0.0,
                     star_reduction=0.8, star_kernel=sk)
        r, ms = timed(ge, img)
        fname = os.path.join(OUTPUT_DIR, f'02_star_k{sk:02d}.png')
        save(fname, r, f'kernel={sk} {ms:.0f}ms')

    # ------------------------------------------------------------------
    # 3. Sweep enhancement (gamma boost via Frangi) — USM=0 pour isoler
    # ------------------------------------------------------------------
    print('\n=== 3. Sweep enhancement (Frangi+gamma, USM=0) ===')
    ge.apply_preset(1)
    ge.configure(usm_strength=0.0, star_reduction=0.65)
    for enh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        ge.configure(enhancement=enh)
        r, ms = timed(ge, img)
        fname = os.path.join(OUTPUT_DIR, f'03_enh_{enh:.1f}.png')
        save(fname, r, f'enh={enh} {ms:.0f}ms')

    # ------------------------------------------------------------------
    # 4. Sweep USM strength — enhancement=0 pour isoler
    # ------------------------------------------------------------------
    print('\n=== 4. Sweep USM strength (enhancement=0) ===')
    ge.apply_preset(1)
    ge.configure(enhancement=0.0, star_reduction=0.65)
    for usm in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        ge.configure(usm_strength=usm)
        r, ms = timed(ge, img)
        fname = os.path.join(OUTPUT_DIR, f'04_usm_{usm:.1f}.png')
        save(fname, r, f'usm={usm} {ms:.0f}ms')

    # ------------------------------------------------------------------
    # 5. Combinaisons (enh + USM ensemble)
    # ------------------------------------------------------------------
    print('\n=== 5. Combinaisons enhancement + USM ===')
    ge.apply_preset(1)
    ge.configure(star_reduction=0.65)
    combos = [(1.0, 0.5), (1.5, 1.0), (2.0, 1.0), (2.0, 1.5), (2.5, 1.5), (1.0, 2.0)]
    for enh, usm in combos:
        ge.configure(enhancement=enh, usm_strength=usm)
        r, ms = timed(ge, img)
        fname = os.path.join(OUTPUT_DIR, f'05_enh{enh:.1f}_usm{usm:.1f}.png')
        save(fname, r, f'e={enh} u={usm} {ms:.0f}ms')

    # ------------------------------------------------------------------
    # 6. Comparaisons côte-à-côte
    # ------------------------------------------------------------------
    print('\n=== 6. Comparaisons côte-à-côte ===')
    configs = [
        ('Spirale_enh2_usm1',  1, dict(enhancement=2.0, usm_strength=1.0)),
        ('Tranche_enh2.5_usm1.5', 2, dict(enhancement=2.5, usm_strength=1.5)),
        ('Ellip_enh1.5_usm0.5',   0, dict(enhancement=1.5, usm_strength=0.5)),
    ]
    for name, preset, cfg in configs:
        ge.apply_preset(preset)
        ge.configure(**cfg)
        r, ms = timed(ge, img)
        cmp = side_by_side(img, r, 'Original', name)
        fname = os.path.join(OUTPUT_DIR, f'06_compare_{name}.png')
        save(fname, cmp, f'{ms:.0f}ms')

    # ------------------------------------------------------------------
    # 7. Benchmark n_scales
    # ------------------------------------------------------------------
    print('\n=== 7. Benchmark n_scales ===')
    ge.apply_preset(1)
    ge.configure(enhancement=2.0, usm_strength=1.0, star_reduction=0.65)
    for n in [2, 3, 4, 6]:
        ge.configure(n_scales=n)
        r, ms = timed(ge, img)
        fname = os.path.join(OUTPUT_DIR, f'07_nscales_{n}.png')
        save(fname, r, f'n={n} {ms:.0f}ms')

    print('\nTerminé.')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run_tests(path)
