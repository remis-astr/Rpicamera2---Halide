#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes d'étirement d'histogramme pour PNG
"""

import numpy as np
import cv2
from .config import StretchMethod


def stretch_linear(data, clip_low=1.0, clip_high=99.5):
    """
    Étirement linéaire simple
    
    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    stretched = (data - vmin) / (vmax - vmin)
    return np.clip(stretched, 0, 1)


def stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5):
    """
    Étirement arc-sinus hyperbolique (recommandé pour astro)
    
    Args:
        data: Image (float array)
        factor: Facteur d'étirement (5-50)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    # Normaliser d'abord
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Appliquer asinh
    stretched = np.arcsinh(normalized * factor) / np.arcsinh(factor)
    return stretched


def stretch_log(data, factor=100.0, clip_low=1.0, clip_high=99.5):
    """
    Étirement logarithmique
    
    Args:
        data: Image (float array)
        factor: Facteur d'étirement (10-200)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Log stretch
    stretched = np.log1p(normalized * factor) / np.log1p(factor)
    return stretched


def stretch_sqrt(data, clip_low=1.0, clip_high=99.5):
    """
    Étirement racine carrée (bon pour objets brillants)
    
    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    return np.sqrt(normalized)


def stretch_histogram(data, clip_low=1.0, clip_high=99.5):
    """
    Égalisation d'histogramme
    
    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Convertir en uint8 pour cv2.equalizeHist
    img_8 = (normalized * 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img_8)
    
    return equalized / 255.0


def stretch_auto(data, clip_low=0.1, clip_high=99.9):
    """
    Auto-stretch adaptatif (type SIRIL)

    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)

    Returns:
        Image étirée (0-1)
    """
    # Estimer fond du ciel
    background = np.percentile(data, 5)
    data_clean = np.maximum(data - background, 0)

    # Percentiles adaptatifs
    vmin = np.percentile(data_clean, clip_low)
    vmax = np.percentile(data_clean, clip_high)

    if vmax == vmin:
        return np.zeros_like(data)

    normalized = np.clip((data_clean - vmin) / (vmax - vmin), 0, 1)

    # Appliquer asinh doux
    factor = 5.0
    stretched = np.arcsinh(normalized * factor) / np.arcsinh(factor)

    return stretched


def stretch_mtf(data, midtone=0.2, shadows=0.0, highlights=1.0, clip_low=1.0, clip_high=99.5):
    """
    Midtone Transfer Function (style PixInsight)

    Args:
        data: Image (float array)
        midtone: Point médian (0-1, défaut 0.2)
        shadows: Clipping bas (0-0.5, défaut 0.0)
        highlights: Clipping haut (0.5-1.0, défaut 1.0)
        clip_low: Percentile bas pour normalisation (%)
        clip_high: Percentile haut pour normalisation (%)

    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)

    if vmax == vmin:
        return np.zeros_like(data)

    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)

    # Appliquer shadow / highlight clip
    h_s = max(highlights - shadows, 1e-6)
    clipped = np.clip((normalized - shadows) / h_s, 0, 1)

    # Formule MTF PixInsight : M(x, m) = (m-1)*x / ((2m-1)*x - m)
    m = midtone
    denom = (2 * m - 1) * clipped - m
    mask = np.abs(denom) > 1e-10
    stretched = np.where(mask, (m - 1) * clipped / denom, 0.0)
    stretched = np.where(clipped == 0, 0.0, stretched)
    stretched = np.where(clipped == 1, 1.0, stretched)
    return np.clip(stretched, 0, 1)


def stretch_ghs(data, D=3.0, b=0.13, SP=0.2, LP=0.0, HP=0.0, clip_low=0.0, clip_high=99.97, B=None, normalize_output=True):
    """
    Generalized Hyperbolic Stretch (GHS) - Implémentation IDENTIQUE à RPiCamera2.py
    Conforme à RPiCamera2 et ghsastro.co.uk

    Args:
        data: Image (float array, 0-1)
        D: Stretch factor (0.0 à 10.0) - force de l'étirement (défaut: 3.0)
        b: Local intensity (-5.0 à 20.0) - concentration du contraste (défaut: 0.13)
        SP: Symmetry point (0.0 à 1.0) - point focal du contraste (défaut: 0.2)
        LP: Protect shadows (0.0 à SP) - protection basses lumières (défaut: 0.0)
        HP: Protect highlights (SP à 1.0) - protection hautes lumières (défaut: 0.0)
        clip_low: Percentile bas (%) - normalisation pré-traitement (défaut: 0.0)
        clip_high: Percentile haut (%) - normalisation pré-traitement (défaut: 99.97)
        B: [DEPRECATED] Ancien paramètre, utilisez 'b' à la place
        normalize_output: Si True, normalise le résultat pour couvrir [0,1] (défaut: True)
                         Mettre à False pour RAW où les données sont déjà dans [0,1]

    Returns:
        Image étirée (0-1 float)

    Notes:
        - Normalisation par percentiles appliquée AVANT la transformation GHS
        - Normalisation finale (si activée) garantit que le résultat couvre [0, 1]
        - Pour YUV/RGB: normalize_output=True (données ne couvrent pas [0,1])
        - Pour RAW après ISP: normalize_output=False (données déjà dans [0,1])
    """
    # Rétro-compatibilité: si B est fourni et b n'est pas modifié
    if B is not None and b == 0.13:
        b = B

    # Normalisation par percentiles AVANT GHS (crucial pour images brutes)
    if clip_low > 0 or clip_high < 100:
        vmin = np.percentile(data, clip_low)
        vmax = np.percentile(data, clip_high)

        if vmax > vmin:
            data = np.clip((data - vmin) / (vmax - vmin), 0, 1)

    # Transformation GHS complète (IDENTIQUE à RPiCamera2.py ghs_stretch)
    epsilon = 1e-10
    img_float = np.clip(data.astype(np.float64), epsilon, 1.0 - epsilon)

    if abs(D) < epsilon:
        return img_float.astype(np.float32)

    # Contraintes : 0 <= LP <= SP <= HP <= 1
    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))

    # =========================================================================
    # FONCTIONS DE TRANSFORMATION DE BASE T(x) selon la valeur de b
    # =========================================================================
    def T_base(x, D, b):
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            result = np.log1p(D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            exponent = (b + 1.0) / b
            result = (1.0 - np.power(base, exponent)) / (D * (b + 1.0))
        elif abs(b) < epsilon:
            result = 1.0 - np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = 1.0 - 1.0 / (1.0 + D * x)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = 1.0 - np.power(base, -1.0 / b)

        return result

    def T_prime(x, D, b):
        x = np.asarray(x, dtype=np.float64)

        if abs(b - (-1.0)) < epsilon:
            return D / (1.0 + D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            return np.power(base, 1.0 / b)
        elif abs(b) < epsilon:
            return D * np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            return D / np.power(1.0 + D * x, 2)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            return D * np.power(base, -(1.0 / b + 1.0))

    # =========================================================================
    # CONSTRUCTION DE LA TRANSFORMATION COMPLÈTE (identique RPiCamera2.py)
    # =========================================================================

    # T3(x) = T(x - SP) pour x >= SP (transformation centrée sur SP)
    def T3(x):
        return T_base(x - SP, D, b)

    def T3_prime(x):
        return T_prime(x - SP, D, b)

    # T2(x) = -T(SP - x) pour LP <= x < SP (symétrie autour de SP)
    def T2(x):
        return -T_base(SP - x, D, b)

    def T2_prime(x):
        return T_prime(SP - x, D, b)

    # Valeurs aux bornes pour les segments linéaires
    T2_LP = float(T2(LP))
    T2_prime_LP = float(T2_prime(LP))
    T3_HP = float(T3(HP))
    T3_prime_HP = float(T3_prime(HP))

    # T1(x) = T2'(LP) * (x - LP) + T2(LP) pour x < LP (linéaire - protection shadows)
    def T1(x):
        return T2_prime_LP * (x - LP) + T2_LP

    # T4(x) = T3'(HP) * (x - HP) + T3(HP) pour x >= HP (linéaire - protection highlights)
    def T4(x):
        return T3_prime_HP * (x - HP) + T3_HP

    # Valeurs pour la normalisation (transformation doit aller de 0 à 1)
    T1_0 = float(T1(0.0))
    T4_1 = float(T4(1.0))
    norm_range = T4_1 - T1_0

    if abs(norm_range) < epsilon:
        return img_float.astype(np.float32)  # Pas de transformation possible

    # =========================================================================
    # APPLICATION DE LA TRANSFORMATION PAR RÉGION
    # =========================================================================

    img_stretched = np.zeros_like(img_float)

    # Masques pour les 4 régions
    mask1 = img_float < LP                          # Région 1: 0 <= x < LP (linéaire)
    mask2 = (img_float >= LP) & (img_float < SP)    # Région 2: LP <= x < SP (symétrie)
    mask3 = (img_float >= SP) & (img_float < HP)    # Région 3: SP <= x < HP (principale)
    mask4 = img_float >= HP                         # Région 4: HP <= x <= 1 (linéaire)

    # Appliquer les transformations par région
    if np.any(mask1):
        img_stretched[mask1] = T1(img_float[mask1])
    if np.any(mask2):
        img_stretched[mask2] = T2(img_float[mask2])
    if np.any(mask3):
        img_stretched[mask3] = T3(img_float[mask3])
    if np.any(mask4):
        img_stretched[mask4] = T4(img_float[mask4])

    # NORMALISATION FINALE: garantit que le résultat couvre [0, 1]
    # Pour YUV/RGB: nécessaire car données ne couvrent pas [0,1]
    # Pour RAW après ISP: désactivé car données déjà dans [0,1]
    if normalize_output:
        img_stretched = (img_stretched - T1_0) / norm_range

    return np.clip(img_stretched, 0.0, 1.0).astype(np.float32)


def apply_stretch(data, method=StretchMethod.ASINH, **params):
    """
    Applique la méthode d'étirement spécifiée

    Args:
        data: Image à étirer (float array)
        method: Méthode ('off', 'linear', 'asinh', 'log', 'sqrt', 'histogram', 'auto', 'ghs') ou StretchMethod enum
        **params: Paramètres (factor, clip_low, clip_high, ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP)

    Returns:
        Image étirée (0-1)
    """
    # Convertir string en enum si nécessaire
    if isinstance(method, str):
        method_map = {
            'off': StretchMethod.OFF,
            'linear': StretchMethod.LINEAR,
            'asinh': StretchMethod.ASINH,
            'log': StretchMethod.LOG,
            'sqrt': StretchMethod.SQRT,
            'histogram': StretchMethod.HISTOGRAM,
            'auto': StretchMethod.AUTO,
            'ghs': StretchMethod.GHS,
            'mtf': StretchMethod.MTF,
        }
        method = method_map.get(method.lower(), StretchMethod.ASINH)

    if method == StretchMethod.OFF:
        # Pas de stretch - juste clip [0, 1] SANS normalisation
        # IMPORTANT: Ne PAS normaliser pour préserver l'histogramme original
        return np.clip(data, 0, 1)

    elif method == StretchMethod.LINEAR:
        return stretch_linear(
            data,
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.ASINH:
        return stretch_asinh(
            data,
            factor=params.get('factor', 10.0),
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.LOG:
        return stretch_log(
            data,
            factor=params.get('factor', 100.0),
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.SQRT:
        return stretch_sqrt(
            data,
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.HISTOGRAM:
        return stretch_histogram(
            data,
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.GHS:
        return stretch_ghs(
            data,
            D=params.get('ghs_D', 3.0),
            b=params.get('ghs_b', params.get('ghs_B', 0.13)),  # Rétro-compatibilité avec ghs_B
            SP=params.get('ghs_SP', 0.2),
            LP=params.get('ghs_LP', 0.0),
            HP=params.get('ghs_HP', 0.0),
            clip_low=params.get('clip_low', 0.0),
            clip_high=params.get('clip_high', 99.97),
            normalize_output=params.get('normalize_output', True)  # False pour RAW après ISP
        )

    elif method == StretchMethod.MTF:
        return stretch_mtf(
            data,
            midtone=params.get('mtf_midtone', 0.2),
            shadows=params.get('mtf_shadows', 0.0),
            highlights=params.get('mtf_highlights', 1.0),
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    else:  # AUTO
        return stretch_auto(
            data,
            clip_low=params.get('clip_low', 0.1),
            clip_high=params.get('clip_high', 99.9)
        )
