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


def apply_stretch(data, method=StretchMethod.ASINH, **params):
    """
    Applique la méthode d'étirement spécifiée
    
    Args:
        data: Image à étirer (float array)
        method: Méthode ('linear', 'asinh', 'log', 'sqrt', 'histogram', 'auto')
        **params: Paramètres (factor, clip_low, clip_high)
    
    Returns:
        Image étirée (0-1)
    """
    if method == StretchMethod.LINEAR:
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
    
    else:  # AUTO
        return stretch_auto(
            data,
            clip_low=params.get('clip_low', 0.1),
            clip_high=params.get('clip_high', 99.9)
        )
