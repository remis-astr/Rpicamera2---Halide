#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
libastrostack - Bibliothèque de live stacking pour astrophotographie
=====================================================================

API principale pour l'intégration dans RPiCamera.py et autres applications.

Usage:
    from libastrostack import LiveStackSession, StackingConfig
    
    config = StackingConfig()
    session = LiveStackSession(config)
    session.start()
    
    result = session.process_image_data(camera_array)
    session.save_result("output.fit")
    session.stop()
"""

__version__ = "2.0.0"
__author__ = "AstroStack Team"

# Imports publics
from .config import (
    StackingConfig,
    AlignmentMode,
    QualityConfig,
    StretchMethod
)

from .session import LiveStackSession

# API simplifiée
__all__ = [
    'LiveStackSession',
    'StackingConfig',
    'AlignmentMode',
    'QualityConfig',
    'StretchMethod',
]
