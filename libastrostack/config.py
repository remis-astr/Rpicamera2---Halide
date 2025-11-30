#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration classes pour libastrostack
"""


class AlignmentMode:
    """Modes d'alignement disponibles"""
    NONE = "none"               # Pas d'alignement (images empilées telles quelles)
    TRANSLATION = "translation"  # X, Y seulement
    ROTATION = "rotation"        # X, Y, rotation (recommandé)
    AFFINE = "affine"           # X, Y, rotation, scale, shear


class StretchMethod:
    """Méthodes d'étirement PNG"""
    LINEAR = "linear"
    ASINH = "asinh"
    LOG = "log"
    SQRT = "sqrt"
    HISTOGRAM = "histogram"
    AUTO = "auto"


class QualityConfig:
    """Configuration du contrôle qualité"""

    def __init__(self):
        # Activation
        self.enable = True

        # Seuils de qualité
        self.max_fwhm = 12.0              # FWHM maximale en pixels
        self.max_ellipticity = 0.4        # Ellipticité max (0=rond, 1=ligne)
        self.min_stars = 10               # Nombre minimum d'étoiles
        self.max_drift = 50.0             # Drift max en pixels
        self.min_sharpness = 0.3          # Netteté minimale (0-1)

        # Seuils d'alignement (pour éviter les transformations aberrantes)
        self.max_rotation = 5.0           # Rotation max en degrés
        self.min_scale = 0.95             # Scale minimum (défaut 0.95)
        self.max_scale = 1.05             # Scale maximum (défaut 1.05)
        self.min_inliers_ratio = 0.3      # Ratio minimum d'inliers RANSAC (30%)
        
        # Paramètres détection étoiles
        self.star_detection_sigma = 5.0   # Seuil sigma pour détection
        self.min_star_separation = 10     # Distance min entre étoiles (px)
        
        # Statistiques (remplies pendant exécution)
        self.rejected_images = []         # Liste des images rejetées
        self.rejection_reasons = {}       # Raisons de rejet par image


class StackingConfig:
    """Configuration principale du stacking"""
    
    def __init__(self):
        # Mode d'alignement
        self.alignment_mode = AlignmentMode.ROTATION
        
        # Configuration qualité
        self.quality = QualityConfig()
        
        # Paramètres PNG
        self.auto_png = True
        self.png_stretch_method = StretchMethod.ASINH
        self.png_stretch_factor = 10.0
        self.png_clip_low = 1.0           # Percentile bas (%)
        self.png_clip_high = 99.5         # Percentile haut (%)
        
        # Paramètres affichage (RPiCamera)
        self.preview_refresh_interval = 5  # Rafraîchir preview toutes les N images
        
        # Paramètres sauvegarde DNG (mode hybride)
        self.save_dng_mode = "accepted"   # "none", "accepted", "all"
        self.output_directory = "/media/admin/THKAILAR/Stacks/"
        self.save_rejected_list = True
        
        # Paramètres alignement avancés
        self.max_stars_alignment = 50     # Nb max étoiles pour alignement
        
        # Statistiques (remplies pendant exécution)
        self.num_stacked = 0
        self.total_exposure = 0.0
        self.noise_level = 0.0
        self.is_color = None
    
    def validate(self):
        """Valide la configuration"""
        errors = []
        
        if self.quality.max_fwhm <= 0:
            errors.append("max_fwhm doit être > 0")
        
        if not 0 <= self.quality.max_ellipticity <= 1:
            errors.append("max_ellipticity doit être entre 0 et 1")
        
        if self.quality.min_stars < 0:
            errors.append("min_stars doit être >= 0 (0 = désactivé)")
        
        if self.png_stretch_factor <= 0:
            errors.append("png_stretch_factor doit être > 0")
        
        if not 0 <= self.png_clip_low < self.png_clip_high <= 100:
            errors.append("Percentiles invalides")
        
        if self.save_dng_mode not in ["none", "accepted", "all"]:
            errors.append("save_dng_mode invalide")
        
        if errors:
            raise ValueError("Configuration invalide: " + ", ".join(errors))
        
        return True
