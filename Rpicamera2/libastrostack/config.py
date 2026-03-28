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
    OFF = "off"                     # Pas de stretch
    LINEAR = "linear"
    ASINH = "asinh"
    LOG = "log"
    SQRT = "sqrt"
    HISTOGRAM = "histogram"
    AUTO = "auto"
    GHS = "ghs"                     # Generalized Hyperbolic Stretch
    MTF = "mtf"                     # Midtone Transfer Function (PixInsight)


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
        self.min_inliers_ratio = 0.5      # Ratio minimum d'inliers RANSAC (50%)
        
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

        # Soustraction black level RAW (indépendante de l'ISP software)
        # Valeur en ADU 12-bit natif du capteur. 0 = désactivé.
        # IMX585 standard : 256 ADU (= 6.25% de 4096)
        self.raw_black_level = 256

        # Soustraction BL per-canal Bayer (avant débayérisation dans debayer_raw_array)
        # Élimine le FPN 2×2 causé par les déséquilibres inter-canal (R, G1, G2, B).
        # None = désactivé (comportement historique, BL global appliqué après débayérisation).
        # tuple (R, G1, G2, B) en ADU 12-bit natif pour correction manuelle.
        self.bl_per_channel = None

        # Si True, estimer automatiquement les 4 BL par percentile bas sur chaque
        # sous-canal Bayer (robuste au fond sombre dominant en imagerie astronomique).
        # Ignoré si bl_per_channel est fourni.
        self.bl_auto_estimate = False

        # Suppression de gradient de fond (lumière parasite, vignetage)
        # Algorithme par grille de médiane/percentile (mesh-based background estimation)
        self.gradient_removal = False
        self.gradient_removal_tiles = 8   # Taille de la grille n×n (8 = grille 8×8)
        self.awb_auto = False             # AWB auto (grey-world) pour preview stack RAW12

        # Paramètres ISP (Image Signal Processor)
        self.isp_enable = False            # Activer l'ISP (pour RAW12/16)
        self.isp_config_path = None        # Chemin vers config ISP (ou None = auto-calibration)
        self.isp_calibration_frames = None # Images de référence pour calibration (RAW, YUV)
        self.video_format = None           # Format vidéo source: 'yuv420', 'raw12', 'raw16', None=auto

        # Calibration automatique ISP (nouveau)
        self.isp_auto_calibrate_method = 'histogram_peaks'  # Méthode: 'none', 'histogram_peaks', 'gray_world'
        self.isp_auto_calibrate_after = 10       # Calibrer après N frames stackées (0 = désactivé)
        self.isp_recalibrate_interval = 0        # Recalibrer tous les N frames (0 = jamais - calibration unique)
        self.isp_auto_update_only_wb = True      # Si True, ne met à jour que les gains RGB (préserve gamma, contrast, etc.)

        # Paramètres PNG
        self.auto_png = True
        self.png_bit_depth = None          # None=auto (détecté), 8, ou 16
        self.png_stretch_method = StretchMethod.ASINH
        self.png_stretch_factor = 10.0
        self.png_clip_low = 1.0           # Percentile bas (%)
        self.png_clip_high = 99.5         # Percentile haut (%)

        # Paramètres GHS (Generalized Hyperbolic Stretch) - 5 paramètres
        self.ghs_D = 3.0                  # Stretch factor (0.0 à 10.0) - force de l'étirement
        self.ghs_b = 0.13                 # Local intensity (-5.0 à 20.0) - concentration du contraste
        self.ghs_SP = 0.2                 # Symmetry point (0.0 à 1.0) - point focal du contraste
        self.ghs_LP = 0.0                 # Protect shadows (0.0 à SP) - protection basses lumières
        self.ghs_HP = 0.0                 # Protect highlights (SP à 1.0) - protection hautes lumières

        # NOUVEAU: Ajustement automatique SP pour RAW
        # Compense la différence entre preview hardware ISP et ISP software
        # En mode RAW, ajuste SP au pic d'histogramme réel des données
        self.ghs_auto_adjust_sp = True    # Auto-ajuster SP pour RAW (recommandé)

        # Paramètres FITS
        self.fits_linear = True            # Sauvegarder FITS en linéaire (True=RAW, False=stretched)

        # Paramètres affichage (RPiCamera)
        self.preview_refresh_interval = 5  # Rafraîchir preview toutes les N images

        # Paramètres sauvegarde DNG (mode hybride)
        self.save_dng_mode = "accepted"   # "none", "accepted", "all"
        self.output_directory = "/home/admin/stacks/"
        self.save_rejected_list = True

        # Paramètres alignement avancés
        self.max_stars_alignment = 50     # Nb max étoiles pour alignement

        # Paramètres méthode de stacking
        self.stacking_method = 'mean'     # 'mean', 'kappa_sigma', 'winsorized', 'median'
        self.stacking_kappa = 2.5         # Kappa pour sigma-clipping (kappa_sigma/winsorized)

        # Limite de frames (0 = illimité)
        self.max_frames = 0

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
