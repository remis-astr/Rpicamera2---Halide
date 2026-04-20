#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration étendue pour libastrostack
========================================

Ce module étend config.py avec les nouvelles options pour:
- Méthodes de stacking avancées (Kappa-Sigma, Winsorized, etc.)
- Drizzle pour super-résolution
- Configuration unifiée

Usage:
    from libastrostack.config_advanced import AdvancedStackingConfig
    
    config = AdvancedStackingConfig()
    config.stacking.method = StackMethod.KAPPA_SIGMA
    config.stacking.kappa = 2.5
    config.drizzle.enable = True
    config.drizzle.scale = 2.0

Auteur: libastrostack Team
Version: 1.0.0
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# Enums
# =============================================================================

class StackMethod(Enum):
    """Méthodes de combinaison disponibles"""
    MEAN = "mean"                   # Moyenne simple (rapide)
    MEDIAN = "median"               # Médiane (robuste, -20% SNR)
    KAPPA_SIGMA = "kappa_sigma"     # Sigma-clipping itératif (recommandé)
    WINSORIZED = "winsorized"       # Winsorized sigma-clipping
    WEIGHTED = "weighted"           # Moyenne pondérée par qualité


class DrizzleKernel(Enum):
    """Types de kernel pour drizzle"""
    POINT = "point"       # Rapide, moins précis
    SQUARE = "square"     # Bon compromis (recommandé)
    GAUSSIAN = "gaussian" # Plus lent, meilleur lissage


class AlignmentMode(Enum):
    """Modes d'alignement disponibles"""
    NONE = "none"               # Pas d'alignement
    TRANSLATION = "translation" # X, Y seulement (étoiles)
    ROTATION = "rotation"       # X, Y + rotation (étoiles, recommandé)
    AFFINE = "affine"           # Complet (étoiles)
    # Modes planétaires (Soleil, Lune, planètes)
    DISK = "disk"               # Alignement sur le limbe du disque
    SURFACE = "surface"         # Corrélation de phase sur les détails
    PLANETARY = "planetary"     # Alias pour SURFACE (rétrocompatibilité)


class PlanetaryMode(Enum):
    """Modes d'alignement planétaire (sous-modes de DISK/SURFACE)"""
    DISK = "disk"           # Alignement sur le limbe du disque
    SURFACE = "surface"     # Corrélation de phase sur les détails
    HYBRID = "hybrid"       # Disque + surface combinés


# Importer les enums de lucky_imaging comme source unique de vérité.
# Évite la duplication et les risques de désynchronisation entre les deux modules.
from .lucky_imaging import ScoreMethod as LuckyScoreMethod, StackMethod as LuckyStackMethod


class StretchMethod(Enum):
    """Méthodes d'étirement (copie de config.py)"""
    OFF = "off"                     # Pas de stretch
    LINEAR = "linear"
    ASINH = "asinh"
    LOG = "log"
    SQRT = "sqrt"
    HISTOGRAM = "histogram"
    AUTO = "auto"
    GHS = "ghs"                     # Generalized Hyperbolic Stretch
    MTF = "mtf"                     # Midtone Transfer Function (PixInsight)


# =============================================================================
# Dataclasses de configuration
# =============================================================================

@dataclass
class StackingMethodConfig:
    """Configuration des méthodes de stacking avancées"""
    
    # Méthode de base
    method: StackMethod = StackMethod.KAPPA_SIGMA
    
    # Paramètres Kappa-Sigma / Winsorized
    kappa: float = 2.5              # Seuil sigma pour rejection (1.5-4.0)
    iterations: int = 3             # Itérations max (1-10)
    min_images_for_clip: int = 3    # Minimum d'images pour activer sigma-clip
    
    # Paramètres moyenne pondérée
    weight_by_fwhm: bool = True     # Pondérer par FWHM
    weight_by_stars: bool = True    # Pondérer par nombre d'étoiles
    weight_by_sharpness: bool = True  # Pondérer par netteté
    
    # Mode streaming (mémoire limitée)
    streaming_mode: bool = False    # Activer pour RPi/ressources limitées
    
    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        if not 0.5 <= self.kappa <= 5.0:
            errors.append("kappa doit être entre 0.5 et 5.0")
        
        if not 1 <= self.iterations <= 20:
            errors.append("iterations doit être entre 1 et 20")
        
        if self.min_images_for_clip < 2:
            errors.append("min_images_for_clip doit être >= 2")
        
        if errors:
            raise ValueError("Config stacking invalide: " + ", ".join(errors))
        
        return True


@dataclass
class DrizzleConfig:
    """Configuration du drizzle pour super-résolution"""
    
    # Activation
    enable: bool = False
    
    # Paramètres principaux
    scale: float = 2.0              # Facteur super-résolution (1.0-4.0)
    pixfrac: float = 0.8            # Drop size ratio (0.1-1.0)
    kernel: DrizzleKernel = DrizzleKernel.SQUARE
    
    # Options avancées
    fill_value: float = 0.0         # Valeur pour pixels sans données
    min_coverage: float = 0.5       # Couverture min pour pixel valide
    
    # Optimisation
    use_fast_drizzle: bool = True   # Utiliser version vectorisée
    
    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        if not 1.0 <= self.scale <= 4.0:
            errors.append("scale doit être entre 1.0 et 4.0")
        
        if not 0.1 <= self.pixfrac <= 1.0:
            errors.append("pixfrac doit être entre 0.1 et 1.0")
        
        if not 0.0 <= self.min_coverage <= 1.0:
            errors.append("min_coverage doit être entre 0.0 et 1.0")
        
        if errors:
            raise ValueError("Config drizzle invalide: " + ", ".join(errors))
        
        return True
    
    def get_output_scale(self) -> float:
        """Retourne le facteur d'échelle effectif"""
        return self.scale if self.enable else 1.0


@dataclass
class PlanetaryConfig:
    """Configuration pour l'alignement planétaire (Soleil, Lune, planètes)"""
    
    # Activation et mode
    enable: bool = False
    mode: PlanetaryMode = PlanetaryMode.SURFACE
    
    # Détection de disque
    disk_min_radius: int = 50       # Rayon minimum en pixels
    disk_max_radius: int = 2000     # Rayon maximum en pixels
    disk_threshold: int = 30        # Seuil Canny pour détection bord
    disk_margin: int = 10           # Marge autour du disque (pixels)
    disk_use_ellipse: bool = False  # Détecter ellipse au lieu de cercle
    
    # Corrélation de surface
    surface_window_size: int = 256  # Taille fenêtre FFT (puissance de 2)
    surface_upsample: int = 10      # Facteur upsampling sub-pixel
    surface_highpass: bool = True   # Appliquer filtre passe-haut
    surface_roi_center: bool = True # ROI au centre du disque
    
    # Seuils de qualité
    max_shift: float = 100.0        # Décalage max accepté (pixels)
    min_correlation: float = 0.3    # Corrélation minimum
    max_rotation: float = 2.0       # Rotation max (degrés)
    
    # Post-traitement
    apply_derotation: bool = False  # Compenser rotation planétaire
    rotation_rate: float = 0.0      # Degrés/seconde
    
    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        if self.disk_min_radius <= 0:
            errors.append("disk_min_radius doit être > 0")
        
        if self.disk_max_radius <= self.disk_min_radius:
            errors.append("disk_max_radius doit être > disk_min_radius")
        
        if self.surface_window_size < 64:
            errors.append("surface_window_size doit être >= 64")
        
        if not 0.0 <= self.min_correlation <= 1.0:
            errors.append("min_correlation doit être entre 0 et 1")
        
        if errors:
            raise ValueError("Config planétaire invalide: " + ", ".join(errors))
        
        return True


@dataclass
class LuckyImagingConfig:
    """Configuration pour le Lucky Imaging (planétaire haute vitesse)"""
    
    # Activation
    enable: bool = False
    
    # Buffer circulaire
    buffer_size: int = 100              # Nombre d'images dans le buffer
    
    # Sélection
    keep_percent: float = 10.0          # Pourcentage d'images à garder (1-100)
    keep_count: int = 0                 # OU nombre fixe (0 = utiliser keep_percent)
    min_score: float = 0.0              # Score minimum absolu (0 = désactivé)
    
    # Scoring
    score_method: LuckyScoreMethod = LuckyScoreMethod.LAPLACIAN
    score_roi_percent: float = 50.0     # % central de l'image pour scoring
    
    # Stacking
    stack_method: LuckyStackMethod = LuckyStackMethod.MEAN
    stack_interval: int = 0             # 0 = stack quand buffer plein
    auto_stack: bool = True             # Stacker automatiquement
    sigma_clip_kappa: float = 2.5       # Kappa pour sigma clipping
    
    # Alignement
    align_enabled: bool = True          # Activer l'alignement (legacy)
    align_mode: int = 1                 # 0=off, 1=surface/phase, 2=disk/Hough, 3=hybride
    max_shift: float = 50.0             # Décalage max accepté en pixels (0=désactivé)
    align_method: str = "phase"         # "phase" (FFT) ou "ecc"
    align_roi_percent: float = 80.0     # % central pour alignement
    
    # Performance
    downscale_scoring: float = 1.0      # Downscale pour scoring rapide

    # ── Mode Pool Élite (buffer_mode="elite", RGB8 uniquement) ───────────────
    buffer_mode: str = "ring"           # "ring" (classique) ou "elite"
    elite_pool_size: int = 100          # Taille du pool (20-300)
    elite_stack_interval: float = 5.0  # Intervalle de stack en secondes (2-15)
    elite_entry_mode: str = "min"       # "min" (> pire) ou "mean" (> moyenne pool)
    elite_score_clip: bool = True       # Sigma-clipping des scores avant stack
    elite_score_kappa: float = 2.0      # Kappa pour sigma-clipping (1.5-4.0)

    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        if self.buffer_size < 10:
            errors.append("buffer_size doit être >= 10")
        
        if not 1.0 <= self.keep_percent <= 100.0:
            errors.append("keep_percent doit être entre 1 et 100")
        
        if not 0.0 <= self.score_roi_percent <= 100.0:
            errors.append("score_roi_percent doit être entre 0 et 100")
        
        if self.downscale_scoring <= 0 or self.downscale_scoring > 1.0:
            errors.append("downscale_scoring doit être entre 0 et 1.0")
        
        if errors:
            raise ValueError("Config Lucky Imaging invalide: " + ", ".join(errors))
        
        return True
    
    def get_keep_count(self) -> int:
        """Retourne le nombre d'images à garder"""
        if self.keep_count > 0:
            return min(self.keep_count, self.buffer_size)
        return max(1, int(self.buffer_size * self.keep_percent / 100.0))


@dataclass 
class QualityConfig:
    """Configuration du contrôle qualité (étendue)"""
    
    # Activation
    enable: bool = True
    
    # Seuils de qualité image
    max_fwhm: float = 12.0              # FWHM max en pixels
    max_ellipticity: float = 0.4        # Ellipticité max (0=rond, 1=ligne)
    min_stars: int = 10                 # Nombre minimum d'étoiles
    max_drift: float = 50.0             # Drift max en pixels
    min_sharpness: float = 0.3          # Netteté minimale (0-1)
    
    # Seuils d'alignement
    max_rotation: float = 5.0           # Rotation max en degrés
    min_scale: float = 0.95             # Scale minimum
    max_scale: float = 1.05             # Scale maximum
    min_inliers_ratio: float = 0.3      # Ratio min inliers RANSAC
    
    # Paramètres détection
    star_detection_sigma: float = 5.0   # Seuil sigma détection
    min_star_separation: int = 10       # Distance min entre étoiles
    
    # Stats runtime
    rejected_images: list = field(default_factory=list)
    rejection_reasons: dict = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        if self.max_fwhm <= 0:
            errors.append("max_fwhm doit être > 0")
        
        if not 0 <= self.max_ellipticity <= 1:
            errors.append("max_ellipticity doit être entre 0 et 1")
        
        if self.min_stars < 0:
            errors.append("min_stars doit être >= 0")
        
        if errors:
            raise ValueError("Config qualité invalide: " + ", ".join(errors))
        
        return True


@dataclass
class OutputConfig:
    """Configuration des sorties"""

    # Répertoire
    output_directory: str = "/home/admin/stacks/"

    # FITS
    auto_fits: bool = True
    fits_compress: bool = False
    fits_linear: bool = True            # FITS linéaire (RAW) vs stretched

    # PNG
    auto_png: bool = True
    png_bit_depth: Optional[int] = None # None=auto, 8, ou 16
    png_stretch_method: StretchMethod = StretchMethod.ASINH
    png_stretch_factor: float = 10.0
    png_clip_low: float = 1.0           # Percentile bas (%)
    png_clip_high: float = 99.5         # Percentile haut (%)

    # Paramètres GHS (Generalized Hyperbolic Stretch)
    ghs_D: float = 0.0                  # Linked stretching parameter (-1.0 à 1.0)
    ghs_B: float = 0.0                  # Blackpoint parameter
    ghs_SP: float = 0.5                 # Symmetry point (0-1)

    # Rapports
    save_rejected_list: bool = True
    save_quality_report: bool = True
    save_weight_map: bool = False       # Sauver carte des poids
    
    def validate(self) -> bool:
        """Valide la configuration"""
        _factor_methods = {StretchMethod.ASINH, StretchMethod.LINEAR, StretchMethod.SQRT,
                           StretchMethod.HISTOGRAM, StretchMethod.AUTO}
        if self.png_stretch_method in _factor_methods and self.png_stretch_factor <= 0:
            raise ValueError("png_stretch_factor doit être > 0")
        
        if not 0 <= self.png_clip_low < self.png_clip_high <= 100:
            raise ValueError("Percentiles PNG invalides")
        
        return True


# =============================================================================
# Configuration principale unifiée
# =============================================================================

@dataclass
class AdvancedStackingConfig:
    """
    Configuration principale unifiée pour libastrostack
    
    Regroupe toutes les options dans une structure cohérente.
    Compatible avec l'ancien StackingConfig via méthodes de conversion.
    """
    
    # Mode d'alignement
    alignment_mode: AlignmentMode = AlignmentMode.ROTATION
    max_stars_alignment: int = 50

    # ISP (Image Signal Processor)
    isp_enable: bool = False
    isp_config_path: Optional[str] = None
    isp_calibration_frames: Optional[tuple] = None
    video_format: Optional[str] = None  # 'yuv420', 'raw12', 'raw16', None=auto

    # Calibration RAW (black level capteur + suppression gradient)
    raw_black_level: int = 256        # ADU 12-bit natif capteur, 0 = désactivé
    gradient_removal: bool = False    # Suppression gradient de fond
    gradient_removal_tiles: int = 8   # Taille de la grille n×n
    gradient_removal_flat_strength: int = 0  # 0=BG seulement, 100=correction vignetage complète
    gradient_removal_poly_degree: int = 2    # Degré polynôme 2D (1=linéaire, 2=quad, 3=cubique, 4=quartique)
    gradient_removal_sigma: float = 2.0      # Seuil σ masquage tuiles (0.5=agressif, 5.0=minimal)
    awb_auto: bool = False            # AWB auto (grey-world) pour preview stack RAW12

    # BL per-canal Bayer (correction FPN 2×2 avant débayérisation)
    # None = désactivé. tuple (R, G1, G2, B) en ADU 12-bit pour correction manuelle.
    bl_per_channel: object = None     # Optional[tuple]
    bl_auto_estimate: bool = False    # True = estimation auto par percentile bas

    # ISP Auto-calibration
    isp_auto_calibrate_method: str = 'histogram_peaks'  # 'none', 'histogram_peaks', 'gray_world'
    isp_auto_calibrate_after: int = 10       # Calibrer après N frames stackées (0 = désactivé)
    isp_recalibrate_interval: int = 0        # Recalibrer tous les N frames (0 = jamais)
    isp_auto_update_only_wb: bool = True     # Si True, ne met à jour que les gains RGB

    # Sous-configurations
    stacking: StackingMethodConfig = field(default_factory=StackingMethodConfig)
    drizzle: DrizzleConfig = field(default_factory=DrizzleConfig)
    planetary: PlanetaryConfig = field(default_factory=PlanetaryConfig)
    lucky: LuckyImagingConfig = field(default_factory=LuckyImagingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Preview (pour RPiCamera)
    preview_refresh_interval: int = 5

    # DNG (mode hybride)
    save_dng_mode: str = "accepted"  # "none", "accepted", "all"
    
    # Stats runtime
    num_stacked: int = 0
    total_exposure: float = 0.0
    noise_level: float = 0.0
    is_color: Optional[bool] = None
    
    def validate(self) -> bool:
        """Valide toute la configuration"""
        self.stacking.validate()
        self.drizzle.validate()
        self.planetary.validate()
        self.lucky.validate()
        self.quality.validate()
        self.output.validate()
        
        if self.save_dng_mode not in ["none", "accepted", "all"]:
            raise ValueError("save_dng_mode invalide")
        
        return True
    
    def to_legacy_config(self) -> 'LegacyStackingConfig':
        """
        Convertit vers l'ancien format StackingConfig
        
        Pour compatibilité avec le code existant.
        """
        legacy = LegacyStackingConfig()
        
        # Alignement
        legacy.alignment_mode = self.alignment_mode.value
        legacy.max_stars_alignment = self.max_stars_alignment
        
        # Qualité
        legacy.quality.enable = self.quality.enable
        legacy.quality.max_fwhm = self.quality.max_fwhm
        legacy.quality.max_ellipticity = self.quality.max_ellipticity
        legacy.quality.min_stars = self.quality.min_stars
        legacy.quality.max_drift = self.quality.max_drift
        legacy.quality.min_sharpness = self.quality.min_sharpness
        legacy.quality.max_rotation = self.quality.max_rotation
        legacy.quality.min_scale = self.quality.min_scale
        legacy.quality.max_scale = self.quality.max_scale
        legacy.quality.min_inliers_ratio = self.quality.min_inliers_ratio
        legacy.quality.star_detection_sigma = self.quality.star_detection_sigma
        legacy.quality.min_star_separation = self.quality.min_star_separation
        
        # Canvas expandable
        legacy.canvas_margin_frac = getattr(self, 'canvas_margin_frac', 0.0)

        # ISP (nouveaux champs)
        legacy.isp_enable = getattr(self, 'isp_enable', False)
        legacy.isp_config_path = getattr(self, 'isp_config_path', None)
        legacy.isp_calibration_frames = getattr(self, 'isp_calibration_frames', None)
        legacy.video_format = getattr(self, 'video_format', None)

        # ISP Auto-calibration
        legacy.isp_auto_calibrate_method = getattr(self, 'isp_auto_calibrate_method', 'histogram_peaks')
        legacy.isp_auto_calibrate_after = getattr(self, 'isp_auto_calibrate_after', 10)
        legacy.isp_recalibrate_interval = getattr(self, 'isp_recalibrate_interval', 0)
        legacy.isp_auto_update_only_wb = getattr(self, 'isp_auto_update_only_wb', True)

        # PNG
        legacy.auto_png = self.output.auto_png
        legacy.png_bit_depth = getattr(self.output, 'png_bit_depth', None)
        legacy.png_stretch_method = self.output.png_stretch_method.value
        legacy.png_stretch_factor = self.output.png_stretch_factor
        legacy.png_clip_low = getattr(self, 'png_clip_low', self.output.png_clip_low)
        legacy.png_clip_high = getattr(self, 'png_clip_high', self.output.png_clip_high)

        # GHS - Lire depuis niveau racine (configure()) ou fallback sur output
        legacy.ghs_D = getattr(self, 'ghs_D', self.output.ghs_D)
        legacy.ghs_b = getattr(self, 'ghs_b', getattr(self.output, 'ghs_b', self.output.ghs_B))
        legacy.ghs_SP = getattr(self, 'ghs_SP', self.output.ghs_SP)
        legacy.ghs_LP = getattr(self, 'ghs_LP', getattr(self.output, 'ghs_LP', 0.0))
        legacy.ghs_HP = getattr(self, 'ghs_HP', getattr(self.output, 'ghs_HP', 0.0))
        legacy.ghs_auto_adjust_sp = getattr(self, 'ghs_auto_adjust_sp', True)

        # Log/MTF stretch params
        legacy.log_factor = getattr(self, 'log_factor', 100.0)
        legacy.mtf_midtone = getattr(self, 'mtf_midtone', 0.2)
        legacy.mtf_shadows = getattr(self, 'mtf_shadows', 0.0)
        legacy.mtf_highlights = getattr(self, 'mtf_highlights', 1.0)

        # FITS
        legacy.fits_linear = getattr(self.output, 'fits_linear', True)

        # Autres
        legacy.preview_refresh_interval = self.preview_refresh_interval
        legacy.save_dng_mode = self.save_dng_mode
        legacy.output_directory = self.output.output_directory
        legacy.save_rejected_list = self.output.save_rejected_list
        
        # Calibration RAW (black level capteur + suppression gradient)
        legacy.raw_black_level = self.raw_black_level
        legacy.gradient_removal = self.gradient_removal
        legacy.gradient_removal_tiles = self.gradient_removal_tiles
        legacy.gradient_removal_flat_strength = self.gradient_removal_flat_strength
        legacy.gradient_removal_poly_degree = self.gradient_removal_poly_degree
        legacy.gradient_removal_sigma = self.gradient_removal_sigma
        legacy.awb_auto = self.awb_auto

        # BL per-canal Bayer
        legacy.bl_per_channel = self.bl_per_channel
        legacy.bl_auto_estimate = self.bl_auto_estimate

        # Stats
        legacy.num_stacked = self.num_stacked
        legacy.total_exposure = self.total_exposure
        legacy.noise_level = self.noise_level
        legacy.is_color = self.is_color

        return legacy
    
    @classmethod
    def from_legacy_config(cls, legacy: 'LegacyStackingConfig') -> 'AdvancedStackingConfig':
        """
        Crée depuis l'ancien format StackingConfig
        """
        config = cls()
        
        # Alignement
        config.alignment_mode = AlignmentMode(legacy.alignment_mode)
        config.max_stars_alignment = legacy.max_stars_alignment
        
        # Qualité
        config.quality.enable = legacy.quality.enable
        config.quality.max_fwhm = legacy.quality.max_fwhm
        config.quality.max_ellipticity = legacy.quality.max_ellipticity
        config.quality.min_stars = legacy.quality.min_stars
        config.quality.max_drift = legacy.quality.max_drift
        config.quality.min_sharpness = legacy.quality.min_sharpness
        
        # ISP
        config.isp_enable = getattr(legacy, 'isp_enable', False)
        config.isp_config_path = getattr(legacy, 'isp_config_path', None)
        config.isp_calibration_frames = getattr(legacy, 'isp_calibration_frames', None)
        config.video_format = getattr(legacy, 'video_format', None)

        # ISP Auto-calibration
        config.isp_auto_calibrate_method = getattr(legacy, 'isp_auto_calibrate_method', 'histogram_peaks')
        config.isp_auto_calibrate_after = getattr(legacy, 'isp_auto_calibrate_after', 10)
        config.isp_recalibrate_interval = getattr(legacy, 'isp_recalibrate_interval', 0)
        config.isp_auto_update_only_wb = getattr(legacy, 'isp_auto_update_only_wb', True)

        # PNG
        config.output.auto_png = legacy.auto_png
        config.output.png_stretch_method = StretchMethod(legacy.png_stretch_method)
        config.output.png_stretch_factor = legacy.png_stretch_factor

        # Autres
        config.preview_refresh_interval = legacy.preview_refresh_interval
        config.save_dng_mode = legacy.save_dng_mode
        
        return config
    
    def print_summary(self):
        """Affiche un résumé de la configuration"""
        print("\n" + "=" * 60)
        print("CONFIGURATION LIBASTROSTACK")
        print("=" * 60)
        
        print(f"\n[ALIGNEMENT]")
        print(f"  Mode: {self.alignment_mode.value}")
        print(f"  Max étoiles: {self.max_stars_alignment}")
        
        print(f"\n[STACKING]")
        print(f"  Méthode: {self.stacking.method.value}")
        if self.stacking.method in [StackMethod.KAPPA_SIGMA, StackMethod.WINSORIZED]:
            print(f"  Kappa: {self.stacking.kappa}")
            print(f"  Itérations: {self.stacking.iterations}")
        print(f"  Mode streaming: {'Oui' if self.stacking.streaming_mode else 'Non'}")
        
        print(f"\n[DRIZZLE]")
        print(f"  Activé: {'Oui' if self.drizzle.enable else 'Non'}")
        if self.drizzle.enable:
            print(f"  Scale: {self.drizzle.scale}x")
            print(f"  Pixfrac: {self.drizzle.pixfrac}")
            print(f"  Kernel: {self.drizzle.kernel.value}")
        
        print(f"\n[PLANÉTAIRE]")
        print(f"  Activé: {'Oui' if self.planetary.enable else 'Non'}")
        if self.planetary.enable:
            print(f"  Mode: {self.planetary.mode.value}")
            print(f"  Rayon disque: {self.planetary.disk_min_radius}-{self.planetary.disk_max_radius} px")
            print(f"  Fenêtre FFT: {self.planetary.surface_window_size}")
            print(f"  Corrélation min: {self.planetary.min_correlation}")
            print(f"  Décalage max: {self.planetary.max_shift} px")
        
        print(f"\n[LUCKY IMAGING]")
        print(f"  Activé: {'Oui' if self.lucky.enable else 'Non'}")
        if self.lucky.enable:
            print(f"  Buffer: {self.lucky.buffer_size} images")
            print(f"  Garder: {self.lucky.keep_percent}%")
            if self.lucky.keep_count > 0:
                print(f"  Garder (fixe): {self.lucky.keep_count} images")
            print(f"  Méthode score: {self.lucky.score_method.value}")
            print(f"  Méthode stack: {self.lucky.stack_method.value}")
            print(f"  ROI score: {self.lucky.score_roi_percent}%")
            print(f"  Alignement: {'Oui' if self.lucky.align_enabled else 'Non'}")
        
        print(f"\n[QUALITÉ]")
        print(f"  Activé: {'Oui' if self.quality.enable else 'Non'}")
        if self.quality.enable:
            print(f"  FWHM max: {self.quality.max_fwhm} px")
            print(f"  Ellipticité max: {self.quality.max_ellipticity}")
            print(f"  Étoiles min: {self.quality.min_stars}")
            print(f"  Drift max: {self.quality.max_drift} px")
            print(f"  Netteté min: {self.quality.min_sharpness}")
        
        print(f"\n[SORTIE]")
        print(f"  Répertoire: {self.output.output_directory}")
        print(f"  PNG auto: {'Oui' if self.output.auto_png else 'Non'}")
        print(f"  Stretch: {self.output.png_stretch_method.value}")
        print(f"  DNG: {self.save_dng_mode}")
        
        print("=" * 60)


# =============================================================================
# Classe legacy pour compatibilité
# =============================================================================

class LegacyQualityConfig:
    """Ancien format QualityConfig (pour compatibilité)"""
    
    def __init__(self):
        self.enable = True
        self.max_fwhm = 12.0
        self.max_ellipticity = 0.4
        self.min_stars = 10
        self.max_drift = 50.0
        self.min_sharpness = 0.3
        self.max_rotation = 5.0
        self.min_scale = 0.95
        self.max_scale = 1.05
        self.min_inliers_ratio = 0.3
        self.star_detection_sigma = 5.0
        self.min_star_separation = 10
        self.rejected_images = []
        self.rejection_reasons = {}


class LegacyStackingConfig:
    """Ancien format StackingConfig (pour compatibilité)"""

    def __init__(self):
        self.alignment_mode = "rotation"
        self.quality = LegacyQualityConfig()

        # Paramètres ISP (nouveaux, pour compatibilité)
        self.isp_enable = False
        self.isp_config_path = None
        self.isp_calibration_frames = None
        self.video_format = None

        # Calibration automatique ISP
        self.isp_auto_calibrate_method = 'histogram_peaks'  # Méthode: 'none', 'histogram_peaks', 'gray_world'
        self.isp_auto_calibrate_after = 10       # Calibrer après N frames stackées (0 = désactivé)
        self.isp_recalibrate_interval = 0        # Recalibrer tous les N frames (0 = jamais - calibration unique)
        self.isp_auto_update_only_wb = True      # Si True, ne met à jour que les gains RGB (préserve gamma, contrast, etc.)

        # Paramètres PNG
        self.auto_png = True
        self.png_bit_depth = None
        self.png_stretch_method = "asinh"
        self.png_stretch_factor = 10.0
        self.png_clip_low = 1.0
        self.png_clip_high = 99.5

        # Paramètres GHS
        self.ghs_D = 0.0
        self.ghs_B = 0.0
        self.ghs_SP = 0.5

        # Paramètres FITS
        self.fits_linear = True

        self.preview_refresh_interval = 5
        self.save_dng_mode = "accepted"
        self.output_directory = "/home/admin/stacks/"
        self.save_rejected_list = True
        self.max_stars_alignment = 50
        self.max_frames = 0              # 0 = illimité (I11)
        self.num_stacked = 0
        self.total_exposure = 0.0
        self.noise_level = 0.0
        self.is_color = None
    
    def validate(self):
        return True


# =============================================================================
# Presets de configuration
# =============================================================================

def get_preset_fast() -> AdvancedStackingConfig:
    """
    Preset optimisé pour la vitesse (live stacking temps réel)
    
    - Méthode: moyenne simple
    - Pas de drizzle
    - Contrôle qualité basique
    """
    config = AdvancedStackingConfig()
    
    config.stacking.method = StackMethod.MEAN
    config.stacking.streaming_mode = True
    
    config.drizzle.enable = False
    
    config.quality.enable = True
    config.quality.max_fwhm = 15.0  # Plus tolérant
    config.quality.min_stars = 5
    
    config.preview_refresh_interval = 3
    
    return config


def get_preset_quality() -> AdvancedStackingConfig:
    """
    Preset optimisé pour la qualité (post-processing)
    
    - Méthode: Kappa-Sigma
    - Drizzle 2x
    - Contrôle qualité strict
    """
    config = AdvancedStackingConfig()
    
    config.stacking.method = StackMethod.KAPPA_SIGMA
    config.stacking.kappa = 2.5
    config.stacking.iterations = 5
    config.stacking.streaming_mode = False
    
    config.drizzle.enable = True
    config.drizzle.scale = 2.0
    config.drizzle.pixfrac = 0.8
    config.drizzle.kernel = DrizzleKernel.SQUARE
    
    config.quality.enable = True
    config.quality.max_fwhm = 8.0
    config.quality.min_stars = 15
    config.quality.min_sharpness = 0.4
    
    return config


def get_preset_balanced() -> AdvancedStackingConfig:
    """
    Preset équilibré (défaut recommandé)
    
    - Méthode: Kappa-Sigma (streaming)
    - Pas de drizzle
    - Contrôle qualité modéré
    """
    config = AdvancedStackingConfig()
    
    config.stacking.method = StackMethod.KAPPA_SIGMA
    config.stacking.kappa = 2.5
    config.stacking.iterations = 3
    config.stacking.streaming_mode = True
    
    config.drizzle.enable = False
    
    config.quality.enable = True
    config.quality.max_fwhm = 12.0
    config.quality.min_stars = 10
    
    return config


def get_preset_planetary() -> AdvancedStackingConfig:
    """
    Preset pour imagerie planétaire (Soleil, Lune, planètes)
    
    - Alignement: Surface (corrélation de phase)
    - Méthode: Kappa-Sigma
    - Drizzle 2x optionnel
    - Contrôle qualité adapté (pas d'étoiles)
    """
    config = AdvancedStackingConfig()
    
    # Alignement planétaire
    config.alignment_mode = AlignmentMode.SURFACE
    config.planetary.enable = True
    config.planetary.mode = PlanetaryMode.SURFACE
    config.planetary.surface_window_size = 256
    config.planetary.min_correlation = 0.3
    config.planetary.max_shift = 50.0
    
    # Stacking
    config.stacking.method = StackMethod.KAPPA_SIGMA
    config.stacking.kappa = 2.0  # Plus strict pour planétaire
    config.stacking.iterations = 5
    config.stacking.streaming_mode = False  # Garder toutes les frames
    
    # Drizzle optionnel mais recommandé
    config.drizzle.enable = True
    config.drizzle.scale = 2.0
    config.drizzle.pixfrac = 0.7
    
    # Qualité adaptée (pas d'étoiles sur disque planétaire)
    config.quality.enable = True
    config.quality.min_stars = 0  # Désactivé pour planétaire
    config.quality.max_fwhm = 20.0  # Plus tolérant
    config.quality.min_sharpness = 0.2
    
    return config


def get_preset_solar() -> AdvancedStackingConfig:
    """
    Preset optimisé pour imagerie solaire
    
    - Mode: Surface (détails taches, granulation)
    - Haute tolérance au décalage (turbulence)
    """
    config = get_preset_planetary()
    
    # Paramètres solaires spécifiques
    config.planetary.disk_min_radius = 200  # Soleil = grand disque
    config.planetary.disk_max_radius = 2000
    config.planetary.max_shift = 100.0  # Plus de turbulence
    config.planetary.surface_highpass = True  # Accentuer granulation
    
    return config


def get_preset_lunar() -> AdvancedStackingConfig:
    """
    Preset optimisé pour imagerie lunaire
    
    - Mode: Hybride (limbe + cratères)
    - Supporte phases partielles
    """
    config = get_preset_planetary()
    
    # Paramètres lunaires spécifiques
    config.planetary.mode = PlanetaryMode.HYBRID
    config.planetary.disk_use_ellipse = True  # Phases partielles
    config.planetary.disk_min_radius = 100
    config.planetary.surface_roi_center = True
    
    return config


def get_preset_lucky_fast() -> AdvancedStackingConfig:
    """
    Preset Lucky Imaging rapide
    
    - Buffer: 50 images
    - Garder: 20%
    - Scoring: Laplacian (très rapide)
    """
    config = AdvancedStackingConfig()
    
    config.lucky.enable = True
    config.lucky.buffer_size = 50
    config.lucky.keep_percent = 20.0
    config.lucky.score_method = LuckyScoreMethod.LAPLACIAN
    config.lucky.stack_method = LuckyStackMethod.MEAN
    config.lucky.align_enabled = True
    config.lucky.score_roi_percent = 50.0
    
    # Désactiver le QC standard (Lucky fait sa propre sélection)
    config.quality.enable = False
    
    return config


def get_preset_lucky_quality() -> AdvancedStackingConfig:
    """
    Preset Lucky Imaging haute qualité
    
    - Buffer: 200 images
    - Garder: 5%
    - Scoring: Tenengrad (plus précis)
    - Stack: Sigma-clip
    """
    config = AdvancedStackingConfig()
    
    config.lucky.enable = True
    config.lucky.buffer_size = 200
    config.lucky.keep_percent = 5.0
    config.lucky.score_method = LuckyScoreMethod.TENENGRAD
    config.lucky.stack_method = LuckyStackMethod.SIGMA_CLIP
    config.lucky.sigma_clip_kappa = 2.5
    config.lucky.align_enabled = True
    config.lucky.score_roi_percent = 60.0
    
    config.quality.enable = False
    
    return config


def get_preset_lucky_aggressive() -> AdvancedStackingConfig:
    """
    Preset Lucky Imaging très sélectif
    
    - Buffer: 500 images
    - Garder: 1% (seulement les meilleures)
    - Pour conditions de seeing difficiles
    """
    config = AdvancedStackingConfig()
    
    config.lucky.enable = True
    config.lucky.buffer_size = 500
    config.lucky.keep_percent = 1.0
    config.lucky.score_method = LuckyScoreMethod.TENENGRAD
    config.lucky.stack_method = LuckyStackMethod.SIGMA_CLIP
    config.lucky.sigma_clip_kappa = 2.0
    config.lucky.align_enabled = True
    config.lucky.score_roi_percent = 40.0
    
    config.quality.enable = False
    
    return config


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test Configuration Avancée ===\n")
    
    # Test création
    config = AdvancedStackingConfig()
    config.validate()
    config.print_summary()
    
    # Test presets
    print("\n--- Preset FAST ---")
    fast = get_preset_fast()
    fast.print_summary()
    
    print("\n--- Preset QUALITY ---")
    quality = get_preset_quality()
    quality.print_summary()
    
    # Test conversion legacy
    print("\n--- Test conversion legacy ---")
    legacy = config.to_legacy_config()
    print(f"Legacy alignment_mode: {legacy.alignment_mode}")
    print(f"Legacy max_fwhm: {legacy.quality.max_fwhm}")
    
    restored = AdvancedStackingConfig.from_legacy_config(legacy)
    print(f"Restored alignment_mode: {restored.alignment_mode.value}")
    
    print("\n=== Tests terminés ===")
