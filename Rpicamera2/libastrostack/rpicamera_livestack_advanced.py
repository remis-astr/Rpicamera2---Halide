#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'intégration des fonctionnalités avancées pour RPiCamera.py
===================================================================

Ce module étend rpicamera_livestack.py avec le support des nouvelles
fonctionnalités:
- Méthodes de stacking avancées (Kappa-Sigma, Winsorized, etc.)
- Drizzle pour super-résolution
- Mode batch pour post-processing

Usage dans RPiCamera.py:
    from rpicamera_livestack_advanced import RPiCameraLiveStackAdvanced
    
    livestack = RPiCameraLiveStackAdvanced(camera_params)
    livestack.configure(
        stacking_method='kappa_sigma',
        kappa=2.5,
        drizzle_enable=True,
        drizzle_scale=2.0
    )
    livestack.start()

Auteur: libastrostack Team
Version: 1.0.0
"""

import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import queue
from typing import Optional, Dict, Any, List

# Imports relatifs depuis libastrostack
from . import LiveStackSession, StackingConfig, AlignmentMode, StretchMethod
from .stacker_advanced import AdvancedStacker, StackMethod
from .drizzle import DrizzleStacker, DrizzleStackerFast
from .aligner_planetary import PlanetaryAligner, PlanetaryMode, PlanetaryConfig
from .lucky_imaging import (
    LuckyImagingStacker, ElitePoolStacker, BufferMode, LuckyConfig,
    ScoreMethod as LuckyScoreMethod,
    StackMethod as LuckyStackMethod,
    RPiCameraLuckyImaging, create_lucky_session,
    QualityScorer
)
from .config_advanced import (
    AdvancedStackingConfig, 
    get_preset_fast, 
    get_preset_quality, 
    get_preset_balanced,
    get_preset_planetary,
    get_preset_solar,
    get_preset_lunar,
    get_preset_lucky_fast,
    get_preset_lucky_quality,
    get_preset_lucky_aggressive,
)


class RPiCameraLiveStackAdvanced:
    """
    Wrapper Live Stacking avancé pour RPiCamera.py
    
    Étend RPiCameraLiveStack avec:
    - Support des méthodes de stacking avancées
    - Support du drizzle
    - Mode batch pour post-processing
    - Statistiques détaillées
    """
    
    # Mapping des méthodes de stacking
    STACKING_METHODS = {
        'mean': StackMethod.MEAN,
        'median': StackMethod.MEDIAN,
        'kappa_sigma': StackMethod.KAPPA_SIGMA,
        'winsorized': StackMethod.WINSORIZED,
        'weighted': StackMethod.WEIGHTED,
    }
    
    def __init__(self, camera_params: Dict, output_dir: str = "/home/admin/stacks"):
        """
        Args:
            camera_params: dict avec paramètres caméra RPiCamera
                - exposure: temps exposition (ms)
                - gain: gain/ISO
                - red: balance rouge
                - blue: balance bleue
            output_dir: Répertoire de sortie
        """
        self.camera_params = camera_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = AdvancedStackingConfig()
        
        # Session standard (alignement + qualité)
        self.session: Optional[LiveStackSession] = None
        
        # Stacker avancé (optionnel)
        self.advanced_stacker: Optional[AdvancedStacker] = None
        self.use_advanced_stacking = False
        
        # Drizzle (optionnel)
        self.drizzle_stacker: Optional[DrizzleStacker] = None
        self.use_drizzle = False
        
        # Alignement planétaire (optionnel)
        self.planetary_aligner: Optional[PlanetaryAligner] = None
        self.use_planetary = False
        
        # Lucky Imaging (optionnel)
        self.lucky_stacker: Optional[LuckyImagingStacker] = None
        self.use_lucky = False
        self.last_lucky_result: Optional[np.ndarray] = None  # Dernier résultat Lucky pour affichage continu
        
        # État
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.last_preview = None
        self.last_preview_update = 0
        
        # Thread sauvegarde
        self.save_queue = queue.Queue()
        self.save_thread = None
        
        # Buffer pour stacking avancé (stocke les images alignées)
        self.aligned_buffer: List[np.ndarray] = []
        self.transform_buffer: List[Dict] = []
        self.max_buffer_size = 100  # Limite mémoire
        
        # Statistiques
        self.start_time = None
        self.stats = {
            'total_frames': 0,
            'accepted_frames': 0,
            'rejected_frames': 0,
            'stacking_method': 'mean',
            'drizzle_enabled': False,
            'memory_mb': 0.0,
        }
    
    def configure(self, **kwargs):
        """
        Configure les paramètres de stacking
        
        Paramètres disponibles:
        
        Alignement:
            - alignment_mode: "none", "translation", "rotation", "affine",
                             "disk", "surface", "planetary"
        
        Contrôle qualité:
            - enable_qc: bool
            - max_fwhm: float
            - max_ellipticity: float
            - min_stars: int
            - max_drift: float
            - min_sharpness: float
        
        Stacking avancé:
            - stacking_method: "mean", "median", "kappa_sigma", "winsorized", "weighted"
            - kappa: float (pour kappa_sigma/winsorized)
            - iterations: int (pour kappa_sigma/winsorized)
            - streaming_mode: bool (économie mémoire)
        
        Planétaire (Soleil, Lune, planètes):
            - planetary_enable: bool (activer alignement planétaire)
            - planetary_mode: 0/1/2 ou "disk"/"surface"/"hybrid"
                * disk (0): Alignement sur le limbe (Hough circle)
                * surface (1): Corrélation FFT sur détails (recommandé)
                * hybrid (2): Disque + surface combinés (Lune)
            - planetary_disk_min: int (rayon min disque en pixels, 50-500)
            - planetary_disk_max: int (rayon max disque en pixels, 200-2000)
            - planetary_disk_threshold: int (seuil Canny, 10-100)
            - planetary_disk_margin: int (marge autour disque, 5-50)
            - planetary_disk_ellipse: bool (détecter ellipse vs cercle)
            - planetary_window: int (taille fenêtre FFT, 128/256/512)
            - planetary_upsample: int (upsampling sub-pixel, 1-20)
            - planetary_highpass: bool (filtre passe-haut pour détails)
            - planetary_roi_center: bool (ROI au centre du disque)
            - planetary_corr: int/float (corrélation min, 20-80 ou 0.2-0.8)
            - planetary_max_shift: float (décalage max en pixels)
            - planetary_max_rotation: float (rotation max en degrés)
            - planetary_derotation: bool (compenser rotation planétaire)
            - planetary_rotation_rate: float (degrés/seconde)
        
        Lucky Imaging (haute cadence):
            - lucky_enable: bool
            - lucky_buffer_size: int (50-500)
            - lucky_keep_percent: float (1-50)
            - lucky_score_method: "laplacian", "gradient", "sobel", "tenengrad"
            - lucky_stack_method: "mean", "median", "sigma_clip"
            - lucky_align_enabled: bool
            - lucky_score_roi_percent: float (20-100)
        
        Drizzle:
            - drizzle_enable: bool
            - drizzle_scale: float (1.0-4.0)
            - drizzle_pixfrac: float (0.1-1.0)
            - drizzle_kernel: "point", "square", "gaussian"
        
        Sortie:
            - png_stretch: "linear", "asinh", "log", "sqrt"
            - png_factor: float
            - preview_refresh: int
            - save_dng: "none", "accepted", "all"
        
        Presets:
            - preset: "fast", "quality", "balanced", 
                     "planetary", "solar", "lunar",
                     "lucky_fast", "lucky_quality", "lucky_aggressive"
        """
        # Appliquer preset si spécifié
        if 'preset' in kwargs:
            preset = kwargs.pop('preset')
            if preset == 'fast':
                self.config = get_preset_fast()
            elif preset == 'quality':
                self.config = get_preset_quality()
            elif preset == 'planetary':
                self.config = get_preset_planetary()
            elif preset == 'solar':
                self.config = get_preset_solar()
            elif preset == 'lunar':
                self.config = get_preset_lunar()
            elif preset == 'lucky_fast':
                self.config = get_preset_lucky_fast()
                self.use_lucky = True
            elif preset == 'lucky_quality':
                self.config = get_preset_lucky_quality()
                self.use_lucky = True
            elif preset == 'lucky_aggressive':
                self.config = get_preset_lucky_aggressive()
                self.use_lucky = True
            else:
                self.config = get_preset_balanced()
        
        # Alignement
        if 'alignment_mode' in kwargs:
            mode = kwargs['alignment_mode'].lower()
            from .config_advanced import AlignmentMode
            mode_map = {
                'none': AlignmentMode.NONE,
                'off': AlignmentMode.NONE,
                'translation': AlignmentMode.TRANSLATION,
                'rotation': AlignmentMode.ROTATION,
                'affine': AlignmentMode.AFFINE,
                # Modes planétaires
                'disk': AlignmentMode.DISK,
                'surface': AlignmentMode.SURFACE,
                'planetary': AlignmentMode.PLANETARY,
            }
            self.config.alignment_mode = mode_map.get(mode, AlignmentMode.ROTATION)
            
            # Activer automatiquement le mode planétaire si nécessaire
            if mode in ['disk', 'surface', 'planetary']:
                self.config.planetary.enable = True
                self.use_planetary = True
        
        # Configuration planétaire
        if 'planetary_enable' in kwargs:
            self.config.planetary.enable = bool(kwargs['planetary_enable'])
            self.use_planetary = self.config.planetary.enable
        if 'planetary_mode' in kwargs:
            mode = kwargs['planetary_mode'].lower() if isinstance(kwargs['planetary_mode'], str) else kwargs['planetary_mode']
            if isinstance(mode, int):
                # Support valeur numérique (0=disk, 1=surface, 2=hybrid)
                mode_list = [PlanetaryMode.DISK, PlanetaryMode.SURFACE, PlanetaryMode.HYBRID]
                self.config.planetary.mode = mode_list[min(mode, 2)]
            else:
                mode_map = {
                    'disk': PlanetaryMode.DISK,
                    'surface': PlanetaryMode.SURFACE,
                    'hybrid': PlanetaryMode.HYBRID,
                }
                self.config.planetary.mode = mode_map.get(mode, PlanetaryMode.SURFACE)
        if 'planetary_disk_min' in kwargs:
            self.config.planetary.disk_min_radius = int(kwargs['planetary_disk_min'])
        if 'disk_min_radius' in kwargs:  # Alias
            self.config.planetary.disk_min_radius = int(kwargs['disk_min_radius'])
        if 'planetary_disk_max' in kwargs:
            self.config.planetary.disk_max_radius = int(kwargs['planetary_disk_max'])
        if 'disk_max_radius' in kwargs:  # Alias
            self.config.planetary.disk_max_radius = int(kwargs['disk_max_radius'])
        if 'planetary_disk_threshold' in kwargs:
            self.config.planetary.disk_threshold = int(kwargs['planetary_disk_threshold'])
        if 'planetary_disk_margin' in kwargs:
            self.config.planetary.disk_margin = int(kwargs['planetary_disk_margin'])
        if 'planetary_disk_ellipse' in kwargs:
            self.config.planetary.disk_use_ellipse = bool(kwargs['planetary_disk_ellipse'])
        if 'surface_window_size' in kwargs:
            self.config.planetary.surface_window_size = int(kwargs['surface_window_size'])
        if 'planetary_window' in kwargs:  # Alias court
            self.config.planetary.surface_window_size = int(kwargs['planetary_window'])
        if 'planetary_upsample' in kwargs:
            self.config.planetary.surface_upsample = int(kwargs['planetary_upsample'])
        if 'planetary_highpass' in kwargs:
            self.config.planetary.surface_highpass = bool(kwargs['planetary_highpass'])
        if 'planetary_roi_center' in kwargs:
            self.config.planetary.surface_roi_center = bool(kwargs['planetary_roi_center'])
        if 'min_correlation' in kwargs:
            self.config.planetary.min_correlation = float(kwargs['min_correlation'])
        if 'planetary_corr' in kwargs:  # Alias (÷100 si entier)
            val = kwargs['planetary_corr']
            if isinstance(val, int) and val > 1:
                val = val / 100.0  # Conversion entier → float
            self.config.planetary.min_correlation = float(val)
        if 'planetary_max_shift' in kwargs:
            self.config.planetary.max_shift = float(kwargs['planetary_max_shift'])
        if 'planetary_max_rotation' in kwargs:
            self.config.planetary.max_rotation = float(kwargs['planetary_max_rotation'])
        if 'planetary_derotation' in kwargs:
            self.config.planetary.apply_derotation = bool(kwargs['planetary_derotation'])
        if 'planetary_rotation_rate' in kwargs:
            self.config.planetary.rotation_rate = float(kwargs['planetary_rotation_rate'])
        
        # Configuration Lucky Imaging
        if 'lucky_enable' in kwargs:
            self.config.lucky.enable = bool(kwargs['lucky_enable'])
            self.use_lucky = self.config.lucky.enable
            # Forcer l'advanced_stacker en mode streaming pour le mode cumulatif Lucky
            if self.use_lucky:
                self.use_advanced_stacking = True
                self.config.stacking.streaming_mode = True
        if 'lucky_buffer_size' in kwargs:
            new_size = int(kwargs['lucky_buffer_size'])
            old_size = self.config.lucky.buffer_size
            self.config.lucky.buffer_size = new_size
            # Mise à jour dynamique : recréer le buffer avec la nouvelle taille
            print(f"[DEBUG LIVESTACK] lucky_buffer_size reçu: {old_size} → {new_size}")
            print(f"[DEBUG LIVESTACK] self.lucky_stacker={self.lucky_stacker}, self.is_running={self.is_running}")
            if self.lucky_stacker and self.is_running:
                print(f"[DEBUG LIVESTACK] Appel self.lucky_stacker.configure(buffer_size={new_size})")
                self.lucky_stacker.configure(buffer_size=new_size)
                print(f"[LUCKY] ✓ Buffer recréé avec {new_size} images (anciennes données perdues)")
            else:
                print(f"[DEBUG LIVESTACK] ÉCHEC - lucky_stacker non actif ou non running")

        if 'lucky_keep_percent' in kwargs:
            new_percent = float(kwargs['lucky_keep_percent'])
            self.config.lucky.keep_percent = new_percent
            # Mise à jour dynamique : modifier config du stacker actif
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.config.keep_percent = new_percent
                print(f"[LUCKY] keep_percent mis à jour: {new_percent}%")

        if 'lucky_keep_count' in kwargs:
            new_count = int(kwargs['lucky_keep_count'])
            self.config.lucky.keep_count = new_count
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.config.keep_count = new_count

        if 'lucky_min_score' in kwargs:
            new_score = float(kwargs['lucky_min_score'])
            self.config.lucky.min_score = new_score
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.config.min_score = new_score

        if 'lucky_score_method' in kwargs:
            method = kwargs['lucky_score_method'].lower()
            # LuckyScoreMethod == lucky_imaging.ScoreMethod (import direct, source unique)
            method_map = {
                'laplacian': LuckyScoreMethod.LAPLACIAN,
                'gradient': LuckyScoreMethod.GRADIENT,
                'sobel': LuckyScoreMethod.SOBEL,
                'tenengrad': LuckyScoreMethod.TENENGRAD,
            }
            new_method = method_map.get(method, LuckyScoreMethod.LAPLACIAN)
            self.config.lucky.score_method = new_method
            # Mise à jour dynamique : même type, assignment direct
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.config.score_method = new_method
                # Recréer le scorer avec la nouvelle méthode
                self.lucky_stacker.scorer = QualityScorer(self.lucky_stacker.config)
                print(f"[LUCKY] score_method mis à jour: {method}")

        if 'lucky_stack_method' in kwargs:
            method = kwargs['lucky_stack_method'].lower()
            # LuckyStackMethod == lucky_imaging.StackMethod (import direct, source unique)
            method_map = {
                'mean': LuckyStackMethod.MEAN,
                'median': LuckyStackMethod.MEDIAN,
                'sigma_clip': LuckyStackMethod.SIGMA_CLIP,
            }
            new_method = method_map.get(method, LuckyStackMethod.MEAN)
            self.config.lucky.stack_method = new_method
            print(f"[DEBUG] lucky_stack_method configure: method='{method}', self.lucky_stacker={self.lucky_stacker}, self.is_running={self.is_running}")
            # Mise à jour dynamique : même type, assignment direct
            if self.lucky_stacker:
                self.lucky_stacker.config.stack_method = new_method
                print(f"[LUCKY] stack_method mis à jour: {method} → {new_method}")
            else:
                print(f"[LUCKY] stack_method sera appliqué au prochain start(): {method}")

        if 'lucky_align_mode' in kwargs:
            new_mode = int(kwargs['lucky_align_mode'])
            self.config.lucky.align_mode = new_mode
            self.config.lucky.align_enabled = (new_mode != 0)
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.configure(align_mode=new_mode)
                print(f"[LUCKY] align_mode mis à jour: {new_mode}")

        if 'lucky_align_enabled' in kwargs:
            new_align = bool(kwargs['lucky_align_enabled'])
            self.config.lucky.align_enabled = new_align
            # Backward compat : sync align_mode
            if not new_align:
                self.config.lucky.align_mode = 0
            elif getattr(self.config.lucky, 'align_mode', 0) == 0:
                self.config.lucky.align_mode = 1
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.config.align_enabled = new_align
                print(f"[LUCKY] align_enabled mis à jour: {new_align}")

        if 'lucky_score_roi_percent' in kwargs:
            new_roi = float(kwargs['lucky_score_roi_percent'])
            self.config.lucky.score_roi_percent = new_roi
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.config.score_roi_percent = new_roi
                # Recréer le scorer pour mettre à jour le ROI
                self.lucky_stacker.scorer = QualityScorer(self.lucky_stacker.config)
                print(f"[LUCKY] score_roi_percent mis à jour: {new_roi}%")

        if 'lucky_max_shift' in kwargs:
            new_shift = float(kwargs['lucky_max_shift'])
            self.config.lucky.max_shift = new_shift
            if self.lucky_stacker and self.is_running:
                self.lucky_stacker.configure(max_shift=new_shift)
                print(f"[LUCKY] max_shift mis à jour: {new_shift:.0f}px {'(désactivé)' if new_shift == 0 else ''}")

        # ── Pool Élite ────────────────────────────────────────────────────────
        if 'lucky_buffer_mode' in kwargs:
            self.config.lucky.buffer_mode = str(kwargs['lucky_buffer_mode'])
        if 'lucky_elite_pool_size' in kwargs:
            new_size = int(kwargs['lucky_elite_pool_size'])
            self.config.lucky.elite_pool_size = new_size
            if self.lucky_stacker and isinstance(self.lucky_stacker, ElitePoolStacker):
                self.lucky_stacker.configure(elite_pool_size=new_size)
        if 'lucky_elite_stack_interval' in kwargs:
            new_interval = float(kwargs['lucky_elite_stack_interval'])
            self.config.lucky.elite_stack_interval = new_interval
            if self.lucky_stacker and isinstance(self.lucky_stacker, ElitePoolStacker):
                self.lucky_stacker.configure(elite_stack_interval=new_interval)
        if 'lucky_elite_entry_mode' in kwargs:
            new_mode = str(kwargs['lucky_elite_entry_mode'])   # "min" | "mean"
            self.config.lucky.elite_entry_mode = new_mode
            if self.lucky_stacker and isinstance(self.lucky_stacker, ElitePoolStacker):
                self.lucky_stacker.configure(elite_entry_mode=new_mode)
        if 'lucky_elite_score_clip' in kwargs:
            new_clip = bool(kwargs['lucky_elite_score_clip'])
            self.config.lucky.elite_score_clip = new_clip
            if self.lucky_stacker and isinstance(self.lucky_stacker, ElitePoolStacker):
                self.lucky_stacker.configure(elite_score_clip=new_clip)
        if 'lucky_elite_score_kappa' in kwargs:
            new_kappa = float(kwargs['lucky_elite_score_kappa'])
            self.config.lucky.elite_score_kappa = new_kappa
            if self.lucky_stacker and isinstance(self.lucky_stacker, ElitePoolStacker):
                self.lucky_stacker.configure(elite_score_kappa=new_kappa)

        # Contrôle qualité
        if 'enable_qc' in kwargs:
            self.config.quality.enable = bool(kwargs['enable_qc'])
        if 'max_fwhm' in kwargs:
            self.config.quality.max_fwhm = float(kwargs['max_fwhm'])
        if 'max_ellipticity' in kwargs:
            self.config.quality.max_ellipticity = float(kwargs['max_ellipticity'])
        if 'min_stars' in kwargs:
            self.config.quality.min_stars = int(kwargs['min_stars'])
        if 'max_drift' in kwargs:
            self.config.quality.max_drift = float(kwargs['max_drift'])
        if 'min_sharpness' in kwargs:
            self.config.quality.min_sharpness = float(kwargs['min_sharpness'])
        
        # Stacking avancé
        if 'stacking_method' in kwargs:
            method = kwargs['stacking_method'].lower()
            if method in self.STACKING_METHODS:
                self.config.stacking.method = self.STACKING_METHODS[method]
                # Don't disable advanced stacking if Lucky mode is active
                if not self.use_lucky:
                    self.use_advanced_stacking = method != 'mean'
                self.stats['stacking_method'] = method
        
        if 'kappa' in kwargs:
            self.config.stacking.kappa = float(kwargs['kappa'])
        if 'iterations' in kwargs:
            self.config.stacking.iterations = int(kwargs['iterations'])
        if 'streaming_mode' in kwargs:
            self.config.stacking.streaming_mode = bool(kwargs['streaming_mode'])
        
        # Drizzle
        if 'drizzle_enable' in kwargs:
            self.config.drizzle.enable = bool(kwargs['drizzle_enable'])
            self.use_drizzle = self.config.drizzle.enable
            self.stats['drizzle_enabled'] = self.use_drizzle
        if 'drizzle_scale' in kwargs:
            self.config.drizzle.scale = float(kwargs['drizzle_scale'])
        if 'drizzle_pixfrac' in kwargs:
            self.config.drizzle.pixfrac = float(kwargs['drizzle_pixfrac'])
        if 'drizzle_kernel' in kwargs:
            from .config_advanced import DrizzleKernel
            kernel_map = {
                'point': DrizzleKernel.POINT,
                'square': DrizzleKernel.SQUARE,
                'gaussian': DrizzleKernel.GAUSSIAN,
            }
            self.config.drizzle.kernel = kernel_map.get(
                kwargs['drizzle_kernel'], 
                DrizzleKernel.SQUARE
            )
        
        # Sortie
        if 'png_stretch' in kwargs:
            from .config_advanced import StretchMethod
            stretch_map = {
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
            stretch_method = stretch_map.get(kwargs['png_stretch'], StretchMethod.ASINH)
            # Stocker dans les deux emplacements pour compatibilité
            self.config.output.png_stretch_method = stretch_method
            self.config.png_stretch_method = stretch_method
            # Propagation live si session déjà démarrée
            if self.session is not None:
                self.session.config.png_stretch_method = kwargs['png_stretch']
            print(f"[DEBUG] png_stretch configuré: {kwargs['png_stretch']} → {stretch_method}")
        if 'png_factor' in kwargs:
            self.config.output.png_stretch_factor = float(kwargs['png_factor'])
        if 'png_clip_low' in kwargs:
            self.config.png_clip_low = float(kwargs['png_clip_low'])
        if 'png_clip_high' in kwargs:
            self.config.png_clip_high = float(kwargs['png_clip_high'])

        # Paramètres GHS - stocker à la racine de config pour compatibilité session.py
        ghs_updated = False
        if 'ghs_D' in kwargs:
            self.config.ghs_D = float(kwargs['ghs_D'])
            ghs_updated = True
        if 'ghs_b' in kwargs:
            self.config.ghs_b = float(kwargs['ghs_b'])
            ghs_updated = True
        if 'ghs_B' in kwargs:  # Rétro-compatibilité
            self.config.ghs_b = float(kwargs['ghs_B'])
            ghs_updated = True
        if 'ghs_SP' in kwargs:
            self.config.ghs_SP = float(kwargs['ghs_SP'])
            ghs_updated = True
        if 'ghs_LP' in kwargs:
            self.config.ghs_LP = float(kwargs['ghs_LP'])
            ghs_updated = True
        if 'ghs_HP' in kwargs:
            self.config.ghs_HP = float(kwargs['ghs_HP'])
            ghs_updated = True
        if 'ghs_auto_adjust_sp' in kwargs:
            self.config.ghs_auto_adjust_sp = bool(kwargs['ghs_auto_adjust_sp'])

        # Paramètres Log
        if 'log_factor' in kwargs:
            self.config.log_factor = float(kwargs['log_factor'])
            if self.session is not None:
                self.session.config.log_factor = float(kwargs['log_factor'])

        # Paramètres MTF
        if 'mtf_midtone' in kwargs:
            self.config.mtf_midtone = float(kwargs['mtf_midtone'])
            if self.session is not None:
                self.session.config.mtf_midtone = float(kwargs['mtf_midtone'])
        if 'mtf_shadows' in kwargs:
            self.config.mtf_shadows = float(kwargs['mtf_shadows'])
            if self.session is not None:
                self.session.config.mtf_shadows = float(kwargs['mtf_shadows'])
        if 'mtf_highlights' in kwargs:
            self.config.mtf_highlights = float(kwargs['mtf_highlights'])
            if self.session is not None:
                self.session.config.mtf_highlights = float(kwargs['mtf_highlights'])

        # Debug GHS si modifié
        if ghs_updated:
            print(f"[DEBUG GHS CONFIGURE] D={getattr(self.config, 'ghs_D', '?')}, "
                  f"b={getattr(self.config, 'ghs_b', '?')}, SP={getattr(self.config, 'ghs_SP', '?')}, "
                  f"LP={getattr(self.config, 'ghs_LP', '?')}, HP={getattr(self.config, 'ghs_HP', '?')}")
        if 'preview_refresh' in kwargs:
            self.config.preview_refresh_interval = int(kwargs['preview_refresh'])
        if 'save_dng' in kwargs:
            self.config.save_dng_mode = kwargs['save_dng']

        # ISP (Image Signal Processor)
        print(f"[DEBUG CONFIGURE] isp_enable dans kwargs: {'isp_enable' in kwargs}")
        print(f"[DEBUG CONFIGURE] isp_config_path dans kwargs: {'isp_config_path' in kwargs}")
        if 'isp_enable' in kwargs:
            print(f"[DEBUG CONFIGURE] isp_enable reçu: {kwargs['isp_enable']}")
            self.config.isp_enable = bool(kwargs['isp_enable'])
        if 'isp_config_path' in kwargs:
            print(f"[DEBUG CONFIGURE] isp_config_path reçu: {kwargs['isp_config_path']}")
            self.config.isp_config_path = kwargs['isp_config_path']

        # Video format (yuv420, raw12, raw16)
        if 'video_format' in kwargs:
            print(f"[DEBUG CONFIGURE] video_format reçu: {kwargs['video_format']}")
            self.config.video_format = kwargs['video_format']
            # Mettre à jour la session si elle existe déjà (configuration dynamique)
            if hasattr(self, 'session') and self.session:
                self.session.config.video_format = kwargs['video_format']
                print(f"[CONFIG] ✓ Format vidéo mis à jour: {kwargs['video_format']}")

        # Calibration RAW : black level capteur + suppression gradient
        if 'raw_black_level' in kwargs:
            self.config.raw_black_level = int(kwargs['raw_black_level'])
            if hasattr(self, 'session') and self.session:
                self.session.config.raw_black_level = int(kwargs['raw_black_level'])
                print(f"[CONFIG] ✓ Black level RAW mis à jour: {kwargs['raw_black_level']}")
        if 'gradient_removal' in kwargs:
            self.config.gradient_removal = bool(kwargs['gradient_removal'])
            if hasattr(self, 'session') and self.session:
                self.session.config.gradient_removal = bool(kwargs['gradient_removal'])
                print(f"[CONFIG] ✓ Gradient removal mis à jour: {kwargs['gradient_removal']}")
        if 'gradient_removal_tiles' in kwargs:
            self.config.gradient_removal_tiles = int(kwargs['gradient_removal_tiles'])
            if hasattr(self, 'session') and self.session:
                self.session.config.gradient_removal_tiles = int(kwargs['gradient_removal_tiles'])
        if 'gradient_removal_flat_strength' in kwargs:
            self.config.gradient_removal_flat_strength = int(kwargs['gradient_removal_flat_strength'])
            if hasattr(self, 'session') and self.session:
                self.session.config.gradient_removal_flat_strength = int(kwargs['gradient_removal_flat_strength'])
        if 'gradient_removal_poly_degree' in kwargs:
            self.config.gradient_removal_poly_degree = int(kwargs['gradient_removal_poly_degree'])
            if hasattr(self, 'session') and self.session:
                self.session.config.gradient_removal_poly_degree = int(kwargs['gradient_removal_poly_degree'])
        if 'gradient_removal_sigma' in kwargs:
            self.config.gradient_removal_sigma = float(kwargs['gradient_removal_sigma'])
            if hasattr(self, 'session') and self.session:
                self.session.config.gradient_removal_sigma = float(kwargs['gradient_removal_sigma'])
        if 'awb_auto' in kwargs:
            self.config.awb_auto = bool(kwargs['awb_auto'])
            if hasattr(self, 'session') and self.session:
                self.session.config.awb_auto = bool(kwargs['awb_auto'])

        print(f"[DEBUG CONFIGURE] self.config.isp_enable final: {self.config.isp_enable}")
        print(f"[DEBUG CONFIGURE] self.config.isp_config_path final: {self.config.isp_config_path}")
        print(f"[DEBUG CONFIGURE] self.config.video_format final: {self.config.video_format}")

        print(f"[CONFIG] Méthode: {self.config.stacking.method.value}, "
              f"Drizzle: {'ON' if self.use_drizzle else 'OFF'}")
    
    def start(self):
        """Démarre la session de live stacking"""
        self.config.validate()

        # Créer session standard
        legacy_config = self.config.to_legacy_config()

        # Mapper le format RAW de la caméra vers video_format
        # SAUF si video_format a déjà été configuré par configure() (priorité à configure)
        if self.config.video_format is None and 'raw_format' in self.camera_params:
            raw_format_name = self.camera_params['raw_format']  # 'YUV420', 'SRGGB12', 'SRGGB16'
            format_mapping = {
                'YUV420': 'yuv420',
                'SRGGB12': 'raw12',
                'SRGGB16': 'raw16',
                'RAW12 Bayer': 'raw12',  # Support nouveaux noms de format
                'RAW16 Clear HDR': 'raw16'
            }
            legacy_config.video_format = format_mapping.get(raw_format_name, 'yuv420')
            print(f"[CONFIG] Format vidéo auto-détecté: {legacy_config.video_format} (source: {raw_format_name})")
        elif self.config.video_format is not None:
            # video_format déjà configuré par configure() - ne pas écraser
            print(f"[CONFIG] Format vidéo préconfiguré: {legacy_config.video_format} (via configure)")

        self.session = LiveStackSession(legacy_config)
        self.session.start()

        # En mode Lucky, la session ne détecte pas automatiquement is_color
        # Forcer à None pour que ce soit détecté à la première frame
        if self.use_lucky:
            self.session.is_color = None
        
        # Initialiser stacker avancé si nécessaire
        if self.use_advanced_stacking:
            self.advanced_stacker = AdvancedStacker(legacy_config)
            self.advanced_stacker.set_method(
                self.config.stacking.method,
                kappa=self.config.stacking.kappa,
                iterations=self.config.stacking.iterations
            )
            if self.config.stacking.streaming_mode:
                self.advanced_stacker.enable_streaming(True)
            print(f"[STACKER] Mode avancé: {self.config.stacking.method.value}")
        
        # Initialiser drizzle si nécessaire
        if self.use_drizzle:
            DrizzleClass = DrizzleStackerFast if self.config.drizzle.use_fast_drizzle else DrizzleStacker
            self.drizzle_stacker = DrizzleClass(
                scale=self.config.drizzle.scale,
                pixfrac=self.config.drizzle.pixfrac,
                kernel=self.config.drizzle.kernel.value
            )
            print(f"[DRIZZLE] Scale: {self.config.drizzle.scale}x, "
                  f"Pixfrac: {self.config.drizzle.pixfrac}")
        
        # Initialiser alignement planétaire si nécessaire
        if self.use_planetary or self.config.planetary.enable:
            self.use_planetary = True
            planetary_config = PlanetaryConfig()
            planetary_config.mode = self.config.planetary.mode
            planetary_config.disk_min_radius = self.config.planetary.disk_min_radius
            planetary_config.disk_max_radius = self.config.planetary.disk_max_radius
            planetary_config.surface_window_size = self.config.planetary.surface_window_size
            planetary_config.min_correlation = self.config.planetary.min_correlation
            planetary_config.max_shift = self.config.planetary.max_shift
            planetary_config.surface_highpass = self.config.planetary.surface_highpass
            planetary_config.surface_roi_center = self.config.planetary.surface_roi_center
            
            self.planetary_aligner = PlanetaryAligner(planetary_config)
            print(f"[PLANETARY] Mode: {self.config.planetary.mode.value}, "
                  f"Window: {self.config.planetary.surface_window_size}px")
        
        # Initialiser Lucky Imaging si nécessaire
        if self.use_lucky or self.config.lucky.enable:
            self.use_lucky = True
            lucky_config = LuckyConfig()
            lucky_config.buffer_size = self.config.lucky.buffer_size
            lucky_config.keep_percent = self.config.lucky.keep_percent
            if self.config.lucky.keep_count > 0:
                lucky_config.keep_count = self.config.lucky.keep_count
            lucky_config.min_score = self.config.lucky.min_score
            lucky_config.score_roi_percent = self.config.lucky.score_roi_percent
            lucky_config.align_enabled = self.config.lucky.align_enabled
            lucky_config.align_mode = getattr(self.config.lucky, 'align_mode', 1 if self.config.lucky.align_enabled else 0)
            lucky_config.max_shift = getattr(self.config.lucky, 'max_shift', 50.0)
            lucky_config.auto_stack = self.config.lucky.auto_stack
            
            # LuckyScoreMethod == lucky_imaging.ScoreMethod (même type après unification I2)
            # → assignment direct, aucune conversion nécessaire
            lucky_config.score_method = self.config.lucky.score_method
            lucky_config.stack_method = self.config.lucky.stack_method
            print(f"[DEBUG START] lucky_config.score_method = {lucky_config.score_method}, "
                  f"stack_method = {lucky_config.stack_method}")
            lucky_config.sigma_clip_kappa = self.config.lucky.sigma_clip_kappa

            # Champs Pool Élite
            lucky_config.elite_pool_size      = self.config.lucky.elite_pool_size
            lucky_config.elite_stack_interval = self.config.lucky.elite_stack_interval
            lucky_config.elite_entry_mode     = self.config.lucky.elite_entry_mode
            lucky_config.elite_score_clip     = self.config.lucky.elite_score_clip
            lucky_config.elite_score_kappa    = self.config.lucky.elite_score_kappa

            # CORRECTION BUG IMAGE BLANCHE: Passer le format RAW explicite
            # pour éviter la détection incorrecte basée sur frame.max()
            raw_format_str = None
            if hasattr(self.camera_params, 'raw_format') and self.camera_params.raw_format:
                raw_format_str = self.camera_params.raw_format
            elif 'raw_format' in self.camera_params:
                raw_format_str = self.camera_params['raw_format']

            if raw_format_str:
                # Mapper les formats RPiCamera vers 'raw12' ou 'raw16'
                # raw_formats = ['YUV420 8bit', 'XRGB8888 ISP', 'RAW12 Bayer', 'RAW16 Clear HDR']
                if 'RAW12' in raw_format_str.upper() or 'BAYER' in raw_format_str.upper():
                    lucky_config.raw_format = 'raw12'
                    print(f"[LUCKY CONFIG] Format RAW détecté: {raw_format_str} → raw12")
                elif 'RAW16' in raw_format_str.upper() or 'CLEAR' in raw_format_str.upper() or 'HDR' in raw_format_str.upper():
                    lucky_config.raw_format = 'raw16'
                    print(f"[LUCKY CONFIG] Format RAW détecté: {raw_format_str} → raw16")
                else:
                    # YUV420/XRGB8888 : pas de normalisation RAW nécessaire
                    lucky_config.raw_format = None
                    print(f"[LUCKY CONFIG] Format non-RAW détecté: {raw_format_str}, pas de normalisation RAW")
            else:
                # Fallback: détection intelligente par percentile (plus robuste que max)
                lucky_config.raw_format = None
                lucky_config.raw_normalize_method = 'percentile'
                print(f"[LUCKY CONFIG] Format RAW non spécifié, utilisation détection percentile (robuste)")

            if self.config.lucky.buffer_mode == "elite":
                self.lucky_stacker = ElitePoolStacker(lucky_config)
                print(f"[LUCKY] Mode Pool Élite activé")
            else:
                self.lucky_stacker = LuckyImagingStacker(lucky_config)
                print(f"[LUCKY] Buffer: {self.config.lucky.buffer_size}, "
                      f"Keep: {self.config.lucky.keep_percent}%, "
                      f"Score: {self.config.lucky.score_method.value}")
            self.lucky_stacker.start()
        
        self.is_running = True
        self.is_paused = False
        self.start_time = datetime.now()
        self.frame_count = 0
        self._last_stacks_count = 0  # Compteur pour détecter nouveaux stacks Lucky
        
        print(f"\n[START] Session avancée démarrée")
        self.config.print_summary()

    def _normalize_raw_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convertit une frame RAW en float32 préservant la dynamique complète

        CORRECTION BUG LIVESTACK RAW:
        Les données RAW doivent être préservées en pleine résolution [0-65535]
        pour que l'ISP puisse les traiter correctement.
        NE PAS compresser en 8-bit [0-255] sinon perte de dynamique !

        Args:
            frame: Image brute (uint8, uint16, ou float32)

        Returns:
            Image en float32 avec dynamique préservée
        """
        # Cas 1: Déjà float32 → retourner tel quel
        if frame.dtype == np.float32:
            return frame.copy()

        # Cas 2: uint8 → conversion directe
        if frame.dtype == np.uint8:
            return frame.astype(np.float32)

        # Cas 3: uint16 (RAW 12/16-bit) → conversion SANS compression
        # CORRECTION: Garder la pleine dynamique pour l'ISP
        if frame.dtype == np.uint16:
            raw_format_str = self.camera_params.get('raw_format', '')

            # Log une seule fois pour debug
            if not hasattr(self, '_raw_normalize_logged'):
                max_val = frame.max()
                print(f"[LIVESTACK RAW] Format: {raw_format_str}, dtype: {frame.dtype}, max: {max_val:.0f}")
                print(f"[LIVESTACK RAW] → Conversion en float32 SANS compression (dynamique préservée)")
                self._raw_normalize_logged = True

            # Conversion directe sans compression (garde [0-4095] ou [0-65535])
            return frame.astype(np.float32)

        # Cas 4: Autre dtype → conversion directe
        return frame.astype(np.float32)

    def process_frame(self, image_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Traite une frame de la caméra
        
        Args:
            image_data: Array NumPy (float32) RGB ou MONO
        
        Returns:
            Image empilée courante (pour preview) ou None
        """
        if not self.is_running or self.is_paused:
            return None

        self.stats['total_frames'] += 1

        # CORRECTION COHÉRENCE ISP: Normaliser les frames RAW AVANT traitement
        # pour que LiveStack et LuckyStack envoient les mêmes plages de données à l'ISP
        # (Le mode Lucky a déjà sa propre normalisation, on skip pour lui)
        if not (self.use_lucky and self.lucky_stacker):
            image_data = self._normalize_raw_frame(image_data)

        # Mode Lucky Imaging : utiliser le stacker Lucky
        if self.use_lucky and self.lucky_stacker:
            # Détecter le type couleur pour les stats (si pas encore fait)
            if self.session and self.session.is_color is None:
                self.session.is_color = len(image_data.shape) == 3

            score = self.lucky_stacker.add_frame(image_data)

            # Mettre à jour stats
            lucky_stats = self.lucky_stacker.get_stats()
            self.stats['lucky_buffer_fill'] = lucky_stats.get('buffer_fill', 0)
            self.stats['lucky_buffer_size'] = lucky_stats.get('buffer_size', 0)
            self.stats['lucky_avg_score'] = lucky_stats.get('avg_score', 0)
            current_stacks_done = lucky_stats.get('stacks_done', 0)
            self.stats['lucky_stacks_done'] = current_stacks_done
            self.stats['accepted_frames'] = lucky_stats.get('frames_selected', 0)
            # Clés spécifiques Pool Élite (transparentes pour mode Ring)
            for _k in ('buffer_mode', 'phase', 'accept_rate', 'last_clipped',
                       'next_stack_in', 'min_score', 'max_score', 'total_frames'):
                if _k in lucky_stats:
                    self.stats[_k] = lucky_stats[_k]

            # Détecter si un nouveau stack est disponible via le compteur
            previous_stacks = getattr(self, '_last_stacks_count', 0)
            
            if current_stacks_done > previous_stacks:
                # Nouveau stack disponible !
                self._last_stacks_count = current_stacks_done
                
                # Récupérer le résultat (consomme last_result)
                new_result = self.lucky_stacker.get_result()
                
                if new_result is not None:
                    # MODE CUMULATIF : Envoyer le stack Lucky vers l'advanced_stacker
                    if self.use_advanced_stacking and self.advanced_stacker:
                        # Pondération par le score moyen des frames sélectionnées du buffer
                        buffer_weight = float(lucky_stats.get('avg_score', 1.0)) or 1.0
                        self.advanced_stacker.add_image(new_result, weight=buffer_weight, quality_metrics={})
                        
                        # Mettre à jour le compteur de frames acceptées
                        if hasattr(self.advanced_stacker, 'get_count'):
                            self.stats['accepted_frames'] = self.advanced_stacker.get_count()
                        elif hasattr(self.advanced_stacker, 'frame_count'):
                            self.stats['accepted_frames'] = self.advanced_stacker.frame_count
                        else:
                            self.stats['accepted_frames'] = current_stacks_done

                        # Retourner le master stack accumulé
                        master_stack = self.advanced_stacker.combine()
                        if master_stack is not None:
                            self.last_lucky_result = master_stack.copy()

                            # SOLUTION 2: Mettre à jour la référence d'alignement Lucky
                            # avec le résultat cumulatif pour compenser la dérive
                            self.lucky_stacker.update_alignment_reference(master_stack)

                            self._update_memory_stats()
                            self.frame_count += 1
                            return master_stack
                    else:
                        # Mode non-cumulatif
                        self.last_lucky_result = new_result.copy()
                        self._update_memory_stats()
                        self.frame_count += 1
                        return new_result

            # Pendant le remplissage du buffer, retourner le dernier résultat connu
            if self.last_lucky_result is not None:
                return self.last_lucky_result
            
            return None
        
        # Mode planétaire : utiliser l'aligneur planétaire directement
        if self.use_planetary and self.planetary_aligner:
            aligned_image, params, success = self.planetary_aligner.align(image_data)
            
            if not success:
                self.stats['rejected_frames'] += 1
                return None
            
            self.stats['accepted_frames'] += 1
            transform_params = params
            
        else:
            # Mode standard : traitement via session (qualité + alignement étoiles)
            # process_image_data retourne None si rejetée OU si preview_refresh pas atteint
            # → vérifier num_stacked pour distinguer les deux cas
            _stacked_before = self.session.config.num_stacked
            result = self.session.process_image_data(image_data)
            _stacked_after = self.session.config.num_stacked

            if _stacked_after > _stacked_before:
                # Frame acceptée et empilée par session.stacker
                self.stats['accepted_frames'] += 1
                # session.stacker a déjà empilé → ne pas re-stacker dans advanced_stacker
                self._update_memory_stats()
                self.frame_count += 1
                if result is not None:
                    return result  # Preview refresh atteint
                if self.frame_count % self.config.preview_refresh_interval == 0:
                    return self.get_current_preview()
                return None
            else:
                # Frame réellement rejetée (QC ou alignement échoué)
                self.stats['rejected_frames'] += 1
                return None

        # Stacking avancé
        if self.use_advanced_stacking and self.advanced_stacker:
            quality_metrics = {}
            self.advanced_stacker.add_image(aligned_image, quality_metrics=quality_metrics)
            
            if not self.config.stacking.streaming_mode and self.use_drizzle:
                if len(self.aligned_buffer) < self.max_buffer_size:
                    self.aligned_buffer.append(aligned_image.copy())
                    self.transform_buffer.append(transform_params.copy())
        
        # Drizzle (mode temps réel)
        if self.use_drizzle and self.drizzle_stacker:
            self.drizzle_stacker.add_image(
                aligned_image,
                dx=transform_params.get('dx', 0),
                dy=transform_params.get('dy', 0),
                angle=transform_params.get('angle', 0)
            )
        
        # Mettre à jour stats
        self._update_memory_stats()
        
        # Retourner preview si nécessaire
        self.frame_count += 1
        if self.frame_count % self.config.preview_refresh_interval == 0:
            return self.get_current_preview()
        
        return None
    
    def get_current_preview(self) -> Optional[np.ndarray]:
        """
        Retourne l'image courante pour preview (données brutes sans traitement ISP/stretch)

        Si drizzle est actif, retourne une version downscalée.
        """
        if self.use_drizzle and self.drizzle_stacker:
            drizzle_result = self.drizzle_stacker.combine()
            if drizzle_result is not None:
                # Downscale pour preview
                scale = self.config.drizzle.scale
                if scale > 1.0:
                    import cv2
                    h, w = drizzle_result.shape[:2]
                    new_h, new_w = int(h / scale), int(w / scale)
                    if len(drizzle_result.shape) == 3:
                        preview = cv2.resize(drizzle_result, (new_w, new_h))
                    else:
                        preview = cv2.resize(drizzle_result, (new_w, new_h))
                    return preview
                return drizzle_result

        # CORRECTION: En mode Lucky, retourner le dernier résultat cumulatif stocké
        # au lieu de recalculer avec advanced_stacker.combine()
        if self.use_lucky and hasattr(self, 'last_lucky_result') and self.last_lucky_result is not None:
            return self.last_lucky_result.copy()

        if self.use_advanced_stacking and self.advanced_stacker:
            return self.advanced_stacker.combine()

        if self.session:
            return self.session.get_current_stack()

        return None

    def get_preview_for_display(self) -> Optional[np.ndarray]:
        """
        Retourne l'image courante avec le MÊME traitement que le PNG sauvegardé
        (ISP + stretch complet), mais converti en uint8 pour affichage pygame.

        Ceci garantit que l'affichage à l'écran correspond exactement au PNG sauvegardé,
        seul le bit depth change (8-bit vs 16-bit).

        Returns:
            Array uint8 (H, W, 3) RGB pour pygame, ou None
        """
        if self.session is None:
            # Fallback sur preview basique si pas de session
            return self.get_current_preview()

        try:
            # MODE LUCKY: Traiter manuellement car session.stacker n'a pas le résultat Lucky
            # Le résultat Lucky est dans self.last_lucky_result
            if self.use_lucky and hasattr(self, 'last_lucky_result') and self.last_lucky_result is not None:
                # Appliquer le même pipeline que get_preview_png() manuellement
                # sur self.last_lucky_result
                result = self.last_lucky_result.copy()

                # Remplacer temporairement le stack de la session pour utiliser get_preview_png()
                original_stack = self.session.stacker.stacked_image
                self.session.stacker.stacked_image = result

                try:
                    preview_png = self.session.get_preview_png()
                finally:
                    # Restaurer le stack original
                    self.session.stacker.stacked_image = original_stack

                if preview_png is None:
                    # Fallback si le traitement échoue
                    return self.get_current_preview()

            else:
                # MODE STANDARD (LiveStack): Utiliser get_preview_png() directement
                preview_png = self.session.get_preview_png()

                if preview_png is None:
                    return None

            # Convertir en uint8 si nécessaire (pygame requiert uint8)
            if preview_png.dtype == np.uint16:
                preview_uint8 = (preview_png / 257).astype(np.uint8)
            elif preview_png.dtype == np.uint8:
                preview_uint8 = preview_png
            else:
                preview_uint8 = np.clip(preview_png * 255, 0, 255).astype(np.uint8)

            return preview_uint8

        except Exception as e:
            print(f"[WARN] get_preview_for_display() échoué: {e}")
            import traceback
            traceback.print_exc()
            # Fallback sur preview basique
            return self.get_current_preview()
    
    def get_final_result(self) -> Optional[np.ndarray]:
        """
        Retourne le résultat final (après combinaison complète)
        
        Pour le stacking avancé, effectue la combinaison finale.
        """
        if self.use_drizzle and self.drizzle_stacker:
            return self.drizzle_stacker.combine()
        
        if self.use_advanced_stacking and self.advanced_stacker:
            return self.advanced_stacker.combine()
        
        if self.session:
            return self.session.get_current_stack()
        
        return None
    
    def save(self, filename: Optional[str] = None, raw_format_name: Optional[str] = None):
        """
        Sauvegarde le résultat final

        Args:
            filename: Nom du fichier (sans extension) ou None pour auto
            raw_format_name: Nom du format RAW actuel (YUV420/SRGGB12/SRGGB16)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Format RAW
            raw_format_str = raw_format_name or self.camera_params.get('raw_format', 'YUV420')

            # ISP software (seulement pour formats RAW)
            isp_str = ""
            if self.config.isp_enable and 'RAW' in raw_format_str.upper():
                isp_str = "_isp"

            # Méthode de stacking
            if self.use_lucky:
                method = f"lucky_{self.config.lucky.stack_method.value}"
                # Ajouter paramètres Lucky importants
                keep = int(self.config.lucky.keep_percent)
                score = self.config.lucky.score_method.value
                align_str = "_align" if self.config.lucky.align_enabled else ""
                lucky_params = f"_k{keep}_{score}{align_str}"
            else:
                method = self.config.stacking.method.value
                lucky_params = ""

            # Planétaire
            planetary_str = f"_{self.config.planetary.mode.value}" if self.use_planetary else ""

            filename = f"stack_{raw_format_str}{isp_str}_{method}{lucky_params}{planetary_str}_{timestamp}"

        output_path = self.output_dir / f"{filename}.fit"
        
        # Récupérer résultat final
        result = self.get_final_result()
        
        if result is None:
            print("[WARN] Aucun résultat à sauvegarder")
            return
        
        # Utiliser la session pour sauvegarder (gère FITS + PNG)
        if self.session:
            # Remplacer temporairement le stack de la session
            original_stack = self.session.stacker.stacked_image
            self.session.stacker.stacked_image = result
            
            self.session.save_result(output_path)
            
            # Restaurer
            self.session.stacker.stacked_image = original_stack
        
        # Sauvegarder weight map si drizzle
        if self.use_drizzle and self.drizzle_stacker:
            weight_map = self.drizzle_stacker.get_weight_map()
            if weight_map is not None and self.config.output.save_weight_map:
                weight_path = self.output_dir / f"{filename}_weight.fit"
                from .io import save_fits
                save_fits(weight_map, weight_path, {'TYPE': 'WEIGHT_MAP'})
                print(f"[SAVE] Weight map: {weight_path}")
    
    def stop(self):
        """Arrête la session"""
        self.is_running = False

        if self.session:
            # En mode Lucky, mettre à jour les stats de la session
            if self.use_lucky:
                self.session.config.num_stacked = self.stats.get('accepted_frames', 0)
                self.session.files_processed = self.stats.get('total_frames', 0)
            self.session.stop()
        
        # Stats finales
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        print(f"\n{'='*60}")
        print("[STOP] Session avancée terminée")
        print(f"{'='*60}")
        print(f"  Durée: {duration:.1f}s")
        print(f"  Frames total: {self.stats['total_frames']}")
        print(f"  Frames acceptées: {self.stats['accepted_frames']}")
        print(f"  Frames rejetées: {self.stats['rejected_frames']}")
        print(f"  Méthode: {self.stats['stacking_method']}")
        print(f"  Drizzle: {'ON' if self.stats['drizzle_enabled'] else 'OFF'}")
        print(f"  Mémoire utilisée: {self.stats['memory_mb']:.1f} MB")

        if self.use_advanced_stacking and self.advanced_stacker:
            # En mode Lucky, calculer SNR basé sur le nombre total de frames acceptées
            if self.use_lucky:
                n_selected = self.stats.get('accepted_frames', 0)
                if n_selected > 0:
                    snr = np.sqrt(n_selected)
                    # Appliquer pénalité selon méthode de stacking
                    from .config_advanced import StackMethod
                    if self.config.stacking.method == StackMethod.MEDIAN:
                        snr *= 0.80  # Médiane perd ~20% de SNR
                    print(f"  Amélioration SNR: {snr:.2f}x")
                else:
                    print(f"  Amélioration SNR: 1.00x")
            else:
                snr = self.advanced_stacker.get_snr_improvement()
                print(f"  Amélioration SNR: {snr:.2f}x")
        
        print(f"{'='*60}")
    
    def pause(self):
        """Met en pause le stacking"""
        self.is_paused = True
        print("[PAUSE] Stacking en pause")
    
    def resume(self):
        """Reprend le stacking"""
        self.is_paused = False
        print("[RESUME] Stacking repris")
    
    def reset(self):
        """Réinitialise la session (garde config)"""
        if self.session:
            self.session.reset()
        
        if self.advanced_stacker:
            self.advanced_stacker.reset()
        
        if self.drizzle_stacker:
            self.drizzle_stacker.reset()
        
        if self.planetary_aligner:
            self.planetary_aligner.reset()
        
        if self.lucky_stacker:
            self.lucky_stacker.reset()
        
        self.aligned_buffer.clear()
        self.transform_buffer.clear()
        self.frame_count = 0
        self.last_lucky_result = None  # Réinitialiser le dernier résultat Lucky
        self._last_stacks_count = 0  # Réinitialiser le compteur de stacks
        
        self.stats = {
            'total_frames': 0,
            'accepted_frames': 0,
            'rejected_frames': 0,
            'stacking_method': self.config.stacking.method.value,
            'drizzle_enabled': self.use_drizzle,
            'planetary_enabled': self.use_planetary,
            'lucky_enabled': self.use_lucky,
            'memory_mb': 0.0,
        }
        
        print("[RESET] Session réinitialisée")
    
    def _update_memory_stats(self):
        """Met à jour les statistiques mémoire"""
        total_bytes = 0
        
        if self.advanced_stacker:
            mem = self.advanced_stacker.get_memory_usage()
            total_bytes += mem['bytes']
        
        # Buffer d'images
        for img in self.aligned_buffer:
            total_bytes += img.nbytes
        
        self.stats['memory_mb'] = total_bytes / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques courantes"""
        stats = self.stats.copy()

        # SNR théorique : sqrt(N frames acceptées)
        accepted = stats.get('accepted_frames', 0)
        stats['snr_gain'] = float(np.sqrt(max(1, accepted)))

        if self.session:
            stats['session_stacked'] = self.session.config.num_stacked

        if self.advanced_stacker:
            stats['advanced_stats'] = self.advanced_stacker.stats.copy()

        if self.drizzle_stacker:
            stats['drizzle_stats'] = self.drizzle_stacker.stats.copy()

        if self.planetary_aligner:
            stats['planetary_stats'] = self.planetary_aligner.get_stats()

        if self.lucky_stacker:
            stats['lucky_stats'] = self.lucky_stacker.get_stats()

        return stats


# =============================================================================
# Fonction de commodité pour RPiCamera.py
# =============================================================================

def create_advanced_livestack_session(camera_params: Dict, 
                                      preset: str = 'balanced',
                                      **kwargs) -> RPiCameraLiveStackAdvanced:
    """
    Crée une session de live stacking avancée
    
    Args:
        camera_params: Paramètres caméra
        preset: 'fast', 'quality', ou 'balanced'
        **kwargs: Paramètres supplémentaires
    
    Returns:
        RPiCameraLiveStackAdvanced configurée
    
    Example:
        livestack = create_advanced_livestack_session(
            camera_params,
            preset='quality',
            drizzle_enable=True,
            drizzle_scale=2.0
        )
    """
    livestack = RPiCameraLiveStackAdvanced(camera_params)
    livestack.configure(preset=preset, **kwargs)
    return livestack


# =============================================================================
# DOCUMENTATION PARAMÈTRES POUR RPICAMERA.PY
# =============================================================================
"""
PARAMÈTRES À INTÉGRER DANS RPICAMERA.PY
=======================================

Ajouter ces définitions dans la section des limites de RPiCamera.py:

```python
# =============================================================================
# Limites des paramètres Live Stacking avancé
# =============================================================================

# Stacker avancé
livestack_stacker_limits = [
    'ls_stack_method', 0, 4,           # 0=mean, 1=median, 2=kappa_sigma, 3=winsorized, 4=weighted
    'ls_stack_kappa', 10, 40,          # Kappa × 10 (1.0-4.0 → 10-40)
    'ls_stack_iterations', 1, 10,      # Itérations sigma-clip
]

# Planétaire
livestack_planetary_limits = [
    'ls_planetary_enable', 0, 1,       # 0=off, 1=on
    'ls_planetary_mode', 0, 2,         # 0=disk, 1=surface, 2=hybrid
    'ls_planetary_disk_min', 20, 500,  # Rayon min disque (pixels)
    'ls_planetary_disk_max', 100, 2000,# Rayon max disque (pixels)
    'ls_planetary_threshold', 10, 100, # Seuil Canny (10-100)
    'ls_planetary_margin', 5, 50,      # Marge autour disque (pixels)
    'ls_planetary_ellipse', 0, 1,      # 0=cercle, 1=ellipse
    'ls_planetary_window', 0, 2,       # 0=128, 1=256, 2=512
    'ls_planetary_upsample', 1, 20,    # Upsampling sub-pixel
    'ls_planetary_highpass', 0, 1,     # 0=off, 1=on
    'ls_planetary_roi_center', 0, 1,   # 0=off, 1=on
    'ls_planetary_corr', 10, 90,       # Corrélation min (÷100 → 0.1-0.9)
    'ls_planetary_max_shift', 10, 200, # Décalage max (pixels)
]

# Lucky Imaging
livestack_lucky_limits = [
    'ls_lucky_enable', 0, 1,           # 0=off, 1=on
    'ls_lucky_buffer', 50, 500,        # Taille buffer (nb images)
    'ls_lucky_keep', 1, 50,            # % images à garder
    'ls_lucky_score', 0, 3,            # 0=laplacian, 1=gradient, 2=sobel, 3=tenengrad
    'ls_lucky_stack', 0, 2,            # 0=mean, 1=median, 2=sigma_clip
    'ls_lucky_align', 0, 1,            # 0=off, 1=on
    'ls_lucky_roi', 20, 100,           # % ROI pour scoring
]

# Valeurs par défaut
ls_stack_method = 0           # Mean (compatible original)
ls_stack_kappa = 25           # 2.5 (÷10)
ls_stack_iterations = 3

ls_planetary_enable = 0       # Désactivé
ls_planetary_mode = 1         # Surface (FFT)
ls_planetary_disk_min = 50
ls_planetary_disk_max = 500
ls_planetary_threshold = 30
ls_planetary_margin = 10
ls_planetary_ellipse = 0
ls_planetary_window = 1       # 256
ls_planetary_upsample = 10
ls_planetary_highpass = 1
ls_planetary_roi_center = 1
ls_planetary_corr = 30        # 0.30
ls_planetary_max_shift = 100

ls_lucky_enable = 0
ls_lucky_buffer = 100
ls_lucky_keep = 10
ls_lucky_score = 0            # Laplacian
ls_lucky_stack = 0            # Mean
ls_lucky_align = 1
ls_lucky_roi = 50

# Labels pour menus
stack_methods = ['Mean', 'Median', 'Kappa-Sigma', 'Winsorized', 'Weighted']
planetary_modes = ['Disk', 'Surface', 'Hybrid']
planetary_windows = [128, 256, 512]
lucky_score_methods = ['Laplacian', 'Gradient', 'Sobel', 'Tenengrad']
lucky_stack_methods = ['Mean', 'Median', 'Sigma-Clip']
```

MAPPING VERS configure()
========================

Dans la fonction de mise à jour de RPiCamera.py, appeler configure() ainsi:

```python
def update_livestack_config():
    livestack.configure(
        # Stacker
        stacking_method=stack_methods[ls_stack_method].lower().replace('-', '_'),
        kappa=ls_stack_kappa / 10.0,
        iterations=ls_stack_iterations,
        
        # Planétaire
        planetary_enable=bool(ls_planetary_enable),
        planetary_mode=ls_planetary_mode,
        planetary_disk_min=ls_planetary_disk_min,
        planetary_disk_max=ls_planetary_disk_max,
        planetary_disk_threshold=ls_planetary_threshold,
        planetary_disk_margin=ls_planetary_margin,
        planetary_disk_ellipse=bool(ls_planetary_ellipse),
        planetary_window=planetary_windows[ls_planetary_window],
        planetary_upsample=ls_planetary_upsample,
        planetary_highpass=bool(ls_planetary_highpass),
        planetary_roi_center=bool(ls_planetary_roi_center),
        planetary_corr=ls_planetary_corr / 100.0,
        planetary_max_shift=float(ls_planetary_max_shift),
        
        # Lucky
        lucky_enable=bool(ls_lucky_enable),
        lucky_buffer_size=ls_lucky_buffer,
        lucky_keep_percent=float(ls_lucky_keep),
        lucky_score_method=lucky_score_methods[ls_lucky_score].lower(),
        lucky_stack_method=lucky_stack_methods[ls_lucky_stack].lower().replace('-', '_'),
        lucky_align_enabled=bool(ls_lucky_align),
        lucky_score_roi_percent=float(ls_lucky_roi),
    )
```

LOGIQUE D'ACTIVATION
====================

```
ls_lucky_enable = 1 ?
    │
    ├── OUI ──► Mode LUCKY IMAGING
    │            └─► lucky_imaging.py (buffer + sélection + stack)
    │
    └── NON ──► ls_planetary_enable = 1 ?
                    │
                    ├── OUI ──► Mode PLANÉTAIRE
                    │            │
                    │            └─ ls_planetary_mode :
                    │                 0 → DISK (limbe)
                    │                 1 → SURFACE (FFT)
                    │                 2 → HYBRID (combiné)
                    │
                    └── NON ──► Mode DSO (étoiles)
                                 │
                                 └─ ls_stack_method :
                                      0 → Mean (stacker.py)
                                      1-4 → Avancé (stacker_advanced.py)
```

PRÉSETS RACCOURCIS
==================

Pour simplifier l'interface, vous pouvez ajouter un sélecteur de preset:

```python
ls_preset_limits = ['ls_preset', 0, 8]
ls_preset = 0

presets = [
    'Manuel',           # 0 - Paramètres manuels
    'DSO Rapide',       # 1 - get_preset_fast()
    'DSO Qualité',      # 2 - get_preset_quality()
    'DSO Équilibré',    # 3 - get_preset_balanced()
    'Planétaire',       # 4 - get_preset_planetary()
    'Solaire',          # 5 - get_preset_solar()
    'Lunaire',          # 6 - get_preset_lunar()
    'Lucky Rapide',     # 7 - get_preset_lucky_fast()
    'Lucky Qualité',    # 8 - get_preset_lucky_quality()
]

preset_map = {
    1: 'fast',
    2: 'quality', 
    3: 'balanced',
    4: 'planetary',
    5: 'solar',
    6: 'lunar',
    7: 'lucky_fast',
    8: 'lucky_quality',
}

def apply_preset():
    if ls_preset > 0:
        livestack.configure(preset=preset_map[ls_preset])
```
"""


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test RPiCameraLiveStackAdvanced ===\n")
    
    # Params caméra simulés
    camera_params = {
        'exposure': 5000000,  # 5s
        'gain': 100,
        'red': 12,
        'blue': 20
    }
    
    # Test création avec preset
    print("--- Test preset 'quality' ---")
    livestack = create_advanced_livestack_session(
        camera_params,
        preset='quality',
        drizzle_enable=True,
        drizzle_scale=2.0
    )
    
    # Afficher config
    livestack.config.print_summary()
    
    # Test démarrage
    livestack.start()
    
    # Simuler quelques frames
    np.random.seed(42)
    for i in range(5):
        # Image test
        img = np.random.uniform(0.1, 0.3, (100, 100, 3)).astype(np.float32)
        # Ajouter étoiles simulées
        for _ in range(15):
            y, x = np.random.randint(10, 90, 2)
            img[y-2:y+2, x-2:x+2, :] = 0.8
        
        print(f"\n[FRAME {i+1}]")
        result = livestack.process_frame(img)
        
        if result is not None:
            print(f"  Preview shape: {result.shape}")
    
    # Stats
    print("\n--- Stats ---")
    stats = livestack.get_stats()
    for k, v in stats.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    
    # Stop
    livestack.stop()
    
    print("\n=== Tests terminés ===")
