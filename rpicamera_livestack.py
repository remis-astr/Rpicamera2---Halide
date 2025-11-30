#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'intégration Live Stacking pour RPiCamera.py
Wrapper entre RPiCamera et libastrostack
"""

import sys
import os

# Ajouter path libastrostack (dans le même répertoire)
LIBASTROSTACK_PATH = os.path.dirname(__file__)
sys.path.insert(0, LIBASTROSTACK_PATH)

from libastrostack import LiveStackSession, StackingConfig, AlignmentMode, StretchMethod
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import queue


class RPiCameraLiveStack:
    """
    Wrapper Live Stacking pour RPiCamera.py
    
    Gère:
    - Session libastrostack
    - Thread de sauvegarde DNG
    - Compteurs et stats
    - Génération preview PyGame
    """
    
    def __init__(self, camera_params, output_dir="/media/admin/THKAILAR/Stacks"):
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
        
        # Configuration libastrostack
        self.config = StackingConfig()
        self.session = None
        
        # État
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.last_preview = None
        self.last_preview_update = 0
        
        # Thread sauvegarde
        self.save_queue = queue.Queue()
        self.save_thread = None
        
        # Statistiques
        self.start_time = None
        self.total_frames = 0
        self.accepted_frames = 0
        self.rejected_frames = 0
    
    def configure(self, **kwargs):
        """
        Configure paramètres stacking

        Paramètres disponibles:
        - alignment_mode: "translation", "rotation", "affine"
        - enable_qc: bool (activer/désactiver le contrôle qualité)
        - max_fwhm: float
        - max_ellipticity: float
        - min_stars: int
        - max_drift: float
        - min_sharpness: float
        - max_rotation: float (rotation max en degrés)
        - min_scale: float (scale minimum)
        - max_scale: float (scale maximum)
        - png_stretch: "linear", "asinh", "log", "sqrt", "histogram", "auto"
        - png_factor: float
        - preview_refresh: int (toutes les N images)
        - save_dng: "none", "accepted", "all"
        """
        # Alignement
        if 'alignment_mode' in kwargs:
            self.config.alignment_mode = kwargs['alignment_mode']

        # Contrôle qualité - activation/désactivation
        if 'enable_qc' in kwargs:
            self.config.quality.enable = bool(kwargs['enable_qc'])

        # Contrôle qualité - seuils
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

        # Seuils d'alignement (pour éviter transformations aberrantes)
        if 'max_rotation' in kwargs:
            self.config.quality.max_rotation = float(kwargs['max_rotation'])
        if 'min_scale' in kwargs:
            self.config.quality.min_scale = float(kwargs['min_scale'])
        if 'max_scale' in kwargs:
            self.config.quality.max_scale = float(kwargs['max_scale'])
        if 'min_inliers_ratio' in kwargs:
            self.config.quality.min_inliers_ratio = float(kwargs['min_inliers_ratio'])
        
        # PNG stretch
        if 'png_stretch' in kwargs:
            self.config.png_stretch_method = kwargs['png_stretch']
        if 'png_factor' in kwargs:
            self.config.png_stretch_factor = float(kwargs['png_factor'])
        if 'png_clip_low' in kwargs:
            self.config.png_clip_low = float(kwargs['png_clip_low'])
        if 'png_clip_high' in kwargs:
            self.config.png_clip_high = float(kwargs['png_clip_high'])
        
        # Preview
        if 'preview_refresh' in kwargs:
            self.config.preview_refresh_interval = int(kwargs['preview_refresh'])
        
        # Sauvegarde
        if 'save_dng' in kwargs:
            self.config.save_dng_mode = kwargs['save_dng']
    
    def start(self):
        """Démarre session live stacking"""
        if self.is_running:
            return
        
        print("\n[LIVESTACK] Démarrage session...")
        
        # Créer session
        self.session = LiveStackSession(self.config)
        self.session.start()
        
        # Reset compteurs
        self.frame_count = 0
        self.total_frames = 0
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.start_time = datetime.now()
        
        # Démarrer thread sauvegarde (si nécessaire)
        if self.config.save_dng_mode != "none":
            self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self.save_thread.start()
        
        self.is_running = True
        self.is_paused = False
        
        print("[LIVESTACK] Session démarrée")
    
    def stop(self):
        """Arrête session"""
        if not self.is_running:
            return
        
        print("\n[LIVESTACK] Arrêt session...")
        
        self.is_running = False
        
        # Attendre thread sauvegarde
        if self.save_thread and self.save_thread.is_alive():
            self.save_queue.put(None)  # Signal stop
            self.save_thread.join(timeout=5)
        
        # Arrêter session
        if self.session:
            self.session.stop()
            self.session = None
        
        print("[LIVESTACK] Session arrêtée")
    
    def pause(self):
        """Met en pause (n'empile plus mais continue capture)"""
        self.is_paused = True
        print("[LIVESTACK] Pause")
    
    def resume(self):
        """Reprend empilement"""
        self.is_paused = False
        print("[LIVESTACK] Reprise")
    
    def process_frame(self, camera_array):
        """
        Traite une frame caméra
        
        Args:
            camera_array: Array NumPy (RGB uint8 ou float32)
        
        Returns:
            preview_updated: bool, True si preview rafraîchi
        """
        if not self.is_running or self.is_paused or self.session is None:
            return False
        
        self.frame_count += 1
        self.total_frames += 1
        
        # Convertir en float32 si nécessaire
        if camera_array.dtype == np.uint8:
            image_data = camera_array.astype(np.float32) * 256.0
        else:
            image_data = camera_array.astype(np.float32)
        
        # Traiter avec libastrostack
        result = self.session.process_image_data(image_data)
        
        # Mettre à jour compteurs
        self.accepted_frames = self.session.config.num_stacked
        self.rejected_frames = self.session.files_rejected
        
        # Preview rafraîchi ?
        if result is not None:
            self.last_preview = self.session.get_preview_png()
            self.last_preview_update = self.frame_count
            return True
        
        return False
    
    def get_preview_surface(self, pygame_module, target_size=None):
        """
        Obtient surface PyGame pour affichage
        
        Args:
            pygame_module: Module pygame
            target_size: (width, height) optionnel pour redimensionner
        
        Returns:
            pygame.Surface ou None
        """
        if self.last_preview is None:
            return None
        
        try:
            # Convertir en surface PyGame
            if len(self.last_preview.shape) == 3:
                # RGB : transposer (H,W,C) -> (W,H,C)
                surface = pygame_module.surfarray.make_surface(
                    self.last_preview.transpose(1, 0, 2)
                )
            else:
                # MONO
                surface = pygame_module.surfarray.make_surface(
                    self.last_preview.T
                )
            
            # Redimensionner si demandé
            if target_size:
                surface = pygame_module.transform.scale(surface, target_size)
            
            return surface
        except Exception as e:
            print(f"[LIVESTACK] Erreur preview: {e}")
            return None
    
    def get_stats(self):
        """
        Retourne statistiques pour affichage
        
        Returns:
            dict avec stats
        """
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'total_frames': self.total_frames,
            'accepted': self.accepted_frames,
            'rejected': self.rejected_frames,
            'rate': self.total_frames / elapsed if elapsed > 0 else 0,
            'elapsed': elapsed,
            'snr_gain': np.sqrt(self.accepted_frames) if self.accepted_frames > 0 else 1.0,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'preview_updated': self.last_preview_update
        }
    
    def save_result(self, filename=None):
        """
        Sauvegarde résultat final
        
        Args:
            filename: Nom fichier optionnel (sinon timestamp)
        
        Returns:
            Path du fichier sauvegardé
        """
        if self.session is None or self.accepted_frames == 0:
            print("[LIVESTACK] Rien à sauvegarder")
            return None
        
        # Nom fichier
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stack_{timestamp}.fit"
        
        output_path = self.output_dir / filename
        
        print(f"[LIVESTACK] Sauvegarde: {output_path}")
        
        # Sauvegarder
        self.session.save_result(str(output_path))
        
        print(f"[LIVESTACK] ✓ Sauvegardé")
        
        return output_path
    
    def reset(self):
        """Réinitialise session (nouvelle cible)"""
        if self.session:
            self.session.reset()
        
        self.frame_count = 0
        self.total_frames = 0
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.last_preview = None
        self.last_preview_update = 0
        
        print("[LIVESTACK] Session réinitialisée")
    
    def _save_worker(self):
        """Worker thread pour sauvegarde DNG asynchrone"""
        while True:
            item = self.save_queue.get()
            
            if item is None:  # Signal stop
                break
            
            # Sauvegarder DNG (placeholder)
            # TODO: Implémenter sauvegarde DNG réelle
            pass
    
    def get_config_summary(self):
        """Retourne résumé config pour affichage"""
        return {
            'alignment': self.config.alignment_mode,
            'max_fwhm': self.config.quality.max_fwhm,
            'min_stars': self.config.quality.min_stars,
            'max_drift': self.config.quality.max_drift,
            'stretch': self.config.png_stretch_method,
            'factor': self.config.png_stretch_factor,
            'preview_refresh': self.config.preview_refresh_interval,
            'save_dng': self.config.save_dng_mode
        }


# Fonction helper pour RPiCamera.py
def create_livestack_session(camera_params, output_dir="/media/admin/THKAILAR/Stacks"):
    """
    Crée une session live stacking prête à l'emploi
    
    Usage dans RPiCamera.py:
        from rpicamera_livestack import create_livestack_session
        
        livestack = create_livestack_session({
            'exposure': 10000,
            'gain': 10.0,
            'red': 1.5,
            'blue': 1.2
        })
        
        livestack.start()
    """
    return RPiCameraLiveStack(camera_params, output_dir)
