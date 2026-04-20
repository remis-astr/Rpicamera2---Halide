#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'intégration Live Stacking pour RPiCamera.py
Wrapper entre RPiCamera et libastrostack
"""

import sys
import os

# Imports relatifs depuis libastrostack
from . import LiveStackSession, StackingConfig, AlignmentMode, StretchMethod
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
    
    def __init__(self, camera_params, output_dir=None):
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
        # Utiliser ~/stacks par défaut si aucun répertoire n'est spécifié
        if output_dir is None:
            output_dir = os.path.expanduser("~/stacks")
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

        # Verrou pour accès concurrent (process_frame vs stop/reset)
        self._lock = threading.RLock()

        # Validation cohérence shape entre frames (I2)
        self._session_frame_shape = None

        # Compteur frames sauvegardées individuellement (I7)
        self._save_frame_count = 0

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
        - stacking_method: "mean", "kappa_sigma", "winsorized", "median"
        - kappa: float (seuil sigma pour kappa_sigma/winsorized, défaut 2.5)
        - max_frames: int (limite frames empilées, 0 = illimité)
        - png_stretch: "linear", "asinh", "log", "sqrt", "histogram", "auto"
        - png_factor: float
        - preview_refresh: int (toutes les N images)
        - save_dng: "none", "accepted", "all"
        """
        # Méthode de stacking
        if 'stacking_method' in kwargs:
            self.config.stacking_method = kwargs['stacking_method']
        if 'kappa' in kwargs:
            self.config.stacking_kappa = float(kwargs['kappa'])
        if 'max_frames' in kwargs:
            self.config.max_frames = int(kwargs['max_frames'])

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

        # Calibration RAW
        if 'raw_black_level' in kwargs:
            self.config.raw_black_level = int(kwargs['raw_black_level'])
        if 'gradient_removal' in kwargs:
            self.config.gradient_removal = bool(kwargs['gradient_removal'])
        if 'gradient_removal_tiles' in kwargs:
            self.config.gradient_removal_tiles = int(kwargs['gradient_removal_tiles'])
    
    def start(self):
        """Démarre session live stacking"""
        if self.is_running:
            return
        
        print("\n[LIVESTACK] Démarrage session...")
        
        # Créer session
        self.session = LiveStackSession(self.config)
        self.session.start()
        
        # Reset compteurs et état
        self.frame_count = 0
        self.total_frames = 0
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.start_time = datetime.now()
        self._session_frame_shape = None
        self._save_frame_count = 0
        
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
        
        # Arrêter session (avec verrou pour éviter accès concurrent depuis process_frame)
        with self._lock:
            if self.session:
                self.session.stop()
                self.session = None
            # Libérer ressources mémoire (I10)
            self.last_preview = None
            self._session_frame_shape = None

        self.save_thread = None

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
        if not self.is_running or self.is_paused:
            return False

        with self._lock:
            if self.session is None:
                return False

            self.frame_count += 1
            self.total_frames += 1

            # Convertir en float32 en préservant la plage native
            # uint8  → [0-255] float32  (pas de ×256 : évite valeurs hors-plage ISP)
            # uint16 → [0-65535] float32
            # float32 → inchangé
            image_data = camera_array.astype(np.float32)

            # Validation cohérence shape entre frames (I2)
            if self._session_frame_shape is None:
                self._session_frame_shape = image_data.shape
            elif image_data.shape != self._session_frame_shape:
                print(f"[LIVESTACK] Frame ignorée : shape {image_data.shape} "
                      f"!= attendu {self._session_frame_shape}")
                self.total_frames -= 1
                self.frame_count -= 1
                return False

            # Traiter avec libastrostack
            # Pour save_dng_mode, comparer num_stacked avant/après pour détecter acceptance (I7)
            n_before = self.session.config.num_stacked
            result = self.session.process_image_data(image_data)
            n_after = self.session.config.num_stacked

            # Mettre à jour compteurs (depuis session = source de vérité, I15)
            self.accepted_frames = n_after
            self.rejected_frames = self.session.files_rejected

            # Enregistrer frame individuelle si save_dng_mode actif (I7)
            if self.config.save_dng_mode != 'none':
                frame_accepted = n_after > n_before
                should_save = (
                    self.config.save_dng_mode == 'all' or
                    (self.config.save_dng_mode == 'accepted' and frame_accepted)
                )
                if should_save:
                    self._save_frame_count += 1
                    frame_path = (self.output_dir / "frames" /
                                  f"frame_{self._save_frame_count:05d}.png")
                    self.save_queue.put((image_data.copy(), str(frame_path)))

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

        # Lire compteurs depuis la session (source de vérité) si disponible (I15)
        with self._lock:
            if self.session is not None:
                accepted = self.session.config.num_stacked
                rejected = self.session.files_rejected
            else:
                accepted = self.accepted_frames
                rejected = self.rejected_frames

        return {
            'total_frames': self.total_frames,
            'accepted': accepted,
            'rejected': rejected,
            'rate': self.total_frames / elapsed if elapsed > 0 else 0,
            'elapsed': elapsed,
            'snr_gain': np.sqrt(accepted) if accepted > 0 else 1.0,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'preview_updated': self.last_preview_update
        }
    
    def save_result(self, filename=None, raw_format_name=None):
        """
        Sauvegarde résultat final

        Args:
            filename: Nom fichier optionnel (sinon timestamp)
            raw_format_name: Nom du format RAW actuel (YUV420/SRGGB12/SRGGB16)

        Returns:
            Path du fichier sauvegardé
        """
        if self.session is None or self.accepted_frames == 0:
            print("[LIVESTACK] Rien à sauvegarder")
            return None

        # Nom fichier
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ajouter le format RAW au nom de fichier (priorité au paramètre)
            raw_format_str = raw_format_name or self.camera_params.get('raw_format', 'YUV420')
            filename = f"stack_{raw_format_str}_{timestamp}.fit"

        output_path = self.output_dir / filename
        
        print(f"[LIVESTACK] Sauvegarde: {output_path}")
        
        # Sauvegarder
        self.session.save_result(str(output_path))
        
        print(f"[LIVESTACK] ✓ Sauvegardé")
        
        return output_path
    
    def reset(self):
        """Réinitialise session (nouvelle cible)"""
        with self._lock:
            if self.session:
                self.session.reset()

        self.frame_count = 0
        self.total_frames = 0
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.last_preview = None
        self.last_preview_update = 0
        self._session_frame_shape = None
        self._save_frame_count = 0

        print("[LIVESTACK] Session réinitialisée")
    
    def _save_worker(self):
        """Worker thread pour sauvegarde frames individuelles en PNG 16-bit (I7)"""
        import cv2
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        while True:
            item = self.save_queue.get()

            if item is None:  # Signal stop
                break

            image_data, path = item
            try:
                # Normaliser vers uint16 selon la plage des données
                dmax = float(image_data.max())
                if dmax <= 1.0:
                    data_16 = (np.clip(image_data, 0, 1) * 65535).astype(np.uint16)
                elif dmax <= 300:
                    data_16 = (np.clip(image_data / 255.0, 0, 1) * 65535).astype(np.uint16)
                elif dmax <= 5000:
                    data_16 = (np.clip(image_data / 4095.0, 0, 1) * 65535).astype(np.uint16)
                else:
                    data_16 = np.clip(image_data, 0, 65535).astype(np.uint16)

                # Sauvegarder (pas de conversion couleur : même espace que le stack)
                cv2.imwrite(path, data_16)
            except Exception as e:
                print(f"[LIVESTACK] Erreur sauvegarde frame {path}: {e}")

    def save(self, filename=None, raw_format_name=None):
        """Alias pour save_result() (compatibilité RPiCamera2.py)"""
        return self.save_result(filename, raw_format_name)

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
def create_livestack_session(camera_params, output_dir=None):
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
