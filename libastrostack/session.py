#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session de live stacking - API principale
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import StackingConfig
from .quality import QualityAnalyzer
from .aligner import AdvancedAligner
from .stacker import ImageStacker
from .io import load_image, save_fits
from .stretch import apply_stretch


class LiveStackSession:
    """
    Session de live stacking - API principale
    
    Usage pour RPiCamera.py:
        config = StackingConfig()
        session = LiveStackSession(config)
        session.start()
        
        # Pour chaque frame de la caméra
        result = session.process_image_data(camera_array)
        
        # Sauvegarder
        session.save_result("output.fit")
        session.stop()
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: StackingConfig instance (ou None pour défaut)
        """
        self.config = config if config else StackingConfig()
        self.config.validate()
        
        # Composants
        self.quality_analyzer = QualityAnalyzer(self.config.quality)
        self.aligner = AdvancedAligner(self.config)
        self.stacker = ImageStacker(self.config)
        
        # État
        self.is_running = False
        self.files_processed = 0
        self.files_rejected = 0
        self.files_failed = 0
        self.is_color = None
        self.rotation_angles = []
        
        # Compteur pour rafraîchissement preview
        self._frame_count_since_refresh = 0
    
    def start(self):
        """Démarre la session"""
        print("\n" + "="*60)
        print(">>> LIBASTROSTACK SESSION")
        print("="*60)
        
        print("\n[CONFIG]")
        print(f"  - Mode alignement: {self.config.alignment_mode}")
        print(f"  - Contrôle qualité: {'OUI' if self.config.quality.enable else 'NON'}")
        print(f"  - Étirement PNG: {self.config.png_stretch_method}")
        print(f"  - Preview refresh: toutes les {self.config.preview_refresh_interval} images")
        print(f"  - Save DNG: {self.config.save_dng_mode}")
        
        if self.config.quality.enable:
            print(f"\n[QC] Seuils:")
            print(f"  - FWHM max: {self.config.quality.max_fwhm} px")
            print(f"  - Ellipticité max: {self.config.quality.max_ellipticity}")
            print(f"  - Étoiles min: {self.config.quality.min_stars}")
            print(f"  - Drift max: {self.config.quality.max_drift} px")
            print(f"  - Netteté min: {self.config.quality.min_sharpness}")
        
        self.is_running = True
        print("\n[OK] Session prête !\n")
    
    def process_image_data(self, image_data):
        """
        Traite une image directement depuis un array (pour RPiCamera)
        
        Args:
            image_data: Array NumPy (float32) RGB ou MONO
        
        Returns:
            Image empilée courante (ou None si rejetée)
        """
        if not self.is_running:
            return None
        
        print(f"\n[IMG] Frame {self.files_processed + self.files_rejected + 1}")
        
        try:
            # Détecter type si première image
            if self.is_color is None:
                self.is_color = len(image_data.shape) == 3
                print(f"[DETECT] Mode {'RGB' if self.is_color else 'MONO'}")
            
            # 1. Contrôle qualité
            is_good, metrics, reason = self.quality_analyzer.analyze(image_data)
            
            if is_good:
                print(f"  [QC] OK - FWHM:{metrics.get('median_fwhm', 0):.2f}px, "
                      f"Ell:{metrics.get('median_ellipticity', 0):.2f}, "
                      f"Sharp:{metrics.get('sharpness', 0):.2f}, "
                      f"Stars:{metrics.get('num_stars', 0)}")
            else:
                print(f"  [QC] REJECT - {reason}")
                self.files_rejected += 1
                self.config.quality.rejected_images.append(f"frame_{self.files_processed + self.files_rejected}")
                self.config.quality.rejection_reasons[f"frame_{self.files_processed + self.files_rejected}"] = reason
                return None
            
            # 2. Alignement
            # Si mode OFF, sauter l'alignement
            if self.config.alignment_mode.upper() == "OFF" or self.config.alignment_mode.upper() == "NONE":
                print("  [ALIGN] Mode OFF - Pas d'alignement")
                aligned = image_data
                params = {'dx': 0, 'dy': 0, 'angle': 0, 'scale': 1.0}
                success = True
            else:
                print("  [ALIGN] Alignement...")
                aligned, params, success = self.aligner.align(image_data)

                if not success:
                    print("  [FAIL] Échec alignement")
                    self.files_rejected += 1
                    return None
            
            # Enregistrer rotation
            if 'angle' in params:
                self.rotation_angles.append(params['angle'])
            
            # 3. Empilement
            print("  [STACK] Empilement...")
            result = self.stacker.stack(aligned)
            
            self.files_processed += 1
            self._frame_count_since_refresh += 1
            
            self._print_stats()
            
            # Retourner résultat si rafraîchissement nécessaire
            if self._frame_count_since_refresh >= self.config.preview_refresh_interval:
                self._frame_count_since_refresh = 0
                return result
            
            return None  # Pas de rafraîchissement preview
            
        except Exception as e:
            print(f"[ERROR] {e}")
            self.files_failed += 1
            import traceback
            traceback.print_exc()
            return None
    
    def process_image_file(self, image_path):
        """
        Traite une image depuis un fichier (pour batch processing)
        
        Args:
            image_path: Chemin vers fichier image
        
        Returns:
            Image empilée courante (ou None si rejetée)
        """
        print(f"\n[IMG] {Path(image_path).name}")
        
        image_data = load_image(image_path)
        if image_data is None:
            self.files_failed += 1
            return None
        
        return self.process_image_data(image_data)
    
    def get_current_stack(self):
        """
        Retourne l'image empilée courante
        
        Returns:
            Image empilée (copie) ou None
        """
        return self.stacker.get_result()
    
    def get_preview_png(self):
        """
        Génère preview PNG étiré pour affichage
        
        Returns:
            Array uint8 (0-255) pour affichage direct
        """
        result = self.stacker.get_result()
        if result is None:
            return None
        
        # Appliquer étirement
        is_color = len(result.shape) == 3
        
        if is_color:
            stretched = np.zeros_like(result)
            for i in range(3):
                stretched[:, :, i] = apply_stretch(
                    result[:, :, i],
                    method=self.config.png_stretch_method,
                    factor=self.config.png_stretch_factor,
                    clip_low=self.config.png_clip_low,
                    clip_high=self.config.png_clip_high
                )
        else:
            stretched = apply_stretch(
                result,
                method=self.config.png_stretch_method,
                factor=self.config.png_stretch_factor,
                clip_low=self.config.png_clip_low,
                clip_high=self.config.png_clip_high
            )
        
        # Convertir en uint8
        preview = (stretched * 255).astype(np.uint8)
        
        return preview
    
    def save_result(self, output_path, generate_png=None):
        """
        Sauvegarde résultat final
        
        Args:
            output_path: Chemin de sortie (.fit)
            generate_png: Générer PNG (défaut = config.auto_png)
        """
        result = self.stacker.get_result()
        if result is None:
            print("[WARN] Aucune image empilée")
            return
        
        output_path = Path(output_path)
        
        # Métadonnées FITS
        header_data = {
            'STACKED': self.config.num_stacked,
            'REJECTED': self.files_rejected,
            'ALIGNMOD': self.config.alignment_mode,
        }
        
        if self.rotation_angles:
            header_data['ROTMIN'] = np.min(self.rotation_angles)
            header_data['ROTMAX'] = np.max(self.rotation_angles)
            header_data['ROTMED'] = np.median(self.rotation_angles)
        
        # Sauvegarder FITS
        save_fits(result, output_path, header_data)
        
        print(f"\n[SAVE] FITS: {output_path}")
        print(f"       Acceptées: {self.config.num_stacked}, Rejetées: {self.files_rejected}")
        
        if self.rotation_angles:
            print(f"       Rotation: {np.min(self.rotation_angles):.2f}° à {np.max(self.rotation_angles):.2f}°")
        
        # PNG si demandé
        if generate_png is None:
            generate_png = self.config.auto_png
        
        if generate_png:
            png_path = output_path.with_suffix('.png')
            self._save_png(result, png_path)
        
        # Rapport qualité si rejets
        if self.files_rejected > 0 and self.config.save_rejected_list:
            report_path = output_path.with_suffix('.quality_report.txt')
            self._save_quality_report(report_path)
    
    def stop(self):
        """Arrête la session"""
        self.is_running = False
        print("\n" + "="*60)
        print("[STOP] SESSION TERMINÉE")
        print("="*60)
        self._print_final_stats()
    
    def reset(self):
        """Réinitialise la session (garde config)"""
        self.stacker.reset()
        self.aligner.reference_image = None
        self.aligner.reference_stars = None
        self.files_processed = 0
        self.files_rejected = 0
        self.files_failed = 0
        self.rotation_angles = []
        self._frame_count_since_refresh = 0
        self.config.quality.rejected_images = []
        self.config.quality.rejection_reasons = {}
    
    def _save_png(self, data, png_path):
        """Sauvegarde PNG pure sans surcharge"""
        preview = self.get_preview_png()
        if preview is None:
            return
        
        # Sauvegarder image pure
        import cv2
        
        if len(preview.shape) == 3:
            # Convertir RGB en BGR pour OpenCV
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(png_path), preview_bgr)
        else:
            cv2.imwrite(str(png_path), preview)
        
        print(f"[OK] PNG: {png_path}")
    
    def _save_quality_report(self, report_path):
        """Sauvegarde rapport qualité"""
        with open(report_path, 'w') as f:
            f.write("=== RAPPORT DE CONTRÔLE QUALITÉ ===\n\n")
            f.write(f"Images acceptées: {self.config.num_stacked}\n")
            f.write(f"Images rejetées: {self.files_rejected}\n")
            
            total = self.config.num_stacked + self.files_rejected
            if total > 0:
                f.write(f"Taux d'acceptation: {100*self.config.num_stacked/total:.1f}%\n\n")
            
            f.write("Images rejetées:\n")
            for img_name in self.config.quality.rejected_images:
                reason = self.config.quality.rejection_reasons.get(img_name, "?")
                f.write(f"  - {img_name}: {reason}\n")
        
        print(f"[SAVE] Rapport: {report_path}")
    
    def _print_stats(self):
        """Affiche stats courantes"""
        print(f"  [STATS] Empilées: {self.config.num_stacked}, "
              f"Rejetées: {self.files_rejected}, Échecs: {self.files_failed}")
    
    def _print_final_stats(self):
        """Affiche stats finales"""
        print(f"\n[STATS] Final:")
        print(f"  * Type: {'RGB' if self.is_color else 'MONO'}")
        print(f"  * Mode: {self.config.alignment_mode}")
        print(f"  * Empilées: {self.config.num_stacked}")
        print(f"  * Rejetées: {self.files_rejected}")
        print(f"  * Échecs: {self.files_failed}")
        
        total = self.config.num_stacked + self.files_rejected
        if total > 0:
            print(f"  * Taux acceptation: {100*self.config.num_stacked/total:.1f}%")
        
        if self.rotation_angles:
            print(f"  * Rotation: {np.min(self.rotation_angles):.2f}° à {np.max(self.rotation_angles):.2f}°")
        
        if self.config.num_stacked > 0:
            snr_gain = self.stacker.get_snr_improvement()
            print(f"  * SNR gain: x{snr_gain:.2f}")
