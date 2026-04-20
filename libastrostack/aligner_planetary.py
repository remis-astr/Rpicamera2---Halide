#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignement planétaire pour libastrostack
========================================

Module d'alignement spécialisé pour les objets étendus :
- Soleil (avec ou sans filtre H-alpha)
- Lune (phases partielles ou pleine)
- Planètes (Jupiter, Saturne, Mars, etc.)

Méthodes d'alignement :
- DISK : Détection du limbe (bord) et centrage sur le disque
- SURFACE : Corrélation de phase pour aligner sur les détails de surface
- HYBRID : Combinaison disque + surface pour meilleure précision

Principe :
1. Mode DISK : Détecte le cercle/ellipse du limbe via Hough ou fit,
   puis centre toutes les images sur le centre du disque.
   Idéal pour : Soleil blanc, Lune pleine, disques planétaires.

2. Mode SURFACE : Utilise la corrélation de phase (FFT) pour
   trouver le décalage sub-pixel entre images basé sur les
   détails de surface (taches solaires, cratères lunaires, bandes).
   Idéal pour : H-alpha solaire, détails planétaires.

3. Mode HYBRID : D'abord centrage grossier par disque, puis
   affinement par corrélation de surface.

Références :
- Hough Circle Transform (OpenCV)
- Phase Correlation (Reddy & Chatterji, 1996)
- Ellipse Fitting pour disques non-circulaires

Usage :
    from libastrostack.aligner_planetary import PlanetaryAligner, PlanetaryMode
    
    aligner = PlanetaryAligner(config)
    aligner.set_mode(PlanetaryMode.SURFACE)
    
    aligned, params, success = aligner.align(image)

Auteur: libastrostack Team
Version: 1.0.0
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import warnings


class PlanetaryMode(Enum):
    """Modes d'alignement planétaire"""
    DISK = "disk"           # Alignement sur le limbe du disque
    SURFACE = "surface"     # Corrélation de phase sur les détails
    HYBRID = "hybrid"       # Disque + surface combinés


class PlanetaryConfig:
    """Configuration pour l'alignement planétaire"""
    
    def __init__(self):
        # Mode
        self.mode = PlanetaryMode.SURFACE
        
        # Détection de disque
        self.disk_min_radius = 50       # Rayon minimum en pixels
        self.disk_max_radius = 2000     # Rayon maximum en pixels
        self.disk_threshold = 30        # Seuil Canny pour détection bord
        self.disk_margin = 10           # Marge autour du disque (pixels)
        self.disk_use_ellipse = False   # Détecter ellipse au lieu de cercle
        
        # Corrélation de surface
        self.surface_window_size = 256  # Taille fenêtre FFT (puissance de 2)
        self.surface_upsample = 10      # Facteur upsampling sub-pixel
        self.surface_highpass = True    # Appliquer filtre passe-haut
        self.surface_roi_center = True  # ROI au centre du disque
        
        # Seuils de qualité
        self.max_shift = 100            # Décalage max accepté (pixels)
        self.min_correlation = 0.3      # Corrélation minimum
        self.max_rotation = 2.0         # Rotation max (degrés) - planètes tournent peu
        
        # Post-traitement
        self.apply_derotation = False   # Compenser rotation planétaire
        self.rotation_rate = 0.0        # Degrés/seconde (ex: Jupiter ~0.01°/s)
    
    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        if self.disk_min_radius <= 0:
            errors.append("disk_min_radius doit être > 0")
        
        if self.disk_max_radius <= self.disk_min_radius:
            errors.append("disk_max_radius doit être > disk_min_radius")
        
        if self.surface_window_size < 64:
            errors.append("surface_window_size doit être >= 64")
        
        if self.surface_upsample < 1:
            errors.append("surface_upsample doit être >= 1")
        
        if errors:
            raise ValueError("Config planétaire invalide: " + ", ".join(errors))
        
        return True


class DiskDetector:
    """
    Détecteur de disque (limbe) pour objets étendus
    
    Utilise la transformée de Hough pour cercles ou
    le fit d'ellipse pour disques non-circulaires.
    """
    
    def __init__(self, config: PlanetaryConfig):
        self.config = config
    
    def detect(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Détecte le disque dans l'image
        
        Args:
            image: Image en niveaux de gris (float ou uint8)
        
        Returns:
            Dict avec center_x, center_y, radius (ou None si échec)
            Pour ellipse: ajoute major_axis, minor_axis, angle
        """
        # Convertir en uint8 si nécessaire
        if image.dtype != np.uint8:
            img_norm = image - np.min(image)
            if np.max(img_norm) > 0:
                img_norm = (img_norm / np.max(img_norm) * 255).astype(np.uint8)
            else:
                return None
        else:
            img_norm = image
        
        # Appliquer flou pour réduire bruit
        blurred = cv2.GaussianBlur(img_norm, (9, 9), 2)
        
        if self.config.disk_use_ellipse:
            return self._detect_ellipse(blurred)
        else:
            return self._detect_circle(blurred)
    
    def _detect_circle(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Détection de cercle par Hough"""
        # Détecter contours
        edges = cv2.Canny(image, 
                         self.config.disk_threshold, 
                         self.config.disk_threshold * 2)
        
        # Hough circles
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=image.shape[0] // 2,  # Un seul disque
            param1=50,
            param2=30,
            minRadius=self.config.disk_min_radius,
            maxRadius=min(self.config.disk_max_radius, 
                         min(image.shape) // 2)
        )
        
        if circles is None or len(circles) == 0:
            # Fallback : utiliser les moments de l'image
            return self._detect_by_moments(image)
        
        # Prendre le cercle le plus grand (probablement le disque)
        circle = circles[0][0]
        
        return {
            'center_x': float(circle[0]),
            'center_y': float(circle[1]),
            'radius': float(circle[2]),
            'method': 'hough_circle'
        }
    
    def _detect_ellipse(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Détection d'ellipse par fit de contours"""
        # Seuillage adaptatif
        _, thresh = cv2.threshold(image, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Trouver contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._detect_by_moments(image)
        
        # Prendre le plus grand contour
        largest = max(contours, key=cv2.contourArea)
        
        if len(largest) < 5:  # Minimum pour fitEllipse
            return self._detect_by_moments(image)
        
        # Fit ellipse
        ellipse = cv2.fitEllipse(largest)
        (cx, cy), (width, height), angle = ellipse
        
        return {
            'center_x': float(cx),
            'center_y': float(cy),
            'radius': float((width + height) / 4),  # Rayon moyen
            'major_axis': float(max(width, height) / 2),
            'minor_axis': float(min(width, height) / 2),
            'angle': float(angle),
            'method': 'ellipse_fit'
        }
    
    def _detect_by_moments(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Fallback : détection par moments (centroïde)"""
        # Seuillage
        _, thresh = cv2.threshold(image, np.median(image), 255, cv2.THRESH_BINARY)
        
        # Moments
        M = cv2.moments(thresh)
        
        if M['m00'] == 0:
            return None
        
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        # Estimer rayon par aire
        area = M['m00']
        radius = np.sqrt(area / np.pi)
        
        return {
            'center_x': float(cx),
            'center_y': float(cy),
            'radius': float(radius),
            'method': 'moments'
        }


class SurfaceCorrelator:
    """
    Corrélation de phase pour alignement sub-pixel
    
    Utilise la FFT pour calculer le décalage entre deux images
    basé sur leurs détails de surface.
    """
    
    def __init__(self, config: PlanetaryConfig):
        self.config = config
        self._window = None
        self._window_size = None
    
    def _get_window(self, size: int) -> np.ndarray:
        """Retourne une fenêtre de Hanning 2D"""
        if self._window is None or self._window_size != size:
            hann = np.hanning(size)
            self._window = np.outer(hann, hann)
            self._window_size = size
        return self._window
    
    def _highpass_filter(self, image: np.ndarray) -> np.ndarray:
        """Applique un filtre passe-haut pour accentuer les détails"""
        # Filtre Laplacien + image originale
        laplacian = cv2.Laplacian(image.astype(np.float64), cv2.CV_64F)
        filtered = image.astype(np.float64) - 0.5 * laplacian
        return filtered
    
    def correlate(self, ref_image: np.ndarray, cur_image: np.ndarray,
                  roi: Optional[Tuple[int, int, int, int]] = None
                  ) -> Tuple[float, float, float]:
        """
        Calcule le décalage par corrélation de phase
        
        Args:
            ref_image: Image de référence (grayscale)
            cur_image: Image courante (grayscale)
            roi: (x, y, width, height) région d'intérêt optionnelle
        
        Returns:
            (dx, dy, correlation) - décalage en pixels et score
        """
        # Extraire ROI si spécifié
        if roi is not None:
            x, y, w, h = roi
            ref = ref_image[y:y+h, x:x+w].astype(np.float64)
            cur = cur_image[y:y+h, x:x+w].astype(np.float64)
        else:
            ref = ref_image.astype(np.float64)
            cur = cur_image.astype(np.float64)
        
        # Redimensionner si nécessaire
        target_size = self.config.surface_window_size
        if ref.shape[0] != target_size or ref.shape[1] != target_size:
            ref = cv2.resize(ref, (target_size, target_size))
            cur = cv2.resize(cur, (target_size, target_size))
        
        # Appliquer fenêtre
        window = self._get_window(target_size)
        ref = ref * window
        cur = cur * window
        
        # Filtre passe-haut optionnel
        if self.config.surface_highpass:
            ref = self._highpass_filter(ref)
            cur = self._highpass_filter(cur)
        
        # FFT
        ref_fft = fft2(ref)
        cur_fft = fft2(cur)
        
        # Cross-power spectrum
        cross_power = (ref_fft * np.conj(cur_fft)) / (np.abs(ref_fft * np.conj(cur_fft)) + 1e-10)
        
        # Corrélation de phase
        correlation = np.real(ifft2(cross_power))
        correlation = fftshift(correlation)
        
        # Trouver le pic
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_value = correlation[peak_idx]
        
        # Décalage brut
        dy = peak_idx[0] - target_size // 2
        dx = peak_idx[1] - target_size // 2
        
        # Raffinement sub-pixel par interpolation parabolique
        if 1 < peak_idx[0] < target_size - 2 and 1 < peak_idx[1] < target_size - 2:
            # Y
            y_vals = correlation[peak_idx[0]-1:peak_idx[0]+2, peak_idx[1]]
            if y_vals[0] + y_vals[2] - 2*y_vals[1] != 0:
                dy_sub = (y_vals[0] - y_vals[2]) / (2 * (y_vals[0] + y_vals[2] - 2*y_vals[1]))
                dy += dy_sub
            
            # X
            x_vals = correlation[peak_idx[0], peak_idx[1]-1:peak_idx[1]+2]
            if x_vals[0] + x_vals[2] - 2*x_vals[1] != 0:
                dx_sub = (x_vals[0] - x_vals[2]) / (2 * (x_vals[0] + x_vals[2] - 2*x_vals[1]))
                dx += dx_sub
        
        # Calculer facteur d'échelle si ROI redimensionné
        if roi is not None:
            scale_x = roi[2] / target_size
            scale_y = roi[3] / target_size
            dx *= scale_x
            dy *= scale_y
        
        return float(dx), float(dy), float(peak_value)


class PlanetaryAligner:
    """
    Aligneur principal pour objets planétaires
    
    Combine détection de disque et corrélation de surface
    selon le mode configuré.
    """
    
    def __init__(self, config: Optional[PlanetaryConfig] = None):
        """
        Args:
            config: PlanetaryConfig instance (ou défaut)
        """
        self.config = config if config else PlanetaryConfig()
        self.config.validate()
        
        self.disk_detector = DiskDetector(self.config)
        self.surface_correlator = SurfaceCorrelator(self.config)
        
        # Référence
        self.reference_image: Optional[np.ndarray] = None
        self.reference_disk: Optional[Dict] = None
        self.reference_gray: Optional[np.ndarray] = None
        self.is_color: bool = False
        
        # Stats
        self.stats = {
            'frames_processed': 0,
            'frames_success': 0,
            'avg_correlation': 0.0,
            'avg_shift': 0.0,
        }
    
    def set_mode(self, mode: PlanetaryMode):
        """Change le mode d'alignement"""
        self.config.mode = mode
        print(f"[PLANETARY] Mode: {mode.value}")
    
    def set_reference(self, image: np.ndarray):
        """
        Définit l'image de référence
        
        Args:
            image: Image de référence (float array, RGB ou MONO)
        """
        self.reference_image = image.copy()
        self.is_color = len(image.shape) == 3
        
        # Extraire grayscale
        if self.is_color:
            self.reference_gray = image[:, :, 1].astype(np.float64)  # Canal vert
        else:
            self.reference_gray = image.astype(np.float64)
        
        # Détecter disque de référence
        self.reference_disk = self.disk_detector.detect(self.reference_gray)
        
        if self.reference_disk:
            print(f"[PLANETARY] Référence: disque détecté")
            print(f"            Centre: ({self.reference_disk['center_x']:.1f}, "
                  f"{self.reference_disk['center_y']:.1f})")
            print(f"            Rayon: {self.reference_disk['radius']:.1f} px")
            print(f"            Méthode: {self.reference_disk['method']}")
        else:
            print(f"[PLANETARY] Référence: pas de disque détecté, "
                  f"utilisation corrélation pure")
    
    def align(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """
        Aligne une image sur la référence
        
        Args:
            image: Image à aligner (float array)
        
        Returns:
            (aligned_image, params, success)
        """
        self.stats['frames_processed'] += 1
        
        # Première image = référence
        if self.reference_image is None:
            self.set_reference(image)
            return image.copy(), {'dx': 0, 'dy': 0, 'method': 'reference'}, True
        
        # Vérifier dimensions
        if image.shape != self.reference_image.shape:
            print(f"[PLANETARY] Dimensions différentes")
            return image, {}, False
        
        # Extraire grayscale
        if self.is_color:
            img_gray = image[:, :, 1].astype(np.float64)
        else:
            img_gray = image.astype(np.float64)
        
        # Aligner selon le mode
        if self.config.mode == PlanetaryMode.DISK:
            result = self._align_by_disk(image, img_gray)
        elif self.config.mode == PlanetaryMode.SURFACE:
            result = self._align_by_surface(image, img_gray)
        else:  # HYBRID
            result = self._align_hybrid(image, img_gray)
        
        if result[2]:  # success
            self.stats['frames_success'] += 1
            shift = np.sqrt(result[1].get('dx', 0)**2 + result[1].get('dy', 0)**2)
            self.stats['avg_shift'] = (
                self.stats['avg_shift'] * (self.stats['frames_success'] - 1) + shift
            ) / self.stats['frames_success']
        
        return result
    
    def _align_by_disk(self, image: np.ndarray, img_gray: np.ndarray
                       ) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """Alignement par détection de disque"""
        # Détecter disque courant
        current_disk = self.disk_detector.detect(img_gray)
        
        if current_disk is None:
            print(f"  [DISK] Échec détection")
            return image, {}, False
        
        if self.reference_disk is None:
            print(f"  [DISK] Pas de référence")
            return image, {}, False
        
        # Calculer décalage
        dx = self.reference_disk['center_x'] - current_disk['center_x']
        dy = self.reference_disk['center_y'] - current_disk['center_y']
        
        # Vérifier seuils
        shift = np.sqrt(dx**2 + dy**2)
        if shift > self.config.max_shift:
            print(f"  [DISK] Décalage trop grand: {shift:.1f}px")
            return image, {'dx': dx, 'dy': dy}, False
        
        # Appliquer transformation
        aligned = self._apply_translation(image, dx, dy)
        
        params = {
            'dx': dx,
            'dy': dy,
            'method': 'disk',
            'disk_radius': current_disk['radius'],
            'disk_method': current_disk['method'],
        }
        
        print(f"  [DISK] dx={dx:.2f}, dy={dy:.2f}, r={current_disk['radius']:.1f}")
        
        return aligned, params, True
    
    def _align_by_surface(self, image: np.ndarray, img_gray: np.ndarray
                          ) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """Alignement par corrélation de surface"""
        # Définir ROI (centre du disque ou image entière)
        roi = None
        if self.config.surface_roi_center and self.reference_disk:
            cx = int(self.reference_disk['center_x'])
            cy = int(self.reference_disk['center_y'])
            r = int(self.reference_disk['radius'] * 0.8)  # 80% du rayon
            
            h, w = img_gray.shape
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)
            
            roi = (x1, y1, x2 - x1, y2 - y1)
        
        # Corrélation
        dx, dy, corr = self.surface_correlator.correlate(
            self.reference_gray, img_gray, roi
        )
        
        # Vérifier qualité
        if corr < self.config.min_correlation:
            print(f"  [SURFACE] Corrélation faible: {corr:.3f}")
            return image, {'dx': dx, 'dy': dy, 'correlation': corr}, False
        
        shift = np.sqrt(dx**2 + dy**2)
        if shift > self.config.max_shift:
            print(f"  [SURFACE] Décalage trop grand: {shift:.1f}px")
            return image, {'dx': dx, 'dy': dy, 'correlation': corr}, False
        
        # Appliquer transformation
        aligned = self._apply_translation(image, dx, dy)
        
        # Mettre à jour stats
        n = self.stats['frames_success'] + 1
        self.stats['avg_correlation'] = (
            self.stats['avg_correlation'] * (n - 1) + corr
        ) / n
        
        params = {
            'dx': dx,
            'dy': dy,
            'correlation': corr,
            'method': 'surface',
        }
        
        print(f"  [SURFACE] dx={dx:.2f}, dy={dy:.2f}, corr={corr:.3f}")
        
        return aligned, params, True
    
    def _align_hybrid(self, image: np.ndarray, img_gray: np.ndarray
                      ) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """Alignement hybride : disque grossier + surface fin"""
        # Étape 1 : Alignement grossier par disque
        current_disk = self.disk_detector.detect(img_gray)
        
        dx_coarse, dy_coarse = 0.0, 0.0
        
        if current_disk and self.reference_disk:
            dx_coarse = self.reference_disk['center_x'] - current_disk['center_x']
            dy_coarse = self.reference_disk['center_y'] - current_disk['center_y']
            
            # Appliquer correction grossière pour la corrélation
            img_gray_shifted = self._apply_translation_gray(img_gray, dx_coarse, dy_coarse)
        else:
            img_gray_shifted = img_gray
        
        # Étape 2 : Affinement par corrélation de surface
        roi = None
        if self.reference_disk:
            cx = int(self.reference_disk['center_x'])
            cy = int(self.reference_disk['center_y'])
            r = int(self.reference_disk['radius'] * 0.6)
            
            h, w = img_gray.shape
            roi = (max(0, cx - r), max(0, cy - r),
                   min(2*r, w - max(0, cx - r)),
                   min(2*r, h - max(0, cy - r)))
        
        dx_fine, dy_fine, corr = self.surface_correlator.correlate(
            self.reference_gray, img_gray_shifted, roi
        )
        
        # Décalage total
        dx_total = dx_coarse + dx_fine
        dy_total = dy_coarse + dy_fine
        
        # Vérifier
        shift = np.sqrt(dx_total**2 + dy_total**2)
        if shift > self.config.max_shift:
            print(f"  [HYBRID] Décalage trop grand: {shift:.1f}px")
            return image, {'dx': dx_total, 'dy': dy_total}, False
        
        # Appliquer transformation totale
        aligned = self._apply_translation(image, dx_total, dy_total)
        
        params = {
            'dx': dx_total,
            'dy': dy_total,
            'dx_coarse': dx_coarse,
            'dy_coarse': dy_coarse,
            'dx_fine': dx_fine,
            'dy_fine': dy_fine,
            'correlation': corr,
            'method': 'hybrid',
        }
        
        print(f"  [HYBRID] total=({dx_total:.2f}, {dy_total:.2f}), "
              f"coarse=({dx_coarse:.1f}, {dy_coarse:.1f}), "
              f"fine=({dx_fine:.2f}, {dy_fine:.2f}), corr={corr:.3f}")
        
        return aligned, params, True
    
    def _apply_translation(self, image: np.ndarray, dx: float, dy: float
                           ) -> np.ndarray:
        """Applique une translation à l'image"""
        M = np.array([
            [1, 0, dx],
            [0, 1, dy]
        ], dtype=np.float32)
        
        if self.is_color and len(image.shape) == 3:
            h, w = image.shape[:2]
            aligned = np.zeros_like(image)
            for i in range(image.shape[2]):
                aligned[:, :, i] = cv2.warpAffine(
                    image[:, :, i].astype(np.float32), M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
        else:
            h, w = image.shape[:2]
            aligned = cv2.warpAffine(
                image.astype(np.float32), M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        
        return aligned.astype(image.dtype)
    
    def _apply_translation_gray(self, image: np.ndarray, dx: float, dy: float
                                ) -> np.ndarray:
        """Applique translation à une image grayscale"""
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        h, w = image.shape
        return cv2.warpAffine(
            image.astype(np.float32), M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    
    def reset(self):
        """Réinitialise l'aligneur"""
        self.reference_image = None
        self.reference_disk = None
        self.reference_gray = None
        self.stats = {
            'frames_processed': 0,
            'frames_success': 0,
            'avg_correlation': 0.0,
            'avg_shift': 0.0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        return self.stats.copy()


# =============================================================================
# Classe wrapper pour intégration avec AdvancedAligner existant
# =============================================================================

class PlanetaryAlignerWrapper:
    """
    Wrapper pour utiliser PlanetaryAligner avec l'interface de AdvancedAligner
    
    Permet l'intégration transparente dans session.py
    """
    
    def __init__(self, config):
        """
        Args:
            config: StackingConfig ou AdvancedStackingConfig
        """
        self.config = config
        
        # Créer config planétaire depuis config principale
        self.planetary_config = PlanetaryConfig()
        
        # Mapper les paramètres si disponibles
        if hasattr(config, 'planetary'):
            pc = config.planetary
            if hasattr(pc, 'mode'):
                self.planetary_config.mode = pc.mode
            if hasattr(pc, 'disk_min_radius'):
                self.planetary_config.disk_min_radius = pc.disk_min_radius
            # ... autres paramètres
        
        self.aligner = PlanetaryAligner(self.planetary_config)
        self.is_color = False
    
    def set_reference(self, image: np.ndarray):
        """Définit l'image de référence"""
        self.aligner.set_reference(image)
        self.is_color = len(image.shape) == 3
    
    def align(self, image: np.ndarray) -> Tuple[np.ndarray, Dict, bool]:
        """Aligne une image sur la référence"""
        return self.aligner.align(image)


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test PlanetaryAligner ===\n")
    
    np.random.seed(42)
    
    # Créer image de test : disque avec détails
    h, w = 512, 512
    y, x = np.ogrid[:h, :w]
    center = (256, 256)
    radius = 150
    
    # Disque de base
    disk = ((x - center[0])**2 + (y - center[1])**2 <= radius**2).astype(np.float32)
    disk = disk * 0.8  # Luminosité
    
    # Ajouter "taches" (détails de surface)
    for _ in range(5):
        tx = np.random.randint(center[0] - 100, center[0] + 100)
        ty = np.random.randint(center[1] - 100, center[1] + 100)
        tr = np.random.randint(10, 30)
        spot = ((x - tx)**2 + (y - ty)**2 <= tr**2).astype(np.float32)
        disk -= spot * 0.2
    
    disk = np.clip(disk, 0, 1)
    disk += np.random.normal(0, 0.02, disk.shape).astype(np.float32)
    
    # Image de référence
    ref_image = disk.copy()
    
    # Images décalées
    test_shifts = [(5.5, 3.2), (-8.0, 12.5), (0.3, -0.7)]
    
    # Test chaque mode
    for mode in PlanetaryMode:
        print(f"\n--- Mode: {mode.value} ---")
        
        config = PlanetaryConfig()
        config.mode = mode
        aligner = PlanetaryAligner(config)
        
        aligner.set_reference(ref_image)
        
        for i, (dx, dy) in enumerate(test_shifts):
            # Créer image décalée
            M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            shifted = cv2.warpAffine(disk, M, (w, h))
            
            # Aligner
            aligned, params, success = aligner.align(shifted)
            
            # Vérifier
            err_x = abs(params.get('dx', 0) - dx)
            err_y = abs(params.get('dy', 0) - dy)
            
            status = "✓" if success and err_x < 1 and err_y < 1 else "✗"
            print(f"  Test {i+1}: shift=({dx}, {dy}) -> "
                  f"detected=({params.get('dx', 0):.2f}, {params.get('dy', 0):.2f}) "
                  f"err=({err_x:.2f}, {err_y:.2f}) {status}")
        
        stats = aligner.get_stats()
        print(f"  Stats: {stats['frames_success']}/{stats['frames_processed']} succès, "
              f"avg_shift={stats['avg_shift']:.2f}px")
    
    print("\n=== Tests terminés ===")
