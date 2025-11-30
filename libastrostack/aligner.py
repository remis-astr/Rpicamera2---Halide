#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignement d'images pour libastrostack
"""

import numpy as np
import cv2
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from skimage.feature import peak_local_max
from .config import AlignmentMode


class StarDetector:
    """Détecteur d'étoiles pour alignement"""
    
    def __init__(self, config):
        """
        Args:
            config: StackingConfig instance
        """
        self.config = config
    
    def detect_stars(self, image):
        """
        Détecte les étoiles pour l'alignement
        
        Args:
            image: Image MONO (float array)
        
        Returns:
            Array de positions (N, 2) avec [y, x]
        """
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        threshold = median + self.config.quality.star_detection_sigma * std
        smoothed = ndimage.gaussian_filter(image, sigma=1.5)
        
        peaks = peak_local_max(
            smoothed,
            min_distance=self.config.quality.min_star_separation,
            threshold_abs=threshold,
            num_peaks=self.config.max_stars_alignment
        )
        
        if len(peaks) == 0:
            return np.array([])
        
        # Trier par intensité
        intensities = [image[p[0], p[1]] for p in peaks]
        sorted_indices = np.argsort(intensities)[::-1]
        
        return peaks[sorted_indices]


class AdvancedAligner:
    """Alignement d'images avec support rotation"""
    
    def __init__(self, config):
        """
        Args:
            config: StackingConfig instance
        """
        self.config = config
        self.star_detector = StarDetector(config)
        self.reference_image = None
        self.reference_stars = None
        self.is_color = False
    
    def set_reference(self, image):
        """
        Définit l'image de référence
        
        Args:
            image: Image de référence (float array)
        """
        self.reference_image = image.copy()
        self.is_color = len(image.shape) == 3
        
        # Détecter étoiles de référence
        if self.is_color:
            ref_gray = image[:, :, 1]  # Canal vert
        else:
            ref_gray = image
        
        self.reference_stars = self.star_detector.detect_stars(ref_gray)
        
        shape_str = f"{image.shape} RGB" if self.is_color else f"{image.shape} MONO"
        print(f"Référence: {shape_str}, {len(self.reference_stars)} étoiles")
    
    def align(self, image):
        """
        Aligne une image sur la référence
        
        Args:
            image: Image à aligner (float array)
        
        Returns:
            (aligned_image, params, success)
            - aligned_image: Image alignée
            - params: dict avec dx, dy, angle, scale, etc.
            - success: True si alignement réussi
        """
        if self.reference_image is None:
            self.set_reference(image)
            return image, {'dx': 0, 'dy': 0, 'angle': 0, 'scale': 1.0}, True
        
        if image.shape != self.reference_image.shape:
            print(f"[WARN] Dimensions différentes")
            return image, {}, False
        
        # Détecter étoiles dans image courante
        if self.is_color:
            img_gray = image[:, :, 1]
        else:
            img_gray = image
        
        current_stars = self.star_detector.detect_stars(img_gray)

        # Déterminer le minimum d'étoiles requis
        # Minimum technique selon le mode d'alignement
        if self.config.alignment_mode == AlignmentMode.TRANSLATION:
            tech_min = 1  # Translation nécessite au moins 1 point
        elif self.config.alignment_mode == AlignmentMode.ROTATION:
            tech_min = 2  # Rotation nécessite au moins 2 points
        else:  # AFFINE
            tech_min = 3  # Affine nécessite au moins 3 points

        # Si min_stars est configuré (> 0), l'utiliser, sinon utiliser le minimum technique
        required_stars = max(tech_min, self.config.quality.min_stars) if self.config.quality.min_stars > 0 else tech_min

        if len(current_stars) < required_stars or len(self.reference_stars) < required_stars:
            print(f"[WARN] Pas assez d'étoiles ({len(current_stars)} détectées, {required_stars} requises)")
            return image, {}, False
        
        # Calculer transformation selon le mode
        if self.config.alignment_mode == AlignmentMode.TRANSLATION:
            transform, params = self._compute_translation(
                self.reference_stars, current_stars
            )
        elif self.config.alignment_mode == AlignmentMode.ROTATION:
            transform, params = self._compute_rotation(
                self.reference_stars, current_stars,
                self.reference_image.shape
            )
        else:  # AFFINE
            transform, params = self._compute_affine(
                self.reference_stars, current_stars,
                self.reference_image.shape
            )
        
        if transform is None:
            return image, {}, False

        # Vérifier drift
        drift = np.sqrt(params.get('dx', 0)**2 + params.get('dy', 0)**2)
        if drift > self.config.quality.max_drift:
            print(f"  [REJECT] Drift trop grand: {drift:.1f}px")
            return image, params, False

        # Vérifier rotation (sauf pour mode TRANSLATION)
        if self.config.alignment_mode != AlignmentMode.TRANSLATION:
            angle = abs(params.get('angle', 0))
            if angle > self.config.quality.max_rotation:
                print(f"  [REJECT] Rotation trop grande: {angle:.1f}° (max {self.config.quality.max_rotation}°)")
                return image, params, False

        # Vérifier scale (pour modes ROTATION et AFFINE)
        if self.config.alignment_mode in [AlignmentMode.ROTATION, AlignmentMode.AFFINE]:
            scale = params.get('scale', 1.0)
            if scale < self.config.quality.min_scale or scale > self.config.quality.max_scale:
                print(f"  [REJECT] Scale aberrant: {scale:.4f} (plage {self.config.quality.min_scale}-{self.config.quality.max_scale})")
                return image, params, False

        # Appliquer transformation
        aligned = self._apply_transform(image, transform)
        
        # Afficher infos
        self._print_transform(params)
        
        return aligned, params, True
    
    def _compute_translation(self, ref_stars, cur_stars):
        """Calcule translation par matching d'étoiles"""
        # Utiliser au max 30 étoiles les plus brillantes
        n_ref = min(30, len(ref_stars))
        n_cur = min(30, len(cur_stars))

        ref_pts = ref_stars[:n_ref]
        cur_pts = cur_stars[:n_cur]

        # Calculer toutes les distances entre étoiles ref et cur
        # Pour chaque étoile de référence, trouver la plus proche dans l'image courante
        matches = []
        match_distances = []

        for i, ref_pt in enumerate(ref_pts):
            # Distances à toutes les étoiles courantes
            distances = np.sqrt(np.sum((cur_pts - ref_pt)**2, axis=1))
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            # Accepter seulement les matches proches (< 100px)
            if min_dist < 100:
                matches.append((ref_pt, cur_pts[min_idx]))
                match_distances.append(min_dist)

        # Besoin d'au moins 3 matches
        if len(matches) < 3:
            print(f"  [WARN] Pas assez de matches: {len(matches)}")
            return None, {}

        # Calculer la translation médiane (plus robuste que moyenne)
        # ref - cur pour ramener l'image courante vers la référence
        translations = np.array([ref - cur for ref, cur in matches])
        dy = np.median(translations[:, 0])
        dx = np.median(translations[:, 1])

        params = {
            'dx': dx, 'dy': dy, 'angle': 0, 'scale': 1.0,
            'matches': len(matches),
            'total': len(ref_pts)
        }

        transform = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])

        return transform, params
    
    def _compute_rotation(self, ref_stars, cur_stars, image_shape):
        """Calcule rotation + translation avec OpenCV"""
        n_stars = min(30, len(ref_stars), len(cur_stars))
        
        # Convertir en format OpenCV (x, y)
        ref_pts = ref_stars[:n_stars][:, [1, 0]].astype(np.float32)
        cur_pts = cur_stars[:n_stars][:, [1, 0]].astype(np.float32)
        
        try:
            # Transformation rigide (rotation + translation + scale)
            M, inliers = cv2.estimateAffinePartial2D(
                cur_pts, ref_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=2000,
                confidence=0.99
            )

            if M is None:
                return None, {}

            # Vérifier le ratio d'inliers
            n_inliers = np.sum(inliers) if inliers is not None else 0
            inliers_ratio = n_inliers / len(cur_pts) if len(cur_pts) > 0 else 0

            if inliers_ratio < self.config.quality.min_inliers_ratio:
                print(f"  [WARN] Pas assez d'inliers: {n_inliers}/{len(cur_pts)} ({inliers_ratio*100:.1f}%)")
                return None, {}

            # Extraire paramètres
            angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            dx = M[0, 2]
            dy = M[1, 2]

            params = {
                'dx': dx, 'dy': dy, 'angle': angle, 'scale': scale,
                'inliers': n_inliers,
                'total': len(cur_pts)
            }

            transform = np.vstack([M, [0, 0, 1]])

            return transform, params
        except:
            return None, {}
    
    def _compute_affine(self, ref_stars, cur_stars, image_shape):
        """Calcule transformation affine complète"""
        n_stars = min(30, len(ref_stars), len(cur_stars))
        
        ref_pts = ref_stars[:n_stars][:, [1, 0]].astype(np.float32)
        cur_pts = cur_stars[:n_stars][:, [1, 0]].astype(np.float32)
        
        try:
            M, inliers = cv2.estimateAffine2D(
                cur_pts, ref_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=2000,
                confidence=0.99
            )

            if M is None:
                return None, {}

            # Vérifier le ratio d'inliers
            n_inliers = np.sum(inliers) if inliers is not None else 0
            inliers_ratio = n_inliers / len(cur_pts) if len(cur_pts) > 0 else 0

            if inliers_ratio < self.config.quality.min_inliers_ratio:
                print(f"  [WARN] Pas assez d'inliers: {n_inliers}/{len(cur_pts)} ({inliers_ratio*100:.1f}%)")
                return None, {}

            angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)

            params = {
                'dx': M[0, 2], 'dy': M[1, 2], 'angle': angle, 'scale': scale,
                'inliers': n_inliers,
                'total': len(cur_pts)
            }

            transform = np.vstack([M, [0, 0, 1]])

            return transform, params
        except:
            return None, {}
    
    def _apply_transform(self, image, transform):
        """Applique transformation à l'image"""
        M = transform[:2, :]
        
        if self.is_color and len(image.shape) == 3:
            h, w = image.shape[:2]
            aligned = np.zeros_like(image)
            
            for i in range(3):
                aligned[:, :, i] = cv2.warpAffine(
                    image[:, :, i], M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
        else:
            h, w = image.shape
            aligned = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        
        return aligned
    
    def _print_transform(self, params):
        """Affiche infos transformation"""
        mode = self.config.alignment_mode
        drift = np.sqrt(params.get('dx', 0)**2 + params.get('dy', 0)**2)

        if mode == AlignmentMode.TRANSLATION:
            matches = params.get('matches', 0)
            total = params.get('total', 0)
            print(f"  Translation: drift={drift:.2f}px, matches={matches}/{total}")
        elif mode == AlignmentMode.ROTATION:
            print(f"  Rotation: angle={params.get('angle', 0):.3f}°, "
                  f"drift={drift:.2f}px, "
                  f"inliers={params.get('inliers', 0)}/{params.get('total', 0)}")
        else:
            print(f"  Affine: angle={params.get('angle', 0):.3f}°, "
                  f"scale={params.get('scale', 1):.4f}, "
                  f"drift={drift:.2f}px")
