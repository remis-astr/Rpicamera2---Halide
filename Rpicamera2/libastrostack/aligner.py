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
        
        # Détecter étoiles de référence (luminance pour robustesse sur toutes cibles)
        if self.is_color:
            ref_gray = (0.299 * image[:, :, 0] +
                        0.587 * image[:, :, 1] +
                        0.114 * image[:, :, 2])
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
            img_gray = (0.299 * image[:, :, 0] +
                        0.587 * image[:, :, 1] +
                        0.114 * image[:, :, 2])
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

        # Seuil adaptatif : 15% de la plus petite dimension, entre 50px et 500px
        # 5% était trop restrictif pour des séquences avec dérive > ~50px (images 1080p → 54px seulement)
        # 15% → 162px pour 1080p, bien en dessous de l'espacement moyen des étoiles (~218px pour 43 étoiles)
        if self.reference_image is not None:
            h, w = self.reference_image.shape[:2]
            match_threshold = max(50, min(500, int(0.15 * min(h, w))))
        else:
            match_threshold = 150

        for i, ref_pt in enumerate(ref_pts):
            # Distances à toutes les étoiles courantes
            distances = np.sqrt(np.sum((cur_pts - ref_pt)**2, axis=1))
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            # Accepter seulement les matches proches
            if min_dist < match_threshold:
                matches.append((ref_pt, cur_pts[min_idx]))
                match_distances.append(min_dist)

        # Exiger au moins 25% des étoiles testées ET un minimum absolu de 5
        min_required = max(5, int(0.25 * min(n_ref, n_cur)))
        if len(matches) < min_required:
            print(f"  [WARN] Pas assez de matches: {len(matches)}/{min(n_ref, n_cur)} (requis {min_required})")
            return None, {}

        # ── RANSAC simplifié en 2 passes ─────────────────────────────────────
        # Problème avec grand seuil (162px) : avec une dérive de ~86px, la probabilité
        # qu'une étoile voisine soit plus proche que la vraie cible est ~32% par étoile.
        # La médiane seule ne suffit plus quand la direction de dérive varie d'une frame
        # à l'autre (oscillation du guidage). Solution : passe 1 grossière → passe 2
        # affinée qui ne garde que les paires cohérentes avec la translation consensus.

        # Passe 1 : estimation grossière (médiane brute sur tous les matches)
        translations = np.array([ref - cur for ref, cur in matches])
        dy_rough = float(np.median(translations[:, 0]))
        dx_rough = float(np.median(translations[:, 1]))

        # Passe 2 : ne garder que les matches dont la translation est dans ±refine_tol
        # de la translation grossière.  15px couvre les erreurs de centroïde tout en
        # rejetant les faux appariements (translation aléatoire ≠ consensus).
        refine_tol = 15.0
        refined = [(ref, cur) for ref, cur in matches
                   if (abs((ref[0] - cur[0]) - dy_rough) < refine_tol and
                       abs((ref[1] - cur[1]) - dx_rough) < refine_tol)]

        if len(refined) < min_required:
            # Raffinement trop agressif (cas rare) → réutiliser les matches bruts
            refined = matches

        # Translation finale sur les matches raffinés
        translations_refined = np.array([ref - cur for ref, cur in refined])
        dy = float(np.median(translations_refined[:, 0]))
        dx = float(np.median(translations_refined[:, 1]))

        params = {
            'dx': dx, 'dy': dy, 'angle': 0, 'scale': 1.0,
            'matches': len(refined),
            'total': len(ref_pts),
            'raw_matches': len(matches),
        }

        transform = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])

        return transform, params
    
    def _compute_rotation(self, ref_stars, cur_stars, image_shape):
        """Calcule rotation + translation avec matching d'étoiles puis RANSAC rigide.

        Ancienne implémentation : passait les 30 étoiles ref et 30 étoiles courantes
        triées INDÉPENDAMMENT par luminosité à estimateAffinePartial2D → faux car
        l'algo attend des paires correspondantes.

        Nouvelle implémentation :
          1. Matching NNN (même pipeline que translation, tolérance élargie à 25px
             pour conserver les étoiles déplacées par rotation)
          2. estimateAffinePartial2D sur les paires matchées → rotation + translation
        """
        n_ref = min(30, len(ref_stars))
        n_cur = min(30, len(cur_stars))
        ref_pts_all = ref_stars[:n_ref]
        cur_pts_all = cur_stars[:n_cur]

        # ── Seuil adaptatif identique au mode translation ──────────────────────
        if self.reference_image is not None:
            h, w = self.reference_image.shape[:2]
            match_threshold = max(50, min(500, int(0.15 * min(h, w))))
        else:
            match_threshold = 150

        # ── Matching NNN ───────────────────────────────────────────────────────
        matches = []
        for ref_pt in ref_pts_all:
            distances = np.sqrt(np.sum((cur_pts_all - ref_pt)**2, axis=1))
            min_idx = np.argmin(distances)
            if distances[min_idx] < match_threshold:
                matches.append((ref_pt, cur_pts_all[min_idx]))

        min_required = max(5, int(0.25 * min(n_ref, n_cur)))
        if len(matches) < min_required:
            print(f"  [WARN] Pas assez de matches: {len(matches)}/{min(n_ref, n_cur)}")
            return None, {}

        # ── RANSAC 2-pass pour éliminer les faux matches ──────────────────────
        # Tolérance volontairement large (25px) pour ne PAS éliminer les étoiles
        # dont la translation brute dévie à cause de la rotation de champ.
        # L'étape suivante (estimateAffinePartial2D) gère correctement la rotation.
        translations = np.array([ref - cur for ref, cur in matches])
        dy_rough = float(np.median(translations[:, 0]))
        dx_rough = float(np.median(translations[:, 1]))
        refine_tol = 25.0
        refined = [(ref, cur) for ref, cur in matches
                   if (abs((ref[0] - cur[0]) - dy_rough) < refine_tol and
                       abs((ref[1] - cur[1]) - dx_rough) < refine_tol)]
        if len(refined) < min_required:
            refined = matches

        # ── Ajustement rigide (rotation + translation) sur paires matchées ────
        # Convertir (y, x) → (x, y) pour OpenCV
        matched_ref = np.array([pt[[1, 0]] for pt, _ in refined], dtype=np.float32)
        matched_cur = np.array([pt[[1, 0]] for _, pt in refined], dtype=np.float32)

        try:
            M, inliers = cv2.estimateAffinePartial2D(
                matched_cur, matched_ref,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=2000,
                confidence=0.99
            )

            if M is None:
                return None, {}

            n_inliers = int(np.sum(inliers)) if inliers is not None else 0
            inliers_ratio = n_inliers / len(matched_cur) if len(matched_cur) > 0 else 0

            if inliers_ratio < self.config.quality.min_inliers_ratio:
                print(f"  [WARN] Pas assez d'inliers rotation: {n_inliers}/{len(matched_cur)} "
                      f"({inliers_ratio*100:.1f}%)")
                return None, {}

            angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            dx = float(M[0, 2])
            dy = float(M[1, 2])

            params = {
                'dx': dx, 'dy': dy, 'angle': angle, 'scale': scale,
                'inliers': n_inliers,
                'total': len(matched_ref),
            }

            transform = np.vstack([M, [0, 0, 1]])
            return transform, params

        except Exception as e:
            print(f"  [ALIGN] Erreur _compute_rotation: {e}")
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
        except Exception as e:
            print(f"  [ALIGN] Erreur _compute_affine: {e}")
            return None, {}

    def _apply_transform(self, image, transform):
        """Applique transformation à l'image.

        Les pixels de bordure introduits par le décalage sont mis à NaN (pas 0)
        afin que le stacker (weight_map + np.isfinite) les ignore dans la moyenne.
        Un borderValue=0 causerait un assombrissement progressif des bords quand
        le télescope dérive, car les 0 seraient comptés comme des vraies mesures.
        """
        M = transform[:2, :]

        if self.is_color and len(image.shape) == 3:
            h, w = image.shape[:2]
            # Masque de validité : 1 là où l'image source contribue, 0 aux bordures
            _ones = np.ones((h, w), dtype=np.float32)
            valid_mask = cv2.warpAffine(
                _ones, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ) > 0.5

            aligned = cv2.warpAffine(
                image.astype(np.float32), M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            aligned[~valid_mask] = np.nan
        else:
            h, w = image.shape
            _ones = np.ones((h, w), dtype=np.float32)
            valid_mask = cv2.warpAffine(
                _ones, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ) > 0.5

            aligned = cv2.warpAffine(
                image.astype(np.float32), M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            aligned[~valid_mask] = np.nan

        return aligned
    
    def _print_transform(self, params):
        """Affiche infos transformation"""
        mode = self.config.alignment_mode
        drift = np.sqrt(params.get('dx', 0)**2 + params.get('dy', 0)**2)

        if mode == AlignmentMode.TRANSLATION:
            matches     = params.get('matches', 0)
            raw_matches = params.get('raw_matches', matches)  # avant raffinement RANSAC
            total       = params.get('total', 0)
            dx          = params.get('dx', 0)
            dy          = params.get('dy', 0)
            # Afficher dx/dy pour diagnostiquer les changements de direction de dérive
            if raw_matches != matches:
                print(f"  Translation: drift={drift:.2f}px, dx={dx:+.1f} dy={dy:+.1f}, "
                      f"matches={matches}/{total} (bruts:{raw_matches})")
            else:
                print(f"  Translation: drift={drift:.2f}px, dx={dx:+.1f} dy={dy:+.1f}, "
                      f"matches={matches}/{total}")
        elif mode == AlignmentMode.ROTATION:
            print(f"  Rotation: angle={params.get('angle', 0):.3f}°, "
                  f"drift={drift:.2f}px, "
                  f"inliers={params.get('inliers', 0)}/{params.get('total', 0)}")
        else:
            print(f"  Affine: angle={params.get('angle', 0):.3f}°, "
                  f"scale={params.get('scale', 1):.4f}, "
                  f"drift={drift:.2f}px")
