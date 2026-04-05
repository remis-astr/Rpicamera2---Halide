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

try:
    import astroalign as _aa
    _ASTROALIGN_AVAILABLE = True
except ImportError:
    _ASTROALIGN_AVAILABLE = False


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
        Définit l'image de référence et calcule les dimensions du canvas.

        Si config.canvas_margin_frac > 0, le canvas est élargi de chaque côté
        pour que les frames décalées ne perdent pas leurs bords.
        """
        self.reference_image = image.copy()
        self.is_color = len(image.shape) == 3

        h, w = image.shape[:2]
        frac = getattr(self.config, 'canvas_margin_frac', 0.0)
        self._canvas_mx = int(w * frac)   # marge horizontale (px par côté)
        self._canvas_my = int(h * frac)   # marge verticale  (px par côté)
        self._canvas_w  = w + 2 * self._canvas_mx
        self._canvas_h  = h + 2 * self._canvas_my

        if self._canvas_mx > 0 or self._canvas_my > 0:
            print(f"  [Canvas] {w}×{h} → {self._canvas_w}×{self._canvas_h} "
                  f"(marge {self._canvas_mx}×{self._canvas_my}px par côté)")

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
        Aligne une image sur la référence.

        Si canvas_margin_frac > 0, l'image alignée est placée sur un canvas
        élargi (marge de chaque côté). La frame de référence est aussi centrée
        sur ce canvas. Le stacker reçoit donc toujours des images canvas-size.

        Returns:
            (aligned_image, params, success)
        """
        if self.reference_image is None:
            self.set_reference(image)
            # Placer la frame de référence sur le canvas (décalée de la marge)
            identity = np.eye(3, dtype=np.float64)
            ref_on_canvas = self._apply_transform(image, identity)
            return ref_on_canvas, {'dx': 0, 'dy': 0, 'angle': 0, 'scale': 1.0}, True
        
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

        # Seuil adaptatif : 25% de la plus petite dimension, entre 50px et 600px
        # 15% (162px pour 1080p) était trop restrictif pour des drifts > ~160px
        # qui surviennent après plusieurs frames ou sans guidage.
        if self.reference_image is not None:
            h, w = self.reference_image.shape[:2]
            match_threshold = max(50, min(600, int(0.25 * min(h, w))))
        else:
            match_threshold = 200

        # ── Hough voting pour trouver la translation consensus ────────────────
        # NNN échoue quand le drift est grand (ex: 85px) et qu'un faux voisin
        # est plus proche que la vraie cible. Hough : toutes les paires ref×cur
        # votent pour leur translation ; le vrai drift accumule N votes, les
        # faux matches se dispersent → pic robuste même avec des clusters d'étoiles.
        bin_size = 15  # px par bin
        vote_grid = {}
        candidate_pairs = []  # (ref_pt, cur_pt, dist, dy, dx)
        for ref_pt in ref_pts:
            for cur_pt in cur_pts:
                dist = float(np.sqrt(np.sum((cur_pt - ref_pt)**2)))
                if dist < match_threshold:
                    dy_pair = float(ref_pt[0] - cur_pt[0])
                    dx_pair = float(ref_pt[1] - cur_pt[1])
                    key = (round(dy_pair / bin_size), round(dx_pair / bin_size))
                    vote_grid[key] = vote_grid.get(key, 0) + 1
                    candidate_pairs.append((ref_pt, cur_pt, dist, dy_pair, dx_pair))

        # Exiger au moins 25% des étoiles testées ET un minimum absolu de 5
        min_required = max(5, int(0.25 * min(n_ref, n_cur)))
        if not candidate_pairs:
            print(f"  [WARN] Aucune paire dans le seuil {match_threshold}px "
                  f"(drift > {match_threshold}px ?)")
            return None, {}

        # Pic Hough → translation consensus
        best_key = max(vote_grid, key=lambda k: vote_grid[k])
        best_dy = best_key[0] * bin_size
        best_dx = best_key[1] * bin_size
        peak_votes = vote_grid[best_key]
        peak_drift = float(np.sqrt(best_dx**2 + best_dy**2))
        print(f"  [Hough] consensus: dy={best_dy:.0f} dx={best_dx:.0f} "
              f"drift={peak_drift:.0f}px votes={peak_votes}/{len(candidate_pairs)}")

        # Garder paires dans ±1.5 bins du consensus + exclusivité cur
        tight_tol = bin_size * 1.5  # ±22.5 px
        _best_for_cur = {}  # tuple(cur_pt) → (dist, ref_pt, cur_pt)
        for ref_pt, cur_pt, dist, dy_pair, dx_pair in candidate_pairs:
            if abs(dy_pair - best_dy) < tight_tol and abs(dx_pair - best_dx) < tight_tol:
                key = (cur_pt[0], cur_pt[1])
                if key not in _best_for_cur or dist < _best_for_cur[key][0]:
                    _best_for_cur[key] = (dist, ref_pt, cur_pt)
        refined = [(ref_pt, cur_pt) for _, ref_pt, cur_pt in _best_for_cur.values()]

        if len(refined) < min_required:
            print(f"  [WARN] Pas assez de matches après Hough: {len(refined)}/{min(n_ref, n_cur)} "
                  f"(requis {min_required}, seuil {match_threshold}px)")
            return None, {}

        # Translation finale : médiane sur les paires raffinées (robuste aux outliers résiduels)
        translations_refined = np.array([ref - cur for ref, cur in refined])
        dy = float(np.median(translations_refined[:, 0]))
        dx = float(np.median(translations_refined[:, 1]))

        params = {
            'dx': dx, 'dy': dy, 'angle': 0, 'scale': 1.0,
            'matches': len(refined),
            'total': len(ref_pts),
            'raw_matches': len(candidate_pairs),
        }

        transform = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])

        return transform, params
    
    def _hough_match(self, ref_pts, cur_pts, match_threshold, bin_size=15):
        """Retourne liste de paires (ref_pt, cur_pt) via vote Hough sur la translation.

        Toutes les paires ref×cur dans match_threshold votent pour leur vecteur
        de translation (binné à bin_size px). Le pic = drift consensus.
        Exclusivité : une étoile courante ne peut être utilisée qu'une seule fois.
        """
        vote_grid = {}
        candidate_pairs = []
        for ref_pt in ref_pts:
            for cur_pt in cur_pts:
                dist = float(np.sqrt(np.sum((cur_pt - ref_pt)**2)))
                if dist < match_threshold:
                    dy = float(ref_pt[0] - cur_pt[0])
                    dx = float(ref_pt[1] - cur_pt[1])
                    key = (round(dy / bin_size), round(dx / bin_size))
                    vote_grid[key] = vote_grid.get(key, 0) + 1
                    candidate_pairs.append((ref_pt, cur_pt, dist, dy, dx))

        if not candidate_pairs:
            return []

        best_key = max(vote_grid, key=lambda k: vote_grid[k])
        best_dy = best_key[0] * bin_size
        best_dx = best_key[1] * bin_size
        peak_drift = float(np.sqrt(best_dx**2 + best_dy**2))
        print(f"  [Hough] dy={best_dy:.0f} dx={best_dx:.0f} drift={peak_drift:.0f}px "
              f"votes={vote_grid[best_key]}/{len(candidate_pairs)}")

        tight_tol = bin_size * 1.5  # ±22.5 px
        best_for_cur = {}
        for ref_pt, cur_pt, dist, dy, dx in candidate_pairs:
            if abs(dy - best_dy) < tight_tol and abs(dx - best_dx) < tight_tol:
                key = (cur_pt[0], cur_pt[1])
                if key not in best_for_cur or dist < best_for_cur[key][0]:
                    best_for_cur[key] = (dist, ref_pt, cur_pt)
        return [(ref_pt, cur_pt) for _, ref_pt, cur_pt in best_for_cur.values()]

    def _compute_rotation(self, ref_stars, cur_stars, image_shape):
        """Rotation + translation : Hough en passe 1, astroalign (triangles) en passe 2.

        Passe 1 — Hough : rapide, O(N²). Suffit quand le champ est stable
        (drift cohérent d'une frame à l'autre). Échoue si le drift est trop grand
        ou si les étoiles communes sont trop rares (< 25 % en commun).

        Passe 2 — astroalign : matching par invariants géométriques de triangles,
        indépendant du drift et de l'orientation. Fonctionne avec seulement 30-40 %
        de recouvrement, n'importe quelle amplitude de dérive.
        """
        n_ref = min(50, len(ref_stars))
        n_cur = min(50, len(cur_stars))
        ref_pts_all = ref_stars[:n_ref]
        cur_pts_all = cur_stars[:n_cur]
        min_required = max(5, int(0.25 * min(n_ref, n_cur)))

        if self.reference_image is not None:
            h, w = self.reference_image.shape[:2]
            match_threshold = max(50, min(600, int(0.25 * min(h, w))))
        else:
            match_threshold = 200

        # ── Passe 1 : Hough + estimateAffinePartial2D ────────────────────────
        refined = self._hough_match(ref_pts_all, cur_pts_all, match_threshold)

        if len(refined) >= min_required:
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
                if M is not None:
                    n_inliers = int(np.sum(inliers)) if inliers is not None else 0
                    inliers_ratio = n_inliers / len(matched_cur)
                    if inliers_ratio >= self.config.quality.min_inliers_ratio:
                        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
                        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                        params = {
                            'dx': float(M[0, 2]), 'dy': float(M[1, 2]),
                            'angle': angle, 'scale': scale,
                            'inliers': n_inliers, 'total': len(matched_ref),
                            'method': 'hough',
                        }
                        return np.vstack([M, [0, 0, 1]]), params
                    else:
                        print(f"  [Hough] inliers insuffisants: {n_inliers}/{len(matched_cur)} "
                              f"({inliers_ratio*100:.1f}%) → essai astroalign")
            except Exception as e:
                print(f"  [Hough] erreur RANSAC: {e} → essai astroalign")
        else:
            print(f"  [Hough] matches insuffisants: {len(refined)}/{min(n_ref, n_cur)} "
                  f"(requis {min_required}) → essai astroalign")

        # ── Passe 2 : astroalign (triangle matching) ─────────────────────────
        if not _ASTROALIGN_AVAILABLE:
            print(f"  [FAIL] astroalign non disponible (pip3 install astroalign)")
            return None, {}

        try:
            # astroalign attend (x, y) — nos étoiles sont (row=y, col=x)
            ref_xy = ref_pts_all[:, [1, 0]].astype(np.float64)
            cur_xy = cur_pts_all[:, [1, 0]].astype(np.float64)

            T, (matched_src, matched_dst) = _aa.find_transform(cur_xy, ref_xy)

            # T.params est 3×3 SimilarityTransform (x,y) compatible cv2.warpAffine
            M = T.params[:2, :].astype(np.float32)
            angle = float(np.degrees(T.rotation))
            scale = float(T.scale)
            dx = float(T.translation[0])
            dy = float(T.translation[1])
            drift = float(np.sqrt(dx**2 + dy**2))
            print(f"  [astroalign] angle={angle:.3f}° drift={drift:.1f}px "
                  f"scale={scale:.4f} matches={len(matched_src)}")

            params = {
                'dx': dx, 'dy': dy, 'angle': angle, 'scale': scale,
                'inliers': len(matched_src), 'total': len(matched_src),
                'method': 'astroalign',
            }
            return np.vstack([M, [0, 0, 1]]), params

        except Exception as e:
            print(f"  [FAIL] astroalign: {e}")
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
        """Applique transformation à l'image sur le canvas de sortie.

        Si canvas_margin_frac > 0, compose la transformation avec un décalage
        de marge afin que la frame de référence soit centrée dans le canvas
        élargi. Les frames décalées n'ont donc plus de bord coupé.

        Les pixels de bordure sont mis à NaN pour que le stacker les ignore.
        """
        mx = getattr(self, '_canvas_mx', 0)
        my = getattr(self, '_canvas_my', 0)
        cw = getattr(self, '_canvas_w', None)
        ch = getattr(self, '_canvas_h', None)

        h_img, w_img = image.shape[:2]
        out_w = cw if cw is not None else w_img
        out_h = ch if ch is not None else h_img

        # Composer la transformation avec le décalage de marge (canvas offset)
        # M_out = M_offset @ M_align  où M_offset = [[1,0,mx],[0,1,my]]
        if mx > 0 or my > 0:
            M3 = np.vstack([transform[:2, :], [0, 0, 1]]).astype(np.float64)
            shift3 = np.array([[1, 0, mx], [0, 1, my], [0, 0, 1]], dtype=np.float64)
            M = (shift3 @ M3)[:2, :].astype(np.float32)
        else:
            M = transform[:2, :].astype(np.float32)

        _ones = np.ones((h_img, w_img), dtype=np.float32)
        valid_mask = cv2.warpAffine(
            _ones, M, (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ) > 0.5

        if self.is_color and len(image.shape) == 3:
            aligned = cv2.warpAffine(
                image.astype(np.float32), M, (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            aligned[~valid_mask] = np.nan
        else:
            aligned = cv2.warpAffine(
                image.astype(np.float32), M, (out_w, out_h),
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
            method = params.get('method', 'hough')
            print(f"  Rotation [{method}]: angle={params.get('angle', 0):.3f}°, "
                  f"drift={drift:.2f}px, "
                  f"inliers={params.get('inliers', 0)}/{params.get('total', 0)}")
        else:
            print(f"  Affine: angle={params.get('angle', 0):.3f}°, "
                  f"scale={params.get('scale', 1):.4f}, "
                  f"drift={drift:.2f}px")
