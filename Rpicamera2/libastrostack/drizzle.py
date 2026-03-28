#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithme Drizzle pour libastrostack
=====================================

Implémentation de l'algorithme Drizzle (Variable-Pixel Linear Reconstruction)
pour améliorer la résolution des images empilées en utilisant les décalages
sub-pixel naturels entre les frames.

Principe:
- Chaque pixel source est "drizzlé" (déposé) sur une grille de sortie
  plus fine (2x, 3x, etc.)
- Le paramètre pixfrac (drop size) contrôle la taille du "goutte" déposée
- Les décalages sub-pixel entre frames permettent de récupérer de
  l'information haute fréquence

Références:
- Fruchter & Hook 2002, "Drizzle: A Method for the Linear Reconstruction 
  of Undersampled Images"
- STScI DrizzlePac documentation

Usage:
    from libastrostack.drizzle import DrizzleStacker
    
    drizzler = DrizzleStacker(scale=2.0, pixfrac=0.8)
    
    for image, transform in aligned_images:
        drizzler.add_image(image, transform)
    
    result = drizzler.combine()

Auteur: libastrostack Team
Version: 1.0.0
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import warnings


class DrizzleStacker:
    """
    Stacker avec algorithme Drizzle pour super-résolution
    
    Le Drizzle permet d'obtenir une image de sortie avec une résolution
    supérieure à la résolution native du capteur, en exploitant les
    décalages sub-pixel entre les frames successives.
    
    Paramètres clés:
    - scale: Facteur de sur-échantillonnage (2.0 = double résolution)
    - pixfrac: Taille de la goutte (0.0-1.0, typiquement 0.7-0.9)
        - 1.0 = pixels sources couvrent tout (comme moyenne simple)
        - 0.5 = gouttes plus petites, meilleure résolution mais plus de bruit
        - Valeur recommandée: 0.7-0.8
    """
    
    def __init__(self, scale: float = 2.0, pixfrac: float = 0.8,
                 kernel: str = 'point'):
        """
        Initialise le Drizzle stacker
        
        Args:
            scale: Facteur de sur-échantillonnage (1.0 à 4.0)
            pixfrac: Drop size ratio (0.1 à 1.0)
            kernel: Type de kernel ('point', 'square', 'gaussian')
        """
        if not 1.0 <= scale <= 4.0:
            raise ValueError("scale doit être entre 1.0 et 4.0")
        if not 0.1 <= pixfrac <= 1.0:
            raise ValueError("pixfrac doit être entre 0.1 et 1.0")
        
        self.scale = scale
        self.pixfrac = pixfrac
        self.kernel = kernel
        
        # Images de sortie
        self._output: Optional[np.ndarray] = None
        self._weight_map: Optional[np.ndarray] = None
        self._context_map: Optional[np.ndarray] = None  # Compte des contributions
        
        # Dimensions
        self._input_shape: Optional[Tuple] = None
        self._output_shape: Optional[Tuple] = None
        self._is_color: Optional[bool] = None
        
        # Statistiques
        self.stats = {
            'num_images': 0,
            'total_drops': 0,
            'coverage': 0.0,
        }
        
        print(f"[DRIZZLE] Initialisé: scale={scale}x, pixfrac={pixfrac}, kernel={kernel}")
    
    def _init_output(self, input_shape: Tuple):
        """Initialise les buffers de sortie"""
        self._input_shape = input_shape
        self._is_color = len(input_shape) == 3
        
        if self._is_color:
            h, w, c = input_shape
            out_h = int(h * self.scale)
            out_w = int(w * self.scale)
            self._output_shape = (out_h, out_w, c)
            
            self._output = np.zeros((out_h, out_w, c), dtype=np.float64)
            self._weight_map = np.zeros((out_h, out_w), dtype=np.float64)
            self._context_map = np.zeros((out_h, out_w), dtype=np.int32)
        else:
            h, w = input_shape
            out_h = int(h * self.scale)
            out_w = int(w * self.scale)
            self._output_shape = (out_h, out_w)
            
            self._output = np.zeros((out_h, out_w), dtype=np.float64)
            self._weight_map = np.zeros((out_h, out_w), dtype=np.float64)
            self._context_map = np.zeros((out_h, out_w), dtype=np.int32)
        
        print(f"[DRIZZLE] Output: {self._input_shape} -> {self._output_shape}")
    
    def add_image(self, image: np.ndarray, 
                  transform: Optional[np.ndarray] = None,
                  dx: float = 0.0, dy: float = 0.0,
                  angle: float = 0.0,
                  weight: float = 1.0):
        """
        Ajoute une image au drizzle
        
        Args:
            image: Image source (float array)
            transform: Matrice de transformation 3x3 (optionnel)
            dx, dy: Décalage en pixels (si pas de transform)
            angle: Rotation en degrés (si pas de transform)
            weight: Poids global de l'image
        """
        # Initialiser output si nécessaire
        if self._output is None:
            self._init_output(image.shape)
        
        # Vérifier dimensions
        if image.shape != self._input_shape:
            raise ValueError(f"Dimensions incompatibles: {image.shape} vs {self._input_shape}")
        
        # Construire transformation si non fournie
        if transform is None:
            transform = self._build_transform(dx, dy, angle)
        
        # Drizzle l'image
        if self.kernel == 'point':
            self._drizzle_point(image, transform, weight)
        elif self.kernel == 'square':
            self._drizzle_square(image, transform, weight)
        else:  # gaussian
            self._drizzle_gaussian(image, transform, weight)
        
        self.stats['num_images'] += 1
    
    def _build_transform(self, dx: float, dy: float, angle: float) -> np.ndarray:
        """
        Construit matrice de transformation 3x3
        
        Args:
            dx, dy: Translation
            angle: Rotation en degrés
        
        Returns:
            Matrice 3x3 de transformation
        """
        # Rotation autour du centre
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Centre de l'image source
        if self._is_color:
            cy, cx = self._input_shape[0] / 2, self._input_shape[1] / 2
        else:
            cy, cx = self._input_shape[0] / 2, self._input_shape[1] / 2
        
        # Matrice: translate to origin, rotate, translate back + offset
        # T = T_back @ R @ T_origin
        T_origin = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])
        
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])
        
        T_back = np.array([
            [1, 0, cx + dx],
            [0, 1, cy + dy],
            [0, 0, 1]
        ])
        
        return T_back @ R @ T_origin
    
    def _drizzle_point(self, image: np.ndarray, transform: np.ndarray, 
                       weight: float):
        """
        Drizzle avec kernel point (rapide)
        
        Chaque pixel source est déposé au point correspondant
        dans l'output, sans interpolation.
        """
        h_in, w_in = self._input_shape[:2]
        h_out, w_out = self._output_shape[:2]
        
        # Taille effective du drop
        drop_size = self.pixfrac
        
        # Parcourir pixels source
        for y_in in range(h_in):
            for x_in in range(w_in):
                # Valeur pixel source
                if self._is_color:
                    pixel_value = image[y_in, x_in, :]
                    if np.all(pixel_value == 0):
                        continue
                else:
                    pixel_value = image[y_in, x_in]
                    if pixel_value == 0:
                        continue
                
                # Transformer coordonnées (centre du pixel)
                src_pt = np.array([x_in + 0.5, y_in + 0.5, 1.0])
                dst_pt = transform @ src_pt
                
                # Coordonnées dans output (avec scale)
                x_out = dst_pt[0] * self.scale
                y_out = dst_pt[1] * self.scale
                
                # Pixel output le plus proche
                ix_out = int(x_out)
                iy_out = int(y_out)
                
                # Vérifier bornes
                if 0 <= ix_out < w_out and 0 <= iy_out < h_out:
                    # Déposer
                    w = weight * drop_size
                    if self._is_color:
                        self._output[iy_out, ix_out, :] += pixel_value * w
                    else:
                        self._output[iy_out, ix_out] += pixel_value * w
                    
                    self._weight_map[iy_out, ix_out] += w
                    self._context_map[iy_out, ix_out] += 1
                    self.stats['total_drops'] += 1
    
    def _drizzle_square(self, image: np.ndarray, transform: np.ndarray,
                        weight: float):
        """
        Drizzle avec kernel carré
        
        Chaque pixel source est déposé sur plusieurs pixels output
        selon la taille du drop (pixfrac).
        """
        h_in, w_in = self._input_shape[:2]
        h_out, w_out = self._output_shape[:2]
        
        # Rayon du drop en pixels output
        drop_radius = self.pixfrac * self.scale / 2
        
        for y_in in range(h_in):
            for x_in in range(w_in):
                # Valeur pixel
                if self._is_color:
                    pixel_value = image[y_in, x_in, :]
                    if np.all(pixel_value == 0):
                        continue
                else:
                    pixel_value = image[y_in, x_in]
                    if pixel_value == 0:
                        continue
                
                # Centre du drop dans output
                src_pt = np.array([x_in + 0.5, y_in + 0.5, 1.0])
                dst_pt = transform @ src_pt
                x_center = dst_pt[0] * self.scale
                y_center = dst_pt[1] * self.scale
                
                # Zone affectée
                x_min = max(0, int(x_center - drop_radius))
                x_max = min(w_out, int(x_center + drop_radius) + 1)
                y_min = max(0, int(y_center - drop_radius))
                y_max = min(h_out, int(y_center + drop_radius) + 1)
                
                # Distribuer sur les pixels output
                for iy in range(y_min, y_max):
                    for ix in range(x_min, x_max):
                        # Calcul overlap
                        overlap = self._compute_overlap(
                            x_center, y_center, drop_radius,
                            ix, iy
                        )
                        
                        if overlap > 0:
                            w = weight * overlap
                            if self._is_color:
                                self._output[iy, ix, :] += pixel_value * w
                            else:
                                self._output[iy, ix] += pixel_value * w
                            
                            self._weight_map[iy, ix] += w
                            self._context_map[iy, ix] += 1
                            self.stats['total_drops'] += 1
    
    def _drizzle_gaussian(self, image: np.ndarray, transform: np.ndarray,
                          weight: float):
        """
        Drizzle avec kernel gaussien
        
        Le drop a une distribution gaussienne centrée sur la
        position transformée.
        """
        h_in, w_in = self._input_shape[:2]
        h_out, w_out = self._output_shape[:2]
        
        # Sigma du gaussian en pixels output
        sigma = self.pixfrac * self.scale / 2
        
        # Rayon d'influence (3 sigma)
        radius = int(3 * sigma) + 1
        
        for y_in in range(h_in):
            for x_in in range(w_in):
                if self._is_color:
                    pixel_value = image[y_in, x_in, :]
                    if np.all(pixel_value == 0):
                        continue
                else:
                    pixel_value = image[y_in, x_in]
                    if pixel_value == 0:
                        continue
                
                # Centre
                src_pt = np.array([x_in + 0.5, y_in + 0.5, 1.0])
                dst_pt = transform @ src_pt
                x_center = dst_pt[0] * self.scale
                y_center = dst_pt[1] * self.scale
                
                # Zone
                x_min = max(0, int(x_center) - radius)
                x_max = min(w_out, int(x_center) + radius + 1)
                y_min = max(0, int(y_center) - radius)
                y_max = min(h_out, int(y_center) + radius + 1)
                
                for iy in range(y_min, y_max):
                    for ix in range(x_min, x_max):
                        # Distance au centre
                        dx = (ix + 0.5) - x_center
                        dy = (iy + 0.5) - y_center
                        d2 = dx * dx + dy * dy
                        
                        # Poids gaussien
                        g = np.exp(-d2 / (2 * sigma * sigma))
                        
                        if g > 0.01:  # Seuil
                            w = weight * g
                            if self._is_color:
                                self._output[iy, ix, :] += pixel_value * w
                            else:
                                self._output[iy, ix] += pixel_value * w
                            
                            self._weight_map[iy, ix] += w
                            self._context_map[iy, ix] += 1
                            self.stats['total_drops'] += 1
    
    def _compute_overlap(self, x_center: float, y_center: float,
                         drop_radius: float, px: int, py: int) -> float:
        """
        Calcule l'overlap entre un drop carré et un pixel output
        
        Args:
            x_center, y_center: Centre du drop
            drop_radius: Rayon du drop
            px, py: Coordonnées pixel output
        
        Returns:
            Fraction d'overlap (0-1)
        """
        # Bornes du drop
        drop_x1 = x_center - drop_radius
        drop_x2 = x_center + drop_radius
        drop_y1 = y_center - drop_radius
        drop_y2 = y_center + drop_radius
        
        # Bornes du pixel
        pix_x1, pix_x2 = px, px + 1
        pix_y1, pix_y2 = py, py + 1
        
        # Intersection
        inter_x1 = max(drop_x1, pix_x1)
        inter_x2 = min(drop_x2, pix_x2)
        inter_y1 = max(drop_y1, pix_y1)
        inter_y2 = min(drop_y2, pix_y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        drop_area = (2 * drop_radius) ** 2
        
        return inter_area / drop_area
    
    def combine(self) -> Optional[np.ndarray]:
        """
        Finalise le drizzle et retourne l'image combinée
        
        Returns:
            Image drizzled normalisée
        """
        if self._output is None:
            return None
        
        # Normaliser par les poids
        weight_safe = np.maximum(self._weight_map, 1e-10)
        
        if self._is_color:
            result = np.zeros_like(self._output)
            for c in range(self._output.shape[2]):
                result[:, :, c] = self._output[:, :, c] / weight_safe
        else:
            result = self._output / weight_safe
        
        # Masquer pixels sans contribution
        if self._is_color:
            mask = self._weight_map < 1e-10
            for c in range(result.shape[2]):
                result[:, :, c][mask] = 0
        else:
            result[self._weight_map < 1e-10] = 0
        
        # Stats
        total_pixels = self._weight_map.size
        covered_pixels = np.sum(self._weight_map > 0)
        self.stats['coverage'] = 100.0 * covered_pixels / total_pixels
        
        print(f"\n[DRIZZLE] Résultat:")
        print(f"  - Images: {self.stats['num_images']}")
        print(f"  - Drops: {self.stats['total_drops']:,}")
        print(f"  - Couverture: {self.stats['coverage']:.1f}%")
        print(f"  - Taille: {self._input_shape} -> {self._output_shape}")
        
        return result.astype(np.float32)
    
    def get_weight_map(self) -> Optional[np.ndarray]:
        """Retourne la carte des poids (exposition relative)"""
        return self._weight_map.copy() if self._weight_map is not None else None
    
    def get_context_map(self) -> Optional[np.ndarray]:
        """Retourne la carte du nombre de contributions par pixel"""
        return self._context_map.copy() if self._context_map is not None else None
    
    def reset(self):
        """Réinitialise le drizzle"""
        self._output = None
        self._weight_map = None
        self._context_map = None
        self._input_shape = None
        self._output_shape = None
        self._is_color = None
        self.stats = {
            'num_images': 0,
            'total_drops': 0,
            'coverage': 0.0,
        }


class DrizzleConfig:
    """Configuration pour le drizzle"""
    
    def __init__(self):
        self.enable = False           # Activer drizzle
        self.scale = 2.0              # Facteur de super-résolution
        self.pixfrac = 0.8            # Drop size (0.1-1.0)
        self.kernel = 'square'        # 'point', 'square', 'gaussian'
        
        # Options avancées
        self.fill_value = 0.0         # Valeur pour pixels sans données
        self.weight_type = 'uniform'  # 'uniform', 'exptime', 'ivm'
    
    def validate(self):
        """Valide la configuration"""
        errors = []
        
        if not 1.0 <= self.scale <= 4.0:
            errors.append("scale doit être entre 1.0 et 4.0")
        
        if not 0.1 <= self.pixfrac <= 1.0:
            errors.append("pixfrac doit être entre 0.1 et 1.0")
        
        if self.kernel not in ['point', 'square', 'gaussian']:
            errors.append("kernel invalide")
        
        if errors:
            raise ValueError("Config drizzle invalide: " + ", ".join(errors))
        
        return True


# =============================================================================
# Optimisation NumPy vectorisée
# =============================================================================

class DrizzleStackerFast(DrizzleStacker):
    """
    Version optimisée du DrizzleStacker utilisant des opérations vectorisées
    
    Beaucoup plus rapide que la version de base pour les grandes images,
    mais utilise plus de mémoire.
    """
    
    def _drizzle_point(self, image: np.ndarray, transform: np.ndarray,
                       weight: float):
        """Version vectorisée du drizzle point"""
        h_in, w_in = self._input_shape[:2]
        h_out, w_out = self._output_shape[:2]
        
        # Créer grille de coordonnées
        y_coords, x_coords = np.meshgrid(
            np.arange(h_in) + 0.5,
            np.arange(w_in) + 0.5,
            indexing='ij'
        )
        
        # Aplatir
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        ones = np.ones_like(x_flat)
        
        # Transformer toutes les coordonnées
        src_pts = np.vstack([x_flat, y_flat, ones])  # 3 x N
        dst_pts = transform @ src_pts  # 3 x N
        
        # Coordonnées output
        x_out = (dst_pts[0, :] * self.scale).astype(int)
        y_out = (dst_pts[1, :] * self.scale).astype(int)
        
        # Masque pixels valides
        valid = (x_out >= 0) & (x_out < w_out) & (y_out >= 0) & (y_out < h_out)
        
        # Masque pixels non-nuls
        if self._is_color:
            img_flat = image.reshape(-1, image.shape[2])
            nonzero = np.any(img_flat > 0, axis=1)
        else:
            img_flat = image.flatten()
            nonzero = img_flat > 0
        
        valid = valid & nonzero
        
        # Indices valides
        idx_in = np.where(valid)[0]
        x_out_valid = x_out[valid]
        y_out_valid = y_out[valid]
        
        # Poids
        w = weight * self.pixfrac
        
        # Accumuler (utiliser np.add.at pour éviter les problèmes de race condition)
        if self._is_color:
            for c in range(image.shape[2]):
                values = img_flat[idx_in, c] * w
                np.add.at(self._output[:, :, c], (y_out_valid, x_out_valid), values)
        else:
            values = img_flat[idx_in] * w
            np.add.at(self._output, (y_out_valid, x_out_valid), values)
        
        np.add.at(self._weight_map, (y_out_valid, x_out_valid), w)
        np.add.at(self._context_map, (y_out_valid, x_out_valid), 1)
        
        self.stats['total_drops'] += len(idx_in)


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test DrizzleStacker ===\n")
    
    # Créer image test avec pattern
    np.random.seed(42)
    h, w = 50, 50
    
    # Pattern damier fin
    pattern = np.zeros((h, w), dtype=np.float32)
    pattern[::2, ::2] = 0.8
    pattern[1::2, 1::2] = 0.8
    pattern += np.random.normal(0, 0.02, (h, w)).astype(np.float32)
    
    # Simuler 8 frames avec décalages sub-pixel
    frames = []
    transforms = []
    
    for i in range(8):
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        angle = np.random.uniform(-0.5, 0.5)
        
        frames.append(pattern.copy())
        transforms.append({'dx': dx, 'dy': dy, 'angle': angle})
    
    # Test drizzle 2x
    print("--- Drizzle 2x, pixfrac=0.8 ---")
    drizzler = DrizzleStackerFast(scale=2.0, pixfrac=0.8, kernel='point')
    
    for frame, t in zip(frames, transforms):
        drizzler.add_image(frame, dx=t['dx'], dy=t['dy'], angle=t['angle'])
    
    result = drizzler.combine()
    print(f"Taille sortie: {result.shape}")
    
    # Test drizzle 3x
    print("\n--- Drizzle 3x, pixfrac=0.7, kernel=square ---")
    drizzler = DrizzleStacker(scale=3.0, pixfrac=0.7, kernel='square')
    
    for frame, t in zip(frames, transforms):
        drizzler.add_image(frame, dx=t['dx'], dy=t['dy'], angle=t['angle'])
    
    result = drizzler.combine()
    print(f"Taille sortie: {result.shape}")
    
    print("\n=== Tests terminés ===")
