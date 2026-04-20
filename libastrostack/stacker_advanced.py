#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes de stacking avancées pour libastrostack
================================================

Ce module fournit des algorithmes de combinaison d'images plus sophistiqués
que la simple moyenne glissante, permettant un meilleur rejet des outliers
(satellites, rayons cosmiques, pixels chauds).

Méthodes disponibles:
- MEAN: Moyenne simple (rapide, sensible aux outliers)
- MEDIAN: Médiane (robuste, perte de SNR ~20%)
- KAPPA_SIGMA: Rejet itératif des outliers par sigma-clipping
- WINSORIZED: Sigma-clipping avec remplacement par les bornes
- WEIGHTED: Moyenne pondérée par qualité d'image

Usage:
    from libastrostack.stacker_advanced import AdvancedStacker, StackMethod
    
    stacker = AdvancedStacker(config)
    stacker.set_method(StackMethod.KAPPA_SIGMA, kappa=2.5, iterations=3)
    
    for image in images:
        stacker.add_image(image)
    
    result = stacker.combine()

Auteur: libastrostack Team
Version: 1.0.0
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import warnings


class StackMethod(Enum):
    """Méthodes de combinaison disponibles"""
    MEAN = "mean"                   # Moyenne simple
    MEDIAN = "median"               # Médiane
    KAPPA_SIGMA = "kappa_sigma"     # Sigma-clipping itératif
    WINSORIZED = "winsorized"       # Winsorized sigma-clipping
    WEIGHTED = "weighted"           # Moyenne pondérée par qualité


class AdvancedStacker:
    """
    Stacker avancé avec multiple méthodes de combinaison
    
    Contrairement au stacker de base qui utilise une moyenne glissante,
    ce stacker accumule toutes les images en mémoire puis les combine
    à la fin avec la méthode choisie.
    
    Pour le live stacking avec ressources limitées, utiliser le mode
    'streaming' qui maintient des statistiques running sans stocker
    toutes les images.
    """
    
    def __init__(self, config=None):
        """
        Initialise le stacker avancé
        
        Args:
            config: StackingConfig instance (optionnel)
        """
        self.config = config
        
        # Méthode par défaut
        self._method = StackMethod.KAPPA_SIGMA
        self._method_params = {
            'kappa': 2.5,           # Seuil sigma pour rejection
            'iterations': 3,        # Nombre d'itérations max
            'min_images': 3,        # Minimum d'images pour sigma-clip
        }
        
        # Stockage des images
        self._images: List[np.ndarray] = []
        self._weights: List[float] = []
        self._is_color: Optional[bool] = None
        
        # Mode streaming (pour live stacking mémoire limitée)
        self._streaming_mode = False
        self._running_sum = None
        self._running_sum_sq = None
        self._running_count = None
        self._running_weight_sum = None  # Somme des poids pour moyenne pondérée
        
        # Statistiques
        self.stats = {
            'num_images': 0,
            'pixels_rejected': 0,
            'rejection_rate': 0.0,
        }
    
    def set_method(self, method: StackMethod, **params):
        """
        Configure la méthode de combinaison
        
        Args:
            method: StackMethod enum
            **params: Paramètres spécifiques à la méthode
                - kappa: Seuil sigma (défaut 2.5)
                - iterations: Itérations max (défaut 3)
                - min_images: Minimum pour sigma-clip (défaut 3)
        """
        self._method = method
        self._method_params.update(params)
        
        print(f"[STACKER] Méthode: {method.value}")
        if method in [StackMethod.KAPPA_SIGMA, StackMethod.WINSORIZED]:
            print(f"          Kappa: {self._method_params['kappa']}, "
                  f"Iterations: {self._method_params['iterations']}")
    
    def enable_streaming(self, enable: bool = True):
        """
        Active le mode streaming pour économiser la mémoire
        
        En mode streaming, seules les statistiques running sont maintenues,
        pas les images individuelles. Cela limite les méthodes disponibles
        à MEAN et une approximation de KAPPA_SIGMA.
        
        Args:
            enable: True pour activer
        """
        self._streaming_mode = enable
        if enable:
            print("[STACKER] Mode streaming activé (mémoire limitée)")
    
    def add_image(self, image: np.ndarray, weight: float = 1.0, 
                  quality_metrics: Optional[Dict] = None):
        """
        Ajoute une image au stack
        
        Args:
            image: Image alignée (float32/float64)
            weight: Poids de l'image (pour méthode WEIGHTED)
            quality_metrics: Métriques qualité optionnelles
        """
        # Détecter type d'image
        if self._is_color is None:
            self._is_color = len(image.shape) == 3
        
        # Vérifier cohérence
        if len(image.shape) == 3 and not self._is_color:
            raise ValueError("Image RGB fournie mais mode MONO détecté")
        if len(image.shape) == 2 and self._is_color:
            raise ValueError("Image MONO fournie mais mode RGB détecté")
        
        if self._streaming_mode:
            self._add_streaming(image, weight)
        else:
            self._images.append(image.astype(np.float64))
            
            # Calculer poids automatique si métriques fournies
            if quality_metrics and self._method == StackMethod.WEIGHTED:
                weight = self._compute_weight(quality_metrics)
            self._weights.append(weight)
        
        self.stats['num_images'] += 1
    
    def _add_streaming(self, image: np.ndarray, weight: float):
        """Ajoute image en mode streaming"""
        img = image.astype(np.float64)
        
        if self._running_sum is None:
            self._running_sum = np.zeros_like(img)
            self._running_sum_sq = np.zeros_like(img)
            self._running_count = np.zeros(img.shape[:2], dtype=np.int32)
            self._running_weight_sum = np.zeros(img.shape[:2], dtype=np.float64)

        # Masque pixels valides
        if self._is_color:
            valid = np.any(img > 0, axis=2)
        else:
            valid = img > 0

        # Accumuler (pondéré par weight = score moyen du buffer)
        if self._is_color:
            for c in range(3):
                self._running_sum[:, :, c][valid] += img[:, :, c][valid] * weight
                self._running_sum_sq[:, :, c][valid] += (img[:, :, c][valid] ** 2) * weight
        else:
            self._running_sum[valid] += img[valid] * weight
            self._running_sum_sq[valid] += (img[valid] ** 2) * weight

        self._running_count[valid] += 1
        self._running_weight_sum[valid] += weight
    
    def _compute_weight(self, metrics: Dict) -> float:
        """
        Calcule un poids basé sur les métriques qualité
        
        Args:
            metrics: Dict avec fwhm, sharpness, num_stars, etc.
        
        Returns:
            Poids normalisé (0-1)
        """
        weight = 1.0
        
        # Pénaliser FWHM élevée
        if 'median_fwhm' in metrics:
            fwhm = metrics['median_fwhm']
            weight *= np.exp(-0.1 * max(0, fwhm - 3))
        
        # Favoriser netteté
        if 'sharpness' in metrics:
            weight *= metrics['sharpness']
        
        # Favoriser nombre d'étoiles
        if 'num_stars' in metrics:
            weight *= min(1.0, metrics['num_stars'] / 50)
        
        return max(0.1, min(1.0, weight))
    
    def combine(self) -> Optional[np.ndarray]:
        """
        Combine toutes les images avec la méthode configurée
        
        Returns:
            Image combinée (float64) ou None si pas d'images
        """
        if self._streaming_mode:
            return self._combine_streaming()
        
        if len(self._images) == 0:
            return None
        
        print(f"\n[COMBINE] {len(self._images)} images, méthode: {self._method.value}")
        
        # Empiler en cube 3D (ou 4D pour RGB)
        cube = np.stack(self._images, axis=0)
        weights = np.array(self._weights)
        
        # Appliquer méthode
        if self._method == StackMethod.MEAN:
            result = self._combine_mean(cube, weights)
        elif self._method == StackMethod.MEDIAN:
            result = self._combine_median(cube)
        elif self._method == StackMethod.KAPPA_SIGMA:
            result = self._combine_kappa_sigma(cube)
        elif self._method == StackMethod.WINSORIZED:
            result = self._combine_winsorized(cube)
        elif self._method == StackMethod.WEIGHTED:
            result = self._combine_weighted(cube, weights)
        else:
            result = self._combine_mean(cube, weights)
        
        # Stats finales
        print(f"[COMBINE] Terminé - Pixels rejetés: {self.stats['pixels_rejected']:,} "
              f"({self.stats['rejection_rate']:.2f}%)")
        
        return result
    
    def _combine_streaming(self) -> Optional[np.ndarray]:
        """Combine en mode streaming — moyenne pondérée par le score de chaque buffer."""
        if self._running_sum is None:
            return None

        # Diviseur : somme des poids si disponible, sinon count (rétrocompat)
        if self._running_weight_sum is not None and np.any(self._running_weight_sum > 0):
            divisor = np.maximum(self._running_weight_sum, 1e-10)
        else:
            divisor = np.maximum(self._running_count, 1).astype(np.float64)

        if self._is_color:
            result = np.zeros_like(self._running_sum)
            for c in range(3):
                result[:, :, c] = self._running_sum[:, :, c] / divisor
        else:
            result = self._running_sum / divisor

        return result
    
    def _combine_mean(self, cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Moyenne pondérée simple"""
        if self._is_color:
            # cube shape: (N, H, W, 3)
            weights_expanded = weights[:, np.newaxis, np.newaxis, np.newaxis]
        else:
            # cube shape: (N, H, W)
            weights_expanded = weights[:, np.newaxis, np.newaxis]
        
        weighted_sum = np.sum(cube * weights_expanded, axis=0)
        weight_sum = np.sum(weights)
        
        return weighted_sum / weight_sum
    
    def _combine_median(self, cube: np.ndarray) -> np.ndarray:
        """Médiane pixel par pixel"""
        return np.median(cube, axis=0)
    
    def _combine_kappa_sigma(self, cube: np.ndarray) -> np.ndarray:
        """
        Kappa-Sigma clipping itératif
        
        Rejette les pixels qui s'écartent de plus de kappa*sigma
        de la moyenne, puis recalcule moyenne/sigma sur les
        pixels restants.
        """
        kappa = self._method_params['kappa']
        iterations = self._method_params['iterations']
        min_images = self._method_params['min_images']
        
        n_images = cube.shape[0]
        
        if n_images < min_images:
            print(f"[WARN] Pas assez d'images ({n_images}) pour sigma-clip, "
                  f"utilisation moyenne simple")
            return np.mean(cube, axis=0)
        
        # Masque des pixels valides (True = garder)
        mask = np.ones(cube.shape, dtype=bool)
        
        total_rejected = 0
        
        for iteration in range(iterations):
            # Calculer stats sur pixels masqués
            masked_cube = np.ma.array(cube, mask=~mask)
            mean = np.ma.mean(masked_cube, axis=0)
            std = np.ma.std(masked_cube, axis=0)
            
            # Éviter division par zéro
            std = np.maximum(std, 1e-10)
            
            # Calculer z-scores
            if self._is_color:
                mean_expanded = mean[np.newaxis, :, :, :]
                std_expanded = std[np.newaxis, :, :, :]
            else:
                mean_expanded = mean[np.newaxis, :, :]
                std_expanded = std[np.newaxis, :, :]
            
            z_scores = np.abs(cube - mean_expanded) / std_expanded
            
            # Rejeter outliers
            new_mask = mask & (z_scores <= kappa)
            
            # Compter rejets
            rejected_this_iter = np.sum(mask) - np.sum(new_mask)
            total_rejected += rejected_this_iter
            
            if rejected_this_iter == 0:
                print(f"  Iteration {iteration + 1}: convergé")
                break
            
            print(f"  Iteration {iteration + 1}: {rejected_this_iter:,} pixels rejetés")
            mask = new_mask
        
        # Calculer moyenne finale sur pixels valides
        # S'assurer qu'on garde au moins 1 pixel par position
        count = np.sum(mask, axis=0)
        count = np.maximum(count, 1)
        
        masked_cube = cube * mask
        result = np.sum(masked_cube, axis=0) / count
        
        # Stats
        total_pixels = cube.size
        self.stats['pixels_rejected'] = total_rejected
        self.stats['rejection_rate'] = 100.0 * total_rejected / total_pixels
        
        return result
    
    def _combine_winsorized(self, cube: np.ndarray) -> np.ndarray:
        """
        Winsorized sigma-clipping
        
        Au lieu de rejeter les outliers, les remplace par les
        valeurs limites (mean ± kappa*sigma). Préserve plus
        d'information que le rejet pur.
        """
        kappa = self._method_params['kappa']
        iterations = self._method_params['iterations']
        
        n_images = cube.shape[0]
        result_cube = cube.copy()
        
        total_clipped = 0
        
        for iteration in range(iterations):
            mean = np.mean(result_cube, axis=0)
            std = np.std(result_cube, axis=0)
            std = np.maximum(std, 1e-10)
            
            if self._is_color:
                mean_exp = mean[np.newaxis, :, :, :]
                std_exp = std[np.newaxis, :, :, :]
            else:
                mean_exp = mean[np.newaxis, :, :]
                std_exp = std[np.newaxis, :, :]
            
            lower = mean_exp - kappa * std_exp
            upper = mean_exp + kappa * std_exp
            
            # Compter pixels à clipper
            too_low = result_cube < lower
            too_high = result_cube > upper
            clipped_this_iter = np.sum(too_low) + np.sum(too_high)
            
            if clipped_this_iter == 0:
                print(f"  Iteration {iteration + 1}: convergé")
                break
            
            # Clipper
            result_cube = np.clip(result_cube, lower, upper)
            total_clipped += clipped_this_iter
            
            print(f"  Iteration {iteration + 1}: {clipped_this_iter:,} pixels clippés")
        
        # Stats
        self.stats['pixels_rejected'] = total_clipped
        self.stats['rejection_rate'] = 100.0 * total_clipped / cube.size
        
        return np.mean(result_cube, axis=0)
    
    def _combine_weighted(self, cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Moyenne pondérée par qualité"""
        # Normaliser poids
        weights = weights / np.sum(weights)
        
        if self._is_color:
            weights_exp = weights[:, np.newaxis, np.newaxis, np.newaxis]
        else:
            weights_exp = weights[:, np.newaxis, np.newaxis]
        
        return np.sum(cube * weights_exp, axis=0)
    
    def get_result(self) -> Optional[np.ndarray]:
        """Alias pour combine()"""
        return self.combine()
    
    def get_snr_improvement(self) -> float:
        """
        Estime l'amélioration SNR
        
        Returns:
            Facteur d'amélioration SNR
        """
        n = self.stats['num_images']
        if n == 0:
            return 1.0
        
        # SNR théorique pour moyenne
        base_snr = np.sqrt(n)
        
        # Ajustement selon méthode
        if self._method == StackMethod.MEDIAN:
            # Médiane perd ~20% de SNR vs moyenne
            return base_snr * 0.80
        elif self._method in [StackMethod.KAPPA_SIGMA, StackMethod.WINSORIZED]:
            # Légère perte due aux rejets
            rejection_factor = 1.0 - 0.5 * self.stats['rejection_rate'] / 100
            return base_snr * rejection_factor
        else:
            return base_snr
    
    def reset(self):
        """Réinitialise le stacker"""
        self._images.clear()
        self._weights.clear()
        self._is_color = None
        self._running_sum = None
        self._running_sum_sq = None
        self._running_count = None
        self.stats = {
            'num_images': 0,
            'pixels_rejected': 0,
            'rejection_rate': 0.0,
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Retourne l'utilisation mémoire estimée
        
        Returns:
            Dict avec bytes, MB, num_images
        """
        if self._streaming_mode:
            if self._running_sum is not None:
                bytes_used = (self._running_sum.nbytes + 
                             self._running_sum_sq.nbytes +
                             self._running_count.nbytes)
            else:
                bytes_used = 0
        else:
            bytes_used = sum(img.nbytes for img in self._images)
        
        return {
            'bytes': bytes_used,
            'MB': bytes_used / (1024 * 1024),
            'num_images': len(self._images) if not self._streaming_mode else self.stats['num_images'],
        }


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def sigma_clip_value(values: np.ndarray, kappa: float = 2.5, 
                     iterations: int = 3) -> Tuple[float, float]:
    """
    Applique sigma-clipping sur un array 1D
    
    Args:
        values: Array de valeurs
        kappa: Seuil sigma
        iterations: Itérations max
    
    Returns:
        (mean, std) après clipping
    """
    mask = np.ones(len(values), dtype=bool)
    
    for _ in range(iterations):
        valid = values[mask]
        if len(valid) < 3:
            break
        
        mean = np.mean(valid)
        std = np.std(valid)
        
        if std < 1e-10:
            break
        
        new_mask = mask & (np.abs(values - mean) <= kappa * std)
        
        if np.sum(new_mask) == np.sum(mask):
            break
        
        mask = new_mask
    
    valid = values[mask]
    return np.mean(valid), np.std(valid)


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test AdvancedStacker ===\n")
    
    # Créer images de test avec outliers
    np.random.seed(42)
    
    # Base: gradient + bruit
    h, w = 100, 100
    base = np.linspace(0.2, 0.8, w)[np.newaxis, :] * np.ones((h, 1))
    
    images = []
    for i in range(10):
        img = base + np.random.normal(0, 0.02, (h, w))
        
        # Ajouter quelques outliers (rayons cosmiques)
        if i in [2, 5, 7]:
            n_outliers = 20
            ys = np.random.randint(0, h, n_outliers)
            xs = np.random.randint(0, w, n_outliers)
            img[ys, xs] = 1.0  # Hot pixels
        
        images.append(img.astype(np.float32))
    
    # Test chaque méthode
    for method in StackMethod:
        print(f"\n--- {method.value} ---")
        stacker = AdvancedStacker()
        stacker.set_method(method)
        
        for img in images:
            stacker.add_image(img)
        
        result = stacker.combine()
        
        # Comparer avec base
        error = np.mean(np.abs(result - base))
        print(f"Erreur moyenne vs base: {error:.6f}")
        print(f"SNR improvement: {stacker.get_snr_improvement():.2f}x")
        print(f"Mémoire: {stacker.get_memory_usage()['MB']:.2f} MB")
    
    print("\n=== Tests terminés ===")
