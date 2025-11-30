#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrôle qualité des images pour libastrostack
"""

import numpy as np
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from skimage.feature import peak_local_max


class QualityAnalyzer:
    """Analyse la qualité d'une image et décide si elle doit être empilée"""
    
    def __init__(self, config):
        """
        Args:
            config: QualityConfig instance
        """
        self.config = config
    
    def analyze(self, image):
        """
        Analyse une image et retourne le verdict qualité
        
        Args:
            image: Image à analyser (float array)
        
        Returns:
            (is_good, metrics, reason)
            - is_good: True si image acceptable
            - metrics: dict avec métriques calculées
            - reason: str avec raison de rejet (vide si acceptée)
        """
        if not self.config.enable:
            return True, {}, ""
        
        # Extraire canal pour analyse (vert si RGB)
        if len(image.shape) == 3:
            analyze_image = image[:, :, 1]
        else:
            analyze_image = image
        
        metrics = {}
        
        # 1. Détecter étoiles
        stars = self._detect_stars(analyze_image)
        metrics['num_stars'] = len(stars)

        # Contrôle nombre d'étoiles (sauf si min_stars = 0 = désactivé)
        if self.config.min_stars > 0 and len(stars) < self.config.min_stars:
            return False, metrics, f"Trop peu d'étoiles ({len(stars)} < {self.config.min_stars})"
        
        # 2. Mesurer FWHM et ellipticité
        fwhms = []
        ellipticities = []
        
        for star in stars[:20]:  # Analyser les 20 plus brillantes
            fwhm, ellipticity = self._measure_star(analyze_image, star)
            if fwhm > 0:
                fwhms.append(fwhm)
                ellipticities.append(ellipticity)
        
        if not fwhms:
            return False, metrics, "Impossible de mesurer les étoiles"
        
        metrics['median_fwhm'] = np.median(fwhms)
        metrics['median_ellipticity'] = np.median(ellipticities)
        metrics['sharpness'] = self._measure_sharpness(analyze_image)
        
        # 3. Vérifier seuils
        if metrics['median_fwhm'] > self.config.max_fwhm:
            return False, metrics, f"FWHM trop grande ({metrics['median_fwhm']:.2f} > {self.config.max_fwhm})"
        
        if metrics['median_ellipticity'] > self.config.max_ellipticity:
            return False, metrics, f"Étoiles allongées ({metrics['median_ellipticity']:.2f} > {self.config.max_ellipticity})"
        
        if metrics['sharpness'] < self.config.min_sharpness:
            return False, metrics, f"Image floue ({metrics['sharpness']:.2f} < {self.config.min_sharpness})"
        
        return True, metrics, ""
    
    def _detect_stars(self, image):
        """
        Détecte les étoiles dans une image
        
        Args:
            image: Image MONO (float array)
        
        Returns:
            Array de positions (N, 2) avec [y, x]
        """
        # Estimer fond du ciel
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        
        # Seuil de détection
        threshold = median + self.config.star_detection_sigma * std
        
        # Lisser pour éviter bruit
        smoothed = ndimage.gaussian_filter(image, sigma=1.5)
        
        # Détecter pics
        peaks = peak_local_max(
            smoothed,
            min_distance=self.config.min_star_separation,
            threshold_abs=threshold,
            num_peaks=100
        )
        
        if len(peaks) == 0:
            return np.array([])
        
        # Trier par intensité (plus brillantes en premier)
        intensities = [image[p[0], p[1]] for p in peaks]
        sorted_indices = np.argsort(intensities)[::-1]
        
        return peaks[sorted_indices]
    
    def _measure_star(self, image, star_pos, box_size=15):
        """
        Mesure FWHM et ellipticité d'une étoile
        
        Args:
            image: Image MONO
            star_pos: Position [y, x]
            box_size: Taille de la boîte d'analyse
        
        Returns:
            (fwhm, ellipticity)
        """
        y, x = star_pos
        h, w = image.shape
        
        # Extraire région
        y1 = max(0, y - box_size)
        y2 = min(h, y + box_size)
        x1 = max(0, x - box_size)
        x2 = min(w, x + box_size)
        
        cutout = image[y1:y2, x1:x2].astype(np.float64)
        
        if cutout.size == 0:
            return 0, 0
        
        # Soustraire fond local
        background = np.percentile(cutout, 20)
        cutout = cutout - background
        cutout = np.maximum(cutout, 0)
        
        total = np.sum(cutout)
        if total == 0:
            return 0, 0
        
        # Calculer centroïde
        yy, xx = np.indices(cutout.shape)
        cy = np.sum(yy * cutout) / total
        cx = np.sum(xx * cutout) / total
        
        # Calculer moments d'inertie
        yy_centered = yy - cy
        xx_centered = xx - cx
        
        Mxx = np.sum(cutout * xx_centered**2) / total
        Myy = np.sum(cutout * yy_centered**2) / total
        Mxy = np.sum(cutout * xx_centered * yy_centered) / total
        
        # Calculer valeurs propres
        a = (Mxx + Myy) / 2
        b = np.sqrt(((Mxx - Myy) / 2)**2 + Mxy**2)
        
        lambda1 = a + b
        lambda2 = a - b
        
        if lambda2 <= 0:
            return 0, 1.0
        
        # FWHM = 2.355 * sigma
        fwhm = 2.355 * np.sqrt((lambda1 + lambda2) / 2)
        
        # Ellipticité
        ellipticity = 1.0 - np.sqrt(lambda2 / lambda1)
        
        return fwhm, ellipticity
    
    def _measure_sharpness(self, image):
        """
        Mesure la netteté de l'image (gradient moyen)
        
        Args:
            image: Image MONO
        
        Returns:
            Sharpness (0-1)
        """
        # Calculer gradient
        gy, gx = np.gradient(image.astype(np.float64))
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Normaliser par dynamique de l'image
        image_range = np.percentile(image, 99) - np.percentile(image, 1)
        
        if image_range == 0:
            return 0
        
        return np.mean(gradient_magnitude) / image_range
