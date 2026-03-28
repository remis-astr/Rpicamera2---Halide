#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empilement d'images pour libastrostack
"""

import numpy as np


class ImageStacker:
    """Empile progressivement les images alignées"""

    def __init__(self, config):
        """
        Args:
            config: StackingConfig instance
        """
        self.config = config
        self.stacked_image = None
        self.weight_map = None
        self._running_m2 = None   # Variance courante pour kappa_sigma (Welford)

    def stack(self, image):
        """
        Empile une nouvelle image

        Args:
            image: Image alignée (float array)

        Returns:
            Image empilée courante (copie), ou None si frame invalide
        """
        # Protection NaN/inf : rejeter uniquement les frames ENTIÈREMENT corrompues.
        # Les NaN partiels (bordures introduites par warpAffine lors de l'alignement)
        # sont gérés pixel par pixel dans _stack_mean / _stack_kappa via np.isfinite(),
        # donc on les laisse passer ici pour ne pas gâcher chaque frame alignée.
        n_bad = int(np.sum(~np.isfinite(image)))
        if n_bad > 0:
            if n_bad >= image.size:
                print(f"[STACKER] Frame ignorée : image entièrement invalide (NaN/inf)")
                return self.stacked_image.copy() if self.stacked_image is not None else None
            # NaN partiels (bordures) → laisser passer, traitement per-pixel

        method   = getattr(self.config, 'stacking_method', 'mean')
        kappa    = getattr(self.config, 'stacking_kappa', 2.5)
        is_color = len(image.shape) == 3

        if self.stacked_image is None:
            # Première image : initialisation
            self.stacked_image = image.astype(np.float32)

            if is_color:
                self.weight_map = np.ones(
                    (image.shape[0], image.shape[1]), dtype=np.float32
                )
            else:
                self.weight_map = np.ones_like(image, dtype=np.float32)

            # Init variance pour sigma-clipping (Welford)
            if method in ('kappa_sigma', 'winsorized'):
                self._running_m2 = np.zeros_like(self.stacked_image)

            self.config.num_stacked = 1

        else:
            self.config.num_stacked += 1
            img = image.astype(np.float32)

            if method in ('kappa_sigma', 'winsorized'):
                # Welford's online algorithm + rejet par kappa
                if self._running_m2 is None:
                    self._running_m2 = np.zeros_like(self.stacked_image)
                self._stack_kappa(img, is_color, kappa)
            else:
                # Moyenne glissante (mean / median fallback / autre)
                if method == 'median' and self.config.num_stacked == 2:
                    print("[STACKER] méthode 'median' non disponible en mode streaming → mean")
                self._stack_mean(img, is_color, self.config.num_stacked)

        # Mettre à jour bruit estimé
        self.config.noise_level = np.std(self.stacked_image)

        return self.stacked_image.copy()

    def _stack_mean(self, img, is_color, n):
        """
        Moyenne glissante pondérée par pixel (M4).

        Utilise weight_map comme compteur per-pixel (au lieu du compteur global
        num_stacked) pour calculer correctement la moyenne des pixels qui étaient
        NaN/inf dans certaines frames : un pixel absent de 2 frames sur 5 n'est
        moyenné que sur 3 valeurs, pas 5.
        """
        cnt = self.weight_map  # (H, W) — compteur courant par pixel

        if is_color:
            # Pixel accepté seulement si tous les canaux sont finis
            valid = np.all([np.isfinite(img[:, :, i]) for i in range(3)], axis=0)
            cnt_new = np.where(valid, cnt + 1, cnt)  # (H, W)
            for i in range(3):
                self.stacked_image[:, :, i] = np.where(
                    valid,
                    (self.stacked_image[:, :, i] * cnt + img[:, :, i]) / cnt_new,
                    self.stacked_image[:, :, i]
                )
            self.weight_map = cnt_new
        else:
            valid = np.isfinite(img)
            cnt_new = np.where(valid, cnt + 1, cnt)
            self.stacked_image = np.where(
                valid,
                (self.stacked_image * cnt + img) / cnt_new,
                self.stacked_image
            )
            self.weight_map = cnt_new

    def _stack_kappa(self, img, is_color, kappa):
        """
        Kappa-sigma streaming via algorithme de Welford.

        Chaque pixel est accepté seulement si |x - mean| <= kappa * std.
        La moyenne et la variance courante sont mises à jour incrémentalement
        (algorithme de Welford) pour les pixels acceptés uniquement.
        winsorized = comportement identique en mode streaming.
        """
        cnt = self.weight_map  # (H, W) — compteur par pixel

        if is_color:
            # Masque de rejet unifié : pixel rejeté si l'un de ses canaux
            # dépasse kappa * std (et que cnt > 1 pour avoir une variance estimée)
            if np.any(cnt > 1):
                var  = self._running_m2 / np.maximum(cnt[:, :, None] - 1, 1)
                std  = np.sqrt(np.maximum(var, 0))
                d    = img - self.stacked_image          # (H, W, 3)
                over = (cnt[:, :, None] > 1) & (std > 0) & (np.abs(d) > kappa * std)
                reject = np.any(over, axis=2)            # (H, W)
            else:
                reject = np.zeros_like(cnt, dtype=bool)

            valid  = np.all([np.isfinite(img[:, :, i]) for i in range(3)], axis=0)
            accept = valid & ~reject                     # (H, W)

            cnt_new = np.where(accept, cnt + 1, cnt)     # (H, W)

            for i in range(3):
                d1 = img[:, :, i] - self.stacked_image[:, :, i]
                new_mean = np.where(
                    accept,
                    self.stacked_image[:, :, i] + d1 / np.maximum(cnt_new, 1),
                    self.stacked_image[:, :, i]
                )
                d2 = img[:, :, i] - new_mean
                self._running_m2[:, :, i] = np.where(
                    accept,
                    self._running_m2[:, :, i] + d1 * d2,
                    self._running_m2[:, :, i]
                )
                self.stacked_image[:, :, i] = new_mean

            self.weight_map = cnt_new

        else:
            if np.any(cnt > 1):
                var    = self._running_m2 / np.maximum(cnt - 1, 1)
                std    = np.sqrt(np.maximum(var, 0))
                d      = img - self.stacked_image
                reject = (cnt > 1) & (std > 0) & (np.abs(d) > kappa * std)
            else:
                reject = np.zeros_like(cnt, dtype=bool)

            valid   = np.isfinite(img)
            accept  = valid & ~reject
            cnt_new = np.where(accept, cnt + 1, cnt)

            d1      = img - self.stacked_image
            new_mean = np.where(
                accept,
                self.stacked_image + d1 / np.maximum(cnt_new, 1),
                self.stacked_image
            )
            d2 = img - new_mean
            self._running_m2 = np.where(
                accept,
                self._running_m2 + d1 * d2,
                self._running_m2
            )
            self.stacked_image = new_mean
            self.weight_map    = cnt_new

    def get_result(self):
        """
        Retourne l'image empilée finale

        Returns:
            Image empilée (copie), ou None si vide
        """
        if self.stacked_image is None:
            return None

        return self.stacked_image.copy()

    def get_snr_improvement(self):
        """
        Calcule le gain SNR théorique

        Returns:
            Facteur de gain SNR (ex: 7.07 pour 50 images)
        """
        if self.config.num_stacked == 0:
            return 1.0

        return np.sqrt(self.config.num_stacked)

    def reset(self):
        """Réinitialise le stacker"""
        self.stacked_image = None
        self.weight_map    = None
        self._running_m2   = None
        self.config.num_stacked = 0
        self.config.noise_level = 0.0
