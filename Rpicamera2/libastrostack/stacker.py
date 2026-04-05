#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empilement d'images pour libastrostack
"""

import ctypes
import os
import numpy as np

# =============================================================================
# Chargement optionnel du pipeline Halide (ls_stack_mean_frame + ls_stack_kappa_frame)
# =============================================================================
_halide_ls_available = False
_hlib_stacker = None

def _load_halide():
    global _halide_ls_available, _hlib_stacker
    _so = os.path.join(os.path.dirname(__file__), "halide", "jsk_halide.so")
    if not os.path.isfile(_so):
        return
    try:
        lib = ctypes.CDLL(_so)
        _P = ctypes.POINTER(ctypes.c_float)

        # ls_stack_mean_frame(stacked_in, cnt_in, img, stacked_out, cnt_out, w, h)
        lib.ls_stack_mean_frame.restype  = ctypes.c_int
        lib.ls_stack_mean_frame.argtypes = [
            _P, _P, _P,          # stacked_in (3,H,W), cnt_in (H,W), img (3,H,W)
            _P, _P,              # stacked_out (3,H,W), cnt_out (H,W)
            ctypes.c_int, ctypes.c_int,  # w, h
        ]

        # ls_stack_kappa_frame(stacked_in, cnt_in, m2_in, img, kappa,
        #                      stacked_out, cnt_out, m2_out, w, h)
        lib.ls_stack_kappa_frame.restype  = ctypes.c_int
        lib.ls_stack_kappa_frame.argtypes = [
            _P, _P, _P, _P,      # stacked_in, cnt_in, m2_in, img
            ctypes.c_float,      # kappa
            _P, _P, _P,          # stacked_out, cnt_out, m2_out
            ctypes.c_int, ctypes.c_int,  # w, h
        ]

        _hlib_stacker = lib
        _halide_ls_available = True
    except Exception:
        pass

_load_halide()


# =============================================================================
# Wrappers Python pour les pipelines Halide live stack
# =============================================================================

def _halide_ls_stack_mean_frame(stacked_p, cnt, img_p):
    """Mise à jour incrémentale de moyenne RGB — 1 passe NEON.

    Args:
        stacked_p : float32 (3,H,W) C-contiguous — moyenne courante (planaire)
        cnt       : float32 (H,W)   C-contiguous — compteur par pixel
        img_p     : float32 (3,H,W) C-contiguous — nouvelle frame (planaire)

    Returns:
        (stacked_out_p, cnt_out) — mêmes shapes, float32
    """
    _, h, w = stacked_p.shape
    _P = ctypes.POINTER(ctypes.c_float)
    stacked_out_p = np.empty_like(stacked_p)
    cnt_out       = np.empty_like(cnt)
    ret = _hlib_stacker.ls_stack_mean_frame(
        stacked_p.ctypes.data_as(_P),
        cnt.ctypes.data_as(_P),
        img_p.ctypes.data_as(_P),
        stacked_out_p.ctypes.data_as(_P),
        cnt_out.ctypes.data_as(_P),
        w, h,
    )
    if ret != 0:
        raise RuntimeError(f"ls_stack_mean_frame Halide erreur: {ret}")
    return stacked_out_p, cnt_out


def _halide_ls_stack_kappa_frame(stacked_p, cnt, m2_p, img_p, kappa):
    """Welford incrémental + rejet kappa-sigma RGB — 1 passe NEON.

    Args:
        stacked_p : float32 (3,H,W) — moyenne courante (planaire)
        cnt       : float32 (H,W)   — compteur par pixel
        m2_p      : float32 (3,H,W) — variance M2 Welford (planaire)
        img_p     : float32 (3,H,W) — nouvelle frame (planaire)
        kappa     : float — seuil sigma (typique 2.5)

    Returns:
        (stacked_out_p, cnt_out, m2_out_p) — float32, mêmes shapes
    """
    _, h, w = stacked_p.shape
    _P = ctypes.POINTER(ctypes.c_float)
    stacked_out_p = np.empty_like(stacked_p)
    cnt_out       = np.empty_like(cnt)
    m2_out_p      = np.empty_like(m2_p)
    ret = _hlib_stacker.ls_stack_kappa_frame(
        stacked_p.ctypes.data_as(_P),
        cnt.ctypes.data_as(_P),
        m2_p.ctypes.data_as(_P),
        img_p.ctypes.data_as(_P),
        ctypes.c_float(kappa),
        stacked_out_p.ctypes.data_as(_P),
        cnt_out.ctypes.data_as(_P),
        m2_out_p.ctypes.data_as(_P),
        w, h,
    )
    if ret != 0:
        raise RuntimeError(f"ls_stack_kappa_frame Halide erreur: {ret}")
    return stacked_out_p, cnt_out, m2_out_p


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
            # Les pixels NaN (marges canvas warpAffine) sont remplacés par 0
            # et leur poids est 0 — ainsi les frames suivantes les remplissent
            # correctement : (0 * 0 + val) / 1 = val (pas de propagation NaN).
            init_valid = np.isfinite(image)
            self.stacked_image = np.where(init_valid, image, 0.0).astype(np.float32)

            if is_color:
                # Poids 0 pour les pixels NaN, 1 pour les pixels valides
                self.weight_map = np.isfinite(image[:, :, 0]).astype(np.float32)
            else:
                self.weight_map = np.isfinite(image).astype(np.float32)

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

        if is_color and _halide_ls_available:
            # ── Chemin Halide (2 passes NEON, ×4-5 vs NumPy) ─────────────────
            # Transpose (H,W,3)→(3,H,W) planaire pour Halide (stride-1 par canal)
            try:
                stacked_p = np.ascontiguousarray(self.stacked_image.transpose(2, 0, 1))
                img_p     = np.ascontiguousarray(img.transpose(2, 0, 1))
                cnt_c     = np.ascontiguousarray(cnt)
                new_stacked_p, new_cnt = _halide_ls_stack_mean_frame(
                    stacked_p, cnt_c, img_p)
                # Halide n'écarte pas les pixels NaN (bordures warpAffine canvas) :
                # il incrémente cnt même si img=NaN → weight_map corrompu.
                # Correction : restaurer cnt et stacked pour les pixels invalides.
                _valid = np.all(np.isfinite(img), axis=2)  # (H, W) bool
                if not _valid.all():
                    new_cnt[~_valid] = cnt_c[~_valid]
                    for _i in range(3):
                        new_stacked_p[_i][~_valid] = stacked_p[_i][~_valid]
                # Retranspose (3,H,W)→(H,W,3) et copie in-place
                np.copyto(self.stacked_image, new_stacked_p.transpose(1, 2, 0))
                np.copyto(self.weight_map, new_cnt)
                return
            except Exception:
                pass  # fallback NumPy ci-dessous

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

        if is_color and _halide_ls_available and self._running_m2 is not None:
            # ── Chemin Halide (1 passe NEON fusionnée, ×8-12 vs NumPy) ───────
            try:
                stacked_p = np.ascontiguousarray(self.stacked_image.transpose(2, 0, 1))
                img_p     = np.ascontiguousarray(img.transpose(2, 0, 1))
                m2_p      = np.ascontiguousarray(self._running_m2.transpose(2, 0, 1))
                cnt_c     = np.ascontiguousarray(cnt)
                new_stacked_p, new_cnt, new_m2_p = _halide_ls_stack_kappa_frame(
                    stacked_p, cnt_c, m2_p, img_p, float(kappa))
                # Correction NaN canvas : Halide incrémente cnt même pour pixels
                # invalides (bordures warpAffine) → restaurer pour pixels NaN.
                _valid = np.all(np.isfinite(img), axis=2)  # (H, W) bool
                if not _valid.all():
                    new_cnt[~_valid] = cnt_c[~_valid]
                    for _i in range(3):
                        new_stacked_p[_i][~_valid] = stacked_p[_i][~_valid]
                        new_m2_p[_i][~_valid]      = m2_p[_i][~_valid]
                np.copyto(self.stacked_image, new_stacked_p.transpose(1, 2, 0))
                np.copyto(self.weight_map,    new_cnt)
                np.copyto(self._running_m2,   new_m2_p.transpose(1, 2, 0))
                return
            except Exception:
                pass  # fallback NumPy ci-dessous

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
