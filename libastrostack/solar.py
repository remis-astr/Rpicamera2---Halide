#!/usr/bin/env python3
"""
SOLAR WHITE LIGHT - Traitement temps réel imagerie solaire

Pipeline : RGB8 ISP (Picamera2 main stream)
  → Niveaux de gris
  → Lucky Imaging (optionnel, buffer circulaire + score Laplacian)
  → Correction assombrissement centre-bord (Limb Darkening)
  → Masquage hors disque
  → CLAHE (contraste adaptatif local)
  → Unsharp Mask adaptatif
  → Déconvolution Lucy-Richardson (optionnel, coûteux)
  → False color (optionnel)
  → BGR pour affichage et sauvegarde

Format entrée : RGB8 ISP via Picamera2 (pas de RAW)
Zoom géré via les modes capteur Picamera2 (comme le mode MOON)
"""

import cv2
import numpy as np
from collections import deque
from libastrostack.aligner_planetary import PlanetaryAligner, PlanetaryMode, PlanetaryConfig


# ============================================================================
# Presets
# ============================================================================

SOLAR_PRESETS = [
    # 0 — Taches (détail taches, plein disque)
    {
        'name':         'Taches',
        'clahe_en':     True,  'clahe_str': 1.0, 'clahe_tile': 8,
        'usm_en':       True,  'usm_sigma': 2.0, 'usm_amount': 2.5,
        'usm_adaptive': True,  'usm_thresh': 120,
        'lr_en':        False, 'lr_iter': 30,    'lr_sigma': 1.0, 'lr_ringing': True,
        'limb_correct': True,  'limb_kernel': 101,
        'mask_disk':    False, 'false_color': 0,
    },
    # 1 — Granulation (texture photosphère)
    {
        'name':         'Granulation',
        'clahe_en':     True,  'clahe_str': 2.0, 'clahe_tile': 8,
        'usm_en':       True,  'usm_sigma': 1.0, 'usm_amount': 3.0,
        'usm_adaptive': True,  'usm_thresh': 120,
        'lr_en':        True,  'lr_iter': 30,    'lr_sigma': 1.0, 'lr_ringing': True,
        'limb_correct': True,  'limb_kernel': 101,
        'mask_disk':    False, 'false_color': 0,
    },
]


# ============================================================================
# Helpers internes
# ============================================================================

def _score_laplacian(gray: np.ndarray) -> float:
    """Score de netteté par variance du Laplacien (rapide)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _fit_circle_kasa(pts_xy: np.ndarray):
    """
    Ajustement de cercle aux points (x, y) par la méthode algébrique de Kasa.
    Fonctionne même si seul un arc partiel est fourni.
    Le centre résultant peut être hors des limites de l'image.

    Args:
        pts_xy : tableau (N, 2) float32/64 de points (x, y) sur le contour

    Returns:
        (cx, cy, r) flottants, ou None si le système est dégénéré
    """
    if len(pts_xy) < 5:
        return None

    x = pts_xy[:, 0].astype(np.float64)
    y = pts_xy[:, 1].astype(np.float64)

    # Centrer pour la stabilité numérique
    xm, ym = x.mean(), y.mean()
    xc = x - xm
    yc = y - ym

    # Système : xc² + yc² = A·xc + B·yc + C
    A_mat = np.column_stack([xc, yc, np.ones(len(xc))])
    b_vec = xc ** 2 + yc ** 2
    try:
        result, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx = result[0] / 2.0 + xm
    cy = result[1] / 2.0 + ym
    r_sq = result[2] + (result[0] / 2.0) ** 2 + (result[1] / 2.0) ** 2
    if r_sq <= 0:
        return None
    return cx, cy, float(np.sqrt(r_sq))


def _lr_deconv(gray: np.ndarray, sigma: float, iterations: int,
               prevent_ringing: bool) -> np.ndarray:
    """
    Déconvolution Lucy-Richardson avec PSF gaussienne.

    Args:
        gray            : Image niveaux de gris uint8
        sigma           : Sigma PSF (0.5–3.0)
        iterations      : Nombre d'itérations (5–60)
        prevent_ringing : Floutage léger des hautes lumières avant déconvolution

    Returns:
        Image uint8 déconvoluée
    """
    k = max(3, int(sigma * 6) | 1)
    psf_1d = cv2.getGaussianKernel(k, sigma)
    psf_2d = (psf_1d @ psf_1d.T).astype(np.float32)
    psf_2d /= psf_2d.sum()

    img = gray.astype(np.float32) / 255.0

    if prevent_ringing:
        bright = img > 0.92
        if bright.any():
            bk = max(3, int(sigma * 4) | 1)
            blurred = cv2.GaussianBlur(img, (bk, bk), sigma * 2)
            img = np.where(bright, blurred, img)

    estimate = img.copy()
    for _ in range(iterations):
        blurred_est = cv2.filter2D(estimate, -1, psf_2d,
                                   borderType=cv2.BORDER_REFLECT)
        blurred_est = np.clip(blurred_est, 1e-7, None)
        ratio_conv  = cv2.filter2D(img / blurred_est, -1, psf_2d,
                                   borderType=cv2.BORDER_REFLECT)
        estimate    = np.clip(estimate * ratio_conv, 0.0, 1.0)

    return np.clip(estimate * 255.0, 0, 255).astype(np.uint8)


# ============================================================================
# SolarProcessor
# ============================================================================

class SolarProcessor:
    """
    Processeur Solar White Light.

    Entrée  : frame RGB uint8 (Picamera2 main stream)
    Sortie  : frame BGR uint8 traitée
    """

    FALSE_COLOR_MAPS = [None,
                        cv2.COLORMAP_HOT,
                        cv2.COLORMAP_VIRIDIS,
                        cv2.COLORMAP_INFERNO]

    def __init__(self):
        # --- Lucky Imaging ---
        self.lucky_frames   = 0        # 0 = OFF ; >0 = taille du buffer
        self.best_pct       = 15.0     # % des meilleures frames à garder (5–50)
        self.align_mode     = 0        # 0=off, 1=disque, 2=surface
        self._lucky_buf     = deque()
        self._lucky_result  = None     # Dernière frame stackée
        self._aligner           = None
        self._aligner_mode_prev = None

        # --- Correction limb darkening ---
        self.limb_correct   = True
        self.limb_kernel    = 101      # impair 51–201
        self._disk_center   = None     # (cx, cy, r) ou None
        self._limb_flat     = None     # Flat synthétique mis en cache

        # --- Masque disque ---
        self.mask_disk      = False

        # --- CLAHE ---
        self.clahe_en       = True
        self.clahe_str      = 1.5      # clipLimit (0.0–4.0)
        self.clahe_tile     = 8        # tileGridSize (4–32)

        # --- Unsharp Mask ---
        self.usm_en         = True
        self.usm_sigma      = 1.2      # 0.5–5.0
        self.usm_amount     = 2.0      # 0.5–5.0
        self.usm_adaptive   = True
        self.usm_thresh     = 120      # seuil adaptatif 0–255

        # --- Lucy-Richardson ---
        self.lr_en          = False
        self.lr_iter        = 30       # 5–60
        self.lr_sigma       = 1.0      # 0.5–3.0
        self.lr_ringing     = True

        # --- False color ---
        self.false_color    = 0        # 0=None, 1=Hot, 2=Viridis, 3=Inferno

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def configure(self, **kwargs):
        """Met à jour un ou plusieurs paramètres."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if 'limb_kernel' in kwargs:
            self._limb_flat = None        # Invalider le flat
        if 'lucky_frames' in kwargs:
            self._lucky_buf.clear()       # Réinitialiser le buffer
            self._lucky_result = None
        if 'align_mode' in kwargs:
            self._aligner = None          # Recréer l'aligneur au prochain batch

    def apply_preset(self, preset_idx: int):
        """Applique un preset (0=Taches, 1=Granulation)."""
        if 0 <= preset_idx < len(SOLAR_PRESETS):
            p = SOLAR_PRESETS[preset_idx]
            self.clahe_en     = p['clahe_en']
            self.clahe_str    = p['clahe_str']
            self.clahe_tile   = p['clahe_tile']
            self.usm_en       = p['usm_en']
            self.usm_sigma    = p['usm_sigma']
            self.usm_amount   = p['usm_amount']
            self.usm_adaptive = p['usm_adaptive']
            self.usm_thresh   = p['usm_thresh']
            self.lr_en        = p['lr_en']
            self.lr_iter      = p['lr_iter']
            self.lr_sigma     = p['lr_sigma']
            self.lr_ringing   = p['lr_ringing']
            self.limb_correct = p['limb_correct']
            self.limb_kernel  = p['limb_kernel']
            self.mask_disk    = p['mask_disk']
            self.false_color  = p['false_color']
            self._limb_flat   = None
            self._lucky_buf.clear()
            self._lucky_result = None

    def force_detect_disk(self, gray: np.ndarray):
        """Force la redétection du disque solaire sur la frame fournie."""
        self._disk_center = None
        self._limb_flat   = None
        self._detect_disk(gray)

    # -------------------------------------------------------------------------
    # Détection du disque (Hough Circle)
    # -------------------------------------------------------------------------

    def _detect_disk(self, gray: np.ndarray):
        """
        Détecte le disque solaire — stratégie en cascade :
          1. Hough Circle (plein disque ou disque quasi-complet dans le champ)
          2. Arc fitting par seuillage + contour (disque partiel, 1/4 et plus)
        """
        h, w = gray.shape
        scale = 2
        small = cv2.resize(gray, (w // scale, h // scale))
        blurred = cv2.GaussianBlur(small, (9, 9), 2)

        # --- Passe Hough (disque entier ou quasi-entier visible) ---
        diag_half = int(np.hypot(h // scale, w // scale) // 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=min(h, w) // (4 * scale),
            param1=80,
            param2=25,
            minRadius=min(h, w) // (8 * scale),
            maxRadius=int(diag_half * 0.95),
        )
        if circles is not None:
            cx, cy, r = circles[0, 0]
            self._disk_center = (int(cx * scale), int(cy * scale),
                                 int(r  * scale))
            self._limb_flat = None
            return

        # --- Fallback : arc fitting (disque partiel) ---
        self._detect_disk_arc(gray)

    def _detect_disk_arc(self, gray: np.ndarray):
        """
        Détecte le disque solaire même si seul un arc partiel est visible
        (typiquement 1/4 à 3/4 du disque, focale longue).

        Méthode :
          1. Seuillage Otsu : sépare le disque brillant du ciel noir
          2. Fermeture morphologique : comble les taches solaires
          3. Contour externe du disque
          4. Ajustement Kasa du meilleur cercle sur l'arc détecté
             → le centre peut être hors de l'image (cas disque > capteur)

        Résultat stocké dans self._disk_center (cx, cy, r) en pixels pleine résolution,
        ou None si la détection échoue (disque trop petit ou image uniforme).
        """
        h, w = gray.shape
        scale = 2
        small = cv2.resize(gray, (w // scale, h // scale))
        hs, ws = small.shape

        # Flou plus prononcé pour lisser la granulation et les taches
        blurred = cv2.GaussianBlur(small, (15, 15), 4)

        # Seuillage Otsu : hypothèse que le fond (ciel) est bien plus sombre
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Si toute l'image est allumée (disque > capteur), aucun limbe visible
        bright_ratio = np.count_nonzero(binary) / binary.size
        if bright_ratio > 0.97:
            self._disk_center = None
            self._limb_flat = None
            return

        # Fermeture morphologique : bouche les taches solaires et la granulation
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

        # Contour externe (limbe solaire)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            self._disk_center = None
            self._limb_flat = None
            return

        # Garder le plus grand contour (le disque)
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 20:
            self._disk_center = None
            self._limb_flat = None
            return

        pts = largest[:, 0, :].astype(np.float32)  # (N, 2) → (x, y)

        # Sous-échantillonner pour la vitesse (500 pts suffisent)
        if len(pts) > 500:
            idx = np.linspace(0, len(pts) - 1, 500, dtype=int)
            pts = pts[idx]

        result = _fit_circle_kasa(pts)
        if result is None:
            self._disk_center = None
            self._limb_flat = None
            return

        cx, cy, r = result

        # Validation : rayon minimal cohérent (> 1/8 du petit côté)
        if r < min(hs, ws) / 8:
            self._disk_center = None
            self._limb_flat = None
            return

        self._disk_center = (int(cx * scale), int(cy * scale), int(r * scale))
        self._limb_flat = None

    # -------------------------------------------------------------------------
    # Flat synthétique (Limb Darkening Correction)
    # -------------------------------------------------------------------------

    def _build_flat(self, gray: np.ndarray) -> np.ndarray:
        """Construit le flat synthétique par grand filtre médian."""
        k = int(self.limb_kernel)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        return cv2.medianBlur(gray, k)

    # -------------------------------------------------------------------------
    # Lucky Imaging
    # -------------------------------------------------------------------------

    def _get_aligner(self):
        """
        Retourne l'aligneur PlanetaryAligner configuré selon align_mode,
        ou None si l'alignement est désactivé (align_mode == 0).
        L'instance est mise en cache et recréée uniquement si le mode change.
        """
        if self.align_mode == 0:
            return None
        if self._aligner is None or self._aligner_mode_prev != self.align_mode:
            cfg = PlanetaryConfig()
            cfg.mode      = PlanetaryMode.DISK if self.align_mode == 1 else PlanetaryMode.SURFACE
            cfg.max_shift = 100   # décalage max accepté (px)
            cfg.disk_min_radius = 50
            cfg.disk_max_radius = 2000
            cfg.surface_window_size = 256
            self._aligner           = PlanetaryAligner(cfg)
            self._aligner_mode_prev = self.align_mode
        return self._aligner

    def _process_lucky(self, gray: np.ndarray) -> np.ndarray:
        """
        Accumule les frames dans un buffer circulaire, sélectionne les
        meilleures par score Laplacian et retourne leur stack (mean).

        Retourne la dernière frame stackée si le buffer n'est pas encore plein.
        Retourne la frame courante directement si lucky_frames == 0.
        """
        n = int(self.lucky_frames)
        if n <= 0:
            return gray

        score = _score_laplacian(gray)
        self._lucky_buf.append((score, gray.copy()))

        # Limiter la taille du buffer
        while len(self._lucky_buf) > n:
            self._lucky_buf.popleft()

        if len(self._lucky_buf) < n:
            # Buffer pas encore plein
            return self._lucky_result if self._lucky_result is not None else gray

        # Sélectionner les meilleures frames
        keep = max(1, int(n * self.best_pct / 100.0))
        sorted_buf = sorted(self._lucky_buf, key=lambda x: x[0], reverse=True)
        best_frames = [f for _, f in sorted_buf[:keep]]

        # Alignement (optionnel) : la 1ère frame (meilleure) devient référence
        aligner = self._get_aligner()
        if aligner is not None and len(best_frames) > 1:
            aligner.reset()
            aligned_frames = []
            for frame in best_frames:
                aligned, _params, success = aligner.align(frame.astype(np.float32))
                aligned_frames.append(
                    np.clip(aligned, 0, 255).astype(np.uint8) if success else frame
                )
            best_frames = aligned_frames

        # Stack par moyenne
        stack = np.mean(best_frames, axis=0).astype(np.uint8)
        self._lucky_result = stack
        self._lucky_buf.clear()
        return stack

    # -------------------------------------------------------------------------
    # Unsharp Mask adaptatif
    # -------------------------------------------------------------------------

    def _apply_usm(self, gray: np.ndarray) -> np.ndarray:
        """Unsharp Mask, optionnellement adaptatif selon la luminosité locale."""
        sigma  = float(self.usm_sigma)
        amount = float(self.usm_amount)
        k = max(3, int(sigma * 6) | 1)
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (k, k), sigma)

        if self.usm_adaptive:
            lum = gray.astype(np.float32)
            # Zones sombres → amount réduit (0.8), zones lumineuses → amount plein
            amt_map = np.where(lum < self.usm_thresh, 0.8, amount).astype(np.float32)
            sharpened = lum + (lum - blurred) * (amt_map - 1.0)
        else:
            f = gray.astype(np.float32)
            sharpened = f + (f - blurred) * (amount - 1.0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Pipeline principal
    # -------------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Traitement Solar White Light complet sur une frame RGB uint8.

        Retourne une frame BGR uint8 prête à l'affichage / sauvegarde.
        """
        if frame_rgb is None or frame_rgb.ndim != 3:
            return frame_rgb

        # [1] RGB → niveaux de gris
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # [2] Lucky Imaging (optionnel)
        gray = self._process_lucky(gray)

        # Détection du disque au premier passage ou si invalidée
        if self._disk_center is None:
            self._detect_disk(gray)

        # [3] Limb Darkening Correction
        if self.limb_correct:
            if self._limb_flat is None:
                self._limb_flat = self._build_flat(gray)
            flat  = self._limb_flat.astype(np.float32)
            flat  = np.where(flat < 1.0, 1.0, flat)
            gray  = np.clip(gray.astype(np.float32) * (flat.mean() / flat),
                            0, 255).astype(np.uint8)

        # [4] Masquage extérieur disque
        if self.mask_disk and self._disk_center is not None:
            cx, cy, r = self._disk_center
            h, w = gray.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            gray = cv2.bitwise_and(gray, gray, mask=mask)

        # [5] CLAHE
        if self.clahe_en:
            tile = max(2, int(self.clahe_tile))
            clahe = cv2.createCLAHE(clipLimit=float(self.clahe_str),
                                    tileGridSize=(tile, tile))
            gray = clahe.apply(gray)

        # [6] Unsharp Mask adaptatif
        if self.usm_en:
            gray = self._apply_usm(gray)

        # [7] Déconvolution Lucy-Richardson (coûteux — frame finale uniquement)
        if self.lr_en:
            gray = _lr_deconv(gray, self.lr_sigma, self.lr_iter, self.lr_ringing)

        # [8] False color ou reconversion BGR
        cmap = self.FALSE_COLOR_MAPS[self.false_color] \
               if 0 <= self.false_color < len(self.FALSE_COLOR_MAPS) else None
        if cmap is not None:
            return cv2.applyColorMap(gray, cmap)
        else:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
