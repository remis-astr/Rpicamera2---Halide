#!/usr/bin/env python3
"""
MINERAL MOON - Amplification des couleurs minéralogiques lunaires

Quatre modes de traitement :

  CLASSIC      : Boost de saturation HSV itératif (méthode originale).
                 Simple, rapide, donne un résultat « naturellement exagéré ».

  DECOR_STRETCH : Décorrélation spectrale par PCA (inspiré NASA Clementine/LRO).
                 Chaque composante principale de couleur est étirée
                 indépendamment pour maximiser les écarts colorimétriques.
                 Révèle des nuances spectrales pratiquement invisibles à l'œil nu.

  FALSE_COLOR  : Faux-couleurs minéralogiques style imagerie planétaire.
                 Calcule l'indice spectral (B−R)/(B+R+G) — proxy titane/fer —
                 et le mappe sur une palette vivide sans vert :
                   Orange → Rouge → Violet → Bleu → Cyan
                   (terminator/fer → hautes terres → mers titanifères)

  COMBINED     : Décorrélation PUIS faux-couleurs sur le résultat amplifié.
                 Les différences subtiles sont d'abord maximisées, puis
                 recolorisées avec la palette scientifique.

Post-traitement commun (tous modes) :
  → Réduction bruit bilatéral (optionnel)
  → Composite LRGB : luminance originale + chrominance traitée (optionnel)
  → BGR uint8 sortie

Note WB : balance des blancs gérée en amont via Picamera2 ColourGains
          (AwbEnable=False + ColourGains=(r, b) dans RPiCamera2.py)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Presets prédéfinis (indices 0-2 : classic ; 3-5 : nouveaux modes)
# ---------------------------------------------------------------------------
MOON_PRESETS = [
    # 0 — Subtil (classique)
    {
        'name': 'Subtil',
        'processing_mode': 'classic',
        'sat_factor': 1.8, 'sat_iter': 6, 'lum_protect': 230,
        'lrgb': False, 'lrgb_w': 0.6,
        'noise': True,  'noise_s': 7,
        'decor_pct': 1.5, 'false_color_intensity': 0.8,
    },
    # 1 — Standard (classique, défaut)
    {
        'name': 'Standard',
        'processing_mode': 'classic',
        'sat_factor': 2.5, 'sat_iter': 4, 'lum_protect': 220,
        'lrgb': False, 'lrgb_w': 0.6,
        'noise': True,  'noise_s': 7,
        'decor_pct': 1.5, 'false_color_intensity': 0.8,
    },
    # 2 — Intense (classique)
    {
        'name': 'Intense',
        'processing_mode': 'classic',
        'sat_factor': 4.0, 'sat_iter': 3, 'lum_protect': 200,
        'lrgb': True,  'lrgb_w': 0.8,
        'noise': True,  'noise_s': 5,
        'decor_pct': 1.5, 'false_color_intensity': 0.8,
    },
    # 3 — Décorrélation spectrale PCA
    {
        'name': 'Décorrélation',
        'processing_mode': 'decor_stretch',
        'sat_factor': 1.0, 'sat_iter': 1, 'lum_protect': 255,
        'lrgb': True,  'lrgb_w': 0.90,
        'noise': True,  'noise_s': 5,
        'decor_pct': 1.5, 'false_color_intensity': 0.8,
    },
    # 4 — Faux-couleurs minéralogiques
    {
        'name': 'Faux-couleurs',
        'processing_mode': 'false_color',
        'sat_factor': 1.0, 'sat_iter': 1, 'lum_protect': 255,
        'lrgb': True,  'lrgb_w': 0.95,
        'noise': True,  'noise_s': 7,
        'decor_pct': 1.5, 'false_color_intensity': 0.85,
    },
    # 5 — Spectral NASA (décorrélation + faux-couleurs)
    {
        'name': 'Spectral NASA',
        'processing_mode': 'combined',
        'sat_factor': 1.0, 'sat_iter': 1, 'lum_protect': 255,
        'lrgb': True,  'lrgb_w': 0.85,
        'noise': True,  'noise_s': 5,
        'decor_pct': 1.0, 'false_color_intensity': 0.80,
    },
]

# Palette faux-couleurs : ancres x=[0, 0.25, 0.5, 0.75, 1.0]
# en canal BGR (pas de vert, transition naturelle scientifique)
_FC_X_ANCHORS = [0.0,  0.25,       0.50,        0.75,       1.0]
_FC_B_VALS    = [0,    0,          160,         220,        220]   # B
_FC_G_VALS    = [70,   0,          0,           40,         200]   # G
_FC_R_VALS    = [220,  200,        180,         80,         20]    # R
#               orange red-orange  violet/mauve blue         cyan


# ---------------------------------------------------------------------------
# Processeur principal
# ---------------------------------------------------------------------------
class MineralMoonProcessor:
    """
    Processeur Mineral Moon : amplifie les couleurs minéralogiques lunaires.

    Entrée  : frame BGR uint8 (depuis Picamera2 main stream)
    Sortie  : frame BGR uint8 traitée

    Modes :
      'classic'       — boost saturation HSV itératif
      'decor_stretch' — décorrélation spectrale PCA
      'false_color'   — faux-couleurs minéralogiques (orange→violet→cyan)
      'combined'      — décorrélation puis faux-couleurs
    """

    def __init__(self):
        # --- Communs ---
        self.processing_mode       = 'classic'
        self.noise_reduction       = True
        self.noise_strength        = 7
        self.lrgb_composite        = False
        self.lrgb_color_weight     = 0.6

        # --- Mode CLASSIC ---
        self.saturation_factor     = 2.5
        self.saturation_iterations = 4
        self.luminance_protect     = 220
        self.contrast_factor       = 1.0

        # --- Mode DECOR_STRETCH ---
        self.decor_percentile      = 1.5   # % clipping aux extrêmes (0.5–5.0)

        # --- Mode FALSE_COLOR ---
        self.false_color_intensity = 0.8   # Force colorisation (0.1–1.0)

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    def configure(self, **kwargs):
        """Met à jour les paramètres du processeur (appel incrémental)."""
        # Communs
        if 'processing_mode' in kwargs:
            m = str(kwargs['processing_mode'])
            if m in ('classic', 'decor_stretch', 'false_color', 'combined'):
                self.processing_mode = m
        if 'noise_reduction' in kwargs:
            self.noise_reduction = bool(kwargs['noise_reduction'])
        if 'noise_strength' in kwargs:
            v = int(kwargs['noise_strength'])
            self.noise_strength = max(3, min(11, v | 1))
        if 'lrgb_composite' in kwargs:
            self.lrgb_composite = bool(kwargs['lrgb_composite'])
        if 'lrgb_color_weight' in kwargs:
            self.lrgb_color_weight = max(0.1, min(1.0, float(kwargs['lrgb_color_weight'])))

        # Classic
        if 'saturation_factor' in kwargs:
            self.saturation_factor = max(1.0, min(20.0, float(kwargs['saturation_factor'])))
        if 'saturation_iterations' in kwargs:
            self.saturation_iterations = max(1, min(8, int(kwargs['saturation_iterations'])))
        if 'luminance_protect' in kwargs:
            self.luminance_protect = max(100, min(255, int(kwargs['luminance_protect'])))
        if 'contrast_factor' in kwargs:
            self.contrast_factor = max(1.0, min(4.0, float(kwargs['contrast_factor'])))

        # Décorrélation
        if 'decor_percentile' in kwargs:
            self.decor_percentile = max(0.5, min(5.0, float(kwargs['decor_percentile'])))

        # Faux-couleurs
        if 'false_color_intensity' in kwargs:
            self.false_color_intensity = max(0.1, min(1.0, float(kwargs['false_color_intensity'])))

    def apply_preset(self, preset_idx: int):
        """Applique un preset prédéfini (0–5)."""
        if not (0 <= preset_idx < len(MOON_PRESETS)):
            return
        p = MOON_PRESETS[preset_idx]
        self.processing_mode       = p.get('processing_mode', 'classic')
        self.saturation_factor     = p.get('sat_factor',  2.5)
        self.saturation_iterations = p.get('sat_iter',    4)
        self.luminance_protect     = p.get('lum_protect', 220)
        self.lrgb_composite        = p.get('lrgb',        False)
        self.lrgb_color_weight     = p.get('lrgb_w',      0.6)
        self.noise_reduction       = p.get('noise',       True)
        self.noise_strength        = p.get('noise_s',     7)
        self.decor_percentile      = p.get('decor_pct',   1.5)
        self.false_color_intensity = p.get('false_color_intensity', 0.8)

    # -----------------------------------------------------------------------
    # Pipeline principal
    # -----------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Traitement Mineral Moon sur une frame BGR uint8.
        Retourne une frame BGR uint8.
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return frame_bgr

        mode = self.processing_mode

        if mode == 'classic':
            result = self._classic_core(frame_bgr)

        elif mode == 'decor_stretch':
            result = self._decor_stretch(frame_bgr)

        elif mode == 'false_color':
            result = self._false_color_mineral(frame_bgr)

        elif mode == 'combined':
            # Étape 1 : décorrélation (amplifie les différences spectrales subtiles)
            decor  = self._decor_stretch(frame_bgr)
            # Étape 2 : faux-couleurs sur le résultat amplifié
            result = self._false_color_mineral(decor)

        else:
            result = self._classic_core(frame_bgr)

        # --- Post-traitement commun ---
        if self.noise_reduction:
            s      = max(3, self.noise_strength | 1)
            result = cv2.bilateralFilter(result, d=s,
                                         sigmaColor=40, sigmaSpace=40)
        if self.lrgb_composite:
            result = self._apply_lrgb(frame_bgr, result)

        return result

    # -----------------------------------------------------------------------
    # Mode CLASSIC
    # -----------------------------------------------------------------------

    def _classic_core(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Boost de saturation itératif en espace HSV (méthode originale)."""
        hsv          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        luminance    = hsv[:, :, 2]
        protect_mask = (luminance < self.luminance_protect).astype(np.float32)
        protect_mask = cv2.GaussianBlur(protect_mask, (21, 21), 0)

        boost = self.saturation_factor ** (1.0 / max(self.saturation_iterations, 1))
        for _ in range(self.saturation_iterations):
            s_boosted     = np.clip(hsv[:, :, 1] * boost, 0, 255)
            hsv[:, :, 1]  = hsv[:, :, 1] * (1 - protect_mask) + s_boosted * protect_mask

        if self.contrast_factor != 1.0:
            v_norm       = hsv[:, :, 2] / 255.0
            v_c          = np.clip((v_norm - 0.5) * self.contrast_factor + 0.5, 0.0, 1.0) * 255.0
            hsv[:, :, 2] = hsv[:, :, 2] * (1 - protect_mask) + v_c * protect_mask

        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # -----------------------------------------------------------------------
    # Mode DECOR_STRETCH (décorrélation spectrale par PCA)
    # -----------------------------------------------------------------------

    def _decor_stretch(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Décorrélation spectrale par Analyse en Composantes Principales.

        Principe (analogue à NASA Clementine/LROC) :
          1. Centrage des données couleur
          2. PCA → axes de variance maximale des 3 canaux
          3. Étirement percentile INDÉPENDANT de chaque composante → [0, 255]
          4. Projection inverse → espace BGR

        Résultat : les infimes différences spectrales (±0.5%) deviennent
        des couleurs vives clairement différenciées.
        """
        h, w   = frame_bgr.shape[:2]
        data   = frame_bgr.astype(np.float64).reshape(-1, 3)

        # Exclure le fond de ciel (pixels quasi-noirs)
        lum_1d = data.max(axis=1)
        valid  = lum_1d > 8.0
        if valid.sum() < 100:
            return frame_bgr

        valid_data = data[valid]
        mean       = np.mean(valid_data, axis=0)
        centered   = valid_data - mean

        # Décomposition en composantes principales (3×3 → très rapide)
        try:
            cov            = np.cov(centered.T)
            eigenvalues, V = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return frame_bgr

        # Trier par variance décroissante
        order = np.argsort(eigenvalues)[::-1]
        V     = V[:, order]

        # Projection sur les axes propres
        projected = centered @ V          # (N_valid, 3)

        # Étirement percentile indépendant de chaque composante → [0, 255]
        p         = self.decor_percentile
        stretched = np.empty_like(projected)
        for i in range(3):
            lo, hi = np.percentile(projected[:, i], [p, 100.0 - p])
            span   = max(hi - lo, 1e-8)
            stretched[:, i] = np.clip(
                (projected[:, i] - lo) / span * 255.0, 0.0, 255.0
            )

        # Projection inverse, recentrée sur gris neutre (128)
        result_valid = (stretched - 127.5) @ V.T + 128.0

        # Reconstruire l'image (fond = noir)
        result_data        = np.zeros_like(data)
        result_data[valid] = np.clip(result_valid, 0.0, 255.0)

        return result_data.reshape(h, w, 3).astype(np.uint8)

    # -----------------------------------------------------------------------
    # Mode FALSE_COLOR (faux-couleurs minéralogiques)
    # -----------------------------------------------------------------------

    def _false_color_mineral(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Faux-couleurs minéralogiques style imagerie planétaire (Clementine/LRO).

        Indice spectral principal : (B − R) / (B + G + R)
          > 0  → excès bleu  = mers riches en titane  → Bleu/Cyan
          ≈ 0  → neutre      = hautes terres (anorthosite) → Violet
          < 0  → excès rouge = oxyde de fer, terminator → Orange/Rouge

        Palette (sans vert, transition scientifique) :
          x = 0.00 : Orange    BGR=(0,   70,  220)
          x = 0.25 : Rouge     BGR=(0,   0,   200)
          x = 0.50 : Violet    BGR=(160, 0,   180)
          x = 0.75 : Bleu      BGR=(220, 40,  80)
          x = 1.00 : Cyan      BGR=(220, 200, 20)

        La luminance originale est conservée (remplacement du canal V HSV).
        """
        img   = frame_bgr.astype(np.float64)
        R     = img[:, :, 2]
        G     = img[:, :, 1]
        B     = img[:, :, 0]
        total = R + G + B + 3.0          # +3 évite division par zéro

        # --- Indice spectral primaire : excès bleu−rouge / total ---
        # Positif → mers titanifères ; négatif → terminator/fer
        idx_primary = (B - R) / total

        # --- Indice secondaire : verdure relative ---
        # Modulation secondaire légère (hautes terres plus vertes → hue décalée)
        idx_secondary = (2.0 * G - R - B) / total

        # Masque pixels lunaires (exclure fond de ciel)
        lum     = img.max(axis=2)
        is_moon = lum > 8.0

        # Normalisation percentile [0, 1] sur les pixels lunaires uniquement
        def pct_norm(arr, p=1.5):
            vals = arr[is_moon]
            if len(vals) < 10:
                return np.full_like(arr, 0.5)
            lo, hi = np.percentile(vals, [p, 100.0 - p])
            span   = max(hi - lo, 1e-10)
            return np.clip((arr - lo) / span, 0.0, 1.0)

        x  = pct_norm(idx_primary)    # 0 = rouge/fer,  1 = bleu/titane
        x2 = pct_norm(idx_secondary)  # 0 = pauvre vert, 1 = riche vert

        # --- Interpolation piecewise linéaire sur la palette ---
        x_flat = x.ravel()
        B_interp = np.interp(x_flat, _FC_X_ANCHORS, _FC_B_VALS).reshape(x.shape)
        G_interp = np.interp(x_flat, _FC_X_ANCHORS, _FC_G_VALS).reshape(x.shape)
        R_interp = np.interp(x_flat, _FC_X_ANCHORS, _FC_R_VALS).reshape(x.shape)

        false_bgr_raw = np.stack([B_interp, G_interp, R_interp], axis=2)  # float64

        # --- Modulation secondaire : léger décalage de teinte via canal G ---
        # Pixels plus « verts » → légère nuance supplémentaire
        green_shift = (x2 - 0.5) * 20.0    # ±10 sur canal G
        false_bgr_raw[:, :, 1] = np.clip(
            false_bgr_raw[:, :, 1] + green_shift, 0, 255
        )

        # --- Intensité de colorisation selon la force du signal minéralogique ---
        # Pixels au centre de la distribution → signal faible, moins saturé
        color_strength = np.abs(x - 0.5) * 2.0          # 0 au centre, 1 aux extrêmes
        intensity      = np.clip(
            color_strength * self.false_color_intensity + 0.15, 0.0, 1.0
        )
        false_bgr_raw *= intensity[:, :, np.newaxis]

        false_bgr = np.clip(false_bgr_raw, 0, 255).astype(np.uint8)

        # --- Conserver la luminance originale (remplacement canal V en HSV) ---
        false_hsv      = cv2.cvtColor(false_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        orig_hsv       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        false_hsv[:, :, 2] = orig_hsv[:, :, 2]    # Luminance originale

        result = cv2.cvtColor(
            np.clip(false_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
        )

        # Fond de ciel → noir
        result[~is_moon] = 0

        return result

    # -----------------------------------------------------------------------
    # Composite LRGB
    # -----------------------------------------------------------------------

    def _apply_lrgb(self, original: np.ndarray,
                    colorized: np.ndarray) -> np.ndarray:
        """
        Mélange la luminance de l'original (netteté, détail fins)
        avec la chrominance du résultat colorisé.
        """
        orig_ycc  = cv2.cvtColor(original,  cv2.COLOR_BGR2YCrCb).astype(np.float32)
        color_ycc = cv2.cvtColor(colorized, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        w              = self.lrgb_color_weight
        comp           = orig_ycc.copy()
        comp[:, :, 1]  = orig_ycc[:, :, 1] * (1 - w) + color_ycc[:, :, 1] * w
        comp[:, :, 2]  = orig_ycc[:, :, 2] * (1 - w) + color_ycc[:, :, 2] * w

        return cv2.cvtColor(
            np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR
        )

    # -----------------------------------------------------------------------
    # Utilitaires
    # -----------------------------------------------------------------------

    def boost_per_pass_str(self) -> str:
        """Retourne une info courte sur le mode actuel (affichée dans le panneau)."""
        mode = self.processing_mode
        if mode == 'classic':
            if self.saturation_iterations > 0:
                bpp = self.saturation_factor ** (1.0 / self.saturation_iterations)
                return f"Classic x{bpp:.2f}/p"
            return "Classic"
        elif mode == 'decor_stretch':
            return f"Décorr. PCA {self.decor_percentile:.1f}%"
        elif mode == 'false_color':
            return f"Faux-couleurs {self.false_color_intensity:.0%}"
        elif mode == 'combined':
            return f"NASA {self.false_color_intensity:.0%}"
        return mode
