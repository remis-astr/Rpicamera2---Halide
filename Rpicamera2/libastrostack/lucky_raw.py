"""
Lucky Stack en domaine Bayer RAW12/16 pour imagerie planétaire.

Pipeline vs Lucky RGB8 :
  RGB8  : débayer × N frames → score RGB → align → stack → post-filtres
  RAW   : score G1 Bayer → align G1×2 → accumule Bayer float32 → débayer ×1 → post-filtres

Avantages :
  - Pas d'artefact ISP (démosaïcisation répétée, compression dynamique, gamma)
  - Score sur données linéaires brutes → discrimination plus fidèle à la turbulence
  - 12 bits préservés jusqu'au stack final

Format d'entrée : uint16 (H, W) Bayer RGGB, espace CSI-2 ×16 (picamera2 unpacked=True)
Format de sortie : float32 (H, W, 3) [0-65535], compatible apply_isp_to_preview()

Pattern Bayer RGGB (IMX585 / SRGGB12) :
  R  = bayer[0::2, 0::2]
  G1 = bayer[0::2, 1::2]   ← canal de référence pour score et alignement
  G2 = bayer[1::2, 0::2]
  B  = bayer[1::2, 1::2]
"""

import cv2
import numpy as np
from collections import deque
import threading
import logging
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction canaux Bayer RGGB
# ---------------------------------------------------------------------------

def _extract_g1(bayer: np.ndarray) -> np.ndarray:
    """Canal G1 (row pair, col impair) pour RGGB — shape (H/2, W/2)."""
    return bayer[0::2, 1::2]


def _apply_bl_per_channel(bayer: np.ndarray,
                          bl_r: float, bl_g1: float,
                          bl_g2: float, bl_b: float) -> np.ndarray:
    """Soustraction black level per-canal en espace CSI-2 ×16.

    Args:
        bayer         : uint16 (H, W) Bayer RGGB, espace CSI-2
        bl_r/g1/g2/b  : BL en ADU 12-bit natif (espace CSI-2 = ADU × 16)

    Returns:
        float32 (H, W) avec BL soustrait, clippé à 0
    """
    f = bayer.astype(np.float32)
    scale = 16.0  # ADU 12-bit → espace CSI-2 ×16
    f[0::2, 0::2] = np.maximum(f[0::2, 0::2] - bl_r  * scale, 0.0)  # R
    f[0::2, 1::2] = np.maximum(f[0::2, 1::2] - bl_g1 * scale, 0.0)  # G1
    f[1::2, 0::2] = np.maximum(f[1::2, 0::2] - bl_g2 * scale, 0.0)  # G2
    f[1::2, 1::2] = np.maximum(f[1::2, 1::2] - bl_b  * scale, 0.0)  # B
    return f


def _estimate_bl_auto(raw: np.ndarray) -> Tuple[float, float, float, float]:
    """Estime le BL de chaque sous-canal par percentile 5% (robuste astro).

    Returns:
        (bl_r, bl_g1, bl_g2, bl_b) en ADU 12-bit natif
    """
    scale = 16.0
    max_bl_adu = 256.0 * 1.3   # Plafond ≈ 1.3× BL nominal IMX585
    results = []
    for sy, sx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        sub = raw[sy::2, sx::2].astype(np.float32)
        p5 = float(np.percentile(sub, 5))
        bl_adu = max(0.0, (p5 / scale) - 1.0)   # -1 ADU de marge
        bl_adu = min(bl_adu, max_bl_adu)
        results.append(bl_adu)
    return tuple(results)


# ---------------------------------------------------------------------------
# Score de netteté sur G1
# ---------------------------------------------------------------------------

def _score_laplacian(g1_roi: np.ndarray) -> float:
    lap = cv2.Laplacian(g1_roi, cv2.CV_32F, ksize=3)
    return float(lap.var())


def _score_tenengrad(g1_roi: np.ndarray) -> float:
    gx = cv2.Sobel(g1_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g1_roi, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx ** 2 + gy ** 2).mean())


def _score_gradient(g1_roi: np.ndarray) -> float:
    gx = cv2.Sobel(g1_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g1_roi, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def _score_sobel(g1_roi: np.ndarray) -> float:
    # Même que gradient, implémentation OpenCV directe
    return _score_gradient(g1_roi)


_SCORE_FNS = {
    'laplacian':     _score_laplacian,
    'gradient':      _score_gradient,
    'sobel':         _score_sobel,
    'tenengrad':     _score_tenengrad,
    'local_variance': lambda r: float(r.var()),
    'psd':           _score_laplacian,   # fallback laplacien
}


def score_bayer_frame(bayer: np.ndarray, method: str = 'laplacian',
                      roi_frac: float = 0.5) -> float:
    """Score de netteté sur canal G1 du pattern Bayer.

    Args:
        bayer    : uint16 ou float32 (H, W) Bayer RGGB en espace CSI-2
        method   : 'laplacian', 'tenengrad', 'gradient', 'sobel', 'local_variance'
        roi_frac : Fraction centrale pour le calcul (0.2 – 1.0)

    Returns:
        Score float (plus élevé = plus net)
    """
    g1 = _extract_g1(bayer).astype(np.float32)
    h, w = g1.shape
    f = max(0.1, min(1.0, roi_frac))
    y0 = int(h * (1.0 - f) * 0.5)
    x0 = int(w * (1.0 - f) * 0.5)
    roi = g1[y0: h - y0, x0: w - x0]
    fn = _SCORE_FNS.get(method, _score_laplacian)
    return fn(roi)


# ---------------------------------------------------------------------------
# Alignement Bayer (décalage toujours multiple de 2 px)
# ---------------------------------------------------------------------------

def _compute_bayer_shift(ref_g1: np.ndarray, frame_g1: np.ndarray,
                         max_shift_half: int = 0) -> Tuple[int, int]:
    """Décalage par corrélation de phase sur G1 (demi-résolution).

    Principe :
      - Phase correlation sur G1 → décalage en pixels G1 (dx_half, dy_half)
      - Arrondi entier en espace G1, puis × 2 → toujours pair en espace Bayer
      - Garantit la préservation du pattern RGGB après translation

    Args:
        ref_g1        : float32 (H/2, W/2) — référence
        frame_g1      : float32 (H/2, W/2) — image à aligner
        max_shift_half: décalage max en pixels G1 (0 = pas de limite).
                        Correspond à max_shift_px // 2.

    Returns:
        (dy_full, dx_full) en pixels Bayer complets (toujours pair).
        (0, 0) si le décalage dépasse max_shift_half.
    """
    ref_max = ref_g1.max()
    frm_max = frame_g1.max()
    if ref_max < 1e-6 or frm_max < 1e-6:
        return 0, 0

    ref_n = ref_g1 / ref_max
    frm_n = frame_g1 / frm_max

    try:
        shift, _resp = cv2.phaseCorrelate(ref_n, frm_n)
    except Exception:
        return 0, 0

    dx_half, dy_half = shift  # OpenCV retourne (x, y)

    if max_shift_half > 0:
        if abs(dx_half) > max_shift_half or abs(dy_half) > max_shift_half:
            return 0, 0  # Rejet : trop grande dérive

    dy_full = int(round(dy_half)) * 2
    dx_full = int(round(dx_half)) * 2
    return dy_full, dx_full


def _shift_bayer(bayer: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Déplace une image Bayer de (dy, dx) pixels entiers pairs.

    Utilise du slicing (pas np.roll) → les bords non remplis restent à zéro
    (ciel noir autour de la planète, pas d'artefact de wrapping).

    Args:
        bayer : float32 (H, W)
        dy    : décalage vertical (positif = vers le bas)
        dx    : décalage horizontal (positif = vers la droite)
    """
    if dy == 0 and dx == 0:
        return bayer

    h, w = bayer.shape
    result = np.zeros_like(bayer)

    sy0 = max(0, -dy);  sy1 = h - max(0, dy)
    dy0 = max(0,  dy);  dy1 = h - max(0, -dy)
    sx0 = max(0, -dx);  sx1 = w - max(0, dx)
    dx0 = max(0,  dx);  dx1 = w - max(0, -dx)

    if sy1 > sy0 and sx1 > sx0:
        result[dy0:dy1, dx0:dx1] = bayer[sy0:sy1, sx0:sx1]

    return result


# ---------------------------------------------------------------------------
# Stacking Bayer
# ---------------------------------------------------------------------------

def _stack_mean(frames: List[np.ndarray]) -> np.ndarray:
    """Moyenne de frames Bayer float32."""
    acc = np.zeros(frames[0].shape, dtype=np.float64)
    for f in frames:
        acc += f
    return (acc / len(frames)).astype(np.float32)


def _stack_sigma_clip(frames: List[np.ndarray], kappa: float = 2.5) -> np.ndarray:
    """Sigma-clipping pixel-wise. Fallback sur moyenne si < 3 frames."""
    if len(frames) < 3:
        return _stack_mean(frames)
    stack = np.stack(frames, axis=0).astype(np.float32)  # (N, H, W)
    mean  = stack.mean(axis=0)
    std   = stack.std(axis=0)
    mask  = np.abs(stack - mean[None]) <= kappa * std[None]
    valid_sum = np.where(mask, stack, 0.0).sum(axis=0)
    valid_cnt = mask.sum(axis=0).clip(1, None).astype(np.float32)
    return (valid_sum / valid_cnt).astype(np.float32)


# ---------------------------------------------------------------------------
# Débayérisation du stack final
# ---------------------------------------------------------------------------

def debayer_bayer_stack(bayer_stack: np.ndarray,
                        red_gain:    float = 1.0,
                        blue_gain:   float = 1.0,
                        global_bl:   float = 0.0) -> np.ndarray:
    """Débayérise un stack Bayer float32 en RGB float32 [0-65535].

    Utilise COLOR_BayerRG2BGR (identique à debayer_raw_array dans RPiCamera2.py).
    Résultat : ch0 = R_physique, ch1 = G, ch2 = B_physique — compatible
    avec apply_isp_to_preview() et toute la chaîne de post-traitement.

    Args:
        bayer_stack : float32 (H, W) Bayer RGGB, espace CSI-2 ×16
        red_gain    : Gain AWB rouge  (ex: red / 10 depuis globals)
        blue_gain   : Gain AWB bleu   (ex: blue / 10 depuis globals)
        global_bl   : Black level global en ADU 12-bit (0 = désactivé, fallback seulement)

    Returns:
        float32 (H, W, 3) [0-65535], compatible apply_isp_to_preview()
    """
    # Correction per-canal BL (FPN 2×2) sur le stack moyenné.
    # Toujours appliquée : si per-frame déjà corrigé → percentile5 ≈ 0 → no-op.
    # Si non corrigé → supprime le quadrillage Bayer résiduel (offsets R/G1/G2/B).
    bl_r, bl_g1, bl_g2, bl_b = _estimate_bl_auto(bayer_stack)
    if max(bl_r, bl_g1, bl_g2, bl_b) > 0.0:
        bayer_stack = _apply_bl_per_channel(bayer_stack, bl_r, bl_g1, bl_g2, bl_b)
    elif global_bl > 0.0:
        # Fallback BL global seulement si l'auto-estimation donne 0 partout
        bayer_stack = np.maximum(bayer_stack - global_bl * 16.0, 0.0)

    bayer_u16 = np.clip(bayer_stack, 0.0, 65535.0).astype(np.uint16)

    # Débayérisation bilinéaire uint16 (VNG non supporté en uint16 par OpenCV)
    # Sur un stack moyenné (bruit réduit), le bilinéaire est suffisant.
    bgr = cv2.cvtColor(bayer_u16, cv2.COLOR_BayerRG2BGR)  # uint16 (H, W, 3)

    # Float32 + gains AWB (ch0=R_phys, ch2=B_phys — même comportement que debayer_raw_array)
    rgb = bgr.astype(np.float32)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * red_gain,  0.0, 65535.0)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * blue_gain, 0.0, 65535.0)
    return rgb


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class BayerLuckyStacker:
    """Lucky Stack planétaire en domaine Bayer RAW12/16.

    Interface identique au lucky stack RGB8 (même globals, mêmes stats,
    même post-pipeline) — seul le traitement interne change.

    Exemple d'utilisation :
        stacker = BayerLuckyStacker(buffer_size=50, keep_percent=20.0)

        # Boucle de capture :
        score = stacker.add_frame(raw_uint16)   # (H, W) uint16

        # Quand lucky_raw_active et nouveau stack :
        if stacker.new_stack_available():
            result = stacker.get_result(red_gain, blue_gain, global_bl)
            # result : float32 (H, W, 3) [0-65535]
            # → apply_isp_to_preview() → apply_lucky_post_stack_filters()
    """

    def __init__(
        self,
        buffer_size:   int   = 50,
        keep_percent:  float = 20.0,
        score_method:  str   = 'laplacian',
        score_roi:     float = 0.50,
        align_enabled: bool  = True,
        max_shift_px:  int   = 30,
        stack_method:  str   = 'mean',
        sigma_kappa:   float = 2.5,
        bl_auto:       bool  = False,
        bl_r:          float = 0.0,
        bl_g1:         float = 0.0,
        bl_g2:         float = 0.0,
        bl_b:          float = 0.0,
    ):
        self.buffer_size   = max(2, int(buffer_size))
        self.keep_percent  = float(keep_percent)
        self.score_method  = score_method
        self.score_roi     = float(score_roi)
        self.align_enabled = bool(align_enabled)
        self.max_shift_px  = int(max_shift_px)
        self.stack_method  = stack_method
        self.sigma_kappa   = float(sigma_kappa)
        self.bl_auto       = bool(bl_auto)
        self.bl_r          = float(bl_r)
        self.bl_g1         = float(bl_g1)
        self.bl_g2         = float(bl_g2)
        self.bl_b          = float(bl_b)

        self._lock:   threading.RLock = threading.RLock()
        self._frames: deque = deque(maxlen=self.buffer_size)
        self._scores: deque = deque(maxlen=self.buffer_size)

        self.frame_count:  int = 0
        self.stacks_done:  int = 0
        self._prev_stacks: int = 0

        self.last_bayer_stack: Optional[np.ndarray] = None
        self._align_ref_g1:    Optional[np.ndarray] = None

        # Stack cumulatif inter-buffer pondéré par score
        # Chaque buffer est pondéré par le score moyen de ses frames sélectionnées
        # → les buffers avec de meilleures frames contribuent davantage au résultat final
        self._cumul_stack:  Optional[np.ndarray] = None   # float64 (H, W) somme pondérée
        self._cumul_count:  int = 0                        # nombre de buffers accumulés
        self._cumul_weight: float = 0.0                    # somme des poids (scores moyens)

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def add_frame(self, raw: np.ndarray) -> float:
        """Ajoute une frame Bayer, déclenche le stack si buffer plein.

        Args:
            raw : uint8 (H, W*2) ou uint16 (H, W) Bayer RGGB, espace CSI-2 ×16
                  Picamera2 livre parfois le Bayer comme buffer uint8 (2 octets/pixel).
                  La conversion uint8→uint16 est faite ici, identique à debayer_raw_array().

        Returns:
            Score de netteté (float ≥ 0)
        """
        # Conversion uint8 → uint16 si picamera2 livre le buffer comme bytes bruts
        # Ex: (480, 1280) uint8 → (480, 640) uint16 pour un Bayer 640×480
        if raw.dtype == np.uint8:
            h = raw.shape[0]
            raw = raw.view(np.uint16).reshape(h, -1)[:, :raw.shape[1] // 2]

        # Black level per-canal (correction FPN 2×2)
        use_bl = self.bl_auto or (self.bl_r + self.bl_g1 + self.bl_g2 + self.bl_b) > 0.0
        if use_bl:
            if self.bl_auto:
                bl_vals = _estimate_bl_auto(raw)
            else:
                bl_vals = (self.bl_r, self.bl_g1, self.bl_g2, self.bl_b)
            processed = _apply_bl_per_channel(raw, *bl_vals)
        else:
            processed = raw.astype(np.float32)

        score = score_bayer_frame(
            processed.astype(np.uint16) if processed.dtype != np.uint16 else processed,
            self.score_method, self.score_roi
        )

        with self._lock:
            self._frames.append(processed)
            self._scores.append(score)
            self.frame_count += 1
            if len(self._frames) >= self.buffer_size:
                self._process_buffer()

        return score

    def new_stack_available(self) -> bool:
        """True si un nouveau stack est prêt (consomme l'indicateur)."""
        result = self.stacks_done > self._prev_stacks
        if result:
            self._prev_stacks = self.stacks_done
        return result

    def get_result(self,
                   red_gain:  float = 1.0,
                   blue_gain: float = 1.0,
                   global_bl: float = 0.0) -> Optional[np.ndarray]:
        """Retourne le stack cumulatif débayérisé, ou None si pas encore de stack.

        Identique au lucky RGB8 : chaque buffer complété est accumulé dans une
        moyenne cumulative → l'image s'améliore progressivement d'un buffer à l'autre.

        Args:
            red_gain  : Gain AWB rouge  (red / 10 depuis globals RPiCamera2)
            blue_gain : Gain AWB bleu   (blue / 10 depuis globals RPiCamera2)
            global_bl : BL global en ADU 12-bit (isp_black_level si FPN désactivé)

        Returns:
            float32 (H, W, 3) [0-65535], ch0=R_phys, ch2=B_phys.
            Compatible avec apply_isp_to_preview().
        """
        if self._cumul_stack is None or self._cumul_count == 0:
            return None
        # Division par la somme des poids (score-weighted mean inter-buffer)
        divisor = self._cumul_weight if self._cumul_weight > 0 else float(self._cumul_count)
        bayer_mean = (self._cumul_stack / divisor).astype(np.float32)
        return debayer_bayer_stack(bayer_mean, red_gain, blue_gain, global_bl)

    def get_stats(self) -> Dict:
        """Statistiques compatibles avec draw_lucky_stats_bar() de RPiCamera2.py."""
        with self._lock:
            scores = list(self._scores)
            n = len(self._frames)
        return {
            # Clés requises par draw_lucky_stats_bar()
            'lucky_buffer_fill':  n,
            'lucky_buffer_size':  self.buffer_size,
            'lucky_stacks_done':  self.stacks_done,
            'total_frames':       self.frame_count,
            'lucky_avg_score':    float(np.mean(scores)) if scores else 0.0,
            # Infos supplémentaires
            'lucky_max_score':    float(max(scores)) if scores else 0.0,
            'keep_percent':       self.keep_percent,
            'buffer_mode':        'ring',  # Toujours ring pour RAW lucky
        }

    def reset(self):
        """Réinitialise complètement le stacker (buffer + cumulatif)."""
        with self._lock:
            self._frames.clear()
            self._scores.clear()
            self.frame_count   = 0
            self.stacks_done   = 0
            self._prev_stacks  = 0
            self.last_bayer_stack = None
            self._align_ref_g1    = None
            self._cumul_stack     = None
            self._cumul_count     = 0
            self._cumul_weight    = 0.0

    def update_config(self, **kwargs):
        """Met à jour la configuration à chaud (sans reset du buffer).

        Paramètres supportés : buffer_size, keep_percent, score_method,
        score_roi, align_enabled, max_shift_px, stack_method, sigma_kappa,
        bl_auto, bl_r, bl_g1, bl_g2, bl_b.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'buffer_size':
                    v = max(2, int(v))
                    if v != self.buffer_size:
                        # Redimensionner le buffer
                        old_frames = list(self._frames)
                        old_scores = list(self._scores)
                        self._frames = deque(maxlen=v)
                        self._scores = deque(maxlen=v)
                        self._frames.extend(old_frames[-v:])
                        self._scores.extend(old_scores[-v:])
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _process_buffer(self):
        """Sélectionne, aligne et stacke les meilleures frames du buffer."""
        frames = list(self._frames)
        scores = list(self._scores)
        n = len(frames)
        if n == 0:
            return

        # Sélection top N%
        keep = max(1, int(n * self.keep_percent / 100.0))
        idx_sorted = sorted(range(n), key=lambda i: scores[i], reverse=True)
        best_frames = [frames[i] for i in idx_sorted[:keep]]

        # Référence = meilleure frame
        ref = best_frames[0]
        ref_g1 = _extract_g1(ref).astype(np.float32)

        max_half = (self.max_shift_px // 2) if self.max_shift_px > 0 else 0

        # Alignement
        aligned = []
        for frame in best_frames:
            if self.align_enabled and len(best_frames) > 1:
                fg1 = _extract_g1(frame).astype(np.float32)
                dy, dx = _compute_bayer_shift(ref_g1, fg1, max_half)
                aligned.append(_shift_bayer(frame, dy, dx))
            else:
                aligned.append(frame)

        # Stacking
        if self.stack_method == 'sigma_clip' and len(aligned) >= 3:
            result = _stack_sigma_clip(aligned, self.sigma_kappa)
        else:
            result = _stack_mean(aligned)

        self.last_bayer_stack = result
        self._align_ref_g1    = ref_g1
        self.stacks_done     += 1

        # Accumulation inter-buffer pondérée par le score moyen des frames sélectionnées
        # Les buffers avec de meilleures frames (score élevé) pèsent davantage
        buffer_score = float(np.mean([scores[i] for i in idx_sorted[:keep]]))
        weight = max(buffer_score, 1e-9)  # Éviter poids nul

        if self._cumul_stack is None:
            self._cumul_stack = result.astype(np.float64) * weight
        else:
            self._cumul_stack += result.astype(np.float64) * weight
        self._cumul_weight += weight
        self._cumul_count  += 1

        # Vider le buffer pour le cycle suivant
        self._frames.clear()
        self._scores.clear()


# ---------------------------------------------------------------------------
# Fabrique
# ---------------------------------------------------------------------------

def create_bayer_lucky_stacker(
    buffer_size:   int   = 50,
    keep_percent:  float = 20.0,
    score_method:  str   = 'laplacian',
    score_roi:     float = 0.50,
    align_enabled: bool  = True,
    max_shift_px:  int   = 30,
    stack_method:  str   = 'mean',
    sigma_kappa:   float = 2.5,
    bl_auto:       bool  = False,
    bl_r:          float = 0.0,
    bl_g1:         float = 0.0,
    bl_g2:         float = 0.0,
    bl_b:          float = 0.0,
) -> BayerLuckyStacker:
    """Crée un BayerLuckyStacker depuis les globals RPiCamera2."""
    return BayerLuckyStacker(
        buffer_size   = buffer_size,
        keep_percent  = keep_percent,
        score_method  = score_method,
        score_roi     = score_roi,
        align_enabled = align_enabled,
        max_shift_px  = max_shift_px,
        stack_method  = stack_method,
        sigma_kappa   = sigma_kappa,
        bl_auto       = bl_auto,
        bl_r          = bl_r,
        bl_g1         = bl_g1,
        bl_g2         = bl_g2,
        bl_b          = bl_b,
    )
