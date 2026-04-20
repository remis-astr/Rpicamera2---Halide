#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP Stacker — Stacking planétaire par Alignment Points
======================================================

Implémentation style AutoStakkert :
1. Grille automatique d'APs sur le masque du disque
2. Corrélation de phase locale par AP (résidus après alignement global)
3. Champ de déplacement dense par interpolation IDW
4. Warp par cv2.remap + stack pondéré par score global

Utilisé exclusivement avec ElitePoolStacker (pool élite uniquement).
Les frames arrivent déjà pré-alignées globalement (translation) — le AP
ne corrige que les déformations atmosphériques locales résiduelles.

Références :
- Christoph Pellegrini, AutoStakkert!3 (2012-2024)
- IDW : Shepard 1968, "A two-dimensional interpolation function..."

Auteur: libastrostack Team
Version: 1.0.0
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any
import time
import logging

try:
    from scipy.fft import fft2 as _fft2, ifft2 as _ifft2
    _FFT_WORKERS = 2   # scipy.fft supporte le parallélisme
    _USE_SCIPY_FFT = True
except ImportError:
    from numpy.fft import fft2 as _fft2, ifft2 as _ifft2
    _FFT_WORKERS = None
    _USE_SCIPY_FFT = False

logger = logging.getLogger(__name__)

# Détecteur de disque (optionnel)
try:
    from .aligner_planetary import DiskDetector, PlanetaryConfig as _PlanetaryConfig
    _HAS_DISK_DETECTOR = True
except ImportError:
    _HAS_DISK_DETECTOR = False
    logger.warning("[AP] aligner_planetary non disponible — détection disque désactivée")


# =============================================================================
# APGrid — placement de la grille sur le disque
# =============================================================================

class APGrid:
    """
    Grille automatique de points d'alignement (APs) sur le masque du disque.

    Place des APs à intervalles réguliers à l'intérieur du disque détecté.
    Si aucun disque n'est détecté, couvre toute l'image avec une marge.
    """

    def __init__(self, window_size: int = 64, spacing_ratio: float = 0.6,
                 margin_ratio: float = 0.85):
        """
        Args:
            window_size:    Taille de la fenêtre d'AP (pixels). Puissance de 2 recommandée.
            spacing_ratio:  Espacement entre APs = window_size × spacing_ratio.
                            0.5 = 50% overlap, 1.0 = pas d'overlap.
            margin_ratio:   APs limités à margin_ratio × rayon_disque (évite les bords flous).
        """
        self.window_size   = window_size
        self.spacing_ratio = spacing_ratio
        self.margin_ratio  = margin_ratio
        self.half          = window_size // 2

        # État après build()
        self.ap_positions: List[Tuple[int, int]] = []
        self.disk_center:  Optional[Tuple[float, float]] = None
        self.disk_radius:  Optional[float] = None

    def build(self, image: np.ndarray,
              disk_info: Optional[Dict[str, Any]] = None) -> List[Tuple[int, int]]:
        """
        Construit la grille d'APs.

        Args:
            image:     Image de référence (H, W) ou (H, W, C).
            disk_info: Dict avec center_x, center_y, radius (issu de DiskDetector).
                       None → couvre toute l'image.

        Returns:
            Liste de positions (cx, cy) des APs.
        """
        h, w = image.shape[:2]
        half    = self.half
        spacing = max(16, int(self.window_size * self.spacing_ratio))

        if disk_info is not None:
            cx = float(disk_info['center_x'])
            cy = float(disk_info['center_y'])
            raw_r = float(disk_info['radius'])
            # Sécurité : rayon irréaliste (ex: fallback moments sur image entière)
            # → capper à 40% de la dimension minimale
            max_sane_r = min(w, h) * 0.40
            if raw_r > max_sane_r:
                print(f"[AP GRID] Rayon {raw_r:.0f}px irréaliste → capé à {max_sane_r:.0f}px")
                raw_r = max_sane_r
            r  = raw_r * self.margin_ratio
            self.disk_center = (cx, cy)
            self.disk_radius = raw_r
        else:
            cx = w / 2.0
            cy = h / 2.0
            r  = min(w, h) / 2.0 * 0.9
            self.disk_center = (cx, cy)
            self.disk_radius = None

        x_start = max(half, int(cx - r))
        x_end   = min(w - half, int(cx + r))
        y_start = max(half, int(cy - r))
        y_end   = min(h - half, int(cy + r))

        positions = []
        for ay in range(y_start, y_end, spacing):
            for ax in range(x_start, x_end, spacing):
                dist = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
                if dist <= r:
                    positions.append((ax, ay))

        self.ap_positions = positions
        logger.info(f"[AP GRID] {len(positions)} APs — "
                    f"centre=({cx:.0f},{cy:.0f}), r={r:.0f}px, "
                    f"espacement={spacing}px, fenêtre={self.window_size}px")
        return positions

    def get_patch(self, image: np.ndarray,
                  ap_pos: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extrait le patch centré sur ap_pos. Retourne None si hors-image."""
        cx, cy = ap_pos
        h, w = image.shape[:2]
        half = self.half
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        if y1 < 0 or y2 > h or x1 < 0 or x2 > w:
            return None
        return image[y1:y2, x1:x2]


# =============================================================================
# APLocalCorrelator — corrélation de phase locale par AP
# =============================================================================

class APLocalCorrelator:
    """
    Corrélation de phase sub-pixel pour chaque AP.

    Calcule le déplacement résiduel local (dx, dy) entre le patch de référence
    et le patch de la frame courante, ainsi qu'un score de qualité [0-1].
    """

    def __init__(self, window_size: int = 64, highpass: bool = True):
        """
        Args:
            window_size: Taille des patches (doit être identique à APGrid.window_size).
            highpass:    Filtre passe-haut avant FFT (accentue les détails fins).
        """
        self.window_size = window_size
        self.highpass    = highpass
        self._hann: Optional[np.ndarray] = None

    def _get_hann(self) -> np.ndarray:
        if self._hann is None:
            h = np.hanning(self.window_size)
            self._hann = np.outer(h, h).astype(np.float32)
        return self._hann

    @staticmethod
    def _to_gray(patch: np.ndarray) -> np.ndarray:
        """Canal vert (robuste) ou grayscale direct — float32."""
        if len(patch.shape) == 3:
            return patch[:, :, 1].astype(np.float32)
        return patch.astype(np.float32)

    def _highpass_filter(self, img: np.ndarray) -> np.ndarray:
        lap = cv2.Laplacian(img, cv2.CV_32F)
        return img - 0.3 * lap

    @staticmethod
    def _fft2(arr: np.ndarray) -> np.ndarray:
        """FFT2 avec scipy si disponible (float32 natif, ~8× plus rapide)."""
        if _USE_SCIPY_FFT:
            return _fft2(arr, workers=_FFT_WORKERS)
        return _fft2(arr)

    @staticmethod
    def _ifft2(arr: np.ndarray) -> np.ndarray:
        if _USE_SCIPY_FFT:
            return _ifft2(arr, workers=_FFT_WORKERS)
        return _ifft2(arr)

    def correlate_patch(self, ref_patch: np.ndarray,
                        cur_patch: np.ndarray) -> Tuple[float, float, float]:
        """
        Corrélation de phase entre deux patches.

        Returns:
            (dx, dy, quality)
            dx, dy  : déplacement sub-pixel (cur → ref), float
            quality : score de qualité [0-1] basé sur le rapport pic/bruit
        """
        ref = self._to_gray(ref_patch)
        cur = self._to_gray(cur_patch)

        sz = self.window_size
        if ref.shape[0] != sz:
            ref = cv2.resize(ref, (sz, sz))
            cur = cv2.resize(cur, (sz, sz))

        # Fenêtrage Hanning
        win = self._get_hann()
        ref = ref * win
        cur = cur * win

        if self.highpass:
            ref = self._highpass_filter(ref)
            cur = self._highpass_filter(cur)

        # FFT + cross-power spectrum normalisé
        ref_f = self._fft2(ref)
        cur_f = self._fft2(cur)
        prod  = ref_f * np.conj(cur_f)
        denom = np.abs(prod) + 1e-7
        cross = prod / denom

        # Corrélation de phase (sans fftshift — pic à (0,0) pour shift nul)
        corr = np.real(self._ifft2(cross))

        half     = sz // 2
        flat_idx = int(np.argmax(corr))
        r_raw    = flat_idx // sz
        c_raw    = flat_idx  % sz
        peak     = float(corr[r_raw, c_raw])

        # Coordonnées non-shiftées → décalage signé [-sz/2, sz/2)
        dy = float(r_raw if r_raw < half else r_raw - sz)
        dx = float(c_raw if c_raw < half else c_raw - sz)

        # Raffinement parabolique sub-pixel
        if 0 < r_raw < sz - 1 and 0 < c_raw < sz - 1:
            vy = corr[r_raw - 1:r_raw + 2, c_raw]
            vx = corr[r_raw, c_raw - 1:c_raw + 2]
            d = float(vy[0] + vy[2] - 2.0 * vy[1])
            if abs(d) > 1e-10:
                dy += (vy[0] - vy[2]) / (2.0 * d)
            d = float(vx[0] + vx[2] - 2.0 * vx[1])
            if abs(d) > 1e-10:
                dx += (vx[0] - vx[2]) / (2.0 * d)

        # Qualité : rapport pic / niveau moyen de la corrélation
        local_mean = float(np.mean(np.abs(corr)))
        quality    = min(1.0, max(0.0, peak / (local_mean * sz + 1e-10)))

        return dx, dy, quality

    def compute_fields(self,
                       frames: List[np.ndarray],
                       reference: np.ndarray,
                       ap_positions: List[Tuple[int, int]],
                       grid: APGrid) -> List[Dict[str, np.ndarray]]:
        """
        Calcule les champs de déplacement pour toutes les frames.

        Optimisation batch : pour chaque AP, les N patches de toutes les frames
        sont traités simultanément via np.fft.fft2 batché — shape (N, sz, sz).
        Cela exploite la vectorisation numpy et évite N appels FFT séparés.

        Args:
            frames:       Frames pré-alignées (uint8 ou float32, RGB ou MONO).
            reference:    Image de référence (même format).
            ap_positions: Liste de (cx, cy).
            grid:         APGrid instance pour l'extraction de patches.

        Returns:
            Liste de dicts par frame :
                'ap_dx'      : (N_aps,) float32
                'ap_dy'      : (N_aps,) float32
                'ap_quality' : (N_aps,) float32
        """
        n_frames = len(frames)
        n_aps    = len(ap_positions)
        sz       = self.window_size
        win      = self._get_hann()
        half     = sz // 2

        # Résultats initialisés à zéro
        all_dx      = np.zeros((n_frames, n_aps), dtype=np.float32)
        all_dy      = np.zeros((n_frames, n_aps), dtype=np.float32)
        all_quality = np.zeros((n_frames, n_aps), dtype=np.float32)

        # Grayscale float32 de toutes les frames + référence (calcul unique)
        ref_gray    = self._to_gray(reference)
        frames_gray = [self._to_gray(f) for f in frames]

        # Boucle sur les APs — une FFT de référence par AP, réutilisée pour toutes les frames
        for ap_idx, pos in enumerate(ap_positions):
            # Patch de référence → FFT (float32)
            ref_p = grid.get_patch(ref_gray, pos)
            if ref_p is None:
                continue

            rg = ref_p.astype(np.float32)
            if rg.shape[0] != sz:
                rg = cv2.resize(rg, (sz, sz))
            rg = rg * win
            if self.highpass:
                rg = self._highpass_filter(rg)
            ref_fft = self._fft2(rg)   # (sz, sz) complex

            # Boucle sur les frames — float32, scipy FFT
            for fi, fg in enumerate(frames_gray):
                cur_p = grid.get_patch(fg, pos)
                if cur_p is None:
                    continue

                cg = cur_p.astype(np.float32)
                if cg.shape[0] != sz:
                    cg = cv2.resize(cg, (sz, sz))
                cg = cg * win
                if self.highpass:
                    cg = self._highpass_filter(cg)
                cur_fft = self._fft2(cg)

                # Corrélation de phase (sans fftshift — traitement direct du pic)
                conj_cur = np.conj(cur_fft)
                prod     = ref_fft * conj_cur
                denom    = np.abs(prod) + 1e-7
                cross    = prod / denom
                corr     = np.real(self._ifft2(cross))   # pic à (0,0) pour shift nul

                flat_idx = int(np.argmax(corr))
                r_raw    = flat_idx // sz
                c_raw    = flat_idx  % sz
                peak     = float(corr[r_raw, c_raw])

                # Convertir coordonnées non-shiftées → décalage signé [-sz/2, sz/2)
                dy = float(r_raw if r_raw < half else r_raw - sz)
                dx = float(c_raw if c_raw < half else c_raw - sz)

                # Raffinement parabolique sub-pixel (dans corr non-shiftée)
                if 0 < r_raw < sz - 1 and 0 < c_raw < sz - 1:
                    vy = corr[r_raw - 1:r_raw + 2, c_raw]
                    vx = corr[r_raw, c_raw - 1:c_raw + 2]
                    d  = float(vy[0] + vy[2] - 2.0 * vy[1])
                    if abs(d) > 1e-10:
                        dy += float((vy[0] - vy[2]) / (2.0 * d))
                    d = float(vx[0] + vx[2] - 2.0 * vx[1])
                    if abs(d) > 1e-10:
                        dx += float((vx[0] - vx[2]) / (2.0 * d))

                local_mean = float(np.mean(np.abs(corr)))
                quality    = min(1.0, max(0.0, peak / (local_mean * sz + 1e-10)))

                all_dx[fi,      ap_idx] = dx
                all_dy[fi,      ap_idx] = dy
                all_quality[fi, ap_idx] = quality

        # Assembler en liste de dicts (une entrée par frame)
        return [
            {'ap_dx': all_dx[i], 'ap_dy': all_dy[i], 'ap_quality': all_quality[i]}
            for i in range(n_frames)
        ]


# =============================================================================
# DisplacementFieldBuilder — champ dense par IDW
# =============================================================================

class DisplacementFieldBuilder:
    """
    Champ de déplacement dense à partir des vecteurs AP sparse.

    Interpolation IDW (Inverse Distance Weighting, Shepard 1968) sur une
    grille basse résolution, puis upsampling bilinéaire vers la taille pleine.

    Optimisation clé : la matrice de poids IDW dépend uniquement des positions
    des APs (pas de leurs valeurs). Elle est précalculée une fois via
    precompute() et réutilisée pour toutes les frames, réduisant le coût
    par frame à un simple produit matrice-vecteur.
    """

    def __init__(self, idw_power: float = 2.0, grid_step: int = 8):
        """
        Args:
            idw_power:  Exposant IDW (2 = standard, plus grand = plus localisé).
            grid_step:  Résolution de la grille intermédiaire (pixels).
                        8 = bonne précision, 16 = plus rapide pour grandes images.
        """
        self.idw_power = idw_power
        self.grid_step = grid_step

        # Cache de la matrice de poids (invalidé si positions changent)
        self._cache_key:    Optional[tuple] = None    # (height, width, ap_pos_tuple)
        self._weights_norm: Optional[np.ndarray] = None  # (P, M) float32 normalisé
        self._valid_mask:   Optional[np.ndarray] = None  # (N_aps,) bool
        self._gw = self._gh = 0
        self._xx: Optional[np.ndarray] = None
        self._yy: Optional[np.ndarray] = None

    def precompute(self, height: int, width: int,
                   ap_positions: List[Tuple[int, int]],
                   ap_quality: np.ndarray) -> bool:
        """
        Précalcule la matrice de poids IDW pour une configuration d'APs.
        Doit être appelé avant le premier build_remap d'une session.

        Returns True si des APs valides ont été trouvés.
        """
        valid = ap_quality > 0.05
        if not np.any(valid):
            self._valid_mask   = valid
            self._weights_norm = None
            return False

        # Clé de cache : invalider si la géométrie change
        cache_key = (height, width, tuple(ap_positions), tuple(valid.tolist()))
        if cache_key == self._cache_key:
            return True  # déjà calculé

        self._cache_key = cache_key
        self._valid_mask = valid

        valid_pos = np.array(ap_positions)[valid]   # (M, 2)
        step = self.grid_step
        gw   = (width  + step - 1) // step
        gh   = (height + step - 1) // step
        self._gw, self._gh = gw, gh

        gx = (np.arange(gw) * step + step // 2).astype(np.float32)
        gy = (np.arange(gh) * step + step // 2).astype(np.float32)
        gx2d, gy2d = np.meshgrid(gx, gy)

        pts_flat = np.stack([gx2d.ravel(), gy2d.ravel()], axis=1)   # (P, 2)
        diff     = pts_flat[:, np.newaxis, :] - valid_pos[np.newaxis, :, :]  # (P, M, 2)
        dist2    = (np.sum(diff ** 2, axis=-1) + 1.0).astype(np.float32)    # (P, M)

        weights  = (1.0 / dist2) ** (self.idw_power / 2.0)   # (P, M)
        w_sum    = np.maximum(weights.sum(axis=1, keepdims=True), 1e-10)
        self._weights_norm = (weights / w_sum).astype(np.float32)   # (P, M) normalisé

        # Grilles identité (float32, pour map_x/map_y)
        self._xx, self._yy = np.meshgrid(np.arange(width,  dtype=np.float32),
                                          np.arange(height, dtype=np.float32))
        return True

    def build_remap(self, height: int, width: int,
                    ap_positions: List[Tuple[int, int]],
                    ap_dx: np.ndarray,
                    ap_dy: np.ndarray,
                    ap_quality: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construit les cartes map_x, map_y pour cv2.remap.

        Si precompute() a été appelé au préalable, utilise la matrice de poids
        mise en cache (produit matrice-vecteur, ~10× plus rapide que le calcul
        complet). Sinon, calcule à la volée (compatible usage standalone).

        Returns:
            (map_x, map_y) float32, shape (H, W)
        """
        xx = self._xx if self._xx is not None else np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))[0]
        yy = self._yy if self._yy is not None else np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))[1]

        if not ap_positions:
            return xx, yy

        # Filtrer par masque de validité courant (peut changer selon qualité par frame)
        valid = ap_quality > 0.05
        if not np.any(valid):
            return xx, yy

        valid_dx = ap_dx[valid].astype(np.float32)
        valid_dy = ap_dy[valid].astype(np.float32)

        # Chemin rapide : matrice précalculée disponible et masque identique
        if (self._weights_norm is not None and
                self._valid_mask is not None and
                np.array_equal(valid, self._valid_mask) and
                self._gw > 0):
            W  = self._weights_norm                         # (P, M)
            dx_grid = (W @ valid_dx).reshape(self._gh, self._gw)   # (gh, gw)
            dy_grid = (W @ valid_dy).reshape(self._gh, self._gw)
        else:
            # Calcul complet (fallback ou premier appel sans precompute)
            valid_pos = np.array(ap_positions)[valid]
            step = self.grid_step
            gw   = (width  + step - 1) // step
            gh   = (height + step - 1) // step
            gx   = (np.arange(gw) * step + step // 2).astype(np.float32)
            gy   = (np.arange(gh) * step + step // 2).astype(np.float32)
            gx2d, gy2d = np.meshgrid(gx, gy)
            pts  = np.stack([gx2d.ravel(), gy2d.ravel()], axis=1)
            diff = pts[:, np.newaxis, :] - valid_pos[np.newaxis, :, :]
            dist2 = (np.sum(diff**2, axis=-1) + 1.0).astype(np.float32)
            W    = (1.0 / dist2) ** (self.idw_power / 2.0)
            W    = (W / np.maximum(W.sum(axis=1, keepdims=True), 1e-10)).astype(np.float32)
            dx_grid = (W @ valid_dx).reshape(gh, gw)
            dy_grid = (W @ valid_dy).reshape(gh, gw)

        disp_x = cv2.resize(dx_grid, (width, height), interpolation=cv2.INTER_LINEAR)
        disp_y = cv2.resize(dy_grid, (width, height), interpolation=cv2.INTER_LINEAR)

        return xx + disp_x, yy + disp_y


# =============================================================================
# APStackEngine — orchestration complète
# =============================================================================

class APStackEngine:
    """
    Moteur de stacking planétaire par Alignment Points.

    S'intègre dans ElitePoolStacker via l'attribut ap_engine :
        stacker.ap_engine = APStackEngine(...)

    Appelé depuis ElitePoolStacker._do_stack() à la place du simple np.mean.

    Pipeline :
        frames (pré-alignés globalement)
            → détection disque → grille APs
            → corrélation locale par AP par frame
            → clipping déplacements aberrants
            → champ dense IDW par frame
            → cv2.remap par frame
            → weighted mean (poids = score global × qualité AP moyenne)
    """

    def __init__(self,
                 ap_window: int = 64,
                 ap_spacing_ratio: float = 0.6,
                 ap_margin_ratio: float = 0.85,
                 idw_power: float = 2.0,
                 idw_grid_step: int = 8,
                 max_ap_shift: float = 20.0,
                 max_aps: int = 60):
        """
        Args:
            ap_window:        Taille des fenêtres AP (pixels). 32/64/128.
            ap_spacing_ratio: Espacement APs = window × ratio (0.5-1.0).
            ap_margin_ratio:  Fraction du rayon disque couverte (0.7-0.95).
            idw_power:        Exposant IDW (1.5-3.0).
            idw_grid_step:    Résolution grille IDW (4-16 pixels).
            max_ap_shift:     Déplacement AP max accepté en pixels.
                              Les AP avec |dx|>max ou |dy|>max sont ignorés.
            max_aps:          Nombre maximum d'APs (évite explosion sur images plein-champ).
        """
        self.max_ap_shift = max_ap_shift
        self.max_aps      = max_aps

        self.grid          = APGrid(window_size=ap_window,
                                    spacing_ratio=ap_spacing_ratio,
                                    margin_ratio=ap_margin_ratio)
        self.correlator    = APLocalCorrelator(window_size=ap_window)
        self.field_builder = DisplacementFieldBuilder(idw_power=idw_power,
                                                      grid_step=idw_grid_step)

        # Statistiques de la dernière exécution
        self.last_n_aps      = 0
        self.last_process_ms = 0.0
        self.last_n_frames   = 0

    # ── API principale ────────────────────────────────────────────────────────

    def process(self,
                frames: List[np.ndarray],
                scores: List[float],
                reference: np.ndarray) -> np.ndarray:
        """
        Stack AP complet.

        Args:
            frames:    Frames pré-alignées globalement (uint8 ou float32, RGB ou MONO).
                       Sorties directes de ElitePool (copies de frames acceptées).
            scores:    Score qualité global par frame (du pool élite, même ordre).
            reference: Image de référence warmup (même format que frames).

        Returns:
            Image stackée float32 [0-255], même shape que frames[0].
        """
        t0 = time.perf_counter()

        if not frames:
            return reference.astype(np.float32)

        h, w      = frames[0].shape[:2]
        is_color  = len(frames[0].shape) == 3
        n_chan    = frames[0].shape[2] if is_color else 1

        # 1. Détecter disque dans la référence
        ref_gray  = self._to_gray(reference)
        disk_info = self._detect_disk(ref_gray)

        # 2. Grille APs
        ap_positions = self.grid.build(reference, disk_info)

        # Limiter le nombre d'APs pour rester sous max_aps
        # (priorité aux APs les plus proches du centre du disque)
        if len(ap_positions) > self.max_aps:
            cx_d = self.grid.disk_center[0] if self.grid.disk_center else w / 2.0
            cy_d = self.grid.disk_center[1] if self.grid.disk_center else h / 2.0
            ap_positions = sorted(
                ap_positions,
                key=lambda p: (p[0] - cx_d) ** 2 + (p[1] - cy_d) ** 2
            )[:self.max_aps]
            print(f"[AP GRID] Limité à {self.max_aps} APs (parmi {len(self.grid.ap_positions)})")

        self.last_n_aps = len(ap_positions)

        if not ap_positions:
            logger.warning("[AP] Aucun AP — fallback moyenne pondérée simple")
            return self._weighted_mean_fallback(frames, scores, is_color)

        # 3. Corrélation locale par AP pour chaque frame
        fields = self.correlator.compute_fields(
            frames, reference, ap_positions, self.grid
        )

        # 4. Clipper les déplacements aberrants (turbulence extreme, dérive)
        for fld in fields:
            bad = ((np.abs(fld['ap_dx']) > self.max_ap_shift) |
                   (np.abs(fld['ap_dy']) > self.max_ap_shift))
            fld['ap_dx'][bad]      = 0.0
            fld['ap_dy'][bad]      = 0.0
            fld['ap_quality'][bad] = 0.0

        # 4b. Précalculer la matrice IDW avec la qualité moyenne par AP
        #     (estimée sur toutes les frames — permet le chemin rapide dans build_remap)
        mean_quality = np.mean(
            np.stack([f['ap_quality'] for f in fields], axis=0), axis=0
        ).astype(np.float32)
        self.field_builder.precompute(h, w, ap_positions, mean_quality)

        # 5. Normaliser les scores globaux [0-1]
        scores_arr = np.array(scores, dtype=np.float32)
        s_max = float(scores_arr.max())
        if s_max > 1e-10:
            scores_arr /= s_max

        # 6. Warp + accumulation pondérée (float32 — suffisant pour 8-bit × N frames)
        if is_color:
            acc     = np.zeros((h, w, n_chan), dtype=np.float32)
        else:
            acc     = np.zeros((h, w), dtype=np.float32)
        w_total = np.zeros((h, w), dtype=np.float32)

        for frame, fld, gs in zip(frames, fields, scores_arr):
            # Champ dense IDW (rapide grâce au cache precompute)
            map_x, map_y = self.field_builder.build_remap(
                h, w, ap_positions,
                fld['ap_dx'], fld['ap_dy'], fld['ap_quality']
            )

            # Warp (cv2.remap → float32 directement)
            warped = cv2.remap(frame.astype(np.float32), map_x, map_y,
                               cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

            # Poids : score global × qualité AP moyenne
            mean_ap_q = float(np.mean(fld['ap_quality']))
            weight    = float(gs) * (0.5 + 0.5 * mean_ap_q)
            w_f32     = np.float32(max(weight, 1e-10))

            acc     += warped * w_f32
            w_total += w_f32

        # 7. Division par les poids
        np.maximum(w_total, np.float32(1e-10), out=w_total)
        if is_color:
            result = acc / w_total[:, :, np.newaxis]
        else:
            result = acc / w_total

        self.last_process_ms = (time.perf_counter() - t0) * 1000
        self.last_n_frames   = len(frames)

        logger.info(f"[AP STACK] {len(frames)} frames, {self.last_n_aps} APs, "
                    f"{self.last_process_ms:.0f}ms")
        print(f"[AP STACK] {len(frames)} frames, {self.last_n_aps} APs, "
              f"{self.last_process_ms:.0f}ms")

        return result

    # ── Internes ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return image[:, :, 1].astype(np.float64)
        return image.astype(np.float64)

    @staticmethod
    def _detect_disk(ref_gray: np.ndarray) -> Optional[Dict[str, Any]]:
        if not _HAS_DISK_DETECTOR:
            return None
        try:
            cfg      = _PlanetaryConfig()
            detector = DiskDetector(cfg)
            return detector.detect(ref_gray)
        except Exception as e:
            logger.warning(f"[AP] Détection disque échouée: {e}")
            return None

    @staticmethod
    def _weighted_mean_fallback(frames: List[np.ndarray],
                                scores: List[float],
                                is_color: bool) -> np.ndarray:
        """Moyenne pondérée simple sans AP (fallback)."""
        stack    = np.array([f.astype(np.float32) for f in frames])
        w        = np.array(scores, dtype=np.float32)
        w        = w / (w.sum() + 1e-10)
        if is_color:
            return np.sum(stack * w[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        return np.sum(stack * w[:, np.newaxis, np.newaxis], axis=0)
