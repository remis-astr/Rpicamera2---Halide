#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lucky Imaging pour libastrostack
================================

Module de stacking planétaire haute vitesse basé sur le principe du Lucky Imaging :
1. Acquisition continue dans un buffer circulaire (ring buffer)
2. Notation rapide de chaque image (score de qualité)
3. Sélection des x% meilleures images
4. Alignement et empilement des images sélectionnées
5. Affichage du résultat

Optimisé pour :
- Haute cadence (>100 fps)
- Faible latence (<10ms par image pour le scoring)
- Imagerie planétaire (Soleil, Lune, planètes)

Paramètres réglables :
- buffer_size : Taille du buffer circulaire (ex: 100, 200, 500)
- keep_percent : Pourcentage d'images à garder (ex: 10%, 20%, 50%)
- score_method : Méthode de scoring (laplacian, gradient, sobel, tenengrad)
- stack_interval : Intervalle de stacking (toutes les N frames)
- min_score : Score minimum absolu pour accepter une image

Usage :
    from libastrostack.lucky_imaging import LuckyImagingStacker, LuckyConfig
    
    config = LuckyConfig(buffer_size=100, keep_percent=10.0)
    stacker = LuckyImagingStacker(config)
    
    for frame in camera_stream:
        stacker.add_frame(frame)
        
        if stacker.is_buffer_full():
            result = stacker.process_buffer()
            display(result)

Auteur: libastrostack Team
Version: 1.0.0
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import threading
import gc  # Pour forcer libération mémoire après process_buffer
import heapq  # Pour ElitePool (min-heap O(log N))
from concurrent.futures import ThreadPoolExecutor

# Aligneur planétaire optionnel (disk/hybrid modes)
try:
    from libastrostack.aligner_planetary import (
        PlanetaryAligner as _PlanetaryAligner,
        PlanetaryMode as _PlanetaryMode,
        PlanetaryConfig as _PlanetaryConfig,
    )
    _HAS_PLANETARY = True
except ImportError:
    _HAS_PLANETARY = False
    _PlanetaryAligner = None
    _PlanetaryMode = None
    _PlanetaryConfig = None


class ScoreMethod(Enum):
    """Méthodes de calcul du score de qualité"""
    LAPLACIAN = "laplacian"             # Variance du Laplacien (rapide, recommandé)
    GRADIENT = "gradient"               # Magnitude du gradient
    SOBEL = "sobel"                     # Filtre de Sobel
    TENENGRAD = "tenengrad"             # Tenengrad (Sobel au carré)
    BRENNER = "brenner"                 # Gradient de Brenner
    FFT = "fft"                         # Analyse fréquentielle (plus lent)
    LOCAL_VARIANCE = "local_variance"   # Variance locale sur patches (micro-détails)
    PSD = "psd"                         # Densité spectrale de puissance haute fréquence


class StackMethod(Enum):
    """Méthodes de combinaison des images"""
    MEAN = "mean"                 # Moyenne simple
    MEDIAN = "median"             # Médiane (plus robuste)
    SIGMA_CLIP = "sigma_clip"     # Moyenne avec rejection sigma


class BufferMode(Enum):
    """Mode du buffer Lucky Stack"""
    RING  = "ring"   # Buffer circulaire classique (sélection top%, stack sur remplissage)
    ELITE = "elite"  # Pool des N meilleures frames (stack périodique, RGB8 uniquement)


@dataclass
class LuckyConfig:
    """Configuration du Lucky Imaging"""
    
    # Buffer circulaire
    buffer_size: int = 100              # Nombre d'images dans le buffer
    
    # Sélection
    keep_percent: float = 10.0          # Pourcentage d'images à garder (1-100)
    keep_count: Optional[int] = None    # OU nombre fixe d'images (prioritaire sur %)
    min_score: float = 0.0              # Score minimum absolu (0 = désactivé)
    
    # Scoring
    score_method: ScoreMethod = ScoreMethod.LAPLACIAN
    score_roi: Optional[Tuple[int, int, int, int]] = None  # ROI pour scoring (x, y, w, h)
    score_roi_percent: float = 50.0     # OU % central de l'image pour scoring
    use_gpu: bool = False               # Utiliser GPU si disponible (OpenCV CUDA)
    
    # Stacking
    stack_method: StackMethod = StackMethod.MEAN
    stack_interval: int = 0             # 0 = stack quand buffer plein, N = toutes les N frames
    auto_stack: bool = True             # Stacker automatiquement quand buffer plein
    sigma_clip_kappa: float = 2.5       # Kappa pour sigma clipping
    
    # Alignement
    align_enabled: bool = True          # Activer l'alignement (legacy, utilisez align_mode)
    align_mode: int = 1                 # 0=off, 1=surface/phase FFT, 2=disk/Hough, 3=hybride
    max_shift: float = 50.0             # Décalage max accepté en pixels (0 = désactivé)
    align_method: str = "phase"         # "phase" (FFT) ou "ecc" (Enhanced Correlation)
    align_roi_percent: float = 80.0     # % central pour alignement
    
    # Performance
    num_threads: int = 4                # Threads pour scoring parallèle
    downscale_scoring: float = 1.0      # Downscale pour scoring (1.0 = pas de downscale)

    # Normalisation RAW (CORRECTION bug image blanche)
    raw_format: Optional[str] = None    # Format RAW explicite: "raw12", "raw16", None=auto-detect
    raw_normalize_method: str = "percentile"  # "max" (ancien) ou "percentile" (robuste)
    raw_percentile_detect: float = 99.0 # Percentile pour détection (ignorer pixels chauds)

    # Statistiques
    keep_history: bool = True           # Garder historique des scores
    history_size: int = 1000            # Taille max de l'historique

    # ── Mode Pool Élite (buffer_mode=ELITE, RGB8 uniquement) ─────────────────
    buffer_mode: BufferMode = BufferMode.RING  # RING=classique, ELITE=pool évolutif
    elite_pool_size: int = 100          # Taille du pool (20-300)
    elite_stack_interval: float = 5.0  # Intervalle de stack en secondes (2-15)
    elite_entry_mode: str = "min"       # "min" (> pire frame) ou "mean" (> moyenne pool)
    elite_score_clip: bool = True       # Sigma-clipping des scores avant stack
    elite_score_kappa: float = 2.0      # Kappa pour le sigma-clipping des scores (1.5-4.0)

    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []

        if self.buffer_size < 10:
            errors.append("buffer_size doit être >= 10")

        if not 1.0 <= self.keep_percent <= 100.0:
            errors.append("keep_percent doit être entre 1 et 100")

        if self.keep_count is not None and self.keep_count < 1:
            errors.append("keep_count doit être >= 1")

        if not 0.0 <= self.score_roi_percent <= 100.0:
            errors.append("score_roi_percent doit être entre 0 et 100")

        if self.downscale_scoring <= 0 or self.downscale_scoring > 1.0:
            errors.append("downscale_scoring doit être entre 0 (exclu) et 1.0")

        if self.raw_format is not None and self.raw_format.lower() not in ['raw12', 'raw16']:
            errors.append("raw_format doit être 'raw12', 'raw16' ou None")

        if self.raw_normalize_method not in ['max', 'percentile']:
            errors.append("raw_normalize_method doit être 'max' ou 'percentile'")

        if not 90.0 <= self.raw_percentile_detect <= 100.0:
            errors.append("raw_percentile_detect doit être entre 90 et 100")

        if errors:
            raise ValueError("Config Lucky Imaging invalide: " + ", ".join(errors))

        return True
    
    def get_keep_count(self) -> int:
        """Retourne le nombre d'images à garder"""
        if self.keep_count is not None:
            return min(self.keep_count, self.buffer_size)
        return max(1, int(self.buffer_size * self.keep_percent / 100.0))


class FrameBuffer:
    """
    Buffer circulaire optimisé pour les images
    
    Stocke les images et leurs scores de manière efficace en mémoire.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.frames: deque = deque(maxlen=max_size)
        self.scores: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
        self.frame_count = 0
        self._lock = threading.Lock()
    
    def add(self, frame: np.ndarray, score: float = 0.0) -> int:
        """
        Ajoute une frame au buffer

        Returns:
            Index de la frame dans le buffer
        """
        with self._lock:
            # Copie conditionnelle : seulement si frame ne possède pas ses données
            # (si c'est une view/référence). Sinon, c'est déjà une nouvelle instance.
            if frame.flags['OWNDATA']:
                # Frame possède ses données (nouvelle instance) → pas besoin de copier
                self.frames.append(frame)
            else:
                # Frame est une view/référence → DOIT copier
                self.frames.append(frame.copy())
            self.scores.append(score)
            self.timestamps.append(time.time())
            self.frame_count += 1
            return len(self.frames) - 1
    
    def update_score(self, index: int, score: float):
        """Met à jour le score d'une frame"""
        with self._lock:
            if 0 <= index < len(self.scores):
                self.scores[index] = score
    
    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Récupère une frame par index"""
        with self._lock:
            if 0 <= index < len(self.frames):
                return self.frames[index]
            return None
    
    def get_frames_by_indices(self, indices: List[int]) -> List[np.ndarray]:
        """Récupère plusieurs frames par indices"""
        with self._lock:
            return [self.frames[i] for i in indices if 0 <= i < len(self.frames)]
    
    def get_best_indices(self, count: int, min_score: float = 0.0) -> List[int]:
        """
        Retourne les indices des N meilleures frames
        
        Args:
            count: Nombre de frames à retourner
            min_score: Score minimum requis
        
        Returns:
            Liste des indices triés par score décroissant
        """
        with self._lock:
            # Créer liste (index, score) filtrée par min_score
            scored = [(i, s) for i, s in enumerate(self.scores) if s >= min_score]
            
            # Trier par score décroissant
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner les N premiers indices
            return [idx for idx, _ in scored[:count]]
    
    def get_statistics(self) -> Dict[str, float]:
        """Retourne statistiques sur les scores"""
        with self._lock:
            if not self.scores:
                return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'count': 0}
            
            scores_array = np.array(self.scores)
            return {
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'count': len(scores_array)
            }
    
    def is_full(self) -> bool:
        """Vérifie si le buffer est plein"""
        return len(self.frames) >= self.max_size
    
    def clear(self):
        """Vide le buffer"""
        with self._lock:
            self.frames.clear()
            self.scores.clear()
            self.timestamps.clear()
    
    def __len__(self) -> int:
        return len(self.frames)


class ElitePool:
    """
    Pool des N meilleures frames basé sur un min-heap.

    Propriétés :
    - Insertion O(log N), accès au minimum O(1)
    - Une frame ne peut entrer que si elle améliore le seuil (min ou mean)
    - Sigma-clipping des scores pour le stack (get_frames_clipped)
    - Thread-safe via RLock
    """

    def __init__(self, max_size: int = 100, entry_mode: str = "min"):
        """
        Args:
            max_size:   Taille maximale du pool (20-300)
            entry_mode: "min" (> score le plus bas) ou "mean" (> score moyen)
        """
        self._max_size   = max_size
        self._entry_mode = entry_mode          # "min" | "mean"
        self._heap: List[Tuple[float, int, np.ndarray]] = []  # min-heap (score, counter, frame)
        self._counter    = 0
        self._lock       = threading.RLock()  # RLock pour appels imbriqués dans propriétés
        self._frames_tested   = 0
        self._frames_accepted = 0

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def is_full(self) -> bool:
        return len(self._heap) >= self._max_size

    @property
    def min_score(self) -> float:
        return self._heap[0][0] if self._heap else 0.0

    @property
    def mean_score(self) -> float:
        if not self._heap:
            return 0.0
        return sum(s for s, _, _ in self._heap) / len(self._heap)

    @property
    def max_score(self) -> float:
        if not self._heap:
            return 0.0
        return max(s for s, _, _ in self._heap)

    @property
    def accept_rate(self) -> float:
        if self._frames_tested == 0:
            return 0.0
        return self._frames_accepted / self._frames_tested * 100.0

    # ── Méthodes ──────────────────────────────────────────────────────────────

    def try_add(self, score: float, frame: np.ndarray) -> bool:
        """Essaie d'insérer une frame. Retourne True si acceptée."""
        with self._lock:
            self._frames_tested += 1

            if len(self._heap) < self._max_size:
                heapq.heappush(self._heap, (score, self._counter, frame.copy()))
                self._counter += 1
                self._frames_accepted += 1
                return True

            # Pool plein : comparer au seuil
            if self._entry_mode == "mean":
                threshold = sum(s for s, _, _ in self._heap) / len(self._heap)
            else:
                threshold = self._heap[0][0]   # min score (O(1))

            if score > threshold:
                heapq.heapreplace(self._heap, (score, self._counter, frame.copy()))
                self._counter += 1
                self._frames_accepted += 1
                return True

            return False

    def get_frames_and_scores(self) -> List[Tuple[float, np.ndarray]]:
        """Retourne [(score, frame), ...] (copies) pour stacking. Thread-safe."""
        with self._lock:
            return [(s, f.copy()) for s, _, f in self._heap]

    def get_frames_clipped(self, kappa: float = 2.0) -> Tuple[List[np.ndarray], int]:
        """
        Sigma-clipping des scores avant stack.
        Retourne (frames_retenues, n_clippées).
        Les frames dont le score < mean - kappa*std sont exclues.
        """
        with self._lock:
            n = len(self._heap)
            if n < 3:
                return [f for _, _, f in self._heap], 0

            scores = np.array([s for s, _, _ in self._heap], dtype=np.float64)
            mean_s = scores.mean()
            std_s  = scores.std()

            if std_s < 1e-9:   # Pool uniforme → garder tout
                return [f for _, _, f in self._heap], 0

            threshold = mean_s - kappa * std_s
            kept      = [f for s, _, f in self._heap if s >= threshold]
            n_clipped = n - len(kept)
            return kept, n_clipped

    def resize(self, new_size: int):
        """Change la taille max du pool à la volée.
        Si réduction : élimine les pires frames jusqu'à la nouvelle limite.
        Si agrandissement : autorise simplement plus de frames à l'avenir.
        """
        with self._lock:
            if new_size == self._max_size:
                return
            self._max_size = new_size
            # Réduction : heappop retire toujours la pire frame (min-heap)
            while len(self._heap) > self._max_size:
                heapq.heappop(self._heap)

    def update_entry_mode(self, mode: str):
        """Change le critère d'entrée à la volée ("min" ou "mean")."""
        self._entry_mode = mode

    def clear(self):
        """Vide le pool et réinitialise les compteurs."""
        with self._lock:
            self._heap.clear()
            self._frames_tested   = 0
            self._frames_accepted = 0


class QualityScorer:
    """
    Calculateur de score de qualité ultra-rapide
    
    Optimisé pour haute cadence (>100 fps).
    Temps cible : <5ms par image.
    """
    
    def __init__(self, config: LuckyConfig):
        self.config = config
        self._roi_cache = None
        
        # Sélectionner la méthode de scoring
        self._score_func = self._get_score_function(config.score_method)
    
    def _get_score_function(self, method: ScoreMethod) -> Callable:
        """Retourne la fonction de scoring appropriée"""
        methods = {
            ScoreMethod.LAPLACIAN: self._score_laplacian,
            ScoreMethod.GRADIENT: self._score_gradient,
            ScoreMethod.SOBEL: self._score_sobel,
            ScoreMethod.TENENGRAD: self._score_tenengrad,
            ScoreMethod.BRENNER: self._score_brenner,
            ScoreMethod.FFT: self._score_fft,
            ScoreMethod.LOCAL_VARIANCE: self._score_local_variance,
            ScoreMethod.PSD: self._score_psd,
        }
        return methods.get(method, self._score_laplacian)
    
    def score(self, image: np.ndarray) -> float:
        """
        Calcule le score de qualité d'une image
        
        Args:
            image: Image (grayscale ou RGB)
        
        Returns:
            Score de qualité (plus élevé = meilleure qualité)
        """
        # Convertir en grayscale si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8) if image.dtype != np.uint8 
                               else image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8) if image.dtype != np.uint8 else image
        
        # Downscale si configuré
        if self.config.downscale_scoring < 1.0:
            new_size = (int(gray.shape[1] * self.config.downscale_scoring),
                       int(gray.shape[0] * self.config.downscale_scoring))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        
        # Extraire ROI si configuré
        gray = self._extract_roi(gray)
        
        # Calculer score
        return self._score_func(gray)
    
    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """Extrait la région d'intérêt pour le scoring"""
        if self.config.score_roi is not None:
            x, y, w, h = self.config.score_roi
            h_img, w_img = image.shape[:2]
            x = min(x, w_img - 1)
            y = min(y, h_img - 1)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            return image[y:y+h, x:x+w]
        
        elif self.config.score_roi_percent < 100.0:
            h, w = image.shape[:2]
            margin_x = int(w * (100 - self.config.score_roi_percent) / 200)
            margin_y = int(h * (100 - self.config.score_roi_percent) / 200)
            return image[margin_y:h-margin_y, margin_x:w-margin_x]
        
        return image
    
    def _score_laplacian(self, gray: np.ndarray) -> float:
        """Score par variance du Laplacien (RECOMMANDÉ - très rapide)"""
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        return float(laplacian.var())

    def _score_gradient(self, gray: np.ndarray) -> float:
        """Score par magnitude du gradient"""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        return float(np.mean(magnitude))

    def _score_sobel(self, gray: np.ndarray) -> float:
        """Score par filtre de Sobel"""
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.abs(sobelx) + np.abs(sobely)))

    def _score_tenengrad(self, gray: np.ndarray) -> float:
        """Score Tenengrad (Sobel au carré) - très sensible au focus"""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(gx**2 + gy**2))

    def _score_brenner(self, gray: np.ndarray) -> float:
        """Score de Brenner (différence horizontale)"""
        diff = gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
        return float(np.mean(diff**2))
    
    def _score_fft(self, gray: np.ndarray) -> float:
        """Score par analyse FFT (plus lent mais robuste)"""
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        # Score = énergie dans les hautes fréquences
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        # Masquer les basses fréquences (centre)
        mask_radius = min(h, w) // 8
        magnitude[cy-mask_radius:cy+mask_radius, cx-mask_radius:cx+mask_radius] = 0
        return float(np.mean(magnitude))

    def _score_local_variance(self, gray: np.ndarray) -> float:
        """Score par variance locale sur patches - sensible aux micro-détails locaux.
        Divise l'image en patches, mesure la variance de chacun et retourne le 90e
        percentile : détecte un détail net même dans une petite zone de l'image."""
        patch_size = max(8, min(32, min(gray.shape) // 8))
        img_f = gray.astype(np.float32)
        ksize = (patch_size, patch_size)
        mean = cv2.boxFilter(img_f, cv2.CV_32F, ksize)
        mean_sq = cv2.boxFilter(img_f * img_f, cv2.CV_32F, ksize)
        var_map = mean_sq - mean * mean
        np.clip(var_map, 0, None, out=var_map)  # Évite les artefacts float négatifs
        flat = var_map.ravel()
        k = max(0, int(0.90 * flat.size) - 1)
        return float(np.partition(flat, k)[k])

    def _score_psd(self, gray: np.ndarray) -> float:
        """Score par densité spectrale de puissance (PSD) haute fréquence.
        Mesure l'énergie (amplitude²) au-delà du quart de la fréquence de Nyquist,
        via masque radial sur rfft2. Plus sensible aux fins détails que _score_fft."""
        f = np.fft.rfft2(gray.astype(np.float32))
        psd = np.abs(f) ** 2
        h, w2 = psd.shape
        r_cut = max(1, min(h, w2) // 4)
        # Distance radiale depuis DC (coin [0,0] pour rfft2)
        Y = np.arange(h, dtype=np.float32)
        Y = np.minimum(Y, h - Y)          # Périodicité verticale
        X = np.arange(w2, dtype=np.float32)
        dist2 = Y[:, None] ** 2 + X[None, :] ** 2
        mask = dist2 > (r_cut ** 2)
        return float(np.mean(psd[mask])) if mask.any() else float(np.mean(psd))


class FrameAligner:
    """
    Aligneur d'images rapide pour Lucky Imaging

    Utilise la corrélation de phase (FFT) pour un alignement sub-pixel rapide.
    """

    def __init__(self, config: LuckyConfig):
        self.config = config
        self.reference: Optional[np.ndarray] = None
        self.reference_is_explicit: bool = False  # Track si référence définie par update_alignment_reference()
        self._planetary_aligner = None  # Instance PlanetaryAligner pour modes 2/3
        # Cache FFT de référence (évite de recalculer FFT(ref) à chaque appel)
        self._ref_fft_cache: Optional[np.ndarray] = None
        self._ref_fft_shape: Optional[Tuple[int, int]] = None  # (h_fft, w_fft) du cache
    
    def _create_planetary_aligner(self):
        """Crée un PlanetaryAligner configuré selon align_mode (2=disk, 3=hybrid)."""
        if not _HAS_PLANETARY:
            print("[ALIGN] PlanetaryAligner non disponible, fallback surface/phase")
            return
        cfg = _PlanetaryConfig()
        mode_map = {2: _PlanetaryMode.DISK, 3: _PlanetaryMode.HYBRID}
        cfg.mode = mode_map.get(self.config.align_mode, _PlanetaryMode.SURFACE)
        cfg.max_shift = int(self.config.max_shift) if self.config.max_shift > 0 else 9999
        cfg.disk_min_radius = 30
        cfg.disk_max_radius = 4000
        cfg.surface_window_size = 256
        cfg.surface_highpass = True
        self._planetary_aligner = _PlanetaryAligner(cfg)
        print(f"[ALIGN] PlanetaryAligner créé — mode={cfg.mode.value}, max_shift={cfg.max_shift}px")

    def set_reference(self, image: np.ndarray, explicit: bool = False):
        """
        Définit l'image de référence

        Args:
            image: Image de référence
            explicit: True si définie par update_alignment_reference() (préserve entre buffers)
        """
        self.reference_is_explicit = explicit

        self._ref_fft_cache = None  # Invalider le cache FFT à chaque nouvelle référence
        self._ref_fft_shape = None

        if self.config.align_mode in (2, 3):
            # Modes disk/hybrid : référence gérée par PlanetaryAligner
            if self._planetary_aligner is None:
                self._create_planetary_aligner()
            if self._planetary_aligner is not None:
                self._planetary_aligner.set_reference(image)
            self.reference = None
            return

        # Mode 1 (surface/phase) : grayscale normalisé + ROI
        if len(image.shape) == 3:
            if image.dtype != np.uint8:
                img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_normalized = image
            self.reference = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2GRAY)
        else:
            if image.dtype != np.uint8:
                self.reference = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                self.reference = image

        # Extraire ROI pour alignement
        self.reference = self._extract_align_roi(self.reference)
    
    def _extract_align_roi(self, image: np.ndarray) -> np.ndarray:
        """Extrait ROI central pour alignement (utilise le même ROI que le scoring)"""
        # OPTIMISATION : Utiliser score_roi_percent pour scoring ET alignement
        # En imagerie planétaire, la cible est au centre → même zone pour les deux
        if self.config.score_roi_percent < 100.0:
            h, w = image.shape[:2]
            margin_x = int(w * (100 - self.config.score_roi_percent) / 200)
            margin_y = int(h * (100 - self.config.score_roi_percent) / 200)
            return image[margin_y:h-margin_y, margin_x:w-margin_x]
        return image
    
    def align(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aligne une image sur la référence.
        Dispatche selon config.align_mode :
          1 = surface/phase FFT (défaut)
          2 = disk/Hough (PlanetaryAligner)
          3 = hybride disque+surface (PlanetaryAligner)

        Returns:
            (image_alignée, params)  — en cas d'échec, retourne l'image originale avec align_failed=True
        """
        # ── Modes 2/3 : PlanetaryAligner ────────────────────────────────────
        if self.config.align_mode in (2, 3):
            if not _HAS_PLANETARY:
                # Fallback mode 1
                pass
            else:
                if self._planetary_aligner is None:
                    self._create_planetary_aligner()
                if self._planetary_aligner is not None:
                    aligned, params, success = self._planetary_aligner.align(image)
                    if not success:
                        return image.copy(), {'dx': 0.0, 'dy': 0.0, 'align_failed': True}
                    return aligned, params

        # ── Mode 1 : surface/phase FFT ───────────────────────────────────────
        if self.reference is None:
            self.set_reference(image)
            return image.copy(), {'dx': 0.0, 'dy': 0.0}

        # Convertir en grayscale avec normalisation correcte
        if len(image.shape) == 3:
            if image.dtype != np.uint8:
                img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_normalized = image
            gray = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2GRAY)
        else:
            if image.dtype != np.uint8:
                gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                gray = image

        # Extraire ROI
        gray_roi = self._extract_align_roi(gray)

        # Calculer décalage par corrélation de phase ou ECC
        if self.config.align_method == "phase":
            dx, dy = self._phase_correlation(self.reference, gray_roi)
        else:
            dx, dy = self._ecc_alignment(self.reference, gray_roi)

        # ── Validation max_shift ─────────────────────────────────────────────
        max_s = self.config.max_shift
        if max_s > 0:
            shift = float(np.sqrt(dx * dx + dy * dy))
            if shift > max_s:
                print(f"[ALIGN] Décalage rejeté: {shift:.1f}px > max {max_s:.0f}px — frame non alignée")
                return image.copy(), {'dx': 0.0, 'dy': 0.0, 'align_failed': True}

        # Appliquer translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        return aligned, {'dx': dx, 'dy': dy}
    
    def _phase_correlation(self, ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
        """Corrélation de phase pour trouver le décalage"""
        # Assurer mêmes dimensions
        h = min(ref.shape[0], img.shape[0])
        w = min(ref.shape[1], img.shape[1])
        ref = ref[:h, :w]
        img = img[:h, :w]

        # DEBUG: Log taille ROI entrante (première fois seulement)
        if not hasattr(self, '_logged_roi_size'):
            print(f"[DEBUG ALIGN] ROI entrante pour FFT: {h}×{w}")
            self._logged_roi_size = True

        # OPTIMISATION : Downscale 50% pour accélérer FFT (4× plus rapide)
        # La précision reste excellente car on rescale le décalage détecté
        scale_factor = 0.5
        h_small = int(h * scale_factor)
        w_small = int(w * scale_factor)

        if h_small > 32 and w_small > 32:  # Downscale seulement si ROI assez grand
            ref_small = cv2.resize(ref, (w_small, h_small), interpolation=cv2.INTER_AREA)
            img_small = cv2.resize(img, (w_small, h_small), interpolation=cv2.INTER_AREA)
            h_fft, w_fft = h_small, w_small
            if not hasattr(self, '_logged_downscale'):
                print(f"[DEBUG ALIGN] Downscale activé: {h}×{w} → {h_fft}×{w_fft}")
                self._logged_downscale = True
        else:
            ref_small = ref
            img_small = img
            h_fft, w_fft = h, w
            scale_factor = 1.0  # Pas de rescale si pas de downscale
            if not hasattr(self, '_logged_no_downscale'):
                print(f"[DEBUG ALIGN] Downscale désactivé (ROI trop petit): {h_small}×{w_small}")
                self._logged_no_downscale = True

        # FFT sur images downscalées (ou originales si trop petites)
        # Cache ref_fft : recalculé uniquement si la shape FFT change (set_reference invalide)
        fft_shape = (h_fft, w_fft)
        if self._ref_fft_cache is None or self._ref_fft_shape != fft_shape:
            self._ref_fft_cache = np.fft.fft2(ref_small.astype(np.float32))
            self._ref_fft_shape = fft_shape
        ref_fft = self._ref_fft_cache
        img_fft = np.fft.fft2(img_small.astype(np.float32))

        # Cross-power spectrum
        cross_power = (ref_fft * np.conj(img_fft)) / (np.abs(ref_fft * np.conj(img_fft)) + 1e-10)

        # Inverse FFT
        correlation = np.real(np.fft.ifft2(cross_power))
        correlation = np.fft.fftshift(correlation)

        # Trouver le pic
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)

        dy = peak_idx[0] - h_fft // 2
        dx = peak_idx[1] - w_fft // 2

        # Rescaler le décalage détecté pour l'appliquer à l'image complète
        dx = dx / scale_factor
        dy = dy / scale_factor

        return float(dx), float(dy)
    
    def _ecc_alignment(self, ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
        """Alignement par Enhanced Correlation Coefficient"""
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            
            _, warp_matrix = cv2.findTransformECC(
                ref.astype(np.float32),
                img.astype(np.float32),
                warp_matrix,
                cv2.MOTION_TRANSLATION,
                criteria
            )
            
            return float(warp_matrix[0, 2]), float(warp_matrix[1, 2])
        except cv2.error:
            return 0.0, 0.0
    
    def reset(self):
        """Réinitialise la référence (sauf si explicite)"""
        if not self.reference_is_explicit:
            self.reference = None
            self.reference_is_explicit = False
            self._ref_fft_cache = None
            self._ref_fft_shape = None
            if self._planetary_aligner is not None:
                self._planetary_aligner.reset()


class LuckyImagingStacker:
    """
    Stacker Lucky Imaging principal
    
    Workflow:
    1. add_frame() : Ajoute frame au buffer + calcule score
    2. is_buffer_full() : Vérifie si buffer prêt
    3. process_buffer() : Sélectionne meilleures + aligne + stack
    4. get_result() : Récupère l'image finale
    """
    
    def __init__(self, config: Optional[LuckyConfig] = None):
        self.config = config if config else LuckyConfig()
        self.config.validate()
        
        # Composants
        self.buffer = FrameBuffer(self.config.buffer_size)
        self.scorer = QualityScorer(self.config)
        self.aligner = FrameAligner(self.config)
        
        # Thread pool pour scoring parallèle
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        # État
        self.is_running = False
        self.last_result: Optional[np.ndarray] = None
        self.last_stack_time = 0.0
        self.total_frames_processed = 0
        self.total_stacks_done = 0
        
        # Historique des scores
        self.score_history: deque = deque(maxlen=self.config.history_size)
        
        # Stats
        self.stats = {
            'frames_total': 0,
            'frames_selected': 0,
            'stacks_done': 0,
            'avg_score': 0.0,
            'best_score': 0.0,
            'worst_score': 0.0,
            'avg_scoring_time_ms': 0.0,
            'avg_stack_time_ms': 0.0,
            'selection_threshold': 0.0,
        }
        
        # Timing
        self._scoring_times: deque = deque(maxlen=100)
        self._stack_times: deque = deque(maxlen=100)

        # Drizzle (optionnel, accumulé cross-buffers)
        self.use_drizzle = False
        self.drizzle_scale = 2.0
        self.drizzle_pixfrac = 0.8
        self.drizzle_kernel = 'square'
        self.drizzle_stacker = None       # DrizzleStackerFast instancié à la demande
        self.last_drizzle_result = None   # Dernier résultat drizzle (float32)
    
    def start(self):
        """Démarre le stacker"""
        self.is_running = True
        self.buffer.clear()
        self.aligner.reset()
        self.score_history.clear()
        self.total_frames_processed = 0
        self.total_stacks_done = 0
        print(f"[LUCKY] Démarré - Buffer: {self.config.buffer_size}, "
              f"Keep: {self.config.keep_percent}%")
    
    def stop(self):
        """Arrête le stacker"""
        self.is_running = False
        # wait=True : s'assurer que tous les futures en cours sont terminés avant de libérer
        self.executor.shutdown(wait=True)
        print(f"[LUCKY] Arrêté - Frames: {self.total_frames_processed}, "
              f"Stacks: {self.total_stacks_done}")

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalise une frame RAW vers float32 [0-255]

        CORRECTION BUG IMAGE BLANCHE:
        Utilise raw_format explicite ou détection intelligente par percentile
        au lieu de frame.max() qui est sensible aux pixels chauds et à la surexposition.

        Args:
            frame: Image brute (uint8, uint16, ou float32)

        Returns:
            Image normalisée en float32 [0-255]
        """
        # Cas 1: Déjà float32 → copier pour éviter corruption
        if frame.dtype == np.float32:
            return frame.astype(np.float32)  # Force nouvelle instance

        # Cas 2: uint8 → conversion directe
        if frame.dtype == np.uint8:
            return frame.astype(np.float32)

        # Cas 3: uint16 (RAW 12/16-bit) → normalisation intelligente
        if frame.dtype == np.uint16:
            # Déterminer le diviseur de normalisation
            divisor = None

            # Option A: Format RAW explicite (recommandé, évite les erreurs)
            if self.config.raw_format is not None:
                raw_fmt = self.config.raw_format.lower()
                if raw_fmt == 'raw12':
                    divisor = 16.0  # 4096 / 256 = 16
                elif raw_fmt == 'raw16':
                    divisor = 256.0  # 65536 / 256 = 256
                else:
                    divisor = 16.0  # Fallback conservateur

                # Log une seule fois
                if not hasattr(self, '_raw_format_logged'):
                    print(f"[LUCKY NORMALIZE] Format RAW explicite: {self.config.raw_format} (diviseur={divisor})")
                    self._raw_format_logged = True

            # Option B: Auto-détection (fallback si raw_format=None)
            else:
                if self.config.raw_normalize_method == 'percentile':
                    # MÉTHODE ROBUSTE: Utiliser np.partition (O(N)) au lieu de
                    # np.percentile (O(N log N)) pour ignorer les pixels chauds / saturés
                    flat = frame.ravel()
                    k = max(0, int(self.config.raw_percentile_detect / 100.0 * flat.size) - 1)
                    ref_val = float(np.partition(flat, k)[k])

                    if ref_val <= 4095:
                        divisor = 16.0
                        detected = 'RAW 12-bit'
                    elif ref_val <= 16383:
                        divisor = 64.0
                        detected = 'RAW 14-bit'
                    else:
                        divisor = 256.0
                        detected = 'RAW 16-bit'

                    # Log une seule fois
                    if not hasattr(self, '_auto_detect_logged'):
                        print(f"[LUCKY NORMALIZE] Auto-détection (percentile {self.config.raw_percentile_detect}%): "
                              f"{detected} (ref_val={ref_val:.1f}, diviseur={divisor})")
                        self._auto_detect_logged = True

                else:
                    # ANCIENNE MÉTHODE (max): Sensible aux pixels chauds et surexposition
                    max_val = frame.max()
                    if max_val <= 4095:
                        divisor = 16.0
                    elif max_val <= 16383:
                        divisor = 64.0
                    else:
                        divisor = 256.0

                    # Log une seule fois avec AVERTISSEMENT
                    if not hasattr(self, '_max_method_warned'):
                        print(f"[LUCKY NORMALIZE] ⚠️  ANCIENNE MÉTHODE (max): max_val={max_val}, diviseur={divisor}")
                        print(f"[LUCKY NORMALIZE] ⚠️  Recommandation: spécifier raw_format='raw12' ou 'raw16' dans la config")
                        self._max_method_warned = True

            # Normaliser
            frame_norm = frame.astype(np.float32) / divisor
            return frame_norm

        # Cas 4: Autre dtype → conversion directe
        return frame.astype(np.float32)

    def add_frame(self, frame: np.ndarray) -> float:
        """
        Ajoute une frame au buffer et calcule son score

        Args:
            frame: Image (RGB ou grayscale, tout dtype)

        Returns:
            Score de qualité de la frame
        """
        if not self.is_running:
            self.start()

        # Normaliser l'image avec la nouvelle méthode robuste
        frame_norm = self._normalize_frame(frame)
        
        # Calculer score (chronométré)
        t0 = time.perf_counter()
        score = self.scorer.score(frame_norm)
        scoring_time = (time.perf_counter() - t0) * 1000
        self._scoring_times.append(scoring_time)
        
        # Ajouter au buffer
        self.buffer.add(frame_norm, score)
        
        # Historique
        if self.config.keep_history:
            self.score_history.append(score)
        
        self.total_frames_processed += 1
        self.stats['frames_total'] = self.total_frames_processed
        
        # Auto-stack si configuré
        if self.config.auto_stack and self.buffer.is_full():
            if self.config.stack_interval == 0 or \
               self.total_frames_processed % self.config.stack_interval == 0:
                self.process_buffer()
        
        return score
    
    def is_buffer_full(self) -> bool:
        """Vérifie si le buffer est plein"""
        return self.buffer.is_full()
    
    def get_buffer_fill(self) -> float:
        """Retourne le taux de remplissage du buffer (0-1)"""
        return len(self.buffer) / self.config.buffer_size
    
    def process_buffer(self) -> Optional[np.ndarray]:
        """
        Traite le buffer : sélectionne les meilleures et les stack
        
        Returns:
            Image stackée ou None si pas assez d'images
        """
        if len(self.buffer) < 2:
            return None
        
        t0 = time.perf_counter()

        # 1. Déterminer combien d'images garder
        keep_count = self.config.get_keep_count()
        keep_count = min(keep_count, len(self.buffer))

        # 2. Sélectionner les meilleures
        t_select = time.perf_counter()
        best_indices = self.buffer.get_best_indices(keep_count, self.config.min_score)

        if not best_indices:
            print(f"[LUCKY] Aucune image au-dessus du seuil min_score={self.config.min_score}")
            return None

        # Calculer le seuil de sélection (score de la dernière image sélectionnée)
        buffer_stats = self.buffer.get_statistics()
        if best_indices:
            scores_list = list(self.buffer.scores)
            selected_scores = [scores_list[i] for i in best_indices]
            self.stats['selection_threshold'] = min(selected_scores) if selected_scores else 0

        # 3. Récupérer les frames sélectionnées, triées par score décroissant
        # (frames[0] = meilleure frame → utilisée comme référence d'alignement)
        selected_frames = self.buffer.get_frames_by_indices(best_indices)
        if selected_scores and len(selected_frames) == len(selected_scores):
            selected_frames = [f for _, f in sorted(
                zip(selected_scores, selected_frames), key=lambda x: x[0], reverse=True)]
        time_select = (time.perf_counter() - t_select) * 1000

        print(f"[LUCKY] Sélection: {len(selected_frames)}/{len(self.buffer)} images "
              f"(seuil={self.stats['selection_threshold']:.1f})")

        # 4. Aligner les frames
        t_align = time.perf_counter()
        _align_active = (getattr(self.config, 'align_mode', 1 if self.config.align_enabled else 0) != 0)
        if _align_active and len(selected_frames) > 1:
            aligned_frames, align_params = self._align_frames(selected_frames)
        else:
            aligned_frames = selected_frames
            align_params = [{'dx': 0.0, 'dy': 0.0, 'angle': 0.0}] * len(selected_frames)
        time_align = (time.perf_counter() - t_align) * 1000

        # 5. Stacker (moyenne/médiane/sigma-clip)
        t_stack = time.perf_counter()
        result = self._stack_frames(aligned_frames)
        time_stack = (time.perf_counter() - t_stack) * 1000

        # 6. Drizzle (optionnel, accumule cross-buffers sur la même session)
        time_drizzle = 0.0
        if self.use_drizzle and aligned_frames:
            t_drizzle = time.perf_counter()
            try:
                from .drizzle import DrizzleStackerFast
                if self.drizzle_stacker is None:
                    self.drizzle_stacker = DrizzleStackerFast(
                        scale=self.drizzle_scale,
                        pixfrac=self.drizzle_pixfrac,
                        kernel=self.drizzle_kernel,
                    )
                for frame, params in zip(aligned_frames, align_params):
                    self.drizzle_stacker.add_image(
                        frame.astype(np.float32),
                        dx=params['dx'],
                        dy=params['dy'],
                        angle=params['angle'],
                    )
                drizzle_out = self.drizzle_stacker.combine()
                if drizzle_out is not None:
                    self.last_drizzle_result = drizzle_out  # float32, même plage que frames
            except Exception as _e:
                print(f"[LUCKY DRIZZLE] Erreur: {_e}")
                self.last_drizzle_result = None
            time_drizzle = (time.perf_counter() - t_drizzle) * 1000

        # Timing et stats
        stack_time = (time.perf_counter() - t0) * 1000
        self._stack_times.append(stack_time)

        self.last_result = result
        self.last_stack_time = time.time()
        self.total_stacks_done += 1

        # Mettre à jour stats
        self._update_stats(buffer_stats, len(selected_frames))

        # DEBUG: Afficher décomposition du temps
        _drz_str = f", Drizzle: {time_drizzle:.1f}ms ({self.drizzle_scale}×)" if self.use_drizzle else ""
        print(f"[LUCKY] Stack #{self.total_stacks_done}: {len(selected_frames)} images, "
              f"{stack_time:.1f}ms total")
        print(f"  └─ Sélection: {time_select:.1f}ms, "
              f"Alignement: {time_align:.1f}ms, "
              f"Stack: {time_stack:.1f}ms{_drz_str}")

        # Vider le buffer pour le prochain cycle
        self.buffer.clear()
        # Reset aligner (mais préserve référence explicite si définie par update_alignment_reference)
        self.aligner.reset()

        # OPTIMISATION : Forcer garbage collection immédiat pour libérer RAM
        # Cela évite l'accumulation de mémoire entre les buffers et accélère
        # le remplissage du buffer suivant (pas d'attente du GC automatique)
        gc.collect()

        return result
    
    def _align_frames(self, frames: List[np.ndarray]):
        """
        Aligne une liste de frames.

        Mode 1 (surface/phase) :
          - La première frame devient référence (sauf si référence explicite définie).
        Modes 2/3 (disk/hybrid) :
          - PlanetaryAligner gère la référence en interne sur le premier appel.
          - Toutes les frames sont passées à align() y compris la première.

        Returns:
            (aligned_frames, params_list) — params_list[i] = {'dx', 'dy', 'angle'}
            La frame de référence a params {'dx': 0.0, 'dy': 0.0, 'angle': 0.0}.
        """
        if not frames:
            return frames, []

        align_mode = getattr(self.config, 'align_mode', 1 if self.config.align_enabled else 0)

        if align_mode in (2, 3):
            if not _HAS_PLANETARY:
                # Fallback silencieux vers mode 1 — loggé une seule fois
                if not hasattr(self, '_logged_planetary_fallback'):
                    print("[LUCKY ALIGN] ⚠ Mode disk/hybrid sélectionné mais PlanetaryAligner "
                          "non disponible — fallback mode 1 (phase correlation)")
                    self._logged_planetary_fallback = True
                # Laisser tomber vers le bloc mode 1 ci-dessous
            else:
                # Disk/Hybrid : PlanetaryAligner auto-set référence sur le 1er appel
                aligned = []
                params_list = []
                failed_count = 0
                for frame in frames:
                    aligned_frame, params = self.aligner.align(frame)
                    if params.get('align_failed', False):
                        failed_count += 1
                    else:
                        aligned.append(aligned_frame)
                        params_list.append({
                            'dx': float(params.get('dx', 0.0)),
                            'dy': float(params.get('dy', 0.0)),
                            'angle': float(params.get('angle', 0.0)),
                        })
                if failed_count > 0:
                    print(f"[LUCKY ALIGN] {failed_count}/{len(frames)} frames rejetées "
                          f"(align_failed) — seuil max_shift ou corrélation trop faible")
                if not aligned:
                    # Fallback : frames originales sans transform
                    return list(frames), [{'dx': 0.0, 'dy': 0.0, 'angle': 0.0}] * len(frames)
                return aligned, params_list

        # Mode 1 (surface/phase) — logique d'origine
        if not self.aligner.reference_is_explicit:
            self.aligner.set_reference(frames[0], explicit=False)
            aligned = [frames[0]]
            params_list = [{'dx': 0.0, 'dy': 0.0, 'angle': 0.0}]  # référence = offset nul
            start_idx = 1
        else:
            aligned = []
            params_list = []
            start_idx = 0
            print(f"[LUCKY ALIGN] Utilisation référence explicite pour aligner {len(frames)} frames")

        to_align = frames[start_idx:]
        if len(to_align) < 3 or self.config.align_method != "phase":
            for frame in to_align:
                aligned_frame, params = self.aligner.align(frame)
                aligned.append(aligned_frame)
                params_list.append({
                    'dx': float(params.get('dx', 0.0)),
                    'dy': float(params.get('dy', 0.0)),
                    'angle': float(params.get('angle', 0.0)),
                })
        else:
            # Parallèle
            futures = [self.executor.submit(self.aligner.align, f) for f in to_align]
            for fut in futures:
                aligned_frame, params = fut.result()
                aligned.append(aligned_frame)
                params_list.append({
                    'dx': float(params.get('dx', 0.0)),
                    'dy': float(params.get('dy', 0.0)),
                    'angle': float(params.get('angle', 0.0)),
                })

        return aligned, params_list
    
    def _stack_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Combine les frames selon la méthode configurée"""
        if not frames:
            return None

        if len(frames) == 1:
            return frames[0].copy()

        # Convertir en array 3D/4D
        stack = np.array(frames)

        # DEBUG: Afficher la méthode utilisée (seulement une fois)
        if not hasattr(self, '_stack_method_shown'):
            print(f"[LUCKY _stack_frames] self.config.stack_method = {self.config.stack_method} (type: {type(self.config.stack_method)})")
            self._stack_method_shown = True

        if self.config.stack_method == StackMethod.MEAN:
            result = np.mean(stack, axis=0)

        elif self.config.stack_method == StackMethod.MEDIAN:
            result = np.median(stack, axis=0)

        elif self.config.stack_method == StackMethod.SIGMA_CLIP:
            result = self._sigma_clip_stack(stack)

        else:
            result = np.mean(stack, axis=0)

        return result.astype(np.float32)
    
    def _sigma_clip_stack(self, stack: np.ndarray) -> np.ndarray:
        """Stack avec rejection sigma-clipping"""
        kappa = self.config.sigma_clip_kappa
        
        # Calculer médiane et écart-type
        median = np.median(stack, axis=0)
        std = np.std(stack, axis=0)
        
        # Masquer les outliers
        lower = median - kappa * std
        upper = median + kappa * std
        
        # Créer masque
        mask = (stack >= lower) & (stack <= upper)
        
        # Moyenne des valeurs non-masquées
        masked_sum = np.sum(stack * mask, axis=0)
        masked_count = np.sum(mask, axis=0)
        masked_count = np.maximum(masked_count, 1)  # Éviter division par zéro
        
        return masked_sum / masked_count
    
    def _update_stats(self, buffer_stats: Dict, selected_count: int):
        """Met à jour les statistiques"""
        self.stats['frames_selected'] = selected_count
        self.stats['stacks_done'] = self.total_stacks_done
        self.stats['avg_score'] = buffer_stats['mean']
        self.stats['best_score'] = buffer_stats['max']
        self.stats['worst_score'] = buffer_stats['min']
        
        if self._scoring_times:
            self.stats['avg_scoring_time_ms'] = np.mean(list(self._scoring_times))
        if self._stack_times:
            self.stats['avg_stack_time_ms'] = np.mean(list(self._stack_times))
    
    def get_result(self) -> Optional[np.ndarray]:
        """
        Retourne le dernier résultat stacké.

        Non-destructif : last_result est conservé pour get_preview() et les appels
        successifs. Utiliser total_stacks_done ou le compteur externe pour détecter
        si un nouveau résultat est disponible (pattern déjà utilisé dans
        rpicamera_livestack_advanced.process_frame via _last_stacks_count).

        Returns:
            Copie du dernier résultat stacké, ou None si aucun stack effectué.
        """
        if self.last_result is None:
            return None
        return self.last_result.copy()

    def get_drizzle_result(self) -> Optional[np.ndarray]:
        """
        Retourne le dernier résultat drizzle accumulé (haute résolution).

        Accumule à travers tous les buffers de la session.
        Appeler reset_drizzle() pour réinitialiser entre sessions.

        Returns:
            Image float32 haute résolution (scale× la taille originale), ou None.
        """
        if self.last_drizzle_result is None:
            return None
        return self.last_drizzle_result.copy()

    def reset_drizzle(self):
        """Réinitialise le drizzle stacker (à appeler entre sessions)."""
        if self.drizzle_stacker is not None:
            self.drizzle_stacker.reset()
        self.last_drizzle_result = None
    
    def get_preview(self, as_uint8: bool = True) -> Optional[np.ndarray]:
        """
        Retourne une preview du dernier résultat

        Args:
            as_uint8: Convertir en uint8 pour affichage
        """
        if self.last_result is None:
            return None

        if as_uint8:
            # Convertir en uint8 SANS normalisation pour préserver l'histogramme
            # Les données sont déjà en float32 [0, 255+], on clip et convertit directement
            result = self.last_result.copy()
            return np.clip(result, 0, 255).astype(np.uint8)

        return self.last_result.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques courantes"""
        stats = self.stats.copy()
        stats['buffer_fill'] = len(self.buffer)
        stats['buffer_size'] = self.config.buffer_size
        stats['buffer_percent'] = 100.0 * len(self.buffer) / self.config.buffer_size
        return stats
    
    def get_score_histogram(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne l'histogramme des scores"""
        if not self.score_history:
            return np.array([]), np.array([])
        
        scores = np.array(list(self.score_history))
        hist, bin_edges = np.histogram(scores, bins=bins)
        return hist, bin_edges
    
    def configure(self, **kwargs):
        """
        Configure les paramètres à la volée
        
        Paramètres modifiables:
        - buffer_size: int
        - keep_percent: float
        - keep_count: int
        - min_score: float
        - score_method: str ("laplacian", "gradient", "sobel", "tenengrad")
        - stack_method: str ("mean", "median", "sigma_clip")
        - align_enabled: bool
        - auto_stack: bool
        """
        if 'buffer_size' in kwargs:
            new_size = int(kwargs['buffer_size'])
            print(f"[DEBUG LUCKY_IMAGING] buffer_size reçu: {self.config.buffer_size} → {new_size}")
            if new_size != self.config.buffer_size:
                old_buffer_size = self.config.buffer_size
                self.config.buffer_size = new_size
                self.buffer = FrameBuffer(new_size)
                print(f"[DEBUG LUCKY_IMAGING] ✓ Buffer recréé: {old_buffer_size} → {new_size} images")
                print(f"[DEBUG LUCKY_IMAGING] Nouveau buffer capacity: {self.buffer.max_size}")
            else:
                print(f"[DEBUG LUCKY_IMAGING] Taille inchangée ({new_size}), buffer non recréé")
        
        if 'keep_percent' in kwargs:
            self.config.keep_percent = float(kwargs['keep_percent'])
        
        if 'keep_count' in kwargs:
            self.config.keep_count = int(kwargs['keep_count']) if kwargs['keep_count'] else None
        
        if 'min_score' in kwargs:
            self.config.min_score = float(kwargs['min_score'])
        
        if 'score_method' in kwargs:
            method_str = kwargs['score_method'].lower()
            method_map = {
                'laplacian': ScoreMethod.LAPLACIAN,
                'gradient': ScoreMethod.GRADIENT,
                'sobel': ScoreMethod.SOBEL,
                'tenengrad': ScoreMethod.TENENGRAD,
                'brenner': ScoreMethod.BRENNER,
                'fft': ScoreMethod.FFT,
                'local_variance': ScoreMethod.LOCAL_VARIANCE,
                'psd': ScoreMethod.PSD,
            }
            if method_str in method_map:
                self.config.score_method = method_map[method_str]
                self.scorer = QualityScorer(self.config)
        
        if 'stack_method' in kwargs:
            method_str = kwargs['stack_method'].lower()
            method_map = {
                'mean': StackMethod.MEAN,
                'median': StackMethod.MEDIAN,
                'sigma_clip': StackMethod.SIGMA_CLIP,
            }
            if method_str in method_map:
                self.config.stack_method = method_map[method_str]
        
        if 'align_mode' in kwargs:
            new_mode = int(kwargs['align_mode'])
            if new_mode != getattr(self.config, 'align_mode', -1):
                self.config.align_mode = new_mode
                self.config.align_enabled = (new_mode != 0)
                self.aligner = FrameAligner(self.config)
                print(f"[LUCKY CONFIG] align_mode: {new_mode}")

        if 'max_shift' in kwargs:
            self.config.max_shift = float(kwargs['max_shift'])

        if 'align_enabled' in kwargs:
            ae = bool(kwargs['align_enabled'])
            self.config.align_enabled = ae
            if not ae:
                self.config.align_mode = 0
            elif getattr(self.config, 'align_mode', 0) == 0:
                self.config.align_mode = 1

        if 'auto_stack' in kwargs:
            self.config.auto_stack = bool(kwargs['auto_stack'])

        if 'score_roi_percent' in kwargs:
            self.config.score_roi_percent = float(kwargs['score_roi_percent'])

        if 'sigma_clip_kappa' in kwargs:
            self.config.sigma_clip_kappa = float(kwargs['sigma_clip_kappa'])

        # Drizzle
        if 'drizzle_enable' in kwargs:
            enable = bool(kwargs['drizzle_enable'])
            if enable != self.use_drizzle:
                self.use_drizzle = enable
                if not enable:
                    self.reset_drizzle()
                print(f"[LUCKY CONFIG] drizzle: {'ON' if enable else 'OFF'}")
        if 'drizzle_scale' in kwargs:
            new_scale = float(kwargs['drizzle_scale'])
            if new_scale != self.drizzle_scale:
                self.drizzle_scale = max(1.0, min(4.0, new_scale))
                # Réinitialiser le stacker si la taille change
                if self.drizzle_stacker is not None:
                    self.drizzle_stacker.reset()
                    self.drizzle_stacker.scale = self.drizzle_scale
                print(f"[LUCKY CONFIG] drizzle_scale: {self.drizzle_scale:.1f}×")
        if 'drizzle_pixfrac' in kwargs:
            self.drizzle_pixfrac = max(0.1, min(1.0, float(kwargs['drizzle_pixfrac'])))
            if self.drizzle_stacker is not None:
                self.drizzle_stacker.pixfrac = self.drizzle_pixfrac
        if 'drizzle_kernel' in kwargs:
            k = str(kwargs['drizzle_kernel']).lower()
            if k in ('point', 'square', 'gaussian'):
                self.drizzle_kernel = k
                if self.drizzle_stacker is not None:
                    self.drizzle_stacker.kernel = k

        # Nouveaux paramètres de normalisation RAW (correction bug image blanche)
        if 'raw_format' in kwargs:
            self.config.raw_format = kwargs['raw_format']
            print(f"[LUCKY CONFIG] raw_format défini: {self.config.raw_format}")

        if 'raw_normalize_method' in kwargs:
            self.config.raw_normalize_method = kwargs['raw_normalize_method']

        if 'raw_percentile_detect' in kwargs:
            self.config.raw_percentile_detect = float(kwargs['raw_percentile_detect'])

    def update_alignment_reference(self, reference_image: np.ndarray):
        """
        Met à jour la référence d'alignement sans réinitialiser le stacker

        Utilisé pour aligner les stacks successifs sur le résultat cumulatif
        au lieu de réinitialiser la référence à chaque buffer.

        Args:
            reference_image: Nouvelle image de référence (résultat cumulatif)
        """
        if self.config.align_enabled:
            # Marquer comme référence explicite pour préserver entre buffers
            self.aligner.set_reference(reference_image, explicit=True)
            print(f"[LUCKY] Référence d'alignement mise à jour (dérive compensée entre buffers)")

    def reset(self):
        """Réinitialise le stacker (garde la config)"""
        self.buffer.clear()
        self.aligner.reset()
        self.score_history.clear()
        self.last_result = None
        self._scoring_times.clear()
        self._stack_times.clear()
        self.stats = {
            'frames_total': 0,
            'frames_selected': 0,
            'stacks_done': 0,
            'avg_score': 0.0,
            'best_score': 0.0,
            'worst_score': 0.0,
            'avg_scoring_time_ms': 0.0,
            'avg_stack_time_ms': 0.0,
            'selection_threshold': 0.0,
        }
        print("[LUCKY] Reset")


# =============================================================================
# Pool Élite — stacker alternatif (RGB8 uniquement)
# =============================================================================

class ElitePoolStacker:
    """
    Mode 'Pool Élite' pour Lucky Stack RGB8.

    Différences avec LuckyImagingStacker :
    - Buffer fixe : les N meilleures frames sont conservées indéfiniment.
    - Entrée conditionnelle : une frame ne remplace une autre que si son score
      est supérieur au seuil (min ou mean du pool).
    - Alignement à l'entrée sur une référence fixe (choisie après warmup).
    - Stack périodique (timer background), pas déclenché par le remplissage.
    - Sigma-clipping des scores optionnel avant chaque stack.
    - RGB8/YUV uniquement (pas de normalisation RAW).

    API compatible avec LuckyImagingStacker pour intégration transparente
    dans rpicamera_livestack_advanced.py.
    """

    def __init__(self, config: Optional['LuckyConfig'] = None):
        self.config = config if config else LuckyConfig()

        # Pool min-heap
        self._pool = ElitePool(
            max_size   = self.config.elite_pool_size,
            entry_mode = self.config.elite_entry_mode,
        )

        # Scorer et aligner (réutilisent l'infrastructure existante)
        self.scorer  = QualityScorer(self.config)
        self.aligner = FrameAligner(self.config)

        # Warmup : accumulation pour choisir la référence d'alignement
        self._warmup_frames: List[Tuple[float, np.ndarray]] = []
        self._warmup_needed  = max(5, min(20, self.config.elite_pool_size // 5))
        self._reference_ready = False

        # État
        self._is_running = False
        self._phase      = "waiting"   # "waiting" | "filling" | "active"
        self.total_frames_processed = 0
        self.total_stacks_done      = 0
        self._frames_accepted       = 0

        # Résultat
        self.last_result: Optional[np.ndarray] = None
        self._result_lock    = threading.Lock()
        self._last_stack_time: float = 0.0
        self._last_clipped_count: int = 0

        # Thread de stack périodique
        self._stop_event  = threading.Event()
        self._stack_now   = threading.Event()
        self._stack_thread: Optional[threading.Thread] = None

        # Stats (clés compatibles LuckyImagingStacker + clés elite)
        self.stats: Dict[str, Any] = {
            'buffer_fill':   0,
            'buffer_size':   self.config.elite_pool_size,
            'avg_score':     0.0,
            'stacks_done':   0,
            'frames_selected': 0,
            'buffer_mode':   'elite',
            'phase':         'waiting',
        }

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self):
        self._is_running = True
        self._phase      = "filling"
        self._stop_event.clear()
        self._last_stack_time = time.time()
        self._stack_thread = threading.Thread(
            target=self._stack_loop, name="ElitePoolStack", daemon=True
        )
        self._stack_thread.start()
        print(f"[ELITE POOL] Démarré — pool={self.config.elite_pool_size} frames, "
              f"intervalle={self.config.elite_stack_interval}s, "
              f"critère={'moyenne' if self.config.elite_entry_mode == 'mean' else 'minimum'}, "
              f"warmup={self._warmup_needed} frames")

    def stop(self):
        self._is_running = False
        self._stop_event.set()
        self._stack_now.set()
        if self._stack_thread is not None:
            self._stack_thread.join(timeout=5.0)
        print(f"[ELITE POOL] Arrêté — Frames: {self.total_frames_processed}, "
              f"Stacks: {self.total_stacks_done}")

    def reset(self):
        """Vide le pool et choisit une nouvelle référence d'alignement."""
        self._pool.clear()
        self._warmup_frames   = []
        self._reference_ready = False
        self._phase           = "filling"
        self._frames_accepted = 0
        self._last_stack_time = time.time()
        # Supprimer la référence (même si explicite)
        self.aligner.reference             = None
        self.aligner.reference_is_explicit = False
        self.aligner.reset()
        with self._result_lock:
            self.last_result = None
        print("[ELITE POOL] Pool réinitialisé — nouvelle référence à choisir")

    # ── Ajout de frames ───────────────────────────────────────────────────────

    def add_frame(self, frame: np.ndarray) -> float:
        """
        Tente d'ajouter une frame dans le pool.
        API compatible avec LuckyImagingStacker.add_frame().
        Retourne le score de qualité.
        Note: n'accepte que du RGB8/uint8 — pas de normalisation RAW.
        """
        if not self._is_running:
            return 0.0

        self.total_frames_processed += 1
        score = self.scorer.score(frame)

        # Phase warmup : accumulation pour choisir la référence d'alignement
        if not self._reference_ready:
            self._warmup_frames.append((score, frame.copy()))
            if len(self._warmup_frames) >= self._warmup_needed:
                self._init_reference_from_warmup()
            self._update_stats()
            return score

        # Pré-filtrage CPU : si pool plein et score < seuil → skip alignement
        if self._pool.is_full:
            threshold = (self._pool.mean_score
                         if self.config.elite_entry_mode == "mean"
                         else self._pool.min_score)
            if score <= threshold:
                self._update_stats()
                return score

        # Alignement contre référence fixe
        aligned = self._align_frame(frame)
        if aligned is None:
            self._update_stats()
            return score

        # Tentative d'insertion dans le pool
        if self._pool.try_add(score, aligned):
            self._frames_accepted += 1

        if self._pool.is_full and self._phase == "filling":
            self._phase = "active"

        self._update_stats()
        return score

    def _init_reference_from_warmup(self):
        """Initialise la référence d'alignement avec la meilleure frame warmup."""
        best_score, best_frame = max(self._warmup_frames, key=lambda x: x[0])
        self.aligner.set_reference(best_frame, explicit=True)
        self._reference_ready = True
        print(f"[ELITE POOL] Référence définie (score={best_score:.4f}, "
              f"{len(self._warmup_frames)} frames warmup)")
        # Insérer les frames warmup dans le pool (alignées contre la référence)
        for ws, wf in self._warmup_frames:
            aligned = self._align_frame(wf)
            if aligned is not None and self._pool.try_add(ws, aligned):
                self._frames_accepted += 1
        self._warmup_frames = []
        if self._pool.is_full:
            self._phase = "active"

    def _align_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Aligne une frame contre la référence fixe. Retourne None si échec."""
        if not self.config.align_enabled or not self._reference_ready:
            return frame.copy()
        try:
            aligned, params = self.aligner.align(frame)
            if params.get('align_failed', False):
                return None
            return aligned
        except Exception as e:
            print(f"[ELITE POOL] Alignement échoué: {e}")
            return None

    # ── Thread de stack périodique ────────────────────────────────────────────

    def _stack_loop(self):
        """Thread de stack périodique (s'exécute en arrière-plan)."""
        while not self._stop_event.is_set():
            self._stack_now.wait(timeout=float(self.config.elite_stack_interval))
            self._stack_now.clear()
            if self._stop_event.is_set():
                break
            if self._pool.size >= 3:
                self._do_stack()

    def _do_stack(self):
        """Effectue le stack de tous les frames du pool."""
        try:
            # Sigma-clipping des scores (optionnel)
            if self.config.elite_score_clip:
                frames, n_clipped = self._pool.get_frames_clipped(self.config.elite_score_kappa)
                self._last_clipped_count = n_clipped
            else:
                frames = [f for _, f in self._pool.get_frames_and_scores()]
                self._last_clipped_count = 0

            if not frames:
                return

            # Conserver en float32 [0, 255] — cohérent avec LuckyImagingStacker._normalize_frame()
            # (get_preview_for_display inject last_lucky_result dans session.stacker qui attend [0-255])
            arr = np.stack([f.astype(np.float32) for f in frames], axis=0)

            method = self.config.stack_method
            if method == StackMethod.MEDIAN:
                result = np.median(arr, axis=0)
            elif method == StackMethod.SIGMA_CLIP:
                result = self._sigma_clip_stack(arr, self.config.sigma_clip_kappa)
            else:  # MEAN
                result = arr.mean(axis=0)

            with self._result_lock:
                self.last_result = result   # float32 [0, 1]
                self.total_stacks_done += 1
                self._last_stack_time = time.time()

            print(f"[ELITE POOL] Stack #{self.total_stacks_done}: "
                  f"{len(frames)}/{self._pool.size} frames utilisées "
                  f"({self._last_clipped_count} clippées σ), "
                  f"pool={self._pool.size}/{self.config.elite_pool_size}")
            self._update_stats()

        except Exception as e:
            print(f"[ELITE POOL] Erreur stack: {e}")

    @staticmethod
    def _sigma_clip_stack(arr: np.ndarray, kappa: float) -> np.ndarray:
        """Sigma-clip pixel-level sur array (N, H, W, C) float32 [0-255]."""
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        mask = np.abs(arr - mean) <= kappa * std
        out  = np.where(mask, arr, np.nan)
        with np.errstate(all='ignore'):
            result = np.nanmean(out, axis=0)
        return np.nan_to_num(result, nan=mean)

    # ── API compatible LuckyImagingStacker ────────────────────────────────────

    def get_result(self) -> Optional[np.ndarray]:
        """Retourne le dernier stack (non-destructif). Compatible LuckyImagingStacker."""
        with self._result_lock:
            if self.last_result is None:
                return None
            return self.last_result.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques courantes. Compatible LuckyImagingStacker."""
        self._update_stats()
        return self.stats.copy()

    def update_alignment_reference(self, _new_ref: np.ndarray):
        """No-op : la référence Elite est fixée au warmup et ne change pas."""
        pass

    def configure(self, **kwargs):
        """Met à jour la configuration à la volée."""
        if 'elite_pool_size' in kwargs:
            new_size = int(kwargs['elite_pool_size'])
            self.config.elite_pool_size = new_size
            self._pool.resize(new_size)   # ← propager dans le heap existant
        if 'elite_stack_interval' in kwargs:
            self.config.elite_stack_interval = float(kwargs['elite_stack_interval'])
        if 'elite_entry_mode' in kwargs:
            self.config.elite_entry_mode = str(kwargs['elite_entry_mode'])
            self._pool.update_entry_mode(self.config.elite_entry_mode)
        if 'elite_score_clip' in kwargs:
            self.config.elite_score_clip = bool(kwargs['elite_score_clip'])
        if 'elite_score_kappa' in kwargs:
            self.config.elite_score_kappa = float(kwargs['elite_score_kappa'])
        if 'score_method' in kwargs:
            _map = {
                'laplacian':      ScoreMethod.LAPLACIAN,
                'gradient':       ScoreMethod.GRADIENT,
                'sobel':          ScoreMethod.SOBEL,
                'tenengrad':      ScoreMethod.TENENGRAD,
                'local_variance': ScoreMethod.LOCAL_VARIANCE,
                'psd':            ScoreMethod.PSD,
            }
            self.config.score_method = _map.get(str(kwargs['score_method']).lower(),
                                                  ScoreMethod.LAPLACIAN)
            self.scorer = QualityScorer(self.config)
        if 'stack_method' in kwargs:
            _map = {
                'mean':       StackMethod.MEAN,
                'median':     StackMethod.MEDIAN,
                'sigma_clip': StackMethod.SIGMA_CLIP,
            }
            self.config.stack_method = _map.get(str(kwargs['stack_method']).lower(),
                                                  StackMethod.MEAN)

    # ── Interne ───────────────────────────────────────────────────────────────

    def _update_stats(self):
        next_in = max(0.0, self.config.elite_stack_interval
                      - (time.time() - self._last_stack_time))
        self.stats.update({
            # Clés compatibles LuckyImagingStacker
            'buffer_fill':     self._pool.size,
            'buffer_size':     self.config.elite_pool_size,
            'avg_score':       self._pool.mean_score,
            'stacks_done':     self.total_stacks_done,
            'frames_selected': self._frames_accepted,
            # Clés spécifiques Elite
            'buffer_mode':     'elite',
            'phase':           self._phase,
            'min_score':       self._pool.min_score,
            'max_score':       self._pool.max_score,
            'accept_rate':     self._pool.accept_rate,
            'last_clipped':    self._last_clipped_count,
            'next_stack_in':   next_in,
            'total_frames':    self.total_frames_processed,
        })


# =============================================================================
# Wrapper pour intégration RPiCamera
# =============================================================================

class RPiCameraLuckyImaging:
    """
    Wrapper Lucky Imaging pour intégration dans RPiCamera.py
    
    Gère:
    - Configuration via paramètres simples
    - Preview pour affichage PyGame
    - Statistiques pour OSD
    """
    
    def __init__(self, output_dir: str = "/home/admin/stacks/lucky"):
        self.output_dir = output_dir
        self.stacker: Optional[LuckyImagingStacker] = None
        self.config = LuckyConfig()
        
        # État
        self.is_running = False
        self.frame_count = 0
        self.last_preview = None
        self.start_time = None
    
    def configure(self, **kwargs):
        """
        Configure le Lucky Imaging
        
        Paramètres:
        - buffer_size: int (défaut: 100)
        - keep_percent: float (défaut: 10.0)
        - keep_count: int (optionnel, prioritaire sur keep_percent)
        - min_score: float (défaut: 0)
        - score_method: str ("laplacian", "gradient", "sobel", "tenengrad")
        - stack_method: str ("mean", "median", "sigma_clip")
        - align_enabled: bool (défaut: True)
        - auto_stack: bool (défaut: True)
        - score_roi_percent: float (défaut: 50.0)
        """
        if 'buffer_size' in kwargs:
            self.config.buffer_size = int(kwargs['buffer_size'])
        if 'keep_percent' in kwargs:
            self.config.keep_percent = float(kwargs['keep_percent'])
        if 'keep_count' in kwargs:
            self.config.keep_count = int(kwargs['keep_count']) if kwargs['keep_count'] else None
        if 'min_score' in kwargs:
            self.config.min_score = float(kwargs['min_score'])
        if 'score_method' in kwargs:
            method_map = {
                'laplacian':      ScoreMethod.LAPLACIAN,
                'gradient':       ScoreMethod.GRADIENT,
                'sobel':          ScoreMethod.SOBEL,
                'tenengrad':      ScoreMethod.TENENGRAD,
                'local_variance': ScoreMethod.LOCAL_VARIANCE,
                'psd':            ScoreMethod.PSD,
            }
            self.config.score_method = method_map.get(kwargs['score_method'].lower(),
                                                       ScoreMethod.LAPLACIAN)
        if 'stack_method' in kwargs:
            method_map = {
                'mean': StackMethod.MEAN,
                'median': StackMethod.MEDIAN,
                'sigma_clip': StackMethod.SIGMA_CLIP,
            }
            self.config.stack_method = method_map.get(kwargs['stack_method'].lower(),
                                                       StackMethod.MEAN)
        if 'align_mode' in kwargs:
            new_mode = int(kwargs['align_mode'])
            self.config.align_mode = new_mode
            self.config.align_enabled = (new_mode != 0)
            if self.stacker:
                self.stacker.configure(align_mode=new_mode)
        if 'max_shift' in kwargs:
            self.config.max_shift = float(kwargs['max_shift'])
            if self.stacker:
                self.stacker.configure(max_shift=self.config.max_shift)
        if 'align_enabled' in kwargs:
            ae = bool(kwargs['align_enabled'])
            self.config.align_enabled = ae
            if not ae:
                self.config.align_mode = 0
            elif getattr(self.config, 'align_mode', 0) == 0:
                self.config.align_mode = 1
        if 'auto_stack' in kwargs:
            self.config.auto_stack = bool(kwargs['auto_stack'])
        if 'score_roi_percent' in kwargs:
            self.config.score_roi_percent = float(kwargs['score_roi_percent'])
    
    def start(self):
        """Démarre le Lucky Imaging"""
        self.config.validate()
        self.stacker = LuckyImagingStacker(self.config)
        self.stacker.start()
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"[LUCKY] Session démarrée")
        print(f"  Buffer: {self.config.buffer_size} images")
        print(f"  Garder: {self.config.keep_percent}%")
        print(f"  Méthode score: {self.config.score_method.value}")
        print(f"  Méthode stack: {self.config.stack_method.value}")
    
    def stop(self):
        """Arrête le Lucky Imaging"""
        if self.stacker:
            self.stacker.stop()
        self.is_running = False
        print(f"[LUCKY] Session arrêtée - {self.frame_count} frames traitées")
    
    def process_frame(self, image_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Traite une frame
        
        Args:
            image_data: Image de la caméra
        
        Returns:
            Image stackée si disponible, None sinon
        """
        if not self.is_running or self.stacker is None:
            return None
        
        # Ajouter la frame
        score = self.stacker.add_frame(image_data)
        self.frame_count += 1
        
        # Mettre à jour preview si nouveau résultat
        if self.stacker.last_result is not None:
            self.last_preview = self.stacker.get_preview(as_uint8=True)
            return self.last_preview
        
        return None
    
    def get_preview_surface(self, pygame_module, target_size=None):
        """
        Retourne une surface PyGame pour affichage
        
        Args:
            pygame_module: Module pygame
            target_size: (width, height) optionnel
        
        Returns:
            pygame.Surface ou None
        """
        if self.last_preview is None:
            return None
        
        try:
            preview = self.last_preview
            
            if len(preview.shape) == 3:
                surface = pygame_module.surfarray.make_surface(
                    preview.transpose(1, 0, 2)
                )
            else:
                # Grayscale -> RGB
                preview_rgb = np.stack([preview, preview, preview], axis=-1)
                surface = pygame_module.surfarray.make_surface(
                    preview_rgb.transpose(1, 0, 2)
                )
            
            if target_size:
                surface = pygame_module.transform.scale(surface, target_size)
            
            return surface
        except Exception as e:
            print(f"[LUCKY] Erreur preview: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques pour affichage OSD"""
        if self.stacker is None:
            return {}
        
        stats = self.stacker.get_stats()
        
        # Ajouter infos supplémentaires
        elapsed = time.time() - self.start_time if self.start_time else 0
        stats['elapsed_time'] = elapsed
        stats['fps'] = self.frame_count / elapsed if elapsed > 0 else 0
        
        return stats
    
    def get_osd_text(self) -> List[str]:
        """Retourne les lignes de texte pour l'OSD"""
        stats = self.get_stats()
        
        lines = [
            f"LUCKY: {stats.get('buffer_fill', 0)}/{stats.get('buffer_size', 0)}",
            f"Score: {stats.get('avg_score', 0):.0f} (>{stats.get('selection_threshold', 0):.0f})",
            f"Stacks: {stats.get('stacks_done', 0)}",
            f"FPS: {stats.get('fps', 0):.1f}",
        ]
        
        return lines
    
    def save_result(self, filename: str = None):
        """Sauvegarde le dernier résultat"""
        if self.stacker is None or self.stacker.last_result is None:
            print("[LUCKY] Pas de résultat à sauvegarder")
            return
        
        import os
        from datetime import datetime
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lucky_{timestamp}"
        
        # Sauvegarder en PNG SANS normalisation pour préserver l'histogramme
        result = self.stacker.last_result
        # Clip directement à [0, 255] sans normaliser
        result_8bit = np.clip(result, 0, 255).astype(np.uint8)
        
        png_path = os.path.join(self.output_dir, f"{filename}.png")
        
        if len(result_8bit.shape) == 3:
            cv2.imwrite(png_path, cv2.cvtColor(result_8bit, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(png_path, result_8bit)
        
        print(f"[LUCKY] Sauvegardé: {png_path}")
        
        # Sauvegarder stats
        stats_path = os.path.join(self.output_dir, f"{filename}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write("=== LUCKY IMAGING STATS ===\n\n")
            for key, value in self.get_stats().items():
                f.write(f"{key}: {value}\n")
        
        return png_path
    
    def reset(self):
        """Réinitialise la session"""
        if self.stacker:
            self.stacker.reset()
        self.frame_count = 0
        self.last_preview = None
        self.start_time = time.time()


# =============================================================================
# Factory function
# =============================================================================

def create_lucky_session(preset: str = 'default', **kwargs) -> RPiCameraLuckyImaging:
    """
    Crée une session Lucky Imaging avec preset
    
    Presets:
    - 'default': Équilibré (100 images, 10%)
    - 'fast': Rapide (50 images, 20%)
    - 'quality': Haute qualité (200 images, 5%)
    - 'aggressive': Très sélectif (200 images, 1%)
    
    Args:
        preset: Nom du preset
        **kwargs: Paramètres supplémentaires
    
    Returns:
        RPiCameraLuckyImaging configuré
    """
    session = RPiCameraLuckyImaging()
    
    # Appliquer preset
    if preset == 'fast':
        session.configure(
            buffer_size=50,
            keep_percent=20.0,
            score_method='laplacian',
            stack_method='mean',
        )
    elif preset == 'quality':
        session.configure(
            buffer_size=200,
            keep_percent=5.0,
            score_method='tenengrad',
            stack_method='sigma_clip',
        )
    elif preset == 'aggressive':
        session.configure(
            buffer_size=200,
            keep_percent=1.0,
            score_method='tenengrad',
            stack_method='sigma_clip',
        )
    else:  # default
        session.configure(
            buffer_size=100,
            keep_percent=10.0,
            score_method='laplacian',
            stack_method='mean',
        )
    
    # Appliquer paramètres supplémentaires
    session.configure(**kwargs)
    
    return session


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test Lucky Imaging ===\n")
    
    np.random.seed(42)
    
    # Créer images de test avec qualité variable
    def create_test_image(quality: float) -> np.ndarray:
        """Crée une image test avec netteté variable"""
        h, w = 256, 256
        
        # Image de base : disque avec détails
        y, x = np.ogrid[:h, :w]
        center = (128, 128)
        radius = 80
        
        disk = ((x - center[0])**2 + (y - center[1])**2 <= radius**2).astype(np.float32)
        
        # Ajouter détails (taches)
        for _ in range(5):
            tx = np.random.randint(60, 196)
            ty = np.random.randint(60, 196)
            tr = np.random.randint(5, 15)
            spot = ((x - tx)**2 + (y - ty)**2 <= tr**2).astype(np.float32)
            disk -= spot * 0.3
        
        disk = np.clip(disk * 200 + 30, 0, 255)
        
        # Appliquer flou selon qualité (moins de flou = meilleure qualité)
        blur_sigma = (1.0 - quality) * 5 + 0.5
        disk = cv2.GaussianBlur(disk, (0, 0), blur_sigma)
        
        # Ajouter bruit
        noise = np.random.normal(0, 5, disk.shape)
        disk = np.clip(disk + noise, 0, 255)
        
        return disk.astype(np.float32)
    
    # Test 1: Scoring
    print("--- Test Scoring ---")
    config = LuckyConfig()
    scorer = QualityScorer(config)
    
    for q in [0.2, 0.5, 0.8, 1.0]:
        img = create_test_image(q)
        score = scorer.score(img)
        print(f"  Qualité={q:.1f} -> Score={score:.1f}")
    
    # Test 2: Buffer et sélection
    print("\n--- Test Buffer et Sélection ---")
    config = LuckyConfig(buffer_size=20, keep_percent=25.0)
    stacker = LuckyImagingStacker(config)
    stacker.start()
    
    # Ajouter 20 images avec qualité aléatoire
    qualities = np.random.uniform(0.2, 1.0, 20)
    for i, q in enumerate(qualities):
        img = create_test_image(q)
        score = stacker.add_frame(img)
        print(f"  Frame {i+1}: qualité={q:.2f}, score={score:.1f}")
    
    # Le buffer devrait être plein et auto-stacké
    stats = stacker.get_stats()
    print(f"\n  Résultat: {stats['frames_selected']}/{stats['buffer_fill']} images sélectionnées")
    print(f"  Seuil de sélection: {stats['selection_threshold']:.1f}")
    print(f"  Score moyen: {stats['avg_score']:.1f}")
    print(f"  Temps scoring: {stats['avg_scoring_time_ms']:.2f}ms")
    print(f"  Temps stack: {stats['avg_stack_time_ms']:.2f}ms")
    
    # Test 3: Wrapper RPiCamera
    print("\n--- Test Wrapper RPiCamera ---")
    session = create_lucky_session('quality')
    session.start()
    
    for i in range(50):
        q = np.random.uniform(0.3, 1.0)
        img = create_test_image(q)
        result = session.process_frame(img)
        
        if result is not None:
            print(f"  Stack généré à la frame {i+1}")
    
    osd = session.get_osd_text()
    print(f"\n  OSD: {osd}")
    
    session.stop()
    
    print("\n=== Tests terminés ===")
