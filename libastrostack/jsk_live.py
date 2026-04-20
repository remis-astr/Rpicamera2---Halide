#!/usr/bin/env python3
"""
JSK LIVE - Module de traitement HDR et Denoise pour RPi5

Pipeline: RAW12 -> Stack (1-4) -> HDR -> Debayer -> Denoise -> Stretch -> 8bits

Adapté du code HDR original (cupy) pour fonctionner sur RPi5 (numpy CPU)
"""

import numpy as np
import cv2
import threading
import ctypes
import os

# ============================================================================
# Chargement optionnel du pipeline Halide AOT (JSK accéléré ×16)
# ============================================================================
_halide_available = False
_hlib = None

def _load_halide():
    global _halide_available, _hlib
    _so = os.path.join(os.path.dirname(__file__), "halide", "jsk_halide.so")
    if not os.path.isfile(_so):
        return
    try:
        lib = ctypes.CDLL(_so)
        _argtypes_3 = [
            ctypes.POINTER(ctypes.c_uint16),  # input  (H×W uint16, 12-bit ×16)
            ctypes.POINTER(ctypes.c_uint8),   # output (planaire 3×H×W uint8)
            ctypes.c_int, ctypes.c_int,       # w, h
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # r_gain, g_gain, b_gain (×256)
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # w0, w1, w2
        ]
        lib.jsk_hdr_median3.restype  = ctypes.c_int
        lib.jsk_hdr_median3.argtypes = _argtypes_3
        lib.jsk_hdr_mean3.restype    = ctypes.c_int
        lib.jsk_hdr_mean3.argtypes   = _argtypes_3
        lib.jsk_hdr_median2.restype  = ctypes.c_int
        lib.jsk_hdr_median2.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,  # w0, w1 seulement
        ]
        lib.jsk_denoise_median3x3.restype  = ctypes.c_int
        lib.jsk_denoise_median3x3.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # input  (H×W, canal unique)
            ctypes.POINTER(ctypes.c_uint8),  # output (H×W, canal unique)
            ctypes.c_int, ctypes.c_int,      # w, h
        ]
        lib.jsk_denoise_guided.restype  = ctypes.c_int
        lib.jsk_denoise_guided.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # input  (W×H×3) planaire
            ctypes.POINTER(ctypes.c_uint8),  # output (W×H×3) planaire
            ctypes.c_int, ctypes.c_int,      # w, h
            ctypes.c_float,                  # eps (régularisation, espace [0, 255²])
        ]
        _hlib = lib
        _halide_available = True
    except Exception:
        pass

_load_halide()


def _halide_hdr(fn, raw12, w0=100, w1=100, w2=100):
    """
    Appelle un pipeline Halide AOT.
    Entrée  : RAW12 uint16 espace 12-bit (0-4095), shape (H,W).
    Sortie  : RGB uint8 C-contiguous (H,W,3) via cv2.merge des plans R/G/B.
    Gains AWB = 1.0 (appliqués ensuite par apply_color_contrast via LUT).
    """
    h, w = raw12.shape
    # Halide attend CSI-2 ×16 (0-65520) : scale 12-bit → ×16 en place
    raw16 = np.ascontiguousarray(raw12 << 4)
    out_planar = np.empty((3, h, w), dtype=np.uint8)
    ret = fn(
        raw16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out_planar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w, h,
        256, 256, 256,  # gains neutres (AWB géré par LUT après)
        int(w0), int(w1), int(w2),
    )
    if ret != 0:
        raise RuntimeError(f"Halide pipeline erreur: {ret}")
    # cv2.merge produit (H,W,3) C-contiguous — ~8× plus rapide que ascontiguousarray(transpose)
    return cv2.merge([out_planar[0], out_planar[1], out_planar[2]])


def _halide_guided_filter(image, eps=200.0):
    """
    Guided filter Halide (radius=8) sur image RGB (H,W,3) uint8.
    Préserve les bords, lisse le bruit texturé.
    ~30 ms pour 3840×2160 vs ~500 ms bilateral d=10 (×16 speedup).

    eps [0, 255²] : régularisation
        50  → forte préservation des bords (lissage minimal)
        200 → équilibre (défaut)
        1000 → lissage agressif
    """
    h, w = image.shape[:2]
    # Convertir (H,W,3) interleaved → (3,H,W) planaire pour Halide
    # np.ascontiguousarray force une copie — inévitable pour le layout change
    in_planar  = np.ascontiguousarray(image.transpose(2, 0, 1))  # (3,H,W)
    out_planar = np.empty((3, h, w), dtype=np.uint8)
    ret = _hlib.jsk_denoise_guided(
        in_planar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out_planar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w, h,
        ctypes.c_float(float(eps)),
    )
    if ret != 0:
        raise RuntimeError(f"Halide guided filter erreur: {ret}")
    return cv2.merge([out_planar[0], out_planar[1], out_planar[2]])


def _halide_median3x3_interleaved(image):
    """
    Médiane 3×3 Halide sur image RGB (H,W,3) uint8.
    Traite les 3 canaux séparément (chaque canal est C-contiguous dans
    le layout planaire intermédiaire) puis recompose via cv2.merge.
    ~19 ms pour 3840×2160 vs ~83 ms OpenCV (×4 speedup).
    """
    h, w = image.shape[:2]
    out_planar = np.empty((3, h, w), dtype=np.uint8)
    for c in range(3):
        ch = np.ascontiguousarray(image[:, :, c])
        ret = _hlib.jsk_denoise_median3x3(
            ch.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out_planar[c].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h,
        )
        if ret != 0:
            raise RuntimeError(f"Halide median3x3 erreur canal {c}: {ret}")
    return cv2.merge([out_planar[0], out_planar[1], out_planar[2]])


# ============================================================================
# HDR Processing - Adapté pour RAW 12 bits et RPi5 (numpy au lieu de cupy)
# ============================================================================

def HDR_compute_12bit(image_12b, method="Median", bits_to_clip=2, type_bayer=cv2.COLOR_BayerRG2RGB, weights=None):
    """
    Crée une image HDR à partir d'une seule frame RAW 12 bits.
    Génère (bits_to_clip + 1) frames virtuelles par retrait successif
    des bits de poids fort, puis les fusionne.

    Avec bits_to_clip=2, on obtient 3 images :
        12-bit : seuil = 4095 (image originale)
        11-bit : seuil = 2047 (1 bit retiré)
        10-bit : seuil = 1023 (2 bits retirés)

    Args:
        image_12b: Image RAW 12 bits (numpy array, 1 canal Bayer)
        method: Méthode de fusion - "Median", "Mean", ou "Mertens"
        bits_to_clip: Nombre de bits de poids fort à retirer (1-3)
        type_bayer: Pattern Bayer pour debayering (cv2.COLOR_BayerXX2RGB)
        weights: Liste de poids (0-100) pour chaque image, longueur = bits_to_clip+1.
                 Utilisé pour la méthode Mean (moyenne pondérée).
                 None = poids égaux.

    Returns:
        Image HDR 8 bits RGB (numpy array)
    """
    image_float = image_12b.astype(np.float32)
    n_images = bits_to_clip + 1

    # Générer les seuils par retrait réel de bits de poids fort
    thresholds = []
    for i in range(n_images):
        bit_depth = 12 - i
        thresholds.append(2 ** bit_depth - 1)

    # Générer les images clippées (du seuil le plus haut au plus bas)
    # np.clip() évite les copies intermédiaires inutiles
    img_list = []
    for thres in thresholds:
        img_8b = (np.clip(image_float, 0, thres) / thres * 255.0).astype(np.uint8)
        img_list.append(img_8b)

    # Préparer les poids normalisés pour la fusion
    if weights is not None:
        w = [max(0, w) for w in weights[:n_images]]
    else:
        w = [100] * n_images
    w_sum = sum(w)
    if w_sum == 0:
        w = [1.0] * n_images
        w_sum = float(n_images)
    w_norm = [float(x) / w_sum for x in w]

    # Fusionner selon la méthode choisie
    if method == "Mertens":
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        HDR_image_gray = np.clip(res_mertens * 255, 0, 255).astype(np.uint8)
    elif method == "Median":
        img_stack = np.stack(img_list, axis=0)
        HDR_image_gray = np.median(img_stack, axis=0).astype(np.uint8)
    elif method == "Mean":
        # Moyenne pondérée par les poids utilisateur
        img_stack = np.stack(img_list, axis=0).astype(np.float32)
        HDR_image_gray = np.average(img_stack, axis=0, weights=w_norm).astype(np.uint8)
    else:
        # Fallback: pas de HDR, juste normalisation
        HDR_image_gray = (image_float / 4095.0 * 255.0).astype(np.uint8)

    # Debayering vers RGB
    HDR_image_rgb = cv2.cvtColor(HDR_image_gray, type_bayer)

    return HDR_image_rgb


def hdr_mean_raw_bayer(image_12b, bits_to_clip=2, weights=None):
    """
    Fusion HDR Mean pondérée en espace Bayer RAW (pas de debayer).
    Retourne un array float32 2D Bayer [0, 4095] pour stacking aval.

    Args:
        image_12b: Image RAW 12 bits (numpy array 2D, pattern Bayer)
        bits_to_clip: Nombre de bits de poids fort à retirer (0-3)
        weights: Liste de poids (0-100) par niveau, longueur = bits_to_clip+1.
                 None = poids égaux.

    Returns:
        Image Bayer float32 [0, 4095]
    """
    if bits_to_clip <= 0:
        return image_12b.astype(np.float32)

    image_float = image_12b.astype(np.float32)
    n_images = bits_to_clip + 1

    # Seuils : 4095, 2047, 1023, 511 ...
    thresholds = [2 ** (12 - i) - 1 for i in range(n_images)]

    # Images clippées normalisées à [0, 4095]
    img_list = []
    for thres in thresholds:
        img_norm = np.clip(image_float, 0, thres) / thres * 4095.0
        img_list.append(img_norm)

    # Poids normalisés
    if weights is not None:
        w = [max(0, v) for v in weights[:n_images]]
    else:
        w = [100] * n_images
    w_sum = sum(w)
    if w_sum == 0:
        w = [1.0] * n_images
        w_sum = float(n_images)
    w_norm = [float(x) / w_sum for x in w]

    # Moyenne pondérée → reste en float32 Bayer
    img_stack = np.stack(img_list, axis=0)
    return np.average(img_stack, axis=0, weights=w_norm).astype(np.float32)


def HDR_bypass_12bit(image_12b, type_bayer=cv2.COLOR_BayerRG2RGB):
    """
    Bypass HDR: simple conversion 12bits -> 8bits avec debayering.
    Utilisé quand HDR method = OFF.

    Args:
        image_12b: Image RAW 12 bits
        type_bayer: Pattern Bayer

    Returns:
        Image 8 bits RGB
    """
    # Simple scaling 12 bits -> 8 bits
    image_8b = (image_12b.astype(np.float32) / 4095.0 * 255.0).astype(np.uint8)
    # Debayering
    return cv2.cvtColor(image_8b, type_bayer)


# ============================================================================
# Stacking - Moyenne simple de N images
# ============================================================================

def stack_images(images):
    """
    Empile plusieurs images par moyenne.

    Args:
        images: Liste d'images numpy (même dimensions)

    Returns:
        Image moyennée (même type que l'entrée)
    """
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]

    # Stack et moyenne
    stack = np.stack(images, axis=0).astype(np.float32)
    mean_img = np.mean(stack, axis=0)

    # Retourner au type original
    return mean_img.astype(images[0].dtype)


# ============================================================================
# Denoise Filters
# ============================================================================

def denoise_bilateral(image, strength=5):
    """
    Filtre bilatéral - préserve les bords.

    Args:
        image: Image RGB 8 bits
        strength: Intensité (1-10)

    Returns:
        Image filtrée
    """
    # Paramètres adaptés à l'intensité
    d = 5 + strength  # Diamètre du voisinage
    sigma_color = 10 + strength * 8  # Sigma couleur
    sigma_space = 10 + strength * 8  # Sigma spatial

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_gaussian(image, strength=5):
    """
    Flou gaussien - très rapide mais floute les détails.

    Args:
        image: Image RGB 8 bits
        strength: Intensité (1-10)

    Returns:
        Image filtrée
    """
    # Taille du kernel (doit être impair)
    ksize = 3 + (strength // 2) * 2
    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def denoise_median(image, strength=5):
    """
    Filtre médian - bon contre le bruit sel/poivre.

    Args:
        image: Image RGB 8 bits
        strength: Intensité (1-10)

    Returns:
        Image filtrée
    """
    # Taille du kernel (doit être impair)
    ksize = 3 + (strength // 3) * 2
    if ksize % 2 == 0:
        ksize += 1

    return cv2.medianBlur(image, ksize)


def apply_denoise(image, denoise_type, strength=5):
    """
    Applique le filtre de denoise sélectionné.

    denoise_type : 0=OFF  1=Bilateral  2=Gaussian  3=Median  4=Guided(Halide)
    strength     : 1-10
    (FastNLM supprimé : ~15 s/frame, incompatible vidéo)
    """
    if denoise_type == 0:
        return image
    elif denoise_type == 1:
        return denoise_bilateral(image, strength)
    elif denoise_type == 2:
        return denoise_gaussian(image, strength)
    elif denoise_type == 3:
        return denoise_median(image, strength)
    elif denoise_type == 4:
        # Guided filter Halide — eps ∈ [0, 255²], mappé depuis strength 1-10
        # strength=1 → eps=25  (préservation bords forte)
        # strength=5 → eps=200 (équilibre)
        # strength=10→ eps=800 (lissage fort)
        eps = 25.0 * (1.6 ** (strength - 1))
        if _halide_available:
            try:
                return _halide_guided_filter(image, eps=eps)
            except Exception:
                pass  # fallback bilateral
        return denoise_bilateral(image, strength)
    else:
        return image


# ============================================================================
# Pipeline complet JSK LIVE
# ============================================================================

class JSKLiveProcessor:
    """
    Processeur JSK LIVE avec buffer de stacking.
    """

    def __init__(self):
        # Paramètres par défaut
        self.stack_count = 1       # 1-6 images à empiler
        self.hdr_bits_clip = 2     # 1-3 bits à clipper
        self.hdr_method = 1        # 0=OFF, 1=Median, 2=Mean, 3=Mertens
        self.denoise_type = 0      # 0=OFF, 1=Bilateral, 2=Gaussian, 3=Median
        self.denoise_strength = 5  # 1-10
        self.hdr_weights = [100, 100, 100, 100, 100, 100]  # Poids HDR par niveau de bit (0-100)
        self.bayer_pattern = cv2.COLOR_BayerRG2BGR
        self.crop_square = False   # True = crop carré 1080×1080 centré avant traitement

        # Soustraction fond de ciel (avant HDR, espace 12-bit 0-4095)
        self.bg_offset = 0         # 0-1000 ADU 12-bit (0 = désactivé)

        # Couleur & Contraste (LUT par canal pour perf maximale)
        self.color_enabled = False # True = actif
        self.r_gain = 1.0          # 0.5-2.0 (1.0 = neutre)
        self.g_gain = 1.0
        self.b_gain = 1.0
        self.contrast = 1.0        # 0.5-2.0 (1.0 = neutre)
        self._lut_dirty = True     # Recalculer les LUT au prochain appel
        self._lut_r = None
        self._lut_g = None
        self._lut_b = None

        # Stacking RAW incrémental (thread-safe via lock)
        # Le rolling mean est sur les frames RAW brutes (avant HDR/debayer),
        # ce qui garantit que chaque frame capturée contribue au stack,
        # même si le thread HDR+denoise est encore occupé.
        self._raw_buf = []           # Buffer circulaire de frames RAW uint16 (mode Window)
        self._raw_sum = None         # Somme courante float32 (mode Window)
        self._raw_lock = threading.Lock()

        # Mode EMA (Exponential Moving Average)
        self.use_ema = False         # True = EMA, False = Rolling Window
        self._ema = None             # Accumulateur EMA float32 (alpha = 2/(stack_count+1))

        # Méthodes HDR (pour affichage)
        self.hdr_methods = ["OFF", "Median", "Mean", "Mertens"]
        self.denoise_types = ["OFF", "Bilateral", "Gaussian", "Median", "Guided"]

    @property
    def ready(self):
        """True si des données sont disponibles pour process_current()."""
        if self.use_ema:
            return self._ema is not None
        return bool(self._raw_buf)

    def configure(self, **kwargs):
        """Configure les paramètres du processeur."""
        if 'stack_count' in kwargs:
            new_sc = max(1, min(10, int(kwargs['stack_count'])))
            if new_sc != self.stack_count:
                self.stack_count = new_sc
                # Tailler le buffer immédiatement si stack_count diminue
                with self._raw_lock:
                    while len(self._raw_buf) > self.stack_count:
                        oldest = self._raw_buf.pop(0)
                        if self._raw_sum is not None:
                            self._raw_sum -= oldest.astype(np.float32)
            else:
                self.stack_count = new_sc
        if 'hdr_bits_clip' in kwargs:
            self.hdr_bits_clip = max(0, min(3, kwargs['hdr_bits_clip']))
        if 'hdr_method' in kwargs:
            self.hdr_method = max(0, min(3, kwargs['hdr_method']))
        if 'denoise_type' in kwargs:
            self.denoise_type = max(0, min(4, kwargs['denoise_type']))
        if 'denoise_strength' in kwargs:
            self.denoise_strength = max(1, min(10, kwargs['denoise_strength']))
        if 'hdr_weights' in kwargs:
            self.hdr_weights = list(kwargs['hdr_weights'][:4])
        if 'bayer_pattern' in kwargs:
            self.bayer_pattern = kwargs['bayer_pattern']
        if 'crop_square' in kwargs:
            self.crop_square = bool(kwargs['crop_square'])
        # Couleur & Contraste — marquer LUT comme à recalculer si paramètre changé
        if 'color_enabled' in kwargs:
            self.color_enabled = bool(kwargs['color_enabled'])
        if 'r_gain' in kwargs:
            self.r_gain = max(0.5, min(2.0, float(kwargs['r_gain'])))
            self._lut_dirty = True
        if 'g_gain' in kwargs:
            self.g_gain = max(0.5, min(2.0, float(kwargs['g_gain'])))
            self._lut_dirty = True
        if 'b_gain' in kwargs:
            self.b_gain = max(0.5, min(2.0, float(kwargs['b_gain'])))
            self._lut_dirty = True
        if 'contrast' in kwargs:
            self.contrast = max(0.5, min(2.0, float(kwargs['contrast'])))
            self._lut_dirty = True
        if 'bg_offset' in kwargs:
            self.bg_offset = max(0, min(1000, int(kwargs['bg_offset'])))
        if 'use_ema' in kwargs:
            new_ema = bool(kwargs['use_ema'])
            if new_ema != self.use_ema:
                self.use_ema = new_ema
                # Réinitialiser l'accumulateur au changement de mode
                with self._raw_lock:
                    self._ema = None

    def _build_luts(self):
        """Précalcule les LUT R/G/B (contraste + gain canal). Très rapide."""
        x = np.arange(256, dtype=np.float32)
        # Contraste autour du point médian 128
        x = (x - 128.0) * self.contrast + 128.0
        self._lut_r = np.clip(x * self.r_gain, 0, 255).astype(np.uint8)
        self._lut_g = np.clip(x * self.g_gain, 0, 255).astype(np.uint8)
        self._lut_b = np.clip(x * self.b_gain, 0, 255).astype(np.uint8)
        self._lut_dirty = False

    def apply_color_contrast(self, image):
        """Applique la correction couleur/contraste via LUT (quasi sans coût CPU)."""
        if not self.color_enabled:
            return image
        if self._lut_dirty:
            self._build_luts()
        r = cv2.LUT(image[:, :, 0], self._lut_r)
        g = cv2.LUT(image[:, :, 1], self._lut_g)
        b = cv2.LUT(image[:, :, 2], self._lut_b)
        return cv2.merge([r, g, b])

    def clear_buffer(self):
        """Vide le buffer RAW de stacking (Window et EMA)."""
        with self._raw_lock:
            self._raw_buf = []
            self._raw_sum = None
            self._ema = None

    def add_raw_frame(self, raw_frame):
        """
        Ajoute une frame RAW au rolling mean (main thread, chaque frame capturée).
        Thread-safe via lock. O(H×W) ≈ 1ms → non-bloquant.
        Le crop carré est appliqué ici si actif.
        """
        rf = raw_frame
        if self.crop_square:
            h, w = rf.shape[:2]
            crop_size = min(h, w)
            x_start = (w - crop_size) // 2
            if x_start % 2 != 0:
                x_start += 1
            rf = rf[:crop_size, x_start:x_start + crop_size]

        rf_f = rf.astype(np.float32)
        with self._raw_lock:
            if self.use_ema:
                # Mode EMA : pas de buffer, juste une mise à jour O(H×W)
                # alpha = 2/(N+1) → équivalent N frames de poids moyen
                alpha = 2.0 / (self.stack_count + 1)
                if self._ema is None or self._ema.shape != rf_f.shape:
                    self._ema = rf_f.copy()
                else:
                    # ema += alpha * (new - ema)  →  in-place, une seule alloc temp
                    self._ema += alpha * (rf_f - self._ema)
            else:
                # Mode Rolling Window
                # Guard résolution : réinitialiser si la taille change (binning, crop toggle)
                if self._raw_sum is not None and self._raw_sum.shape != rf_f.shape:
                    self._raw_sum = None
                    self._raw_buf = []

                self._raw_buf.append(rf)
                # Retirer les frames excédentaires (stack_count peut avoir changé via configure)
                while len(self._raw_buf) > self.stack_count:
                    oldest = self._raw_buf.pop(0)
                    if self._raw_sum is not None:
                        self._raw_sum -= oldest.astype(np.float32)
                # Ajouter la nouvelle frame à la somme courante
                if self._raw_sum is None:
                    self._raw_sum = rf_f.copy()
                else:
                    self._raw_sum += rf_f

    def process_current(self):
        """
        Traite la moyenne RAW courante : HDR → debayer → denoise → LUT.
        Appelé depuis le thread de traitement (HDR+denoise = 40-150ms).
        Le crop a déjà été appliqué dans add_raw_frame().
        """
        with self._raw_lock:
            if self.use_ema:
                if self._ema is None:
                    return None
                raw_mean = np.clip(self._ema, 0, 4095).astype(np.uint16)
            else:
                if self._raw_sum is None or not self._raw_buf:
                    return None
                n = len(self._raw_buf)
                raw_mean = np.clip(self._raw_sum / n, 0, 4095).astype(np.uint16)

        # Soustraction fond de ciel — sur moyenne RAW avant HDR (espace 12-bit)
        if self.bg_offset > 0:
            raw_mean = np.clip(raw_mean.astype(np.int32) - self.bg_offset, 0, 4095).astype(np.uint16)

        # HDR + debayer sur la moyenne RAW (hors lock)
        if self.hdr_method == 0 or self.hdr_bits_clip == 0:
            rgb_image = HDR_bypass_12bit(raw_mean, self.bayer_pattern)
        elif _halide_available and self.hdr_bits_clip in (1, 2):
            # Chemin Halide AOT : ×16 speedup sur Median/Mean 2-3 niveaux
            w = self.hdr_weights
            w0, w1, w2 = int(w[0]), int(w[1]), (int(w[2]) if len(w) > 2 else 100)
            try:
                if self.hdr_method == 1 and self.hdr_bits_clip == 2:    # Median 3 niveaux
                    rgb_image = _halide_hdr(_hlib.jsk_hdr_median3, raw_mean, w0, w1, w2)
                elif self.hdr_method == 1 and self.hdr_bits_clip == 1:  # Median 2 niveaux
                    rgb_image = _halide_hdr(_hlib.jsk_hdr_median2, raw_mean, w0, w1)
                elif self.hdr_method == 2 and self.hdr_bits_clip == 2:  # Mean 3 niveaux
                    rgb_image = _halide_hdr(_hlib.jsk_hdr_mean3, raw_mean, w0, w1, w2)
                else:
                    # Mean 2 niveaux : pas de variante Halide → fallback numpy
                    raise NotImplementedError
            except Exception:
                method_name = self.hdr_methods[self.hdr_method]
                rgb_image = HDR_compute_12bit(
                    raw_mean, method=method_name,
                    bits_to_clip=self.hdr_bits_clip,
                    type_bayer=self.bayer_pattern,
                    weights=self.hdr_weights
                )
        else:
            method_name = self.hdr_methods[self.hdr_method]
            rgb_image = HDR_compute_12bit(
                raw_mean,
                method=method_name,
                bits_to_clip=self.hdr_bits_clip,
                type_bayer=self.bayer_pattern,
                weights=self.hdr_weights
            )

        # Denoise + LUT couleur/contraste
        result = apply_denoise(rgb_image, self.denoise_type, self.denoise_strength)
        result = self.apply_color_contrast(result)
        return result

    def process(self, raw_frame=None):
        """Backward-compat : add_raw_frame + process_current en un appel."""
        if raw_frame is None:
            return None
        self.add_raw_frame(raw_frame)
        return self.process_current()

    def process_single(self, raw_frame):
        """Alias de process() pour compatibilité."""
        return self.process(raw_frame)


# ============================================================================
# Video Recording Helper
# ============================================================================

class JSKVideoRecorder:
    """
    Enregistreur vidéo MP4 pour JSK LIVE.
    """

    def __init__(self):
        self.writer = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        self.output_path = None
        self.fps = 25  # FPS par défaut

    def start(self, output_path, width, height, fps=25):
        """
        Démarre l'enregistrement.

        Args:
            output_path: Chemin du fichier MP4
            width, height: Dimensions de la vidéo
            fps: Images par seconde
        """
        import time

        self.fps = fps
        self.output_path = output_path
        self.rec_width = width
        self.rec_height = height

        # Codec H264
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if self.writer.isOpened():
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            return True
        else:
            self.writer = None
            return False

    def write_frame(self, frame_rgb):
        """
        Écrit une frame (RGB) dans la vidéo.
        Redimensionne automatiquement si les dimensions ne correspondent pas.
        """
        if not self.is_recording or self.writer is None:
            return False

        # Convertir RGB -> BGR pour OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Redimensionner si les dimensions ne correspondent pas au VideoWriter
        h, w = frame_bgr.shape[:2]
        if w != self.rec_width or h != self.rec_height:
            frame_bgr = cv2.resize(frame_bgr, (self.rec_width, self.rec_height))

        self.writer.write(frame_bgr)
        self.frame_count += 1
        return True

    def stop(self):
        """
        Arrête l'enregistrement et finalise le fichier.
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.is_recording = False
        return self.output_path

    def get_elapsed_time(self):
        """Retourne le temps écoulé en secondes."""
        import time
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def get_elapsed_str(self):
        """Retourne le temps écoulé formaté MM:SS."""
        elapsed = int(self.get_elapsed_time())
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02d}:{seconds:02d}"
