#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session de live stacking - API principale
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import StackingConfig
from .quality import QualityAnalyzer
from .aligner import AdvancedAligner
from .stacker import ImageStacker
from .io import load_image, save_fits, save_png_auto
from .stretch import apply_stretch
from .isp import ISP, ISPCalibrator


class LiveStackSession:
    """
    Session de live stacking - API principale
    
    Usage pour RPiCamera.py:
        config = StackingConfig()
        session = LiveStackSession(config)
        session.start()
        
        # Pour chaque frame de la caméra
        result = session.process_image_data(camera_array)
        
        # Sauvegarder
        session.save_result("output.fit")
        session.stop()
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: StackingConfig instance (ou None pour défaut)
        """
        self.config = config if config else StackingConfig()
        self.config.validate()

        # Composants
        self.quality_analyzer = QualityAnalyzer(self.config.quality)
        self.aligner = AdvancedAligner(self.config)
        self.stacker = ImageStacker(self.config)

        # ISP (Image Signal Processor)
        self.isp = None
        print(f"[DEBUG ISP INIT] isp_enable={self.config.isp_enable}, isp_config_path={self.config.isp_config_path}")
        if self.config.isp_enable:
            if self.config.isp_config_path:
                # Charger config ISP existante
                print(f"[DEBUG ISP] Chargement de {self.config.isp_config_path}...")
                try:
                    self.isp = ISP.load_config(self.config.isp_config_path)
                    print(f"[DEBUG ISP] ✓ Chargé avec succès")
                except Exception as e:
                    print(f"[DEBUG ISP] ✗ Erreur: {e}")
                    # Créer ISP avec config par défaut en cas d'erreur
                    self.isp = ISP()
                    print(f"[DEBUG ISP] → ISP créé avec config par défaut")
            else:
                # Créer ISP avec config par défaut (sera configuré par le panneau RAW)
                print(f"[DEBUG ISP] Pas de chemin de config, création ISP avec config par défaut")
                self.isp = ISP()
                print(f"[DEBUG ISP] ✓ ISP créé avec config par défaut")
        else:
            print(f"[DEBUG ISP] ISP désactivé (isp_enable=False)")

        # État
        self.is_running = False
        self.files_processed = 0
        self.files_rejected = 0
        self.files_failed = 0
        self.is_color = None
        self.rotation_angles = []

        # Compteur pour rafraîchissement preview
        self._frame_count_since_refresh = 0

        # Tracking calibration automatique ISP
        self._isp_calibrated = False
        self._last_isp_calibration_frame = 0
    
    def start(self):
        """Démarre la session"""
        print("\n" + "="*60)
        print(">>> LIBASTROSTACK SESSION")
        print("="*60)

        print("\n[CONFIG]")
        print(f"  - Mode alignement: {self.config.alignment_mode}")
        print(f"  - Contrôle qualité: {'OUI' if self.config.quality.enable else 'NON'}")
        print(f"  - ISP activé: {'OUI' if self.config.isp_enable else 'NON'}")
        if self.config.isp_enable:
            if self.isp:
                print(f"    • Config chargée: {self.config.isp_config_path}")
            else:
                print(f"    • Mode: Auto-calibration")
            if self.config.isp_auto_calibrate_method != 'none':
                print(f"    • Calibration auto: {self.config.isp_auto_calibrate_method}")
                print(f"    • Calibration après: {self.config.isp_auto_calibrate_after} frames")
                if self.config.isp_recalibrate_interval > 0:
                    print(f"    • Recalibration tous les: {self.config.isp_recalibrate_interval} frames")
                if self.config.isp_auto_update_only_wb:
                    print(f"    • Mode: Mise à jour WB uniquement (préserve gamma, contrast, etc.)")
                else:
                    print(f"    • Mode: Remplacement complet de la config ISP")
        print(f"  - Format vidéo: {self.config.video_format or 'Auto-détection'}")
        print(f"  - PNG bit depth: {self.config.png_bit_depth or 'Auto'}")
        print(f"  - Étirement PNG: {self.config.png_stretch_method}")
        print(f"  - FITS: {'Linéaire (RAW)' if self.config.fits_linear else 'Stretched'}")
        print(f"  - Preview refresh: toutes les {self.config.preview_refresh_interval} images")
        print(f"  - Save DNG: {self.config.save_dng_mode}")
        
        if self.config.quality.enable:
            print(f"\n[QC] Seuils:")
            print(f"  - FWHM max: {self.config.quality.max_fwhm} px")
            print(f"  - Ellipticité max: {self.config.quality.max_ellipticity}")
            print(f"  - Étoiles min: {self.config.quality.min_stars}")
            print(f"  - Drift max: {self.config.quality.max_drift} px")
            print(f"  - Netteté min: {self.config.quality.min_sharpness}")
        
        self.is_running = True
        print("\n[OK] Session prête !\n")
    
    def process_image_data(self, image_data):
        """
        Traite une image directement depuis un array (pour RPiCamera)
        
        Args:
            image_data: Array NumPy (float32) RGB ou MONO
        
        Returns:
            Image empilée courante (ou None si rejetée)
        """
        if not self.is_running:
            return None

        # Vérifier limite max_frames (I11)
        if self.config.max_frames > 0 and self.config.num_stacked >= self.config.max_frames:
            return self.stacker.get_result()

        try:
            # Détecter type si première image
            if self.is_color is None:
                self.is_color = len(image_data.shape) == 3

            # 1. Contrôle qualité
            is_good, metrics, reason = self.quality_analyzer.analyze(image_data)

            if not is_good:
                print(f"  [QC] REJECT - {reason}")
                self.files_rejected += 1
                self.config.quality.rejected_images.append(f"frame_{self.files_processed + self.files_rejected}")
                self.config.quality.rejection_reasons[f"frame_{self.files_processed + self.files_rejected}"] = reason
                return None

            # 2. Alignement
            if self.config.alignment_mode.upper() == "OFF" or self.config.alignment_mode.upper() == "NONE":
                aligned = image_data
                params = {'dx': 0, 'dy': 0, 'angle': 0, 'scale': 1.0}
                success = True
            else:
                aligned, params, success = self.aligner.align(image_data)

                if not success:
                    print("  [FAIL] Échec alignement")
                    self.files_rejected += 1
                    return None

            # Enregistrer rotation
            if 'angle' in params:
                self.rotation_angles.append(params['angle'])

            # 3. Empilement
            result = self.stacker.stack(aligned)

            self.files_processed += 1
            self._frame_count_since_refresh += 1

            # 4. Calibration automatique ISP (si activée)
            self._check_and_calibrate_isp_if_needed()

            # Retourner résultat si rafraîchissement nécessaire
            if self._frame_count_since_refresh >= self.config.preview_refresh_interval:
                self._frame_count_since_refresh = 0
                return result

            return None  # Pas de rafraîchissement preview
            
        except Exception as e:
            print(f"[ERROR] {e}")
            self.files_failed += 1
            import traceback
            traceback.print_exc()
            return None
    
    def process_image_file(self, image_path):
        """
        Traite une image depuis un fichier (pour batch processing)
        
        Args:
            image_path: Chemin vers fichier image
        
        Returns:
            Image empilée courante (ou None si rejetée)
        """
        print(f"\n[IMG] {Path(image_path).name}")
        
        image_data = load_image(image_path)
        if image_data is None:
            self.files_failed += 1
            return None
        
        return self.process_image_data(image_data)
    
    def get_current_stack(self):
        """
        Retourne l'image empilée courante
        
        Returns:
            Image empilée (copie) ou None
        """
        return self.stacker.get_result()
    
    def _remove_gradient(self, image, n_tiles=8):
        """
        Supprime le gradient de fond (lumière parasite, vignetage) par estimation
        de fond sur une grille de tuiles (mesh-based background estimation).

        Utilise le 25e percentile de chaque tuile pour estimer le fond du ciel
        tout en rejetant les étoiles et objets brillants. Le fond estimé est
        interpolé bicubiquement vers la résolution complète puis soustrait.

        Args:
            image: float32 (H, W, C) normalisé [0-1]
            n_tiles: nombre de tuiles par axe (ex: 8 → grille 8×8)

        Returns:
            float32 (H, W, C) gradient soustrait, clipé à [0, 1], non renormalisé
        """
        import cv2
        H, W = image.shape[:2]
        is_color = len(image.shape) == 3
        C = image.shape[2] if is_color else 1

        tile_h = max(1, H // n_tiles)
        tile_w = max(1, W // n_tiles)
        n_y = H // tile_h
        n_x = W // tile_w

        if n_y < 2 or n_x < 2:
            return image  # Image trop petite pour la grille demandée

        if is_color:
            bg_small = np.zeros((n_y, n_x, C), dtype=np.float32)
            for i in range(n_y):
                for j in range(n_x):
                    y0, y1 = i * tile_h, min(H, (i + 1) * tile_h)
                    x0, x1 = j * tile_w, min(W, (j + 1) * tile_w)
                    bg_small[i, j] = np.percentile(image[y0:y1, x0:x1], 25, axis=(0, 1))
        else:
            bg_small = np.zeros((n_y, n_x), dtype=np.float32)
            for i in range(n_y):
                for j in range(n_x):
                    y0, y1 = i * tile_h, min(H, (i + 1) * tile_h)
                    x0, x1 = j * tile_w, min(W, (j + 1) * tile_w)
                    bg_small[i, j] = np.percentile(image[y0:y1, x0:x1], 25)

        bg_full = cv2.resize(bg_small, (W, H), interpolation=cv2.INTER_CUBIC)
        bg_full = np.clip(bg_full, 0, None)

        return np.clip(image - bg_full, 0, 1).astype(np.float32)

    def get_preview_png(self):
        """
        Génère preview PNG étiré pour affichage

        Returns:
            Array uint8 ou uint16 (selon config) pour affichage/sauvegarde
        """
        # Compteur pour logs périodiques (toutes les 5 frames)
        if not hasattr(self, '_preview_log_count'):
            self._preview_log_count = 0
        self._preview_log_count += 1
        _do_log = False

        result = self.stacker.get_result()
        if result is None:
            return None

        is_color = len(result.shape) == 3

        if _do_log:
            fmt = self.config.video_format or '?'
            if is_color:
                print(f"\n[SESSION LOG #{self._preview_log_count}] get_preview_png() — format={fmt}")
                print(f"  STACK BRUT  : shape={result.shape} dtype={result.dtype} "
                      f"min={result.min():.4f} max={result.max():.4f} mean={result.mean():.4f}")
                print(f"  Canaux (0,1,2) means: "
                      f"ch0={result[:,:,0].mean():.4f}  "
                      f"ch1={result[:,:,1].mean():.4f}  "
                      f"ch2={result[:,:,2].mean():.4f}")

        # NOUVEAU: Appliquer ISP AVANT stretch (si activé ET format RAW uniquement)
        # Pipeline optimal: Stack (linéaire) → ISP → Stretch → PNG
        # L'ISP ne doit s'appliquer QUE sur RAW12/RAW16, JAMAIS sur YUV420 (déjà traité par ISP hardware)
        is_raw_format = self.config.video_format in ['raw12', 'raw16']
        if self.config.isp_enable and self.isp is not None and is_raw_format:
            # debayer_raw_array retourne ch0=R_physique, ch2=B_physique (COLOR_BayerRG2RGB)
            result = self.isp.process(result, return_uint8=False, swap_rb=False,
                                     format_hint=self.config.video_format)  # Reste en float32
            if _do_log and is_color:
                print(f"  APRÈS ISP   : min={result.min():.4f} max={result.max():.4f} "
                      f"ch0={result[:,:,0].mean():.4f} ch1={result[:,:,1].mean():.4f} ch2={result[:,:,2].mean():.4f}")

        # Appliquer étirement
        is_color = len(result.shape) == 3

        # CORRECTION: Normaliser vers [0-1] AVANT stretch pour éviter clip
        # La fonction stretch_ghs() attend des données en [0-1]
        # IMPORTANT: Les données peuvent venir dans 3 plages différentes:
        #   1. Déjà normalisé [0-1] (après ISP software)
        #   2. YUV420: [0-255] (8-bit)
        #   3. RAW12/16: [0-65535] (16-bit haute dynamique, sans ISP)
        # Il faut détecter automatiquement la plage et normaliser en conséquence

        # Détecter automatiquement la plage de données
        max_val = result.max()
        min_val = result.min()


        # IMPORTANT: Le format explicite (video_format) est prioritaire sur max_val.
        # Sinon, un signal RAW très faible (ex: max=208 après soustraction BL) tombe
        # faussement dans le cas YUV420 (factor=255) car max_val <= 256.
        _vfmt = (self.config.video_format or '').lower()
        _is_raw_format = any(k in _vfmt for k in ('raw12', 'raw16', 'raw10', 'srggb'))

        if max_val <= 1.1 and not _is_raw_format:
            # Cas 1: Déjà normalisé [0-1] (typiquement après ISP software)
            normalization_factor = 1.0
            data_type_str = "Déjà normalisé [0-1]"
        elif _is_raw_format or max_val > 256:
            # Cas 2: Données RAW (format explicite OU max_val > 256)
            # Priorité au format déclaré pour ne pas se tromper à faible signal.
            if 'raw12' in _vfmt or 'srggb12' in _vfmt:
                theoretical_max = 4095.0
                normalization_factor = max(theoretical_max, max_val * 1.02)
                data_type_str = f"RAW 12-bit ({self.config.video_format})"
            elif 'raw10' in _vfmt or 'srggb10' in _vfmt:
                theoretical_max = 1023.0
                normalization_factor = max(theoretical_max, max_val * 1.02)
                data_type_str = f"RAW 10-bit ({self.config.video_format})"
            elif 'raw16' in _vfmt or 'srggb16' in _vfmt:
                normalization_factor = 65535.0
                data_type_str = f"RAW 16-bit ({self.config.video_format})"
            elif max_val > 256:
                # Format non spécifié mais valeurs hautes → RAW auto
                normalization_factor = max_val * 1.02 if max_val > 0 else 65535.0
                data_type_str = f"RAW auto (max={max_val:.0f})"
            else:
                # RAW déclaré mais signal très faible → utiliser le max théorique du format
                normalization_factor = 65535.0
                data_type_str = f"RAW (signal faible, factor fixe)"
        else:
            # Cas 3: Données 8-bit (YUV420) : [0-255]
            normalization_factor = 255.0
            data_type_str = "YUV420 8-bit"

        # Normaliser à [0-1] (requis par apply_stretch)
        if _do_log:
            print(f"  NORMALISATION: {data_type_str}  factor={normalization_factor:.1f}  "
                  f"avant: min={result.min():.1f} max={result.max():.1f}")
        result = result / normalization_factor
        result = np.clip(result, 0, 1)
        if _do_log and is_color:
            print(f"  APRÈS NORM  : ch0={result[:,:,0].mean():.4f} ch1={result[:,:,1].mean():.4f} ch2={result[:,:,2].mean():.4f}")

        # ── Soustraction black level RAW (indépendante de l'ISP software) ──
        # Appliquée uniquement si ISP software désactivé (sinon déjà fait dans ISP)
        # IMPORTANT: bl_adu est exprimé en ADU 12-bit natif (0-4095).
        # Les données peuvent être en espace uint16 shifté ×16 par picamera2 (CSI-2 format),
        # donc on calcule la fraction de dynamique directement sur 4095 (indépendant du shift).
        if is_raw_format and not self.config.isp_enable:
            bl_adu = getattr(self.config, 'raw_black_level', 0)
            if bl_adu > 0:
                # Cohérence d'échelle : result a été divisé par normalization_factor.
                # Les données picamera2 RAW12 sont en espace CSI-2 ×16 (0-65520),
                # donc BL en espace normalisé = (bl_adu * 16) / normalization_factor.
                bl_norm = (bl_adu * 16) / normalization_factor
                result = np.clip(result - bl_norm, 0, None)

        # ── Suppression de gradient de fond ──
        # Estimation du fond du ciel par grille de percentile + interpolation bicubique.
        # Appliquée uniquement sur RAW, après normalisation et soustraction BL.
        if is_raw_format and getattr(self.config, 'gradient_removal', False):
            n_tiles = getattr(self.config, 'gradient_removal_tiles', 8)
            result = self._remove_gradient(result, n_tiles=n_tiles)

        # Configurer le clipping par percentiles selon le format
        clip_low = self.config.png_clip_low
        clip_high = self.config.png_clip_high

        if is_raw_format:
            # RAW: normalisation par percentiles (data brute avec possibles pixels chauds)
            # CORRECTION: En mode RAW, FORCER une normalisation minimale si désactivée
            # Les données RAW linéaires ont besoin d'être normalisées pour le stretch
            if clip_low == 0.0 and clip_high >= 100.0:
                # Utiliser des valeurs par défaut raisonnables pour RAW
                clip_low = 0.1    # Exclure le bruit de fond (pixels noirs)
                clip_high = 99.9  # Exclure les pixels chauds/saturés
        else:
            # YUV420/XRGB8888: déjà traité par ISP caméra hardware
            # IMPORTANT: Pour correspondre à la preview (ghs_stretch dans RPiCamera2.py),
            # NE PAS appliquer de normalisation par percentiles
            # La preview divise juste par 255 sans percentiles → même comportement ici
            clip_low = 0.0
            clip_high = 100.0

        # ── Pré-normalisation stable pour RAW (EMA sur vmax + estimation fond de ciel) ──
        # Objectif : éviter que le vmax percentile re-calculé à chaque frame cause des
        # variations de luminosité. Le fond de ciel est estimé pour éviter l'effet laiteux.
        if is_raw_format and clip_high < 100.0:
            _k = max(0, min(result.size - 1, int(result.size * clip_high / 100.0)))
            _vmax_now = float(np.partition(result.ravel(), _k)[_k])
            # Fallback: si le percentile est nul (image très creuse : ciel sombre, peu d'étoiles),
            # utiliser le max réel pour que le stretch révèle quand même les pixels brillants.
            if _vmax_now < 1e-10:
                _vmax_now = float(result.max())

            # Estimation fond de ciel = percentile bas des pixels positifs
            # → donne le niveau du fond (sky glow) au-dessus du BL
            _pos = result[result > 1e-6]
            if len(_pos) > result.size * 0.05:  # Au moins 5% de pixels positifs
                _k2 = max(0, min(_pos.size - 1, int(_pos.size * 0.05)))
                _vmin_now = float(np.partition(_pos, _k2)[_k2])
            else:
                _vmin_now = 0.0

            # EMA (Exponential Moving Average) pour stabiliser vmax et vmin
            ema_alpha = 0.25  # 25% nouveau / 75% historique
            if not hasattr(self, '_stretch_vmax_ema') or self._stretch_vmax_ema <= 0:
                self._stretch_vmax_ema = _vmax_now
                self._stretch_vmin_ema = _vmin_now
            else:
                self._stretch_vmax_ema = ema_alpha * _vmax_now + (1 - ema_alpha) * self._stretch_vmax_ema
                self._stretch_vmin_ema = ema_alpha * _vmin_now + (1 - ema_alpha) * self._stretch_vmin_ema

            _vmin = self._stretch_vmin_ema
            _vmax = self._stretch_vmax_ema

            if _vmax > _vmin + 1e-6:
                # Pré-normaliser : fond de ciel → 0 (noir), étoiles → 1 (blanc)
                result = np.clip((result - _vmin) / (_vmax - _vmin), 0, 1)
            if _do_log:
                print(f"  EMA vmin={_vmin:.4f} vmax={_vmax:.4f}  "
                      f"(now: vmin={_vmin_now:.4f} vmax={_vmax_now:.4f})")
                if is_color:
                    print(f"  APRÈS EMA   : ch0={result[:,:,0].mean():.4f} ch1={result[:,:,1].mean():.4f} ch2={result[:,:,2].mean():.4f}")
            # Les fonctions de stretch recevront des données déjà dans [0,1] :
            # - stretch_ghs : skip interne car clip_low=0, clip_high=100
            # - stretch_asinh : percentile(data,0)=0, percentile(data,100)=1 → pas de changement
            clip_low = 0.0
            clip_high = 100.0

        # (AWB auto grey-world déplacé après le stretch — voir ci-dessous)

        # Récupérer paramètres GHS
        ghs_D = getattr(self.config, 'ghs_D', 3.0)
        ghs_b = getattr(self.config, 'ghs_b', getattr(self.config, 'ghs_B', 0.13))
        ghs_SP = getattr(self.config, 'ghs_SP', 0.2)
        ghs_LP = getattr(self.config, 'ghs_LP', 0.0)
        ghs_HP = getattr(self.config, 'ghs_HP', 0.0)
        ghs_auto_adjust = getattr(self.config, 'ghs_auto_adjust_sp', True)

        # Récupérer paramètres Log et MTF
        _log_factor = getattr(self.config, 'log_factor', 100.0)
        _mtf_midtone = getattr(self.config, 'mtf_midtone', 0.2)
        _mtf_shadows = getattr(self.config, 'mtf_shadows', 0.0)
        _mtf_highlights = getattr(self.config, 'mtf_highlights', 1.0)

        # Pour Log, utiliser log_factor ; pour les autres méthodes, utiliser png_stretch_factor
        _stretch_method = self.config.png_stretch_method
        _is_log = (_stretch_method == 'log' or
                   (hasattr(_stretch_method, 'value') and _stretch_method.value == 'log'))
        _factor = _log_factor if _is_log else self.config.png_stretch_factor

        # Auto-ajustement SP pour RAW UNIQUEMENT
        # Après ISP software, les données RAW ont un histogramme centré ~0.5
        # mais SP configuré est souvent bas (0.04). On ajuste SP au pic réel.
        # NOTE: La normalisation finale GHS est DÉSACTIVÉE pour RAW (normalize_output=False)
        # donc l'auto-ajustement SP ne cause plus de double éclaircissement.
        original_SP = ghs_SP
        is_raw_only = is_raw_format and self.config.video_format in ['raw12', 'raw16']

        if (ghs_auto_adjust and is_raw_only and
            self.config.png_stretch_method == 'ghs' and ghs_SP < 0.15):
            # Calculer le pic d'histogramme des données normalisées
            if is_color:
                gray_data = 0.299 * result[:,:,0] + 0.587 * result[:,:,1] + 0.114 * result[:,:,2]
            else:
                gray_data = result

            # Histogramme avec 256 bins, ignorer les extrêmes
            mask = (gray_data > 0.01) & (gray_data < 0.99)
            if np.sum(mask) > 1000:
                hist, bin_edges = np.histogram(gray_data[mask], bins=256, range=(0.01, 0.99))
                peak_bin = np.argmax(hist)
                peak_value = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0

                # Ajuster SP pour cibler le pic d'histogramme
                if peak_value > ghs_SP * 2:
                    ghs_SP = peak_value * 0.8  # Légèrement en dessous du pic
                    ghs_SP = max(ghs_SP, 0.05)  # Minimum 5%
                    ghs_SP = min(ghs_SP, 0.5)   # Maximum 50%
                    print(f"  [GHS AUTO-SP] Pic histogramme={peak_value:.3f}, SP ajusté: {original_SP:.3f} → {ghs_SP:.3f}")

        # Normalisation finale GHS: TOUJOURS activée.
        # Sans normalisation, T1(0) est négatif → les ombres s'écrasent vers 0
        # → assombrissement progressif avec GHS (arcsinh n'a pas ce problème
        # car il normalise toujours sa sortie à [0..1]).
        normalize_ghs_output = True

        if is_color:
            stretched = np.zeros_like(result, dtype=np.float32)
            for i in range(3):
                stretched[:, :, i] = apply_stretch(
                    result[:, :, i],
                    method=self.config.png_stretch_method,
                    factor=_factor,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    ghs_D=ghs_D,
                    ghs_b=ghs_b,
                    ghs_SP=ghs_SP,
                    ghs_LP=ghs_LP,
                    ghs_HP=ghs_HP,
                    normalize_output=normalize_ghs_output,
                    mtf_midtone=_mtf_midtone,
                    mtf_shadows=_mtf_shadows,
                    mtf_highlights=_mtf_highlights,
                )
        else:
            stretched = apply_stretch(
                result,
                method=self.config.png_stretch_method,
                factor=_factor,
                clip_low=clip_low,
                clip_high=clip_high,
                ghs_D=ghs_D,
                ghs_b=ghs_b,
                ghs_SP=ghs_SP,
                ghs_LP=ghs_LP,
                ghs_HP=ghs_HP,
                normalize_output=normalize_ghs_output,
                mtf_midtone=_mtf_midtone,
                mtf_shadows=_mtf_shadows,
                mtf_highlights=_mtf_highlights,
            )

        # Log après stretch
        if _do_log:
            is_color_s = len(stretched.shape) == 3
            print(f"  STRETCH [{self.config.png_stretch_method}] : "
                  f"min={stretched.min():.4f} max={stretched.max():.4f}")
            if is_color_s:
                print(f"  APRÈS STRETCH: ch0={stretched[:,:,0].mean():.4f} "
                      f"ch1={stretched[:,:,1].mean():.4f} "
                      f"ch2={stretched[:,:,2].mean():.4f}")

        # ── AWB automatique grey-world (RAW uniquement, APRÈS stretch) ──
        # Appliqué APRÈS le stretch car le stretch per-canal (for i in range(3))
        # normalise chaque canal indépendamment et annulerait toute correction pré-stretch.
        # N'affecte que la preview affichée, pas les données du stack.
        if is_raw_format and is_color and getattr(self.config, 'awb_auto', False):
            ch_means = np.array([stretched[:, :, i].mean() for i in range(3)])
            global_mean = ch_means.mean()
            if global_mean > 1e-6:
                for i in range(3):
                    if ch_means[i] > 1e-6:
                        stretched[:, :, i] = np.clip(
                            stretched[:, :, i] * (global_mean / ch_means[i]), 0, 1)
                if _do_log:
                    gains = [global_mean / m if m > 1e-6 else 1.0 for m in ch_means]
                    print(f"  AWB AUTO    : gains ch0×{gains[0]:.3f} ch1×{gains[1]:.3f} ch2×{gains[2]:.3f}")

        # Convertir en uint8 ou uint16 selon configuration
        # NOUVEAU: Support PNG 16-bit

        if self.config.png_bit_depth == 16:
            preview = (stretched * 65535).astype(np.uint16)
            # print(f"     → 16-bit (forcé par config)")
        elif self.config.png_bit_depth == 8:
            preview = (stretched * 255).astype(np.uint8)
            # print(f"     → 8-bit (forcé par config)")
        else:
            # Auto-détection intelligente selon stretch et format
            # CORRECTION: Toujours utiliser 16-bit si stretch activé (même pour YUV420)
            # Car le stretch crée des niveaux intermédiaires → besoin de 16-bit pour histogramme lisse
            if self.config.png_stretch_method != 'off':
                preview = (stretched * 65535).astype(np.uint16)
                # print(f"     → 16-bit (auto: stretch '{self.config.png_stretch_method}' activé → préserve histogramme)")
            elif self.config.video_format and 'raw' in self.config.video_format.lower():
                preview = (stretched * 65535).astype(np.uint16)
                # print(f"     → 16-bit (auto: format RAW)")
            else:
                preview = (stretched * 255).astype(np.uint8)
                # print(f"     → 8-bit (auto: pas de stretch, pas de RAW)")

        return preview
    
    def save_result(self, output_path, generate_png=None):
        """
        Sauvegarde résultat final
        
        Args:
            output_path: Chemin de sortie (.fit)
            generate_png: Générer PNG (défaut = config.auto_png)
        """
        result = self.stacker.get_result()
        if result is None:
            print("[WARN] Aucune image empilée")
            return
        
        output_path = Path(output_path)
        
        # Métadonnées FITS
        header_data = {
            'STACKED': self.config.num_stacked,
            'REJECTED': self.files_rejected,
            'ALIGNMOD': self.config.alignment_mode,
        }
        
        if self.rotation_angles:
            header_data['ROTMIN'] = np.min(self.rotation_angles)
            header_data['ROTMAX'] = np.max(self.rotation_angles)
            header_data['ROTMED'] = np.median(self.rotation_angles)
        
        # Sauvegarder FITS (linéaire ou stretched selon config)
        save_fits(result, output_path, header_data,
                  linear=self.config.fits_linear,
                  format_hint=self.config.video_format)

        print(f"\n[SAVE] FITS: {output_path}")
        print(f"       Type: {'Linéaire (RAW)' if self.config.fits_linear else 'Stretched'}")
        print(f"       Acceptées: {self.config.num_stacked}, Rejetées: {self.files_rejected}")
        
        if self.rotation_angles:
            print(f"       Rotation: {np.min(self.rotation_angles):.2f}° à {np.max(self.rotation_angles):.2f}°")
        
        # PNG si demandé
        if generate_png is None:
            generate_png = self.config.auto_png
        
        if generate_png:
            png_path = output_path.with_suffix('.png')
            self._save_png(result, png_path)
        
        # Rapport qualité si rejets
        if self.files_rejected > 0 and self.config.save_rejected_list:
            report_path = output_path.with_suffix('.quality_report.txt')
            self._save_quality_report(report_path)
    
    def stop(self):
        """Arrête la session"""
        self.is_running = False
        print("\n" + "="*60)
        print("[STOP] SESSION TERMINÉE")
        print("="*60)
        self._print_final_stats()
    
    def reset(self):
        """Réinitialise la session (garde config)"""
        self.stacker.reset()
        self.aligner.reference_image = None
        self.aligner.reference_stars = None
        self.files_processed = 0
        self.files_rejected = 0
        self.files_failed = 0
        self.rotation_angles = []
        self._frame_count_since_refresh = 0
        self.config.quality.rejected_images = []
        self.config.quality.rejection_reasons = {}
        # Réinitialiser EMA vmax/vmin (nouveau stack = nouvelles conditions)
        if hasattr(self, '_stretch_vmax_ema'):
            del self._stretch_vmax_ema
        if hasattr(self, '_stretch_vmin_ema'):
            del self._stretch_vmin_ema
    
    def _save_png(self, data, png_path):
        """Sauvegarde PNG avec détection automatique du bit depth"""
        preview = self.get_preview_png()
        if preview is None:
            return

        import cv2

        # Détecter le bit depth
        bit_depth = 16 if preview.dtype == np.uint16 else 8

        if len(preview.shape) == 3:
            # CORRECTION: Les données sont en RGB (de Picamera2)
            # Les fichiers PNG stockent RGB, donc écrire directement
            # Note: cv2.imwrite() n'effectue PAS de conversion BGR->RGB pour les PNG!
            # Il écrit les données telles quelles. Donc RGB→RGB = correct!
            cv2.imwrite(str(png_path), preview)
        else:
            # Grayscale image
            cv2.imwrite(str(png_path), preview)

        # Afficher infos
        file_size = Path(png_path).stat().st_size / 1024
        print(f"[OK] PNG: {png_path}")
        print(f"       Bit depth: {bit_depth}-bit, Taille: {file_size:.1f} KB")
    
    def _save_quality_report(self, report_path):
        """Sauvegarde rapport qualité"""
        with open(report_path, 'w') as f:
            f.write("=== RAPPORT DE CONTRÔLE QUALITÉ ===\n\n")
            f.write(f"Images acceptées: {self.config.num_stacked}\n")
            f.write(f"Images rejetées: {self.files_rejected}\n")
            
            total = self.config.num_stacked + self.files_rejected
            if total > 0:
                f.write(f"Taux d'acceptation: {100*self.config.num_stacked/total:.1f}%\n\n")
            
            f.write("Images rejetées:\n")
            for img_name in self.config.quality.rejected_images:
                reason = self.config.quality.rejection_reasons.get(img_name, "?")
                f.write(f"  - {img_name}: {reason}\n")
        
        print(f"[SAVE] Rapport: {report_path}")
    
    def _print_stats(self):
        """Affiche stats courantes"""
        print(f"  [STATS] Empilées: {self.config.num_stacked}, "
              f"Rejetées: {self.files_rejected}, Échecs: {self.files_failed}")

    def _check_and_calibrate_isp_if_needed(self):
        """
        Vérifie si la calibration automatique de l'ISP est nécessaire et l'effectue

        Cette méthode est appelée après chaque empilement réussi.
        Elle gère:
        - La calibration initiale après N frames (isp_auto_calibrate_after)
        - La recalibration périodique (isp_recalibrate_interval)
        """
        # Vérifier si ISP est activé
        if not self.config.isp_enable:
            return

        # Vérifier si calibration auto est activée
        method = self.config.isp_auto_calibrate_method
        if method == 'none' or method is None:
            return

        # Vérifier si on doit calibrer
        should_calibrate = False
        calibration_type = ""

        # Calibration initiale
        if (not self._isp_calibrated and
            self.config.isp_auto_calibrate_after > 0 and
            self.config.num_stacked >= self.config.isp_auto_calibrate_after):
            should_calibrate = True
            calibration_type = "initiale"

        # Recalibration périodique
        elif (self._isp_calibrated and
              self.config.isp_recalibrate_interval > 0 and
              (self.config.num_stacked - self._last_isp_calibration_frame) >= self.config.isp_recalibrate_interval):
            should_calibrate = True
            calibration_type = "périodique"

        if not should_calibrate:
            return

        # Effectuer la calibration
        try:
            print(f"\n  [ISP] Calibration automatique {calibration_type} (méthode: {method})...")

            # Récupérer l'image stackée actuelle
            stacked_image = self.stacker.get_result()

            if stacked_image is None:
                print(f"  [ISP] ✗ Impossible de récupérer l'image stackée")
                return

            # Calibrer l'ISP
            from .isp import ISPCalibrator, ISP
            isp_config_auto = ISPCalibrator.calibrate_from_stacked_image(
                stacked_image,
                method=method
            )

            # Fusionner avec la config existante si présente
            if self.isp and self.isp.config and self.config.isp_auto_update_only_wb:
                # Mode fusion : mise à jour UNIQUEMENT des gains RGB
                # Préserve gamma, contrast, saturation, CCM, black_level, etc.
                existing_config = self.isp.config

                # Mettre à jour UNIQUEMENT les gains RGB (calibration auto)
                existing_config.wb_red_gain = isp_config_auto.wb_red_gain
                existing_config.wb_green_gain = isp_config_auto.wb_green_gain
                existing_config.wb_blue_gain = isp_config_auto.wb_blue_gain

                # Mettre à jour les infos de calibration
                if 'auto_calibration' not in existing_config.calibration_info:
                    existing_config.calibration_info['auto_calibration'] = {}
                existing_config.calibration_info['auto_calibration'].update(isp_config_auto.calibration_info)
                existing_config.calibration_info['auto_calibrated_at_frame'] = self.config.num_stacked
                existing_config.calibration_info['mode'] = 'wb_only'

                # Garder l'instance ISP existante (pas besoin de recréer)
                isp_config = existing_config

                print(f"  [ISP] ✓ Gains RGB mis à jour (gamma={existing_config.gamma:.2f}, "
                      f"contrast={existing_config.contrast:.2f}, saturation={existing_config.saturation:.2f} préservés)")
            else:
                # Mode remplacement complet : utiliser toute la calibration auto
                self.isp = ISP(isp_config_auto)
                isp_config = isp_config_auto
                print(f"  [ISP] ✓ Nouvelle config ISP complète créée")

            # Mettre à jour le tracking
            self._isp_calibrated = True
            self._last_isp_calibration_frame = self.config.num_stacked

            print(f"  [ISP] ✓ Calibration réussie! Gains RGB: "
                  f"R={isp_config.wb_red_gain:.3f}, "
                  f"G={isp_config.wb_green_gain:.3f}, "
                  f"B={isp_config.wb_blue_gain:.3f}")

            # Optionnel: sauvegarder la config pour référence
            if self.output_dir:
                config_path = self.output_dir / 'session_isp_config_auto.json'
                self.isp.save_config(config_path)

        except Exception as e:
            print(f"  [ISP] ✗ Erreur lors de la calibration: {e}")
            import traceback
            traceback.print_exc()

    def _print_final_stats(self):
        """Affiche stats finales"""
        print(f"\n[STATS] Final:")
        print(f"  * Type: {'RGB' if self.is_color else 'MONO'}")
        print(f"  * Mode: {self.config.alignment_mode}")
        print(f"  * Empilées: {self.config.num_stacked}")
        print(f"  * Rejetées: {self.files_rejected}")
        print(f"  * Échecs: {self.files_failed}")
        
        total = self.config.num_stacked + self.files_rejected
        if total > 0:
            print(f"  * Taux acceptation: {100*self.config.num_stacked/total:.1f}%")
        
        if self.rotation_angles:
            print(f"  * Rotation: {np.min(self.rotation_angles):.2f}° à {np.max(self.rotation_angles):.2f}°")
        
        if self.config.num_stacked > 0:
            snr_gain = self.stacker.get_snr_improvement()
            print(f"  * SNR gain: x{snr_gain:.2f}")
