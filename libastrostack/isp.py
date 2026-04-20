#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISP (Image Signal Processor) pipeline pour images RAW
Calibration automatique basée sur analyse RAW vs YUV
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ISPConfig:
    """Configuration des paramètres ISP"""
    # Balance des blancs
    wb_red_gain: float = 1.0
    wb_green_gain: float = 1.0
    wb_blue_gain: float = 1.0

    # Black level
    black_level: int = 64

    # Gamma
    gamma: float = 2.2

    # Brightness offset (ajouté pour mapping depuis interface GUI)
    brightness_offset: float = 0.0  # Offset linéaire [-1.0, 1.0]

    # Contraste et saturation
    contrast: float = 1.0
    saturation: float = 1.0

    # Netteté
    sharpening: float = 0.0

    # Swap R/B pour débayeurisation RAW (RGGB → BGR inversion)
    swap_rb: bool = False

    # Color Correction Matrix
    ccm: Optional[np.ndarray] = None

    # Métadonnées de calibration
    calibration_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.ccm is None:
            self.ccm = np.eye(3, dtype=np.float32)


class ISPCalibrator:
    """
    Calibre les paramètres ISP en comparant une image RAW à son équivalent traité (YUV)
    """

    @staticmethod
    def calibrate_from_images(raw_image: np.ndarray,
                             processed_image: np.ndarray,
                             method: str = 'auto') -> ISPConfig:
        """
        Analyse une paire d'images pour extraire les paramètres ISP

        Args:
            raw_image: Image RAW (avant ISP)
            processed_image: Image traitée par l'ISP (YUV420 ou RGB)
            method: Méthode de calibration ('auto', 'simple', 'advanced')

        Returns:
            Configuration ISP calibrée
        """
        config = ISPConfig()

        # Convertir les images en float (0-1)
        raw_float = ISPCalibrator._to_float(raw_image)
        proc_float = ISPCalibrator._to_float(processed_image)

        # Redimensionner si nécessaire
        if raw_float.shape != proc_float.shape:
            h = min(raw_float.shape[0], proc_float.shape[0])
            w = min(raw_float.shape[1], proc_float.shape[1])
            raw_float = raw_float[:h, :w]
            proc_float = proc_float[:h, :w]

        print("\n=== Calibration ISP automatique ===")
        print(f"Image RAW: {raw_float.shape}, range [{raw_float.min():.3f}, {raw_float.max():.3f}]")
        print(f"Image traitée: {proc_float.shape}, range [{proc_float.min():.3f}, {proc_float.max():.3f}]")

        # 1. Estimer la balance des blancs
        wb_gains = ISPCalibrator._estimate_white_balance(raw_float, proc_float)
        config.wb_red_gain = wb_gains[0]
        config.wb_green_gain = wb_gains[1]
        config.wb_blue_gain = wb_gains[2]
        print(f"\n✓ Balance des blancs: R={wb_gains[0]:.3f}, G={wb_gains[1]:.3f}, B={wb_gains[2]:.3f}")

        # Appliquer WB à l'image RAW pour les étapes suivantes
        raw_wb = raw_float.copy()
        for i in range(3):
            raw_wb[:, :, i] *= [wb_gains[0], wb_gains[1], wb_gains[2]][i]
        raw_wb = np.clip(raw_wb, 0, 1)

        # 2. Estimer le gamma
        gamma = ISPCalibrator._estimate_gamma(raw_wb, proc_float)
        config.gamma = gamma
        print(f"✓ Gamma: {gamma:.2f}")

        # Appliquer gamma à l'image RAW
        raw_gamma = np.power(raw_wb, 1.0 / gamma)

        # 3. Estimer le contraste
        contrast = ISPCalibrator._estimate_contrast(raw_gamma, proc_float)
        config.contrast = contrast
        print(f"✓ Contraste: {contrast:.2f}")

        # Appliquer contraste
        raw_contrast = (raw_gamma - 0.5) * contrast + 0.5
        raw_contrast = np.clip(raw_contrast, 0, 1)

        # 4. Estimer la saturation
        saturation = ISPCalibrator._estimate_saturation(raw_contrast, proc_float)
        config.saturation = saturation
        print(f"✓ Saturation: {saturation:.2f}")

        # 5. Calculer l'erreur résiduelle
        error = np.mean(np.abs(raw_contrast - proc_float))
        print(f"\n✓ Erreur résiduelle moyenne: {error:.4f}")

        # Stocker les infos de calibration
        config.calibration_info = {
            'method': method,
            'residual_error': float(error),
            'raw_mean': [float(raw_float[:,:,i].mean()) for i in range(3)],
            'processed_mean': [float(proc_float[:,:,i].mean()) for i in range(3)]
        }

        return config

    @staticmethod
    def calibrate_from_files(raw_path: Path, processed_path: Path) -> ISPConfig:
        """
        Calibre à partir de fichiers images

        Args:
            raw_path: Chemin de l'image RAW
            processed_path: Chemin de l'image traitée

        Returns:
            Configuration ISP
        """
        raw_img = cv2.imread(str(raw_path))
        proc_img = cv2.imread(str(processed_path))

        if raw_img is None:
            raise ValueError(f"Impossible de lire {raw_path}")
        if proc_img is None:
            raise ValueError(f"Impossible de lire {processed_path}")

        # Convertir BGR -> RGB
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

        return ISPCalibrator.calibrate_from_images(raw_img, proc_img)

    @staticmethod
    def _to_float(image: np.ndarray) -> np.ndarray:
        """Convertit en float32 normalisé"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            img = image.astype(np.float32)
            if img.max() > 1.0:
                return img / img.max()
            return img

    @staticmethod
    def _estimate_white_balance(raw: np.ndarray, processed: np.ndarray) -> Tuple[float, float, float]:
        """
        Estime les gains de balance des blancs

        Compare les moyennes des canaux RGB entre raw et processed
        """
        # Utiliser les percentiles pour éviter les outliers
        raw_means = np.array([np.percentile(raw[:,:,i], 50) for i in range(3)])
        proc_means = np.array([np.percentile(processed[:,:,i], 50) for i in range(3)])

        # Calculer les gains relatifs
        gains = proc_means / (raw_means + 1e-6)

        # Normaliser par rapport au vert
        gains = gains / gains[1]

        # Limiter les gains extrêmes
        gains = np.clip(gains, 0.5, 2.0)

        return tuple(gains)

    @staticmethod
    def _estimate_white_balance_from_histogram_peaks(image: np.ndarray,
                                                      bins: int = 256,
                                                      mask_range: Tuple[float, float] = (0.05, 0.95)) -> Tuple[float, float, float]:
        """
        Estime les gains de balance des blancs basé sur les pics d'histogrammes

        Cette méthode analyse les pics (modes) de chaque canal RGB et calcule
        les gains nécessaires pour aligner ces pics. Particulièrement efficace
        pour l'astrophotographie où le fond du ciel domine l'histogramme.

        Args:
            image: Image RGB en float32 (0-1)
            bins: Nombre de bins pour l'histogramme (défaut: 256)
            mask_range: Tuple (min, max) pour masquer les valeurs extrêmes
                       avant de calculer le pic (évite pixels noirs/saturés)

        Returns:
            Tuple (red_gain, green_gain, blue_gain)

        Exemple:
            >>> config = ISPConfig()
            >>> calibrator = ISPCalibrator()
            >>> gains = calibrator._estimate_white_balance_from_histogram_peaks(stacked_image)
            >>> config.wb_red_gain, config.wb_green_gain, config.wb_blue_gain = gains
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("L'image doit être RGB (H, W, 3)")

        peaks = []

        for channel_idx in range(3):
            channel_data = image[:, :, channel_idx].flatten()

            # Masquer les valeurs extrêmes (pixels noirs et saturés)
            mask = (channel_data >= mask_range[0]) & (channel_data <= mask_range[1])
            channel_masked = channel_data[mask]

            if len(channel_masked) == 0:
                # Fallback: utiliser toutes les données
                channel_masked = channel_data

            # Calculer l'histogramme
            hist, bin_edges = np.histogram(channel_masked, bins=bins, range=(0.0, 1.0))

            # Trouver le bin avec le maximum de pixels (pic)
            peak_bin_idx = np.argmax(hist)

            # Calculer la valeur centrale du bin (position du pic)
            peak_value = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2.0

            peaks.append(peak_value)

        peaks = np.array(peaks)

        # Calculer les gains pour aligner tous les pics sur le pic du vert (référence)
        # Si pic_rouge = 0.3 et pic_vert = 0.5, alors gain_rouge = 0.5 / 0.3 = 1.667
        green_peak = peaks[1]

        if green_peak < 1e-3:
            # Image quasi-noire, retourner gains neutres
            print("⚠️  Image trop sombre pour calibration par pics d'histogramme")
            return (1.0, 1.0, 1.0)

        gains = green_peak / (peaks + 1e-6)

        # Limiter les gains extrêmes
        gains = np.clip(gains, 0.5, 2.0)

        # Debug info
        channel_names = ['Rouge', 'Vert', 'Bleu']
        print(f"\n=== Calibration par pics d'histogramme ===")
        for i in range(3):
            print(f"  {channel_names[i]}: pic={peaks[i]:.4f}, gain={gains[i]:.3f}")

        return tuple(gains)

    @staticmethod
    def calibrate_from_stacked_image(image: np.ndarray,
                                     method: str = 'histogram_peaks',
                                     target_brightness: Optional[float] = None) -> ISPConfig:
        """
        Calibre les paramètres ISP à partir d'une image stackée unique

        Cette méthode permet une calibration automatique sans image de référence,
        idéale pour ajuster dynamiquement l'ISP pendant une session de live stacking.

        Args:
            image: Image stackée en RGB (uint8, uint16, ou float32)
            method: Méthode de calibration:
                    - 'histogram_peaks': Aligne les pics d'histogrammes RGB (recommandé)
                    - 'gray_world': Suppose que la moyenne des couleurs est grise
            target_brightness: Luminosité cible après gamma (0-1), None=auto

        Returns:
            Configuration ISP calibrée

        Exemple:
            >>> session = LiveStackSession(config)
            >>> # Après quelques frames stackées...
            >>> stacked = session.stacker.get_stacked_image()
            >>> isp_config = ISPCalibrator.calibrate_from_stacked_image(stacked)
            >>> session.isp = ISP(isp_config)
        """
        config = ISPConfig()

        # Convertir en float32
        img_float = ISPCalibrator._to_float(image)

        print(f"\n=== Calibration ISP depuis image stackée ===")
        print(f"Image: {img_float.shape}, range [{img_float.min():.3f}, {img_float.max():.3f}]")
        print(f"Méthode: {method}")

        # 1. Balance des blancs selon la méthode choisie
        if method == 'histogram_peaks':
            wb_gains = ISPCalibrator._estimate_white_balance_from_histogram_peaks(img_float)
        elif method == 'gray_world':
            # Gray World: suppose que la moyenne des couleurs est neutre
            means = np.array([img_float[:,:,i].mean() for i in range(3)])
            gray_mean = means.mean()
            wb_gains = gray_mean / (means + 1e-6)
            wb_gains = wb_gains / wb_gains[1]  # Normaliser par vert
            wb_gains = np.clip(wb_gains, 0.5, 2.0)
            wb_gains = tuple(wb_gains)
        else:
            raise ValueError(f"Méthode inconnue: {method}")

        config.wb_red_gain = wb_gains[0]
        config.wb_green_gain = wb_gains[1]
        config.wb_blue_gain = wb_gains[2]

        # 2. Estimer gamma basé sur la luminosité
        mean_brightness = img_float.mean()

        if target_brightness is not None:
            # Calculer gamma pour atteindre la cible
            # target = mean^(1/gamma) => gamma = log(mean) / log(target)
            if mean_brightness > 0.01 and target_brightness > 0.01:
                gamma_est = np.log(mean_brightness) / np.log(target_brightness)
                config.gamma = float(np.clip(gamma_est, 1.5, 3.0))
            else:
                config.gamma = 2.2
        else:
            # Gamma adaptatif basé sur la luminosité de l'image
            if mean_brightness < 0.1:
                # Image sombre: gamma plus élevé pour éclaircir
                config.gamma = 2.4
            elif mean_brightness < 0.3:
                config.gamma = 2.2
            else:
                # Image claire: gamma plus bas
                config.gamma = 1.8

        print(f"✓ Luminosité moyenne: {mean_brightness:.3f}")
        print(f"✓ Gamma: {config.gamma:.2f}")

        # 3. Paramètres par défaut pour astrophotographie
        config.contrast = 1.1  # Légèrement augmenté pour les détails
        config.saturation = 1.0  # Neutre
        config.black_level = 64  # Standard pour RAW12
        config.sharpening = 0.0  # Désactivé par défaut

        # Métadonnées
        config.calibration_info = {
            'method': method,
            'source': 'stacked_image',
            'mean_brightness': float(mean_brightness),
            'histogram_peaks': [float(p) for p in
                              ISPCalibrator._estimate_white_balance_from_histogram_peaks(img_float, bins=256)[0:3]
                              if method == 'histogram_peaks']
        }

        print(f"✓ Calibration terminée\n")

        return config

    @staticmethod
    def _estimate_gamma(raw: np.ndarray, processed: np.ndarray) -> float:
        """
        Estime la valeur de gamma

        Utilise plusieurs points de la courbe pour estimer gamma
        """
        # Échantillonner des valeurs à différents niveaux d'intensité
        percentiles = [10, 25, 50, 75, 90]
        gamma_estimates = []

        for p in percentiles:
            raw_val = np.percentile(raw, p)
            proc_val = np.percentile(processed, p)

            if raw_val > 0.01 and proc_val > 0.01:
                # proc = raw^(1/gamma) => gamma = log(raw) / log(proc)
                gamma_est = np.log(raw_val + 1e-6) / np.log(proc_val + 1e-6)
                if 0.5 < gamma_est < 5.0:  # Garder seulement les valeurs raisonnables
                    gamma_estimates.append(gamma_est)

        if gamma_estimates:
            gamma = np.median(gamma_estimates)
        else:
            gamma = 2.2  # Valeur par défaut

        # Limiter à une plage raisonnable
        gamma = np.clip(gamma, 1.5, 3.0)

        return float(gamma)

    @staticmethod
    def _estimate_contrast(raw: np.ndarray, processed: np.ndarray) -> float:
        """
        Estime le facteur de contraste

        Compare les écarts-types (proxy pour le contraste)
        """
        raw_std = np.std(raw)
        proc_std = np.std(processed)

        contrast = proc_std / (raw_std + 1e-6)

        # Limiter à une plage raisonnable
        contrast = np.clip(contrast, 0.8, 1.5)

        return float(contrast)

    @staticmethod
    def _estimate_saturation(raw: np.ndarray, processed: np.ndarray) -> float:
        """
        Estime le facteur de saturation

        Compare la saturation moyenne en HSV
        """
        # Convertir en HSV
        raw_uint8 = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
        proc_uint8 = (np.clip(processed, 0, 1) * 255).astype(np.uint8)

        raw_hsv = cv2.cvtColor(raw_uint8, cv2.COLOR_RGB2HSV)
        proc_hsv = cv2.cvtColor(proc_uint8, cv2.COLOR_RGB2HSV)

        # Comparer le canal S (saturation)
        raw_sat = np.mean(raw_hsv[:, :, 1])
        proc_sat = np.mean(proc_hsv[:, :, 1])

        saturation = proc_sat / (raw_sat + 1e-6)

        # Limiter à une plage raisonnable
        saturation = np.clip(saturation, 0.5, 2.0)

        return float(saturation)


class ISP:
    """
    Image Signal Processor pour traiter les images RAW
    """

    def __init__(self, config: Optional[ISPConfig] = None):
        """
        Initialise l'ISP

        Args:
            config: Configuration ISP (par défaut si None)
        """
        self.config = config or ISPConfig()

    def process(self, image: np.ndarray, return_uint8: bool = False,
                return_uint16: bool = False, swap_rb: bool = None,
                format_hint: str = None) -> np.ndarray:
        """
        Applique le pipeline ISP complet

        Args:
            image: Image d'entrée (uint8, uint16, float32)
            return_uint8: Si True, retourne uint8 (0-255)
            return_uint16: Si True, retourne uint16 (0-65535)
            swap_rb: Si True, inverse les gains R/B. Si None, utilise config.swap_rb
            format_hint: Format de l'image ('raw12', 'raw16', None=auto-détection)
            Si les deux False, retourne float32 (0.0-1.0) [RECOMMANDÉ]

        Returns:
            Image traitée dans le format demandé
        """
        # Utiliser config.swap_rb si non spécifié
        if swap_rb is None:
            swap_rb = self.config.swap_rb
        # Adaptation dynamique des paramètres selon le format
        # RAW12 vs RAW16 Clear HDR ont des caractéristiques différentes
        original_black_level = self.config.black_level
        original_gamma = self.config.gamma

        if format_hint == 'raw16':
            # RAW16 Clear HDR : plage dynamique étendue, black level plus bas
            # Le mode Clear HDR du capteur fusionne déjà les expositions
            self.config.black_level = int(original_black_level * 0.5)  # Black level réduit
            # Gamma légèrement réduit pour préserver détails HDR
            if original_gamma > 2.0:
                self.config.gamma = original_gamma * 0.9
        elif format_hint == 'raw12':
            # RAW12 : profondeur de bit standard, paramètres normaux
            # Utiliser les paramètres par défaut/calibrés
            pass

        # Conversion en float32 (0.0-1.0)
        img = self._to_float(image)

        # Pipeline ISP complet (8 étapes)
        img = self._apply_black_level(img)
        img = self._apply_white_balance(img, swap_rb=swap_rb)
        img = self._apply_ccm(img)  # Correction colorimétrique
        img = self._apply_gamma(img)
        img = self._apply_brightness_offset(img)  # Brightness (après gamma, avant contrast)
        img = self._apply_contrast(img)
        img = self._apply_saturation(img)
        img = self._apply_sharpening(img)  # Netteté (dernière étape)

        # Restaurer les paramètres originaux
        self.config.black_level = original_black_level
        self.config.gamma = original_gamma

        # Conversion format de sortie
        if return_uint8:
            return self._to_uint8(img)
        elif return_uint16:
            return self._to_uint16(img)
        else:
            # Retourner float32 par défaut (haute précision)
            return img

    def _to_float(self, image: np.ndarray) -> np.ndarray:
        """Convertit en float32 normalisé (0-1)"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            img = image.astype(np.float32)
            if img.max() > 1.0:
                return img / img.max()
            return img

    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convertit en uint8 (0-255)"""
        img_clipped = np.clip(image, 0.0, 1.0)
        return (img_clipped * 255).astype(np.uint8)

    def _to_uint16(self, image: np.ndarray) -> np.ndarray:
        """Convertit en uint16 (0-65535)"""
        img_clipped = np.clip(image, 0.0, 1.0)
        return (img_clipped * 65535).astype(np.uint16)

    def _apply_black_level(self, image: np.ndarray) -> np.ndarray:
        """
        Soustrait le niveau de noir

        Si black_level == 0, détecte automatiquement à partir du 1er percentile
        """
        if self.config.black_level == 0:
            # Auto-détection du black level à partir du 1er percentile
            # Cela fonctionne bien pour les images astro avec fond de ciel sombre
            black_auto = np.percentile(image, 1)
            if black_auto > 0.01:  # Seulement si significatif (>1% de la plage)
                print(f"  [ISP] Black level auto-détecté: {black_auto:.3f}")
                img = image - black_auto
                img = np.clip(img, 0, None)
                # Renormaliser à [0, 1]
                if img.max() > 0:
                    img = img / img.max()
                return img
            return image

        # Black level normalisé (supposant 12-bit)
        black_norm = self.config.black_level / 4095.0
        img = np.clip(image - black_norm, 0, 1)

        # Renormaliser
        if img.max() > 0:
            img = img / img.max()

        return img

    def _apply_white_balance(self, image: np.ndarray, swap_rb: bool = False) -> np.ndarray:
        """
        Applique la balance des blancs

        Args:
            image: Image en float32
            swap_rb: Si True, inverse les canaux R/B avant d'appliquer les gains
                     (pour compenser l'inversion dans la débayeurisation RAW)
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        img = image.copy()

        # Si swap_rb=True (pour RAW), inverser les canaux R/B de l'image
        # Car la débayeurisation peut inverser les canaux
        if swap_rb:
            # Inverser les canaux R et B de l'image, puis appliquer les gains normalement
            img = img[:, :, ::-1].copy()  # BGR -> RGB (inverse canaux 0 et 2)

        # ATTENTION: pygame fait un swap R↔B pour l'affichage ([:,:,[2,1,0]])
        # Donc on applique wb_red sur canal 2 et wb_blue sur canal 0 pour que
        # l'effet corresponde visuellement au label du slider
        # INVERSÉ pour compenser le swap pygame [:,:,[2,1,0]]
        img[:, :, 2] *= self.config.wb_red_gain   # Canal 2 natif → Rouge à l'écran
        img[:, :, 1] *= self.config.wb_green_gain
        img[:, :, 0] *= self.config.wb_blue_gain  # Canal 0 natif → Bleu à l'écran

        return np.clip(img, 0, 1)

    def _apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """Applique la correction gamma"""
        if self.config.gamma == 1.0:
            return image
        return np.power(image, 1.0 / self.config.gamma)

    def _apply_brightness_offset(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un offset de luminosité linéaire

        Ajoute/soustrait une valeur constante à tous les pixels.
        Appliqué APRÈS gamma, AVANT contrast pour un effet naturel.

        Args:
            image: Image en float32 [0-1]

        Returns:
            Image avec brightness ajusté, clippé [0-1]
        """
        if self.config.brightness_offset == 0.0:
            return image

        # Ajouter l'offset et clipper pour rester dans [0, 1]
        return np.clip(image + self.config.brightness_offset, 0.0, 1.0)

    def _apply_contrast(self, image: np.ndarray) -> np.ndarray:
        """Applique l'ajustement de contraste"""
        if self.config.contrast == 1.0:
            return image

        img = (image - 0.5) * self.config.contrast + 0.5
        return np.clip(img, 0, 1)

    def _apply_saturation(self, image: np.ndarray) -> np.ndarray:
        """Applique l'ajustement de saturation"""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        if self.config.saturation == 1.0:
            return image

        # HSV pour ajuster saturation
        img_uint8 = self._to_uint8(image)
        img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        img_hsv[:, :, 1] *= self.config.saturation
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img_rgb.astype(np.float32) / 255.0

    def _apply_ccm(self, image: np.ndarray) -> np.ndarray:
        """
        Applique la Color Correction Matrix (CCM)

        Transforme les couleurs via multiplication matricielle RGB_out = CCM × RGB_in
        Standard dans les pipelines ISP pour corriger les aberrations colorimétriques
        du capteur et adapter l'espace colorimétrique.

        Args:
            image: Image RGB en float32 [0-1]

        Returns:
            Image corrigée en float32 [0-1]
        """
        # Uniquement pour les images RGB
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        # Si CCM est identité, skip
        if self.config.ccm is None or np.allclose(self.config.ccm, np.eye(3)):
            return image

        # Appliquer la transformation matricielle
        # Reshape (H, W, 3) → (H*W, 3) pour multiplication matricielle
        h, w, c = image.shape
        img_flat = image.reshape(-1, 3)

        # CCM × RGB (broadcasting)
        # ccm shape: (3, 3), img_flat shape: (H*W, 3)
        # Result: (H*W, 3)
        img_corrected = np.dot(img_flat, self.config.ccm.T)

        # Clip et reshape
        img_corrected = np.clip(img_corrected, 0.0, 1.0)
        return img_corrected.reshape(h, w, c).astype(np.float32)

    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un filtre de netteté (unsharp mask)

        Formule: output = original + amount × (original - blurred)
        - amount = 0 : pas de sharpening
        - amount > 0 : augmente la netteté
        - amount typique : 0.3 à 1.5

        Args:
            image: Image en float32 [0-1]

        Returns:
            Image sharpened en float32 [0-1]
        """
        if self.config.sharpening == 0.0:
            return image

        # Convertir en uint8 pour GaussianBlur (plus rapide)
        img_uint8 = self._to_uint8(image)

        # Appliquer un flou gaussien
        # Kernel size adaptatif basé sur la résolution
        kernel_size = max(3, int(min(image.shape[:2]) / 200) * 2 + 1)  # Impair
        blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)

        # Unsharp mask: original + amount × (original - blurred)
        sharpened = cv2.addWeighted(
            img_uint8, 1.0 + self.config.sharpening,
            blurred, -self.config.sharpening,
            0
        )

        # Convertir en float32 [0-1]
        return sharpened.astype(np.float32) / 255.0

    def save_config(self, path: Path):
        """Sauvegarde la configuration ISP"""
        import json

        config_dict = {
            'wb_red_gain': self.config.wb_red_gain,
            'wb_green_gain': self.config.wb_green_gain,
            'wb_blue_gain': self.config.wb_blue_gain,
            'black_level': self.config.black_level,
            'gamma': self.config.gamma,
            'brightness_offset': self.config.brightness_offset,
            'contrast': self.config.contrast,
            'saturation': self.config.saturation,
            'sharpening': self.config.sharpening,
            'swap_rb': self.config.swap_rb,
            'calibration_info': self.config.calibration_info
        }

        # Sauvegarder la CCM si elle n'est pas l'identité
        if self.config.ccm is not None and not np.allclose(self.config.ccm, np.eye(3)):
            config_dict['ccm'] = self.config.ccm.tolist()

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration ISP sauvegardée: {path}")

    @staticmethod
    def load_config(path: Path) -> 'ISP':
        """Charge une configuration ISP depuis un fichier"""
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Extraire la CCM si présente
        ccm = None
        if 'ccm' in config_dict:
            ccm = np.array(config_dict['ccm'], dtype=np.float32)

        config = ISPConfig(**{k: v for k, v in config_dict.items()
                             if k not in ['calibration_info', 'ccm']})
        config.calibration_info = config_dict.get('calibration_info', {})

        # Assigner la CCM
        if ccm is not None:
            config.ccm = ccm

        return ISP(config)


def quick_calibrate_and_process(raw_image: np.ndarray,
                                reference_processed: Optional[np.ndarray] = None,
                                config: Optional[ISPConfig] = None) -> Tuple[np.ndarray, ISPConfig]:
    """
    Fonction helper pour calibrer et traiter rapidement

    Args:
        raw_image: Image RAW à traiter
        reference_processed: Image de référence traitée (pour calibration)
        config: Configuration existante (skip calibration si fournie)

    Returns:
        Tuple (image traitée, config utilisée)
    """
    if config is None and reference_processed is not None:
        # Calibration automatique
        config = ISPCalibrator.calibrate_from_images(raw_image, reference_processed)
    elif config is None:
        # Configuration par défaut
        config = ISPConfig()

    isp = ISP(config)
    processed = isp.process(raw_image)

    return processed, config
