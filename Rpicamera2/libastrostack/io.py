#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chargement et sauvegarde d'images pour libastrostack
"""

import numpy as np
from pathlib import Path
import cv2

try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False


def load_image(path):
    """
    Charge une image depuis un fichier
    
    Formats supportés:
    - FITS (.fit, .fits)
    - RAW/DNG (.dng, .cr2, .nef, .arw, .raw)
    - Standard (.jpg, .png, .tif)
    
    Args:
        path: Chemin vers le fichier
    
    Returns:
        Image en float32, ou None si erreur
        Shape: (height, width) pour MONO ou (height, width, 3) pour RGB
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    # FITS
    if ext in ['.fit', '.fits']:
        if not HAS_FITS:
            print(f"[ERROR] astropy non disponible pour FITS")
            return None
        
        try:
            with fits.open(path, ignore_missing_end=True, memmap=False) as hdul:
                data = hdul[0].data
                
                # Convertir (3, H, W) en (H, W, 3)
                if len(data.shape) == 3 and data.shape[0] == 3:
                    data = data.transpose(1, 2, 0)
                
                return data.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Lecture FITS {path}: {e}")
            return None
    
    # RAW/DNG
    elif ext in ['.dng', '.cr2', '.nef', '.arw', '.raw']:
        if not HAS_RAWPY:
            print(f"[ERROR] rawpy non disponible pour RAW/DNG")
            return None
        
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
                )
                return rgb.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Lecture RAW {path}: {e}")
            return None
    
    # Images standard
    elif ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        try:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                return None
            
            # Convertir BGR en RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Lecture image {path}: {e}")
            return None
    
    else:
        print(f"[ERROR] Format non supporté: {ext}")
        return None


def save_fits(data, path, header_data=None, linear=True, format_hint=None):
    """
    Sauvegarde image en FITS

    Args:
        data: Image (float array, données du stack)
        path: Chemin de sortie
        header_data: Dictionnaire de métadonnées optionnelles
        linear: Si True, sauvegarde données linéaires (RAW, recommandé)
                Si False, applique stretch avant sauvegarde (legacy)
        format_hint: Format source explicite ('yuv420', 'raw12', 'raw16', None=auto)

    Returns:
        True si succès
    """
    if not HAS_FITS:
        print("[ERROR] astropy non disponible")
        return False

    try:
        is_color = len(data.shape) == 3

        if linear:
            # MODE RAW: Sauvegarder données linéaires normalisées en uint16
            # Utiliser format_hint pour déterminer la plage native des données,
            # puis normaliser vers la plage complète 16-bit (0-65535) pour compatibilité
            # maximale avec les logiciels de post-traitement (PixInsight, Siril, etc.)
            if format_hint:
                f = format_hint.lower()
                if 'yuv' in f or '420' in f:
                    native_max = 255.0
                elif 'raw12' in f or 'srggb12' in f or 'rggb12' in f:
                    native_max = 4095.0
                elif 'raw10' in f or 'srggb10' in f:
                    native_max = 1023.0
                else:
                    native_max = 65535.0
            else:
                # Heuristique basée sur la plage des données
                dmax = float(data.max())
                if dmax > 65535:
                    native_max = dmax
                elif dmax <= 1.0:
                    native_max = 1.0
                elif dmax <= 300:
                    native_max = 255.0
                elif dmax <= 5000:
                    native_max = 4095.0
                else:
                    native_max = 65535.0

            # Normaliser vers uint16 pleine échelle
            if native_max <= 1.0:
                result_data = (np.clip(data, 0, 1) * 65535).astype(np.uint16)
            else:
                result_data = np.clip(data / native_max * 65535, 0, 65535).astype(np.uint16)

            if is_color:
                result = result_data.transpose(2, 0, 1)
                hdu = fits.PrimaryHDU(result)
                hdu.header['COLORTYP'] = 'RGB'
                hdu.header['DATATYPE'] = 'LINEAR'
                hdu.header['COMMENT'] = 'Linear (unstretched) data for post-processing'
            else:
                result = result_data
                hdu = fits.PrimaryHDU(result)
                hdu.header['DATATYPE'] = 'LINEAR'
                hdu.header['COMMENT'] = 'Linear (unstretched) data for post-processing'

        else:
            # MODE LEGACY: Étirer pour FITS (percentiles)
            # Attention: perte d'information, pas optimal pour post-traitement
            if is_color:
                stretched = np.zeros_like(data)
                for i in range(3):
                    vmin = np.percentile(data[:,:,i], 1)
                    vmax = np.percentile(data[:,:,i], 99)
                    stretched[:,:,i] = np.clip((data[:,:,i] - vmin) / (vmax - vmin), 0, 1)

                result_16bit = (stretched * 65535).astype(np.uint16)
                result_16bit = result_16bit.transpose(2, 0, 1)

                hdu = fits.PrimaryHDU(result_16bit)
                hdu.header['COLORTYP'] = 'RGB'
                hdu.header['DATATYPE'] = 'STRETCHED'
                hdu.header['COMMENT'] = 'Stretched data (1-99 percentile)'
            else:
                vmin = np.percentile(data, 1)
                vmax = np.percentile(data, 99)
                stretched = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                result_16bit = (stretched * 65535).astype(np.uint16)

                hdu = fits.PrimaryHDU(result_16bit)
                hdu.header['DATATYPE'] = 'STRETCHED'
                hdu.header['COMMENT'] = 'Stretched data (1-99 percentile)'

        # Ajouter métadonnées utilisateur
        if header_data:
            for key, value in header_data.items():
                hdu.header[key] = value

        hdu.writeto(path, overwrite=True)
        return True

    except Exception as e:
        print(f"[ERROR] Sauvegarde FITS {path}: {e}")
        return False


def save_dng_async(data, path, queue):
    """
    Sauvegarde DNG en arrière-plan (pour mode hybride)
    
    Args:
        data: Image RGB float32
        path: Chemin de sortie
        queue: Queue pour communication thread
    
    Note: Cette fonction est appelée dans un thread séparé
    """
    # TODO: Implémenter sauvegarde DNG
    # Pour l'instant, on sauvegarde en PNG 16-bit
    try:
        # Convertir en uint16
        data_16bit = np.clip(data, 0, 65535).astype(np.uint16)
        
        # Convertir RGB en BGR pour OpenCV
        if len(data_16bit.shape) == 3:
            data_16bit = cv2.cvtColor(data_16bit, cv2.COLOR_RGB2BGR)
        
        # Sauvegarder
        cv2.imwrite(str(path), data_16bit)
        
        if queue:
            queue.put(('success', str(path)))
    except Exception as e:
        if queue:
            queue.put(('error', str(e)))


def detect_source_format(data, format_hint=None):
    """
    Détecte le format source des données pour choisir le bit depth optimal

    Args:
        data: Array numpy (données du stack)
        format_hint: Indice du format source ('yuv420', 'raw12', 'raw16', None)

    Returns:
        Dict avec 'format', 'bit_depth', 'png_bit_depth'
    """
    # Si hint fourni, l'utiliser
    if format_hint:
        hint_lower = format_hint.lower()
        if 'yuv' in hint_lower or '420' in hint_lower:
            return {
                'format': 'YUV420',
                'bit_depth': 8,
                'png_bit_depth': 8,
                'reason': 'YUV420 est déjà 8-bit traité par ISP'
            }
        elif 'raw12' in hint_lower or 'rggb12' in hint_lower or 'srggb12' in hint_lower:
            return {
                'format': 'RAW12',
                'bit_depth': 12,
                'png_bit_depth': 16,
                'reason': 'RAW12 nécessite 16-bit pour préserver les 4096 niveaux'
            }
        elif 'raw16' in hint_lower or 'rggb16' in hint_lower:
            return {
                'format': 'RAW16',
                'bit_depth': 16,
                'png_bit_depth': 16,
                'reason': 'RAW16 nécessite 16-bit complet'
            }

    # Analyse heuristique basée sur les données
    if data.dtype == np.uint8:
        # Déjà 8-bit
        return {
            'format': 'YUV420 (assumed)',
            'bit_depth': 8,
            'png_bit_depth': 8,
            'reason': 'Données en uint8, probablement YUV420'
        }

    # Analyser la plage de valeurs
    data_max = np.max(data)
    data_min = np.min(data)

    if data_max <= 1.0:
        # Données normalisées float (0-1)
        # Estimer le bit depth original basé sur la distribution
        unique_count = len(np.unique(data[::10, ::10]))  # Échantillon

        if unique_count < 500:
            # Peu de valeurs uniques = probablement 8-bit
            return {
                'format': 'YUV420 (detected)',
                'bit_depth': 8,
                'png_bit_depth': 8,
                'reason': f'Peu de valeurs uniques ({unique_count}), probablement 8-bit'
            }
        else:
            # Beaucoup de valeurs = probablement 12/16-bit
            return {
                'format': 'RAW12/16 (detected)',
                'bit_depth': 12,
                'png_bit_depth': 16,
                'reason': f'Nombreuses valeurs uniques ({unique_count}), haute résolution'
            }

    elif data_max <= 255:
        # Valeurs 0-255
        return {
            'format': 'YUV420 (detected)',
            'bit_depth': 8,
            'png_bit_depth': 8,
            'reason': 'Plage 0-255, données 8-bit'
        }

    elif data_max <= 4095:
        # Valeurs 0-4095 (12-bit)
        return {
            'format': 'RAW12 (detected)',
            'bit_depth': 12,
            'png_bit_depth': 16,
            'reason': 'Plage 0-4095, données 12-bit'
        }

    else:
        # Valeurs > 4095 (16-bit ou float)
        return {
            'format': 'RAW16 (detected)',
            'bit_depth': 16,
            'png_bit_depth': 16,
            'reason': f'Plage étendue (max={data_max:.1f}), haute résolution'
        }


def save_png_auto(data, path, format_hint=None, force_bit_depth=None):
    """
    Sauvegarde PNG avec détection automatique du bit depth optimal

    Args:
        data: Image (float array normalisé 0-1)
        path: Chemin de sortie
        format_hint: Indice du format source ('yuv420', 'raw12', 'raw16')
        force_bit_depth: Forcer 8 ou 16-bit (None = auto)

    Returns:
        Dict avec infos de sauvegarde
    """
    # Détecter format source
    format_info = detect_source_format(data, format_hint)

    # Déterminer bit depth à utiliser
    if force_bit_depth:
        bit_depth = force_bit_depth
        reason = f"Forcé à {bit_depth}-bit par configuration"
    else:
        bit_depth = format_info['png_bit_depth']
        reason = format_info['reason']

    # Convertir les données
    if bit_depth == 8:
        # PNG 8-bit
        data_converted = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    else:
        # PNG 16-bit
        data_converted = (np.clip(data, 0, 1) * 65535).astype(np.uint16)

    # Convertir RGB en BGR pour OpenCV
    if len(data_converted.shape) == 3:
        data_converted = cv2.cvtColor(data_converted, cv2.COLOR_RGB2BGR)

    # Sauvegarder
    cv2.imwrite(str(path), data_converted)

    # Infos de retour
    file_size = Path(path).stat().st_size

    return {
        'path': str(path),
        'bit_depth': bit_depth,
        'format_detected': format_info['format'],
        'reason': reason,
        'file_size_kb': file_size / 1024
    }


def is_image_file(path):
    """
    Vérifie si le fichier est une image supportée

    Args:
        path: Chemin du fichier

    Returns:
        True si supporté
    """
    ext = Path(path).suffix.lower()
    return ext in [
        '.fit', '.fits',
        '.dng', '.cr2', '.nef', '.arw', '.raw',
        '.jpg', '.jpeg', '.png', '.tif', '.tiff'
    ]
