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


def save_fits(data, path, header_data=None):
    """
    Sauvegarde image en FITS
    
    Args:
        data: Image (float array)
        path: Chemin de sortie
        header_data: Dictionnaire de métadonnées optionnelles
    
    Returns:
        True si succès
    """
    if not HAS_FITS:
        print("[ERROR] astropy non disponible")
        return False
    
    try:
        # Étirer pour FITS (percentiles)
        is_color = len(data.shape) == 3
        
        if is_color:
            stretched = np.zeros_like(data)
            for i in range(3):
                vmin = np.percentile(data[:,:,i], 1)
                vmax = np.percentile(data[:,:,i], 99)
                stretched[:,:,i] = np.clip((data[:,:,i] - vmin) / (vmax - vmin), 0, 1)
            
            # Convertir en uint16
            result_16bit = (stretched * 65535).astype(np.uint16)
            
            # Transposer pour FITS (3, H, W)
            result_16bit = result_16bit.transpose(2, 0, 1)
            
            hdu = fits.PrimaryHDU(result_16bit)
            hdu.header['COLORTYP'] = 'RGB'
        else:
            vmin = np.percentile(data, 1)
            vmax = np.percentile(data, 99)
            stretched = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            result_16bit = (stretched * 65535).astype(np.uint16)
            
            hdu = fits.PrimaryHDU(result_16bit)
        
        # Ajouter métadonnées
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
