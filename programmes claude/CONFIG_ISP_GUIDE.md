# Guide de configuration ISP pour RPiCamera2

## Problèmes résolus

✅ Les champs ISP ont été ajoutés à toutes les classes de configuration
✅ Les traceurs de debug ont été ajoutés dans `session.py`
✅ La conversion `to_legacy_config()` copie maintenant les paramètres ISP
✅ Le format RAW est maintenant correctement propagé (fix: `camera_params['raw_format']`)
✅ PNG 16-bit automatique pour RAW12/RAW16
✅ Convertisseur libcamera ISP → libastrostack disponible

## Configuration du format vidéo

Pour que le PNG soit sauvegardé en **16-bit** pour les images RAW12/16, vous devez définir le `video_format` dans votre configuration.

### Option 1 : Dans PiLCConfig104.txt

Ajoutez ces lignes dans votre fichier de configuration :

```ini
# Format vidéo source (important pour PNG 16-bit)
video_format=raw12    # ou 'raw16' ou 'yuv420'

# Activer l'ISP (optionnel)
isp_enable=False      # True pour activer
isp_config_path=      # Chemin vers config ISP (vide = auto-calibration)

# Forcer le bit depth PNG (optionnel)
png_bit_depth=        # Vide=auto, 8, ou 16
```

### Option 2 : Dans le code Python

Si vous modifiez directement le code, ajoutez après la lecture de la config :

**Dans RPiCamera2.py :**
```python
# Après avoir chargé la config
config.video_format = "raw12"  # ou "raw16", "yuv420"
config.output.png_bit_depth = None  # Auto-détection (recommandé)
# config.output.png_bit_depth = 16  # Forcer 16-bit

# Optionnel : activer l'ISP
# config.isp_enable = True
# config.isp_config_path = "isp_config.json"
```

**Dans rpicamera_livestack_advanced.py (avant self.session = LiveStackSession) :**
```python
# Option A : Modifier self.config directement
self.config.video_format = "raw12"
self.config.output.png_bit_depth = None  # Auto

# Option B : Modifier legacy_config après conversion
legacy_config = self.config.to_legacy_config()
legacy_config.video_format = "raw12"
legacy_config.png_bit_depth = None  # Auto
self.session = LiveStackSession(legacy_config)
```

## Détection automatique du format

Avec les traceurs activés, vous verrez maintenant dans la console :

```
[DEBUG get_preview_png] Début
  • ISP activé: False
  • ISP instance: False
  • Format vidéo: raw12
  • PNG bit depth config: None
  • Stack result shape: (2180, 3872, 3), dtype: float32
  • Stack result range: [0.023, 0.891]
  → Pas d'ISP (enable=False, isp=False)
  • Stretched range: [0.000, 1.000]
  → Sélection bit depth:
     config.png_bit_depth = None
     config.video_format = raw12
     → 16-bit (auto: 'raw' détecté dans 'raw12')
  ✓ Preview final: dtype=uint16, shape=(2180, 3872, 3)
     range=[0, 65535]
```

## Tableau de décision automatique

| Format vidéo | png_bit_depth | Résultat |
|--------------|---------------|----------|
| `yuv420` | `None` (auto) | **8-bit** (~6 MB) |
| `raw12` | `None` (auto) | **16-bit** (~12 MB) |
| `raw16` | `None` (auto) | **16-bit** (~12 MB) |
| N'importe quel | `8` (forcé) | **8-bit** |
| N'importe quel | `16` (forcé) | **16-bit** |

## Vérification

Après configuration, relancez votre capture et vérifiez :

1. **Dans la console** : Les messages de debug montreront le format détecté
2. **Taille des fichiers** :
   - PNG 8-bit : ~6-8 MB
   - PNG 16-bit : ~12-16 MB
3. **Vérification manuelle** :
   ```python
   import cv2
   import numpy as np
   img = cv2.imread("stack.png", cv2.IMREAD_UNCHANGED)
   print(f"dtype: {img.dtype}")  # uint8 ou uint16
   ```

## Méthode 1 : Utiliser la configuration ISP du capteur (RECOMMANDÉ)

La méthode la plus simple est de convertir la configuration ISP libcamera de votre capteur :

```bash
# Conversion pour lumière du jour (5500K)
python3 convert_libcamera_isp.py /home/admin/Rpicamera\ tests/imx585_lowlight.json \
    -o isp_config_imx585.json -t 5500

# Ou pour d'autres températures de couleur :
# Tungstène (3000K)
python3 convert_libcamera_isp.py imx585_lowlight.json -o isp_tungsten.json -t 3000

# Lumière du jour nuageuse (6500K)
python3 convert_libcamera_isp.py imx585_lowlight.json -o isp_daylight.json -t 6500
```

**Avantages** :
- ✅ Utilise les paramètres calibrés en usine pour votre capteur (IMX585)
- ✅ Inclut black level, CCM (Color Correction Matrix), gamma, balance des blancs
- ✅ Aucune calibration manuelle nécessaire
- ✅ Résultats optimaux pour votre capteur spécifique

**Configuration dans PiLCConfig104.txt** :
```ini
isp_enable=True
isp_config_path=isp_config_imx585.json
video_format=raw12
```

## Méthode 2 : Calibration manuelle (comparaison RAW vs YUV)

Pour calibrer l'ISP manuellement en comparant RAW12 et YUV420 :

```python
# 1. Calibrer l'ISP (une fois)
from libastrostack.isp import ISPCalibrator, ISP
config_isp = ISPCalibrator.calibrate_from_files("raw12_ref.png", "yuv420_ref.png")
ISP(config_isp).save_config("isp_config.json")

# 2. Configurer la session
config.isp_enable = True
config.isp_config_path = "isp_config.json"
config.video_format = "raw12"
config.output.png_bit_depth = None  # Auto 16-bit
config.output.fits_linear = True    # FITS RAW

# 3. La session appliquera automatiquement :
#    RAW12 → Stack → ISP → Stretch → PNG 16-bit
```

## Notes importantes

- **FITS** : Par défaut, le FITS est maintenant sauvegardé en **linéaire (vrai RAW)** avec `fits_linear=True`
- **Performance** : L'ISP est appliqué **1x après le stack**, pas sur chaque frame (optimal)
- **Compatibilité** : Si `video_format=None`, le système choisit 8-bit par sécurité

## Relancez maintenant votre programme !

Les traceurs vous montreront exactement ce qui se passe. 🎯
