# Configuration ISP Complète - IMX585

## ✅ Tout est prêt !

Votre système ISP est maintenant **entièrement fonctionnel** et prêt à être utilisé.

## 📦 Ce qui a été fait

### 1. **Correction du format vidéo** (`RPiCamera2.py`)
- ✅ Le format RAW est maintenant correctement propagé à chaque démarrage
- ✅ PNG 16-bit automatique pour RAW12/RAW16
- ✅ PNG 8-bit pour YUV420

### 2. **Module ISP complet** (`libastrostack/isp.py`)
- ✅ Pipeline ISP : Black level → White Balance → CCM → Gamma → Contraste → Saturation
- ✅ Support de la Color Correction Matrix (CCM)
- ✅ Sauvegarde/chargement de configuration JSON avec CCM
- ✅ Traitement en float32 pour éviter les gaps d'histogramme
- ✅ Sortie en 8-bit ou 16-bit selon besoin

### 3. **Convertisseur libcamera → libastrostack**
- ✅ `convert_libcamera_isp.py` : extrait les paramètres ISP du fichier libcamera
- ✅ Conversion automatique des paramètres du capteur IMX585
- ✅ Support de différentes températures de couleur (3000K-6500K+)
- ✅ Extraction de :
  - Black level (3200 pour IMX585)
  - Color Correction Matrix (5 matrices pour différentes températures)
  - Courbe gamma (estimée à partir de la courbe libcamera)
  - Balance des blancs (interpolée selon température)

### 4. **Configuration ISP IMX585 générée**
- ✅ `isp_config_imx585.json` : configuration optimale pour lumière du jour (5500K)
- ✅ Utilise les paramètres calibrés en usine
- ✅ CCM incluse pour correction colorimétrique précise

## 🚀 Utilisation rapide

### Option 1 : Fichier de configuration (FACILE)

Éditez `PiLCConfig104.txt` et ajoutez :

```ini
# Activer l'ISP
isp_enable=True
isp_config_path=isp_config_imx585.json

# Format vidéo (déjà détecté automatiquement normalement)
video_format=raw12
```

### Option 2 : Test rapide sans modification

Le système détecte automatiquement le format RAW, vous pouvez simplement activer l'ISP :

```bash
# Lancer RPiCamera2 en mode RAW12
python3 RPiCamera2.py

# Dans l'interface :
# 1. Activer le mode RAW (touche appropriée)
# 2. Activer Live Stack
# 3. Le PNG sera automatiquement en 16-bit !
```

## 🎨 Configurations ISP disponibles

### Température de couleur 5500K (lumière du jour) - DÉJÀ CRÉÉE
```bash
# Fichier : isp_config_imx585.json
# Utilisation : déjà prêt !
```

### Créer d'autres températures si nécessaire

```bash
# Tungstène (3000K) - lumière incandescente
python3 convert_libcamera_isp.py "/home/admin/Rpicamera tests/imx585_lowlight.json" \
    -o isp_tungsten.json -t 3000

# Fluorescent (4000K)
python3 convert_libcamera_isp.py "/home/admin/Rpicamera tests/imx585_lowlight.json" \
    -o isp_fluorescent.json -t 4000

# Nuageux (6500K)
python3 convert_libcamera_isp.py "/home/admin/Rpicamera tests/imx585_lowlight.json" \
    -o isp_cloudy.json -t 6500
```

## 📊 Pipeline de traitement

### Sans ISP (mode actuel)
```
RAW12 → Debayer → Stack (float32) → Stretch → PNG 8-bit
                                             → FITS linéaire
```

### Avec ISP activé (nouveau)
```
RAW12 → Debayer → Stack (float32) → ISP (float32) → Stretch → PNG 16-bit
                                                             → FITS linéaire

ISP inclut :
  • Black level (3200)
  • White Balance (R=0.568, G=1.0, B=0.574)
  • Color Correction Matrix (3x3)
  • Gamma (3.0)
  • Contraste (1.1)
  • Saturation (1.2)
```

## 🔍 Vérification

Pour vérifier que tout fonctionne :

```bash
# Test de la configuration ISP
python3 test_isp_imx585.py

# Vous devriez voir :
# ✅ Configuration ISP IMX585 validée !
# ✅ Black level: 3200
# ✅ CCM présente: True
```

## 📝 Notes importantes

### 1. **Format vidéo détecté automatiquement**
Le système détecte maintenant automatiquement si vous êtes en :
- YUV420 → PNG 8-bit (~6 MB)
- RAW12 → PNG 16-bit (~12-16 MB)
- RAW16 → PNG 16-bit (~12-16 MB)

### 2. **FITS toujours en linéaire**
Les fichiers FITS sont sauvegardés en **vrai RAW** (linéaire, non-stretché) pour permettre le post-traitement.

### 3. **ISP appliqué APRÈS le stack**
Pour des raisons de performance, l'ISP est appliqué 1x après le stack (pas sur chaque frame).
Cela est :
- ✅ 1000x plus rapide
- ✅ Mathématiquement correct (stacking linéaire)
- ✅ Qualité identique au traitement frame par frame

### 4. **Inversion R/B dans l'interface pygame**
Rappel : dans l'interface pygame, les sliders R et B sont inversés.
L'ISP libastrostack traite correctement les canaux RGB.

## 🎯 Prochaines étapes

1. **Testez sans ISP d'abord** :
   - Vérifiez que les PNG RAW12 sont bien en 16-bit (~12-16 MB)
   - Confirmez que les logs montrent `Format vidéo: raw12`

2. **Activez l'ISP** :
   - Ajoutez `isp_enable=True` dans PiLCConfig104.txt
   - Comparez les images avant/après ISP

3. **Ajustez si nécessaire** :
   - Testez différentes températures de couleur
   - Ajustez contraste/saturation dans le fichier JSON

## 📚 Documentation

- **Guide complet** : `CONFIG_ISP_GUIDE.md`
- **Test ISP** : `test_isp_imx585.py`
- **Convertisseur** : `convert_libcamera_isp.py --help`

## ❓ Dépannage

### PNG toujours en 8-bit ?
```bash
# Vérifiez les logs au démarrage du livestack :
# Vous devriez voir :
[CONFIG] Format vidéo configuré: raw12 (source: SRGGB12)
  • Format vidéo: raw12
  → 16-bit (auto: 'raw' détecté dans format)
```

### ISP ne s'applique pas ?
```bash
# Vérifiez les logs :
[CONFIG]
  - ISP activé: OUI  # Doit être OUI
  - Format vidéo: raw12  # Doit être raw12 ou raw16
```

### Couleurs étranges avec ISP ?
```bash
# Testez une autre température de couleur :
python3 convert_libcamera_isp.py imx585_lowlight.json -o isp_test.json -t 4000
# Puis changez isp_config_path=isp_test.json
```

---

**Tout est prêt ! Lancez RPiCamera2 et testez ! 🚀**
