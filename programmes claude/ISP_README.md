# ISP (Image Signal Processor) - Documentation

## Vue d'ensemble

L'ISP traite les images RAW pour améliorer leur rendu visuel. Dans notre application astrophotographie, **l'ISP s'applique UNIQUEMENT aux PNG de prévisualisation**, jamais aux fichiers FITS scientifiques.

## Pipeline de traitement

### Fichiers FITS (données scientifiques)
```
RAW → Debayer → Stack → FITS linéaire
                         ↑
                    PAS D'ISP !
```

### Fichiers PNG (prévisualisation écran)
```
RAW → Debayer → Stack → ISP → Stretch → PNG
                         ↑
                    ISP ICI !
```

## Configuration actuelle

**Fichier** : `isp_config_neutral.json`

Cette configuration est **neutre/transparente** - elle ne modifie pas les images :

```json
{
  "black_level": 0,           // Pas de soustraction de noir
  "wb_red_gain": 1.0,         // Pas de correction WB (déjà fait par débayeurisation)
  "wb_green_gain": 1.0,
  "wb_blue_gain": 1.0,
  "ccm": [[1.0, 0.0, 0.0],    // Matrice identité (pas de correction couleur)
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]],
  "gamma": 1.0,               // Linéaire (pas de courbe tonale)
  "contrast": 1.0,            // Pas de boost de contraste
  "saturation": 1.0,          // Saturation normale
  "sharpening": 0.0           // Pas d'accentuation
}
```

## Paramètres ISP disponibles

### 1. Black Level (Niveau de noir)
- **Valeur** : 0 - 4095 (pour 12-bit)
- **Effet** : Soustrait le bruit de fond du capteur
- **Recommandation astro** : 0 (désactivé) ou calibrer avec dark frames

### 2. White Balance (Balance des blancs)
- **Valeurs** : red_gain, green_gain, blue_gain (0.1 - 10.0)
- **Effet** : Corrige la dominante de couleur
- **Recommandation astro** : 1.0, 1.0, 1.0 (déjà géré par débayeurisation)

### 3. CCM (Color Correction Matrix)
- **Valeur** : Matrice 3×3
- **Effet** : Corrige les imperfections du filtre Bayer
- **Recommandation astro** : Matrice identité (problème d'inversion R/B à résoudre)

### 4. Gamma (Courbe tonale)
- **Valeurs** : 1.0 (linéaire), 2.2 (sRGB), 3.0 (contrasté)
- **Effet** : Courbe de réponse tonale
- **Recommandation astro** : 1.0 (linéaire, le stretch gère la courbe)

### 5. Contrast (Contraste)
- **Valeurs** : 0.5 - 2.0 (1.0 = normal)
- **Effet** : Étire/compresse la plage tonale
- **Recommandation astro** : 1.0 (le stretch gère le contraste)

### 6. Saturation
- **Valeurs** : 0.0 - 2.0 (0 = N&B, 1.0 = normal)
- **Effet** : Intensité des couleurs
- **Recommandation astro** : 1.0 (couleurs naturelles) ou 1.1-1.2 (nébuleuses)

### 7. Sharpening (Accentuation)
- **Valeurs** : 0.0 - 2.0
- **Effet** : Renforce les contours
- **Recommandation astro** : 0.0 (amplifie le bruit, à faire en post-traitement)

## Activation/Désactivation

**Dans PiLCConfig104.txt, ligne 84** :
```
isp_enable : 0   # ISP désactivé
isp_enable : 1   # ISP activé
```

## Problèmes connus

### 1. Inversion Rouge/Bleu
Le débayeurisation inverse les gains R/B (lignes 4931-4932 de RPiCamera2.py). La CCM de libcamera n'est donc pas adaptée.

**Solution future** : Adapter la CCM pour tenir compte de cette inversion.

### 2. Ancienne config (isp_config_imx585.json)
Cette config avait des paramètres trop agressifs :
- `black_level: 3200` → Écrasait 78% de la plage dynamique
- `gamma: 3.0` → Courbe trop contrastée
- Résultat : PNG 46MB → 1.5MB, images catastrophiques

**Solution** : Remplacée par `isp_config_neutral.json` (neutre)

## Fichiers du système ISP

- `libastrostack/isp.py` - Module ISP (classe ISP, ISPConfig, ISPCalibrator)
- `isp_config_neutral.json` - Config neutre (actuelle)
- `isp_config_imx585.json` - Config libcamera (non adaptée, ne pas utiliser)
- `convert_libcamera_isp.py` - Script de conversion libcamera → notre format

## Évolutions futures possibles

1. **Calibration du black level** avec dark frames
2. **Adaptation de la CCM** pour l'inversion R/B
3. **Gamma léger** (2.2) pour PNG uniquement si demandé
4. **Saturation douce** (1.1-1.2) pour faire ressortir les couleurs des nébuleuses
5. **Profils ISP** : Neutre, Planétaire, Nébuleuses, etc.
