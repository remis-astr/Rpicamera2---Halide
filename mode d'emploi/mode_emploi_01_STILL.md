# MODE EMPLOI RPiCamera2 - MODE STILL (Photographie)

## 1. VUE D'ENSEMBLE

Le mode **STILL** permet de capturer des images fixes haute résolution avec contrôle précis de l'exposition, de la balance des blancs et du traitement d'image. Il s'agit du mode principal pour la photographie astronomique ou terrestre avec RPiCamera2.

**Déclenchement** : Menu principal → Bouton **STILL** (ligne 0, colonne 0)

**Formats de sortie supportés** :
- **RAW** : DNG (Adobe Digital Negative) ou fichiers .raw Bayer
- **Traités** : JPEG, PNG, BMP (avec traitement ISP intégré)

---

## 2. PARAMÈTRES D'EXPOSITION

### 2.1 Mode d'exposition (mode)

**Valeurs** : 0 (Manuel) | 1-3 (Automatique)

| Mode | Description | Usage |
|------|-------------|-------|
| **0 - Manuel** | Contrôle total exposition/gain | Astrophoto, poses longues |
| **1 - Auto** | AE/AWB automatiques | Usage général |
| **2 - Night** | Optimisé scènes sombres | Paysages nocturnes |
| **3 - Sports** | Priorité vitesse | Action rapide |

**Impact pipeline** :
```
Mode 0 (manuel):
  → Configuration: create_still_configuration()
  → Commandes: --immediate --shutter sspeed --gain gain --awbgains red,blue

Mode 1-3 (auto):
  → Configuration: create_preview_configuration()
  → Commandes: --ev {ev} --metering {meter} --awb {awb_preset}
  → Attente stabilisation: -t {timet} ms (défaut 100ms)
```

**Recommandation** : Mode manuel (0) pour l'astrophotographie pour garantir exposition constante.

### 2.2 Temps d'exposition (speed / sspeed)

**Valeurs** : Index dans tableau `shutters[]` (0-73)

Exemples de valeurs :
```
Index  Vitesse        Microsecondes  Usage
0      1/8000s        125            Plein jour
16     1/125s         8000           Standard
30     1/30s          33333          Intérieur
50     1s             1000000        Pose courte astro
60     10s            10000000       Pose moyenne astro
73     100s           100000000      Pose longue (max)
```

**Limites par caméra** :
| Caméra | Max shutter | Notes |
|--------|-------------|-------|
| IMX585 | 100s | Mode native ou binning |
| Pi HQ | 230s | Excellente sensibilité |
| Pi v3 | 100s | Moderne, performante |
| Pi v2 | 100s | Standard |

**Calcul automatique** :
```python
sspeed = shutters[speed]  # Vitesse en microsecondes
max_frame_duration = max_shutters[Pi_Cam] * 1_000_000  # µs

# Configuration Picamera2
controls["ExposureTime"] = sspeed
controls["FrameDurationLimits"] = (min_duration, max(sspeed, max_frame_duration))
```

**Impact sur le système** :
- Expositions > 80ms → Force `create_still_configuration()` même en mode preview
- Expositions longues (>10s) → Désactive preview pendant capture
- Mode manuel + AWB → Force gains manuels `--awbgains`

### 2.3 Gain analogique (gain)

**Valeurs** : 0 (auto) à max_gain (dépend de la caméra)

| Caméra | Max gain | Équiv. ISO | Mapping |
|--------|----------|------------|---------|
| IMX585 | **3000*** | ~75000 ISO | Non-linéaire (slider_to_gain_nonlinear) |
| Pi HQ | 88 | ~22000 ISO | Linéaire |
| Pi v3 | 64 | ~16000 ISO | Linéaire |
| Pi v2 | 40 | ~10000 ISO | Linéaire |

*IMX585 utilise un mapping non-linéaire spécial :
```python
# Conversion slider → gain réel
def slider_to_gain_nonlinear(slider_value, max_gain=3000):
    if slider_value <= 50:
        return slider_value * 2  # Zone linéaire 0-100
    else:
        # Zone exponentielle 100-3000
        return 100 + (slider_value - 50) * ((max_gain - 100) / 50)
```

**Usage astrophotographie** :
- **Gain faible (0-10)** : Ciel profond, signal fort (poses longues)
- **Gain moyen (10-30)** : Objets faibles, balance bruit/signal
- **Gain élevé (>30)** : Lucky imaging planétaire, vidéo haute cadence

**Impact qualité** :
```
Bruit de lecture = N_read
Bruit total = √(Signal/Gain + N_read²)

→ Gain élevé réduit bruit de lecture mais augmente bruit shot
→ Gain faible améliore SNR si signal suffisant
```

### 2.4 Compensation d'exposition (ev)

**Valeurs** : -10 à +10 (stops)

**Effet** : Ajuste la cible de luminosité en mode automatique uniquement.

```python
# Seulement en mode auto (mode > 0)
controls["ExposureValue"] = ev
```

**Correspondance** :
```
-10 EV : Image très sombre (1/1024 de la lumière)
  0 EV : Exposition neutre
+10 EV : Image très claire (1024× la lumière)
```

**Note** : Ignoré en mode manuel (mode=0).

---

## 3. BALANCE DES BLANCS (AWB)

### 3.1 Mode AWB (awb)

**Valeurs** : 0-7

| Index | Mode | Temp. couleur | Usage |
|-------|------|---------------|-------|
| **0** | **Manuel** | Custom | Astrophoto, contrôle précis |
| 1 | Auto | Variable | Usage général |
| 2 | Incandescent | ~2700K | Ampoules tungstène |
| 3 | Tungsten | ~3200K | Studio |
| 4 | Fluorescent | ~4000K | Néons |
| 5 | Indoor | ~3500K | Intérieur mixte |
| 6 | Daylight | ~5500K | Extérieur jour |
| 7 | Cloudy | ~6500K | Ciel nuageux |

**Pipeline** :
```python
if awb == 0:  # Manuel
    controls["AwbMode"] = controls.AwbModeEnum.Off
    controls["ColourGains"] = (red/10, blue/10)  # Gains manuels
else:  # Auto
    controls["AwbMode"] = awb_modes[awb]
    # Caméra calcule gains automatiquement
```

### 3.2 Gains couleur manuels (red, blue)

**Valeurs** : 0-100 (divisés par 10 pour application)

**Gains réels** : 0.0 à 10.0 (typiquement 1.0-3.0 utilisé)

| Température scène | Gain Rouge | Gain Bleu | Ratio R:G:B |
|-------------------|------------|-----------|-------------|
| Ciel nocturne | 1.5 | 1.2 | 1.5:1:1.2 |
| Lumière naturelle | 2.0 | 1.5 | 2.0:1:1.5 |
| Halogène chaud | 1.0 | 2.5 | 1.0:1:2.5 |
| LED froid | 2.5 | 1.0 | 2.5:1:1.0 |

**Application dans le pipeline** :

1. **Capture RAW** :
   ```python
   # Gains stockés dans métadonnées DNG
   # Appliqués pendant débayérisation
   ```

2. **Débayérisation manuelle** :
   ```python
   rgb_float[:,:,0] *= red/10    # Canal rouge
   rgb_float[:,:,1] *= 1.0       # Canal vert (référence)
   rgb_float[:,:,2] *= blue/10   # Canal bleu
   ```

3. **Capture directe JPEG/PNG** :
   ```
   rpicam-still --awbgains {red/10},{blue/10}
   ```

**Conseil astrophotographie** :
```
Départ : red=15 (1.5), blue=12 (1.2)
Ajuster selon histogramme RGB pour équilibrer canaux
Objectif : Fond de ciel neutre gris (R≈G≈B)
```

---

## 4. QUALITÉ IMAGE

### 4.1 Luminosité (brightness)

**Valeurs** : -100 à +100

**Effet** : Offset linéaire ajouté après débayérisation

```python
controls["Brightness"] = brightness / 100  # Normalisé [-1.0, 1.0]
```

**Application** :
```
Pixel_out = Pixel_in + (brightness/100)
```

**Usage** :
- `brightness < 0` : Assombrir (réduire bruit fond de ciel)
- `brightness = 0` : Neutre (recommandé astrophoto)
- `brightness > 0` : Éclaircir (objets faibles)

**Note** : Préférer ajustement exposition/gain plutôt que brightness pour préserver SNR.

### 4.2 Contraste (contrast)

**Valeurs** : 0-200 (défaut 100 = neutre)

**Effet** : Multiplicateur autour du point milieu (0.5)

```python
controls["Contrast"] = contrast / 100  # Normalisé [0.0, 2.0]
```

**Transformation** :
```
Pixel_out = (Pixel_in - 0.5) × (contrast/100) + 0.5
```

| Valeur | Effet | Usage |
|--------|-------|-------|
| < 100 | Réduit contraste | HDR, scènes contrastées |
| 100 | Neutre | Standard |
| > 100 | Augmente contraste | Objets faibles, planétaire |

**Recommandation astrophoto** : 70-100 (légèrement réduit pour préserver dynamique).

### 4.3 Saturation (saturation)

**Valeurs** : 0-100 (divisé par 10)

**Effet** : Multiplie l'intensité des couleurs

```python
controls["Saturation"] = saturation / 10  # Normalisé [0.0, 10.0]
```

**Transformation** :
```
Chroma = (Pixel_couleur - Luminance) × (saturation/10)
Pixel_out = Luminance + Chroma
```

| Valeur | Effet | Usage |
|--------|-------|-------|
| 0 | Noir & blanc | Éliminer bruit chromatique |
| 5-10 | Désaturé | Naturel, astrophoto ciel profond |
| 10 | Neutre | Standard (défaut) |
| 15-20 | Saturé | Nébuleuses colorées, planétaire |

**Note** : La saturation est appliquée **avant** l'enregistrement JPEG/PNG mais **ignorée** sur les captures RAW.

### 4.4 Netteté (sharpness)

**Valeurs** : 0-100 (divisé par 10)

**Effet** : Filtre de netteté (unsharp mask)

```python
controls["Sharpness"] = sharpness / 10  # Normalisé [0.0, 10.0]
```

**Recommandations** :
- **Astrophoto** : 0-5 (désactiver ou minimal, traitement post)
- **Planétaire** : 15-25 (accentuer détails)
- **Terrestre** : 10-20 (standard)

**Note** : Appliqué dans l'ISP, ignoré pour captures RAW.

### 4.5 Débruitage (denoise)

**Valeurs** : 0-3

| Valeur | Mode | Qualité | Temps | Usage |
|--------|------|---------|-------|-------|
| **0** | Off | - | Instantané | RAW astrophoto |
| **1** | CDN_Off | Faible | Rapide | Vidéo rapide |
| **2** | CDN_Fast | Moyenne | Moyen | Balance qualité/vitesse |
| **3** | CDN_HQ | Élevée | Lent | Qualité maximale |

**Mapping** :
```python
denoise_modes = {
    0: controls.draft.NoiseReductionModeEnum.Off,
    1: controls.draft.NoiseReductionModeEnum.Fast,
    2: controls.draft.NoiseReductionModeEnum.HighQuality,
    3: controls.draft.NoiseReductionModeEnum.Minimal
}
```

**Impact** :
- Mode 0 (Off) : Préserve texture pour traitement post (recommandé astrophoto)
- Mode 3 (HQ) : Maximum réduction bruit mais peut flouter détails fins

**Note** : Débruitage **ignoré pour RAW**, appliqué uniquement sur JPEG/PNG/YUV.

---

## 5. FORMATS ET RÉSOLUTION

### 5.1 Format de sortie (extn)

**Valeurs** : 0-3 + mode RAW

| Index | Format | Qualité | Taille | Usage |
|-------|--------|---------|--------|-------|
| **0** | JPEG | Variable (param quality) | Faible | Standard |
| **1** | PNG | Sans perte | Élevée | Préservation détails |
| **2** | BMP | Sans compression | Très élevée | Traitement brut |
| **3** | DNG | RAW 12/16-bit | Maximale | Astrophoto |

**Configuration** :
```python
extns = ["jpg", "png", "bmp", "dng"]
extension = extns[extn]
```

**Qualité JPEG (quality)** : 0-100
- 80-90 : Bon compromis taille/qualité
- 95-100 : Qualité maximale (recommandé astrophoto si JPEG utilisé)

### 5.2 Format RAW (raw_format)

**Valeurs** : 0 (SRGGB12) | 1 (SRGGB16)

| Format | Bits/pixel | Plage | Taille | Usage |
|--------|-----------|-------|--------|-------|
| **SRGGB12** | 12 | 0-4095 | Moyen | Rapide, suffisant plupart cas |
| **SRGGB16** | 16 | 0-65535 | Élevé | Haute dynamique, objet brillant |

**Impact sur le pipeline** :

1. **Changement format → Recréation Picamera2**
   ```python
   need_recreation = (raw_format != prev_config['raw_format'])
   ```

2. **Configuration stream RAW** :
   ```python
   if raw_format == 0:  # SRGGB12
       raw_config = {"format": "SRGGB12", "size": raw_stream_size}
   else:  # SRGGB16
       raw_config = {"format": "SRGGB16", "size": raw_stream_size}
   ```

3. **Débayérisation** :
   ```python
   # Lecture données brutes
   raw_uint16 = raw_array.view(np.uint16)

   if raw_format == 0:  # 12-bit
       # Valeurs déjà correctes (0-4095)
       pass
   else:  # 16-bit
       # Valeurs étendues (0-65535)
       pass
   ```

**Recommandation** :
- **SRGGB12** : Défaut, suffisant pour 99% des cas
- **SRGGB16** : Soleil, Lune, planètes brillantes (éviter saturation)

### 5.3 Résolution et zoom

**Zoom (zoom)** : 0-5

Le comportement diffère selon la caméra :

#### IMX585 - Modes hardware crop
```
Zoom  Résolution    Mode capteur     FPS max
0     1928×1090     Binning 2×2      178 fps
0     3856×2180     Native 4K        51 fps  (use_native_sensor_mode=1)
1     2880×2160     Crop Mode 2      51 fps
2     1920×1080     Crop Mode 3      100 fps
3     1280×720      Crop Mode 4      150 fps
4-5   800×600       Crop Mode 5      178 fps
```

**Avantages crop hardware** :
- Résolution réduite → FPS augmenté
- Latence réduite (moins de données à transférer)
- Qualité préservée (pas d'interpolation)

#### Autres caméras - ROI logiciel
```python
# Calcul ROI centré
zfs = [1.0, 0.65, 0.5, 0.4, 0.32, 0.27]  # Fractions de résolution
roi_width = int(igw * zfs[zoom])
roi_height = int(igh * zfs[zoom])
zxo = ((igw - roi_width) / 2) / igw  # Centré horizontalement
zyo = ((igh - roi_height) / 2) / igh  # Centré verticalement
```

**Commande CLI** :
```
rpicam-still --roi {zxo},{zyo},{roi_w_norm},{roi_h_norm}
```

### 5.4 Mode capteur natif (use_native_sensor_mode)

**Valeurs** : 0 (Binning) | 1 (Native)

| Mode | Résolution | Qualité | Vitesse | Bruit |
|------|------------|---------|---------|-------|
| **Binning (0)** | Réduite (2×2) | Bonne | Rapide | Moyen |
| **Native (1)** | Maximale | Excellente | Lente | Faible |

**Exemple IMX585** :
```
Binning : 1928×1090 (2×2 pixels moyennés) → 178 fps max
Native  : 3856×2180 (chaque pixel lu) → 51 fps max
```

**Binning 2×2 : Avantages**
- Augmente sensibilité (~4× moins de bruit)
- Réduit taille fichier (4× moins de pixels)
- Accélère traitement (lecture + transfert)

**Native : Avantages**
- Résolution maximale (détails fins)
- Dynamique étendue (moins de saturation pixels brillants)
- Meilleur échantillonnage (plus de précision)

**Recommandation** :
- **Binning** : Preview rapide, vidéo haute cadence, gain sensibilité
- **Native** : Still haute résolution, impression, détails critiques

**Impact système** : Changement → Recréation complète Picamera2

---

## 6. TRAITEMENTS D'IMAGE

### 6.1 Mode HDR (v3_hdr)

**Valeurs** : 0-4 (IMX585 et caméras compatibles)

| Index | Mode | Description | Usage |
|-------|------|-------------|-------|
| **0** | Off | Désactivé | Standard, astrophoto |
| **1** | Single Exposure | HDR mono-expo | Scènes contrastées |
| **2** | Multi Exposure | HDR multi-expo | Paysages dynamique élevée |
| **3** | Night | Optimisé nuit | Faible lumière |
| **4** | Clear HDR 16-bit | Mode 16-bit IMX585 | Max dynamique astrophoto |

**Mode Clear HDR (IMX585 uniquement)** :
```python
if v3_hdr == 4:
    controls["HdrMode"] = controls.HdrModeEnum.Off
    # Active mode capteur 16-bit natif
    # Dynamique étendue : 0-65535 (16384× plus qu'en 12-bit)
```

**Limitations** :
- Multi-Exposure incompatible avec mode manuel (mode=0)
- HDR augmente latence capture
- Changement HDR → Recréation Picamera2

### 6.2 Stretch astro (stretch_mode)

**Valeurs** : 0-2

| Index | Méthode | Usage | Paramètres |
|-------|---------|-------|-----------|
| **0** | Off | Désactivé | - |
| **1** | GHS | Généraliste astrophoto | D, b, SP, LP, HP |
| **2** | Arcsinh | Simple, efficace | factor, p_low, p_high |

**Note** : Stretch appliqué uniquement au **preview pygame**, pas aux fichiers sauvegardés STILL.

Pour sauvegarder avec stretch → Utiliser mode **LIVESTACK** avec `auto_png=True`.

### 6.3 Métadonnées capture

**Sauvegarde automatique** (si `show_cmds=1`) :
```python
metadata = picam2.capture_metadata()
# Contient:
# - ExposureTime (µs)
# - AnalogueGain
# - DigitalGain
# - ColourGains (R, B)
# - SensorTemperature
# - AeLocked, AwbLocked
# - FocusFoM (Figure of Merit)
```

**Stockage DNG** : Métadonnées EXIF complètes incluses automatiquement.

---

## 7. WORKFLOW COMPLET

### 7.1 Pipeline de capture STILL

```
┌─────────────────────────┐
│ Appui bouton STILL      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Vérifier config active  │
│ - Mode manuel/auto      │
│ - Résolution/zoom       │
│ - Format RAW/JPEG       │
└──────────┬──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Recréation   │───── Si changement majeur (format, HDR, zoom)
    │ nécessaire ? │
    └──────┬───────┘
           │ Non (fast path)
           ▼
┌─────────────────────────┐
│ Mise à jour contrôles   │
│ - ExposureTime          │
│ - AnalogueGain          │
│ - ColourGains (AWB)     │
│ - Brightness/Contrast   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Attente stabilisation   │
│ -t {timet} ms           │
│ (défaut 100ms)          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Capture image           │
│ Picamera2:              │
│  capture_file(name='raw')│
│  ou capture_file(name='main')│
│                         │
│ rpicam-still:           │
│  --output filename      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Post-traitement         │
│ - RAW → DNG (si demandé)│
│ - JPEG/PNG (ISP intégré)│
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Sauvegarde fichier      │
│ /home/admin/Pictures/   │
│ horodaté                │
└─────────────────────────┘
```

### 7.2 Exemple configuration astrophoto

**Ciel profond (nébuleuses, galaxies)** :
```
mode = 0                  # Manuel
speed = 50-60             # 1-10s selon objet
gain = 5-15               # Faible à moyen
brightness = 0            # Neutre
contrast = 70-90          # Légèrement réduit
saturation = 10           # Neutre
sharpness = 0             # Off (traitement post)
denoise = 0               # Off (préserver détails)
awb = 0                   # Manuel
red = 15, blue = 12       # Équilibré ciel nocturne
extn = 3                  # DNG (RAW)
raw_format = 1            # SRGGB16 (haute dynamique)
zoom = 0                  # Plein champ
use_native_sensor_mode = 1 # Native (qualité max)
```

**Planétaire (Lune, planètes)** :
```
mode = 0                  # Manuel
speed = 0-10              # 1/8000s - 1/250s (rapide)
gain = 20-40              # Élevé (compensation vitesse)
brightness = 0            # Neutre
contrast = 120-150        # Augmenté (détails)
saturation = 15-20        # Augmenté (couleurs planètes)
sharpness = 20-25         # Élevé (accentuer détails)
denoise = 2-3             # CDN_Fast/HQ (réduire bruit gain élevé)
awb = 6-7                 # Daylight/Cloudy
extn = 1                  # PNG (sans perte, traitement post)
raw_format = 1            # SRGGB16 (objets brillants)
zoom = 1-3                # Crop hardware (selon taille objet)
use_native_sensor_mode = 1 # Native (détails max)
```

**Paysage nocturne** :
```
mode = 0                  # Manuel
speed = 40-50             # 0.5-1s (étoiles sans filé)
gain = 10-20              # Moyen
brightness = 10-20        # Légèrement augmenté
contrast = 100            # Neutre
saturation = 10-12        # Légèrement augmenté
sharpness = 10            # Moyen
denoise = 2               # CDN_Fast
awb = 6                   # Daylight (naturel)
extn = 1                  # PNG
raw_format = 0            # SRGGB12 (suffisant)
zoom = 0                  # Plein champ
use_native_sensor_mode = 1 # Native
```

### 7.3 Raccourcis clavier

| Touche | Action |
|--------|--------|
| **c** | Capture STILL |
| **s** | Sauvegarder configuration |
| **z** | Cycle zoom (0→1→2→3→4→5→0) |
| **m** | Cycle mode exposition (0→1→2→3→0) |
| **ESC** | Sortir de l'application |

---

## 8. DÉPANNAGE

### 8.1 Image trop sombre

**Cause** : Sous-exposition

**Solutions** :
1. Augmenter `speed` (temps d'exposition plus long)
2. Augmenter `gain` (ISO plus élevé)
3. Augmenter `ev` (si mode auto)
4. Augmenter `brightness` (post-traitement, en dernier recours)

### 8.2 Image trop claire / saturée

**Cause** : Surexposition

**Solutions** :
1. Réduire `speed` (temps d'exposition plus court)
2. Réduire `gain`
3. Réduire `ev` (si mode auto)
4. Utiliser `raw_format=1` (16-bit) pour objets très brillants
5. Activer HDR (`v3_hdr=1-4`)

### 8.3 Couleurs incorrectes

**Cause** : Balance des blancs inadaptée

**Solutions** :
1. Passer en mode `awb=0` (manuel)
2. Ajuster `red` et `blue` selon source lumière
3. Utiliser preset AWB adapté (awb=2-7)
4. Vérifier histogramme RGB (doit être équilibré sur fond de ciel)

### 8.4 Bruit excessif

**Cause** : Gain trop élevé, exposition trop courte

**Solutions** :
1. Réduire `gain`
2. Augmenter `speed` (compensation par temps)
3. Activer `denoise=2-3` (si JPEG/PNG)
4. Utiliser mode `use_native_sensor_mode=0` (binning 2×2) pour gain sensibilité
5. Empiler plusieurs images (mode LIVESTACK)

### 8.5 Flou / manque de netteté

**Cause** : Mise au point, vibrations, atmosphère

**Solutions** :
1. Vérifier focus (indicateurs FWHM/HFR si activés)
2. Réduire temps d'exposition si filé
3. Augmenter `sharpness` (10-20) si JPEG/PNG
4. Utiliser déclencheur retardé ou télécommande GPIO
5. Passer en mode LIVESTACK (empilage corrige dérive)

---

## 9. OPTIMISATIONS AVANCÉES

### 9.1 Correction pixels morts

**Activation automatique** (débayérisation manuelle) :
```python
debayer_raw_array(
    raw_array,
    raw_format_str,
    fix_bad_pixels=True,
    sigma_threshold=5.0,      # Détection outliers (5 sigma)
    min_adu_threshold=20.0    # Seuil ADU minimum
)
```

**Algorithme** :
1. Détection pixels aberrants (>5σ de la médiane locale)
2. Remplacement par médiane voisins Bayer
3. Application avant débayérisation (préserve pattern)

### 9.2 Swap canaux R/B

**Usage** : Correction erreur pattern Bayer

```python
debayer_raw_array(
    raw_array,
    raw_format_str,
    swap_rb=True  # Inverse rouge et bleu
)
```

**Symptôme** : Ciel bleu apparaît rouge (ou inverse).

### 9.3 Commandes CLI directes

**Affichage commandes** :
```python
show_cmds = 1  # Dans menu configuration
```

**Exemple sortie** :
```bash
rpicam-still \
  --camera 0 \
  --immediate \
  --shutter 1000000 \
  --gain 10 \
  --awbgains 1.5,1.2 \
  --width 3856 \
  --height 2180 \
  --mode 3856:2180:12:P \
  --denoise off \
  --output /home/admin/Pictures/2025-01-10_12-30-45.dng
```

### 9.4 Métadonnées EXIF personnalisées

**Modification DNG** (post-capture) :
```bash
exiftool -Artist="Votre nom" \
         -Software="RPiCamera2 v2.0" \
         -Copyright="© 2025" \
         image.dng
```

---

## 10. RÉSUMÉ RAPIDE

| Aspect | Paramètre clé | Valeur recommandée astrophoto |
|--------|---------------|-------------------------------|
| **Exposition** | mode, speed, gain | 0 (manuel), 50-60 (1-10s), 5-15 |
| **Balance blancs** | awb, red, blue | 0 (manuel), 15, 12 |
| **Format** | extn, raw_format | 3 (DNG), 1 (SRGGB16) |
| **Résolution** | zoom, use_native | 0 (plein champ), 1 (native) |
| **Qualité** | brightness, contrast | 0, 70-100 |
| **Débruitage** | denoise | 0 (off, RAW) |

**Workflow type** :
1. Configurer paramètres exposition (mode, speed, gain)
2. Ajuster balance blancs (awb, red, blue)
3. Choisir format/résolution (extn, zoom)
4. Appuyer bouton **STILL** ou touche **c**
5. Traiter RAW en post-production (Siril, PixInsight, etc.)

---

**Fichier suivant** : `mode_emploi_02_VIDEO.md`
