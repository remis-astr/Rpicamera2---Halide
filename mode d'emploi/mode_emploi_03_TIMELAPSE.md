# MODE EMPLOI RPiCamera2 - MODE TIMELAPSE

## 1. VUE D'ENSEMBLE

Le mode **TIMELAPSE** permet de capturer une séquence d'images espacées dans le temps, puis optionnellement de les assembler en vidéo. Deux modes sont disponibles : **Standard** et **Allsky** (timelapse longue durée automatisé).

**Déclenchement** : Menu principal → Bouton **TIMELAPSE** (ligne 0, colonne 3)

**Formats de sortie** :
- **Séquence JPEG** : Images individuelles horodatées
- **Vidéo MP4** : Assemblage final via FFmpeg
- **Mode Allsky** : Séquence avec ajustement automatique gain/stretch

---

## 2. MODE STANDARD

### 2.1 Paramètres principaux

#### Intervalle entre captures (tinterval)

**Valeurs** : 0.01 à 10.0 secondes (float avec précision centième)

**Usage** :
```
tinterval = 1.0   →  1 image/seconde
tinterval = 5.0   →  1 image/5 secondes
tinterval = 0.5   →  2 images/seconde
tinterval = 0.01  →  100 images/seconde (max théorique)
```

**Calcul durée totale** :
```python
tduration = tshots × tinterval  # Secondes

Exemple:
tshots=100, tinterval=5.0  → tduration=500s (8min 20s)
```

**Limites pratiques** :
- **Min** : 0.01s (limité par vitesse capture caméra)
- **Optimal** : ≥ 1.0s (temps sauvegarde + stabilisation)
- **Max** : 10.0s (pour intervalles plus longs, utiliser mode Allsky)

#### Nombre de captures (tshots)

**Valeurs** : 1-999 images

**Calcul stockage** :
```
Format JPEG @ 1920×1080, quality=95:
- Taille image : ~2-5 MB
- 100 shots : 200-500 MB
- 500 shots : 1-2.5 GB
```

**Stratégies** :
```
Courte durée, haute fréquence:
  tshots=100, tinterval=1.0  → 100s (1min 40s)
  Usage: Passage nuage, transit ISS

Moyenne durée:
  tshots=200, tinterval=5.0  → 1000s (16min 40s)
  Usage: Coucher soleil, passage comète

Longue durée:
  tshots=500, tinterval=10.0 → 5000s (1h 23min)
  Usage: Voie lactée, mouvement étoiles
```

**Note** : Pour durées > 2h, préférer mode **Allsky** (gestion automatique jour/nuit).

#### Durée totale (tduration)

**Lecture seule** : Calculée automatiquement

```python
tduration = tshots × tinterval
```

**Affichage** : Converti en format HH:MM:SS

```
tduration=90s   → "00:01:30"
tduration=3600s → "01:00:00"
```

### 2.2 Paramètres exposition

Les paramètres d'exposition STILL s'appliquent :

| Paramètre | Recommandation timelapse |
|-----------|--------------------------|
| **mode** | 0 (Manuel) pour cohérence |
| **speed** | Fixe toute la séquence |
| **gain** | Fixe (éviter variations) |
| **awb** | 0 (Manuel) pour couleurs stables |
| **red/blue** | Fixes (équilibre constant) |
| **extn** | 0 (JPEG, taille réduite) |

**Exemple configuration jour → nuit** :

**Fixe (simple mais limité)** :
```python
mode = 0
speed = 20  # 1/60s (compromis jour/nuit)
gain = 10   # Moyen
# ⚠ Image trop sombre nuit, trop claire jour
```

**Mode Allsky (recommandé)** :
```python
allsky_mode = 2  # Auto-Gain
# Ajustement automatique gain selon luminosité
```

### 2.3 Format image

**Extension (extn)** : 0-2 (JPEG recommandé)

| Format | Avantages | Inconvénients | Usage timelapse |
|--------|-----------|---------------|-----------------|
| JPEG | Petit fichier | Compression | ✓ Défaut |
| PNG | Sans perte | Fichier volumineux | Traitement post critique |
| BMP | Aucune compression | Très volumineux | Éviter |

**Qualité JPEG (quality)** : 80-100
- **80-90** : Bon compromis (timelapse final < 1080p)
- **95-100** : Qualité max (traitement post ou zoom numérique)

**Résolution** : Utiliser paramètres STILL (capture_size selon zoom)

### 2.4 Nomenclature fichiers

**Pattern automatique** :
```
YYYYMMDD_HHMMSS_NNNN.jpg

Exemples:
20250110_123045_0001.jpg
20250110_123050_0002.jpg
20250110_123055_0003.jpg
...
20250110_131545_0100.jpg
```

**Timestamp** : Date/heure du début de séquence (identique pour toutes images)
**Index** : 4 digits avec padding zéro (%04d)

**Répertoire** : `/home/admin/Pictures/` (défaut)

### 2.5 Workflow complet

```
┌─────────────────────────┐
│ Appui bouton TIMELAPSE  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Génération timestamp    │
│ YYYYMMDD_HHMMSS         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Boucle tshots fois:     │
│   1. Capture image      │
│   2. Save JPEG (index)  │
│   3. Attente tinterval  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Affichage stats finales │
│ - Durée réelle          │
│ - Images capturées      │
└─────────────────────────┘
```

**Exemple code interne** :
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for i in range(1, tshots + 1):
    filename = f"{timestamp}_{i:04d}.jpg"
    picam2.capture_file(filename)
    time.sleep(tinterval)
```

---

## 3. MODE ALLSKY (Timelapse longue durée)

### 3.1 Vue d'ensemble

Le mode **Allsky** est un timelapse avancé avec ajustement automatique du gain selon la luminosité ambiante, conçu pour des sessions longue durée (heures à jours).

**Activation** : `allsky_mode` = 1 ou 2

| Mode | Description | Usage |
|------|-------------|-------|
| **0 - OFF** | Timelapse standard | Durée courte, expo fixe |
| **1 - ON (gain fixe)** | Allsky sans auto-gain | Test, debug |
| **2 - Auto-Gain** | Allsky avec ajustement | ✓ Recommandé nuit complète |

**Composant** : `AllskyMeanController` (libastrostack/allsky.py)

### 3.2 Principe de fonctionnement

**Algorithme Auto-Gain** :
```
Pour chaque frame:
  1. Capturer image (exposure + gain actuels)
  2. Calculer moyenne luminosité (mean_brightness)
  3. Comparer à cible (allsky_mean_target)

  Si mean_brightness < (target - threshold):
    → Augmenter gain (objet trop sombre)

  Si mean_brightness > (target + threshold):
    → Réduire gain (objet trop clair)

  Sinon:
    → Conserver gain actuel (optimal)
```

**Correction incrémentielle** :
```python
if mean_current < target - threshold:
    new_gain = current_gain × 1.2  # +20%
elif mean_current > target + threshold:
    new_gain = current_gain × 0.8  # -20%
```

**Limites** :
- Gain min : 1.0 (limite matérielle)
- Gain max : `allsky_max_gain` (défaut 200)

### 3.3 Paramètres Allsky

#### Cible luminosité (allsky_mean_target)

**Valeurs** : 10-60 (× 100 pour normalisation 0.0-1.0)

**Correspondance** :
```
allsky_mean_target=10  → 0.10 (10% luminosité, très sombre)
allsky_mean_target=25  → 0.25 (25%, ciel nocturne étoilé)
allsky_mean_target=30  → 0.30 (30%, défaut, équilibré)
allsky_mean_target=50  → 0.50 (50%, clair, aube/crépuscule)
```

**Calcul luminosité moyenne** :
```python
image_rgb = capture_array()
mean_brightness = np.mean(image_rgb) / 255.0  # Normalisé [0-1]
```

**Recommandations** :
- **Ciel profond nocturne** : 20-25 (fond de ciel sombre, étoiles visibles)
- **Paysage nocturne** : 25-35 (détails au sol + étoiles)
- **Aube/crépuscule** : 40-50 (transition jour/nuit)

#### Seuil tolérance (allsky_mean_threshold)

**Valeurs** : 2-15 (× 100 pour normalisation)

**Correspondance** :
```
allsky_mean_threshold=2   → 0.02 (2%, très précis, ajustements fréquents)
allsky_mean_threshold=5   → 0.05 (5%, défaut, équilibré)
allsky_mean_threshold=10  → 0.10 (10%, tolérant, ajustements rares)
```

**Zone morte** :
```
Pas d'ajustement si:
  (target - threshold) < mean_brightness < (target + threshold)

Exemple:
target=30 (0.30), threshold=5 (0.05)
Zone stable: [0.25, 0.35]
```

**Impact** :
- **Seuil faible (2-3)** : Ajustements fréquents, précision élevée, risque oscillations
- **Seuil moyen (5-7)** : Bon compromis (défaut)
- **Seuil élevé (10-15)** : Ajustements rares, moins de bruit, moins réactif

#### Gain maximum (allsky_max_gain)

**Valeurs** : 50-500

**Limites sécurité** :
```
Gain trop élevé → Bruit excessif, saturation
Gain trop faible → Image trop sombre la nuit
```

**Recommandations par caméra** :
```
IMX585 : 100-300 (excellent low-light)
Pi HQ  : 60-150
Pi v3  : 50-100
```

**Stratégie** :
```python
# Début crépuscule : gain faible (~10)
# Nuit noire : gain monte progressivement
# Aube : gain redescend
# Maximum atteint uniquement nuit noire sans Lune
```

#### FPS vidéo finale (allsky_video_fps)

**Valeurs** : 15-60 fps

**Effet** : Vitesse de lecture du timelapse final

```
Images capturées: 1 image/5s pendant 2h = 1440 images
Vidéo 25 fps → Durée = 1440 / 25 = 57.6s
Vidéo 60 fps → Durée = 1440 / 60 = 24s
```

**Recommandations** :
- **15-20 fps** : Lent, détails visibles
- **25-30 fps** : Standard cinéma
- **50-60 fps** : Rapide, fluide

#### Application stretch (allsky_apply_stretch)

**Valeurs** : 0 (OFF) | 1 (ON)

**Effet** : Applique `astro_stretch()` sur chaque JPEG avant assemblage vidéo

**Pipeline** :
```
Mode OFF (0):
  JPEG brut → Vidéo MP4

Mode ON (1):
  JPEG brut → astro_stretch() → JPEG stretched → Vidéo MP4
```

**Avantages stretch** :
- Révèle détails faibles (voie lactée, nébuleuses)
- Homogénéise exposition jour/nuit
- Améliore visibilité étoiles

**Inconvénients** :
- Augmente bruit visible
- Temps traitement accru
- Perd dynamique brute (utiliser si post-traitement non prévu)

**Recommandation** : ON (1) pour timelapse final sans post-traitement.

#### Nettoyage JPEG (allsky_cleanup_jpegs)

**Valeurs** : 0 (Garder) | 1 (Supprimer)

**Effet** : Supprime JPEG individuels après assemblage vidéo

```
Mode 0 (Garder):
  JPEG séquence conservée + vidéo MP4
  Usage: Post-traitement manuel, extraction frames

Mode 1 (Supprimer):
  Seulement vidéo MP4 conservée
  Usage: Économie espace disque, timelapse final uniquement
```

**Calcul espace** :
```
1440 images × 3 MB = 4.3 GB JPEG
Vidéo MP4 (25 fps, H.264) = ~50 MB
Économie = 4.25 GB (98.8%)
```

**Recommandation** :
- Garder (0) si stockage suffisant ou post-traitement prévu
- Supprimer (1) si espace limité et vidéo finale suffit

### 3.4 Workflow Allsky complet

```
┌─────────────────────────┐
│ Activation Allsky       │
│ allsky_mode=2           │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Init AllskyController   │
│ - target=0.30           │
│ - threshold=0.05        │
│ - max_gain=200          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Boucle tshots:          │
│                         │
│ ┌─────────────────────┐ │
│ │ Capture frame       │ │
│ └──────────┬──────────┘ │
│            │             │
│            ▼             │
│ ┌─────────────────────┐ │
│ │ Calcul mean lum.    │ │
│ │ Comparaison target  │ │
│ └──────────┬──────────┘ │
│            │             │
│            ▼             │
│ ┌─────────────────────┐ │
│ │ Ajuster gain        │ │
│ │ (si hors zone morte)│ │
│ └──────────┬──────────┘ │
│            │             │
│            ▼             │
│ ┌─────────────────────┐ │
│ │ Sauver JPEG         │ │
│ │ timestamp_%04d.jpg  │ │
│ └──────────┬──────────┘ │
│            │             │
│            ▼             │
│ ┌─────────────────────┐ │
│ │ Attente tinterval   │ │
│ └──────────┬──────────┘ │
│            │             │
│            └─────────────┤
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ Assemblage vidéo MP4    │
│ via FFmpeg              │
│ (si allsky_video_fps>0) │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Optionnel: stretch JPEG │
│ (si allsky_apply_stretch│
│          =1)            │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Optionnel: cleanup JPEG │
│ (si allsky_cleanup=1)   │
└─────────────────────────┘
```

### 3.5 Assemblage vidéo FFmpeg

**Fonction** : `assemble_allsky_video(pic_dir, timestamp, fps, output_filename)`

**Commande générée** :
```bash
ffmpeg \
  -framerate {allsky_video_fps} \
  -i {pic_dir}/{timestamp}_%04d.jpg \
  -c:v libx264 \
  -preset medium \
  -crf 23 \
  -pix_fmt yuv420p \
  -y \
  -loglevel error \
  {output_filename}
```

**Paramètres FFmpeg** :
- **-framerate** : FPS vidéo finale (allsky_video_fps)
- **-c:v libx264** : Codec H.264
- **-preset medium** : Compromis vitesse/qualité (ultrafast|fast|medium|slow|veryslow)
- **-crf 23** : Qualité constante (18=excellente, 28=moyenne, 51=pauvre)
- **-pix_fmt yuv420p** : Compatibilité maximale (web, mobile)

**Sortie** :
```
/home/admin/Pictures/20250110_183000_timelapse.mp4
```

**Durée vidéo** :
```python
video_duration = frame_count / allsky_video_fps  # Secondes

Exemple:
1440 frames @ 25 fps = 57.6s
```

---

## 4. EXEMPLES DE CONFIGURATION

### 4.1 Timelapse standard coucher de soleil

**Durée** : 30 minutes
**Intervalle** : 5 secondes
**Images** : 360

```python
# Paramètres timelapse
tinterval = 5.0
tshots = 360
tduration = 1800s (30min)

# Exposition
mode = 0           # Manuel
speed = 25         # 1/30s (compromis crépuscule)
gain = 15          # Moyen
brightness = 0
contrast = 100

# Balance blancs
awb = 0            # Manuel
red = 20           # Chaud (coucher soleil)
blue = 10          # Réduit (ambiance chaude)

# Format
extn = 0           # JPEG
quality = 95       # Haute qualité
zoom = 0           # Plein champ

# Allsky
allsky_mode = 0    # OFF (exposition fixe suffisante)
```

**Post-traitement** : Assembler en vidéo 25 fps (14.4s final)

### 4.2 Timelapse Voie Lactée (transition crépuscule → nuit)

**Durée** : 4 heures
**Intervalle** : 10 secondes
**Images** : 1440

```python
# Timelapse
tinterval = 10.0
tshots = 1440
tduration = 14400s (4h)

# Exposition de base
mode = 0           # Manuel
speed = 50         # 1s (pose courte étoiles)
brightness = 0
contrast = 90      # Légèrement réduit

# Allsky (recommandé pour transition jour/nuit)
allsky_mode = 2               # Auto-Gain ✓
allsky_mean_target = 25       # 0.25 (ciel nocturne)
allsky_mean_threshold = 5     # 0.05 (tolérance standard)
allsky_max_gain = 150         # Limite gain (éviter bruit excessif)
allsky_video_fps = 30         # Vidéo finale 30 fps
allsky_apply_stretch = 1      # ✓ Stretch pour révéler détails
allsky_cleanup_jpegs = 0      # Garder JPEG (traitement post)

# Balance blancs
awb = 0            # Manuel
red = 15
blue = 12

# Format
extn = 0           # JPEG
quality = 100      # Max qualité
zoom = 0           # Plein champ
```

**Résultat** :
- Durée vidéo finale : 1440/30 = 48s
- Taille JPEG : ~4-5 GB
- Taille MP4 : ~50-100 MB

### 4.3 Allsky surveillance complète (24h)

**Durée** : 24 heures
**Intervalle** : 60 secondes (économie espace)
**Images** : 1440

```python
# Timelapse
tinterval = 60.0
tshots = 1440
tduration = 86400s (24h)

# Exposition de base
mode = 0           # Manuel
speed = 40         # 0.5s (compromis jour/nuit)

# Allsky agressif
allsky_mode = 2               # Auto-Gain ✓
allsky_mean_target = 30       # 0.30 (équilibré)
allsky_mean_threshold = 7     # 0.07 (tolérance élevée)
allsky_max_gain = 300         # Max élevé (nuit noire)
allsky_video_fps = 25         # Standard
allsky_apply_stretch = 1      # ✓ Révéler détails
allsky_cleanup_jpegs = 1      # ✓ Supprimer JPEG (économie)

# Balance blancs
awb = 1            # Auto (variation jour/nuit OK)

# Format
extn = 0           # JPEG
quality = 85       # Compromis qualité/taille
zoom = 0           # Plein champ (surveillance globale)
```

**Résultat** :
- Images JPEG supprimées après assemblage
- Vidéo MP4 seule : ~150-200 MB
- Économie : ~4-5 GB

### 4.4 Passage ISS / satellite

**Durée** : 5 minutes
**Intervalle** : 0.5 secondes (haute fréquence)
**Images** : 600

```python
# Timelapse
tinterval = 0.5    # 2 images/seconde
tshots = 600
tduration = 300s (5min)

# Exposition
mode = 0           # Manuel
speed = 15         # 1/125s (éviter filé mouvement)
gain = 30          # Élevé (compenser exposition courte)
brightness = 10
contrast = 110

# Balance blancs
awb = 0
red = 15
blue = 12

# Format
extn = 0           # JPEG
quality = 90
zoom = 1-2         # Crop moyen (selon hauteur satellite)

# Allsky
allsky_mode = 0    # OFF (exposition fixe crépuscule)
```

**Post-traitement** :
- Assemblage 25 fps → 24s vidéo
- Stabilisation sur étoiles fixes (PIPP)
- Highlight satellite (tracking)

---

## 5. OPTIMISATION ET CONSEILS

### 5.1 Minimiser dérive temporelle

**Problème** : Intervalle réel > intervalle théorique

**Causes** :
- Temps d'exposition long (speed > tinterval)
- Temps sauvegarde JPEG (carte lente)
- Charge CPU (autres processus)

**Solutions** :
```python
# Assurer temps d'exposition ≪ intervalle
tinterval ≥ 2 × sspeed_seconds

Exemple:
speed=50 (1s) → tinterval ≥ 2.0s (marge sauvegarde)

# Carte SD rapide (UHS-II)
# Fermer applications non nécessaires
# Résolution réduite si possible
```

**Vérification** :
```python
# Timestamps JPEG
import os
from datetime import datetime

files = sorted(glob.glob(f"{timestamp}_*.jpg"))
times = [os.path.getctime(f) for f in files]
intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
mean_interval = np.mean(intervals)
print(f"Intervalle moyen: {mean_interval:.2f}s (théorique: {tinterval}s)")
```

### 5.2 Gestion espace disque

**Calcul pré-capture** :
```python
estimated_size_mb = tshots × jpeg_size_mb
jpeg_size_mb = (width × height × 3) / (1024² × compression_ratio)
compression_ratio ≈ 10-15 (JPEG quality 90-100)

Exemple 1920×1080 quality=95:
jpeg_size_mb ≈ 2-3 MB
1000 shots ≈ 2-3 GB
```

**Vérifier espace disponible** :
```bash
df -h /home/admin/Pictures
# Assurer ≥ 2× espace estimé (marge sécurité)
```

**Nettoyage automatique** :
```python
# Mode Allsky avec cleanup
allsky_cleanup_jpegs = 1
# Conserve seulement MP4 (économie ~98%)
```

### 5.3 Stabilité exposition Allsky

**Éviter oscillations** :
- Augmenter `allsky_mean_threshold` (tolérance)
- Réduire amplitude correction (modification code: 1.2 → 1.1)
- Lisser mesure luminosité (moyenne N dernières frames)

**Logs diagnostic** :
```python
# Activer dans AllskyMeanController
print(f"Frame {i}: mean={mean:.3f}, target={target:.3f}, gain={gain}")
# Observer convergence ou oscillations
```

**Convergence optimale** :
```
Frame 1  : mean=0.15, gain=10  → trop sombre, gain↑
Frame 5  : mean=0.22, gain=15  → encore sombre, gain↑
Frame 10 : mean=0.28, gain=20  → proche cible, stable
Frame 50 : mean=0.30, gain=20  → optimal ✓
```

### 5.4 Qualité vidéo finale

**Ajuster paramètres FFmpeg** :

**Qualité maximale** :
```bash
ffmpeg -crf 18 -preset slow
# Taille fichier +50%, qualité perceptible meilleure
```

**Compression élevée** :
```bash
ffmpeg -crf 28 -preset fast
# Taille fichier -50%, qualité acceptable
```

**Bitrate constant** :
```bash
ffmpeg -b:v 10M -maxrate 10M -bufsize 20M
# Contrôle précis débit (streaming)
```

### 5.5 Stretch optimal frames individuelles

**Éviter sur-traitement** :
```python
# Ajuster paramètres stretch globaux avant timelapse
stretch_p_low = 5      # 0.5% (préserver ombres)
stretch_p_high = 9995  # 99.95% (préserver highlights)
stretch_factor = 50    # Facteur asinh modéré

# OU mode GHS
ghs_preset = 1         # Preset adapté ciel nocturne
```

**Vérifier sur 1 frame test** :
```python
# Capturer 1 image test
# Appliquer stretch
# Vérifier histogramme (pas de clip excessif)
# Ajuster paramètres
# Lancer timelapse
```

---

## 6. DÉPANNAGE

### 6.1 Timelapse interrompu prématurément

**Symptômes** : Moins d'images que tshots configuré

**Causes** :
1. Espace disque saturé
2. Erreur sauvegarde (permissions, carte corrompue)
3. Interruption utilisateur (ESC)
4. Crash application

**Solutions** :
1. Vérifier espace disponible avant démarrage
2. Test carte SD (badblocks, fsck)
3. Ne pas interrompre sauf urgence
4. Logs erreurs : `/var/log/rpicamera.log`

### 6.2 Oscillations gain Allsky

**Symptômes** : Gain varie constamment, pas de stabilisation

**Causes** :
- Threshold trop faible (zone morte étroite)
- Correction trop agressive (facteur 1.2 trop élevé)
- Variation réelle luminosité (nuages, Lune)

**Solutions** :
```python
# Augmenter tolérance
allsky_mean_threshold = 10  # Au lieu de 5

# Réduire amplitude correction (code)
new_gain = current_gain × 1.1  # Au lieu de 1.2

# Lisser mesure (moyenne mobile)
mean_smooth = np.mean(last_5_frames_mean)
```

### 6.3 Vidéo finale saccadée

**Symptômes** : Lecture vidéo MP4 non fluide, sauts

**Causes** :
1. Intervalles réels variables (dérive temporelle)
2. FPS vidéo trop élevé (images insuffisantes)
3. Frames corrompues

**Solutions** :
```bash
# Recréer vidéo avec FPS adapté
frame_count=$(ls {timestamp}_*.jpg | wc -l)
duration_desired=60  # secondes
fps_optimal=$((frame_count / duration_desired))

ffmpeg -framerate $fps_optimal ...

# Vérifier toutes images OK
identify -format "%f %wx%h\n" *.jpg | grep -v "1920x1080"
# Supprimer images incorrectes
```

### 6.4 Couleurs incohérentes entre frames

**Symptômes** : Variations teinte/balance blancs frame à frame

**Causes** :
- AWB auto (awb > 0) avec variations scène
- Gains ColourGains non fixés
- Stretch appliqué différemment

**Solutions** :
```python
# Forcer AWB manuel
awb = 0
red = 15   # Fixe
blue = 12  # Fixe

# Désactiver stretch (post-traitement uniforme)
allsky_apply_stretch = 0
# Stretch manuel après sur toute séquence
```

### 6.5 Assemblage vidéo échoue

**Symptômes** : Erreur FFmpeg, pas de MP4 généré

**Causes** :
1. FFmpeg non installé
2. Pattern fichiers incorrect
3. Images résolutions différentes
4. Espace disque insuffisant

**Solutions** :
```bash
# Vérifier FFmpeg
ffmpeg -version

# Installer si manquant
sudo apt install ffmpeg

# Test assemblage manuel
cd /home/admin/Pictures
ffmpeg -framerate 25 -i 20250110_183000_%04d.jpg \
       -c:v libx264 -crf 23 -pix_fmt yuv420p test.mp4

# Vérifier uniformité images
identify *.jpg | awk '{print $3}' | sort -u
# Toutes doivent être identiques (ex: 1920x1080)
```

---

## 7. RÉSUMÉ RAPIDE

### Mode Standard

| Paramètre | Valeur type | Usage |
|-----------|-------------|-------|
| tinterval | 1-10s | Intervalle captures |
| tshots | 100-500 | Nombre images |
| mode | 0 (Manuel) | Exposition fixe |
| extn | 0 (JPEG) | Format léger |
| quality | 90-100 | Qualité JPEG |

**Workflow** :
1. Configurer tinterval + tshots
2. Fixer exposition (mode=0, speed, gain)
3. Fixer AWB (awb=0, red, blue)
4. Lancer TIMELAPSE
5. Assembler manuellement ou FFmpeg

### Mode Allsky

| Paramètre | Valeur type | Usage |
|-----------|-------------|-------|
| allsky_mode | 2 (Auto-Gain) | Ajustement auto |
| allsky_mean_target | 25-30 | Cible luminosité (0.25-0.30) |
| allsky_mean_threshold | 5-7 | Tolérance |
| allsky_max_gain | 100-300 | Limite gain |
| allsky_video_fps | 25-30 | FPS vidéo finale |
| allsky_apply_stretch | 1 | Stretch frames |
| allsky_cleanup_jpegs | 1 | Supprimer JPEG |

**Workflow** :
1. Activer allsky_mode=2
2. Configurer cibles luminosité
3. Fixer exposition de base (speed)
4. Lancer TIMELAPSE
5. Vidéo MP4 générée automatiquement

---

**Fichier suivant** : `mode_emploi_04_STRETCH.md`
