# MODE EMPLOI RPiCamera2 - MODE VIDEO

## 1. VUE D'ENSEMBLE

Le mode **VIDEO** permet d'enregistrer des séquences vidéo avec différents codecs et formats, optimisé pour l'astrophotographie planétaire et l'acquisition haute cadence.

**Déclenchement** : Menu principal → Bouton **VIDEO** (ligne 0, colonne 1)

**Formats supportés** :
- **H.264** : Compression standard (MP4 sur Pi 5)
- **MJPEG** : Motion JPEG (frame-by-frame)
- **YUV420** : Format brut non compressé
- **SER** : Format astrophoto (YUV, RGB, XRGB)

---

## 2. PARAMÈTRES PRINCIPAUX

### 2.1 Durée vidéo (vlen)

**Valeurs** : 0-3600 secondes

**Effet** : Définit la durée d'enregistrement

```python
video_duration = vlen * 1000  # Converti en millisecondes
```

**Calcul taille fichier estimée** :

| Codec | Bitrate approx | Taille 1min @ 1080p |
|-------|---------------|---------------------|
| H.264 (haute qualité) | 10-25 Mbps | 75-190 MB |
| MJPEG | 50-150 Mbps | 375-1125 MB |
| YUV420 (non compressé) | ~250 Mbps | 1875 MB |
| SER-RGB888 | ~750 Mbps | 5625 MB |

**Recommandations** :
- **Planétaire** : 30-120s (éviter dérive atmosphérique)
- **Tests** : 5-10s (validation setup)
- **Évènements** : 300-600s (éclipse, transit, occultation)

### 2.2 Framerate (fps)

**Valeurs** : 1-180 fps (limité selon caméra/résolution)

**Limites par caméra IMX585** :
```
Résolution    Mode        FPS max
800×600       Crop Mode 5   178 fps
1280×720      Crop Mode 4   150 fps
1920×1080     Crop Mode 3   100 fps
2880×2160     Crop Mode 2    51 fps
3856×2180     Native 4K      51 fps
```

**Impact sur le système** :
```
FPS élevé → Résolution réduite (modes hardware crop)
FPS faible → Résolution maximale disponible

Synchronisation automatique : sync_video_resolution_with_zoom()
```

**Usage astrophoto** :
- **Lucky imaging planétaire** : 100-178 fps (Lune, planètes)
- **Comètes/astéroïdes** : 25-50 fps
- **Occultations stellaires** : 50-100 fps (précision temporelle)
- **Paysages nocturnes** : 15-30 fps

**Note** : Le FPS réel peut être limité par :
- Temps d'exposition (sspeed)
- Bande passante USB/CSI
- Vitesse écriture carte SD/SSD

### 2.3 Format vidéo (vformat)

**Valeurs** : 0-27 (index résolutions natives)

**Résolutions communes** :
```
Index  Résolution   Ratio   Usage
0      640×480      4:3     Tests rapides
1      800×600      4:3     Lucky imaging rapide
2      1280×720     16:9    HD standard
3      1920×1080    16:9    Full HD
10     1920×1080    16:9    Full HD (défaut)
```

**Synchronisation zoom IMX585** :
```python
# Automatic: vformat suit la résolution du zoom actuel
# zoom=0 (native) → vformat=3856×2180
# zoom=1 → vformat=2880×2160
# zoom=2 → vformat=1920×1080
# zoom=3 → vformat=1280×720
# zoom=4-5 → vformat=800×600
```

**Configuration manuelle** :
```python
vwidths = [640, 800, 1280, 1920, 2592, 3280, 4056, 4656, ...]
vheights = [480, 600, 720, 1080, 1944, 2464, 3040, 3496, ...]
# vformat indexe dans ces tableaux
```

**Recommandation** : Laisser synchronisation automatique (mode défaut).

---

## 3. CODECS ET FORMATS

### 3.1 H.264 (codec=0)

**Description** : Codec compression standard haute efficacité

**Paramètres** :
- **profile** : Baseline(0), Main(1), High(2-8)
- **level** : 0-7 (4.1, 4.2, ...)
- **quality** : 0-100 (bitrate variable)

**Profiles H.264** :
```
Index  Profile        Usage
0      Baseline       Compatibilité maximale, faible latence
1      Main           Standard, bon compromis
2      High           Qualité maximale, B-frames
8      High 4:2:2     10-bit (si supporté)
```

**Bitrate estimation** :
```
quality=100 → ~25 Mbps @ 1080p30
quality=80  → ~15 Mbps @ 1080p30
quality=50  → ~8 Mbps @ 1080p30
```

**Avantages** :
- Taille fichier réduite (10-20× vs non compressé)
- Lecture universelle (VLC, navigateurs, etc.)
- Accélération matérielle (GPU encode)

**Inconvénients** :
- Compression avec pertes (détails fins)
- Artefacts de compression (blocking, banding)
- Non recommandé pour traitement astrophoto (préférer SER)

**Commande générée** :
```bash
rpicam-vid \
  --width 1920 --height 1080 \
  --framerate 30 \
  --codec h264 \
  --profile high \
  --level 4.2 \
  --quality 100 \
  --timeout {vlen*1000} \
  --output video.h264
```

**Pi 5 uniquement - MP4** :
```bash
# Codec direct en conteneur MP4 (pas de post-traitement)
rpicam-vid --codec h264 --output video.mp4
```

**Correction timestamps Pi 5** :
Si artefacts temporels (saccades, désynchronisation audio) :
```python
fix_video_timestamps(input_file, fps, quality_preset="ultrafast")
# Réencode via FFmpeg avec timestamps corrigés
```

### 3.2 MJPEG (codec=1)

**Description** : Motion JPEG, séquence d'images JPEG

**Principe** :
```
Frame 1 → JPEG compression → Store
Frame 2 → JPEG compression → Store
...
Chaque frame = image JPEG indépendante
```

**Avantages** :
- Qualité élevée (compression intra-frame uniquement)
- Édition facile (découpe frame exacte)
- Pas d'artefacts inter-frames (B-frames, P-frames)

**Inconvénients** :
- Taille fichier importante (50-150 Mbps)
- Moins d'accélération matérielle
- Qualité inférieure au H.264 à taille égale

**Usage astrophoto** :
- Acquisition rapide haute qualité
- Traitement post avec extraction frames individuelles
- Compatible logiciels planétaires (AutoStakkert, WinJupos)

**Commande** :
```bash
rpicam-vid \
  --codec mjpeg \
  --quality 100 \
  --output video.mjpeg
```

### 3.3 YUV420 (codec=2)

**Description** : Format brut non compressé, sous-échantillonnage chrominance

**Structure** :
```
Y plane  : Luminance full resolution (width × height)
U plane  : Chrominance blue (width/2 × height/2)
V plane  : Chrominance red (width/2 × height/2)

Total size per frame = width × height × 1.5 bytes
```

**Avantages** :
- Aucune perte de compression
- Conversion rapide vers RGB
- Bon compromis qualité/taille (vs RGB888)

**Inconvénients** :
- Fichiers très volumineux (~250 Mbps @ 1080p30)
- Nécessite conversion pour visualisation
- Sous-échantillonnage chrominance (perte résolution couleur)

**Conversion RGB** :
```python
yuv420_to_rgb(yuv_data, width, height)
# Utilise formules YUV → RGB standard (BT.601/BT.709)
```

**Usage** :
- Acquisition intermédiaire (post-traitement requis)
- Pipeline custom (debayer, ISP, etc.)
- Tests sans perte compression

### 3.4 SER - Format astrophotographie

**Description** : Standardformat astrophoto (fichier + frames + timestamps)

**Structure fichier SER** :
```
Header (178 bytes):
  - Signature "LUCAM-RECORDER"
  - Largeur, hauteur, profondeur
  - ColorID (format pixel)
  - FrameCount
  - Observer, Instrument, Telescope
  - DateTime (µs depuis epoch)

Body:
  - Frame 1 (raw pixels)
  - Frame 2
  - ...
  - Frame N

Trailer (SER v3):
  - Timestamp Frame 1 (64-bit µs)
  - Timestamp Frame 2
  - ...
```

#### SER-YUV (codec=4)

**ColorID** : 0 (YUV Planar)

**Pipeline** :
```
1. rpicam-vid --codec yuv420 → temp.yuv
2. convert_yuv420_to_ser(yuv_file, ser_file, width, height, fps)
   ├─ Créer header SER (ColorID=0)
   ├─ Lire frames YUV (Y + U + V planes)
   ├─ Écrire frames SER
   └─ Ajouter timestamps (µs précision)
3. Suppression temp.yuv
```

**Avantages** :
- Compatible logiciels planétaires (PIPP, AutoStakkert)
- Taille modérée (vs RGB)
- Timestamps microsecondes (analyse occultations)

**Inconvénients** :
- Sous-échantillonnage chrominance (4:2:0)
- Conversion nécessaire (YUV → RGB)
- Pas d'accélération lecture logiciels

#### SER-RGB (codec=5)

**ColorID** : 100 (RGB 24-bit)

**Pipeline** :
```
1. Picamera2: capture_array("main") → RGB888 numpy array
2. Écrire directement dans fichier .raw
3. convert_rgb888_to_ser(raw_file, ser_file, width, height, fps)
   ├─ Header SER (ColorID=100, bit_depth=8)
   ├─ Conversion BGR → RGB (OpenCV → Standard)
   ├─ Écrire frames RGB (3 bytes/pixel)
   └─ Timestamps calculés (fps * index)
```

**Avantages** :
- Pleine résolution couleur (pas de sous-échantillonnage)
- Compatible tous logiciels SER
- Pas de conversion nécessaire

**Inconvénients** :
- Taille fichier élevée (~750 Mbps @ 1080p30)
- Carte SD/SSD rapide requise
- Limite FPS (dépend vitesse écriture)

#### SER-XRGB (codec=6)

**ColorID** : 101 (XRGB 32-bit, RGB avec padding alpha)

**Format** :
```
Pixel = [X][R][G][B] = 4 bytes
X (alpha) = ignoré (padding alignement mémoire)
```

**Pipeline** :
```
Picamera2: capture_array("main") en format XRGB8888
→ Écriture directe SER (pas de conversion BGR/RGB)
```

**Avantages** :
- Le plus rapide (pas de conversion)
- Alignement mémoire optimal (4-byte boundary)
- FPS maximal possible

**Inconvénients** :
- Taille fichier maximale (~1 GB/min @ 1080p30)
- Compatibilité limitée (certains logiciels ignorent alpha)
- Gaspillage 25% espace (canal alpha inutilisé)

### 3.5 Comparaison codecs

| Codec | Taille/min @ 1080p30 | Qualité | FPS max | Post-process | Usage astrophoto |
|-------|----------------------|---------|---------|--------------|------------------|
| H.264 | 75-190 MB | Bonne | 100+ | Oui | Preview, partage |
| MJPEG | 375-1125 MB | Très bonne | 60+ | Facile | Extraction frames |
| YUV420 | 1875 MB | Excellente | 100+ | Requis | Pipeline custom |
| SER-YUV | 1875 MB | Excellente | 100+ | Oui | Planétaire standard |
| SER-RGB | 5625 MB | Parfaite | 50-100 | Non | Qualité maximale |
| SER-XRGB | 7500 MB | Parfaite | 100+ | Non | Performance max |

**Recommandation astrophoto planétaire** :
1. **SER-YUV** : Meilleur compromis qualité/taille (défaut)
2. **SER-RGB** : Si couleur critique (Mars, Jupiter)
3. **SER-XRGB** : Si FPS max requis (>100 fps)

---

## 4. CONFIGURATION EXPOSITION

### 4.1 Mode exposition vidéo

**Important** : Les mêmes paramètres que STILL s'appliquent :
- `mode` : 0=Manuel, 1-3=Auto
- `speed` / `sspeed` : Temps d'exposition (µs)
- `gain` : Gain analogique (ISO)

**Contrainte vidéo** :
```
Temps exposition ≤ 1/FPS

Exemple:
fps=100 → Exposition max = 10ms = 10000µs
fps=30  → Exposition max = 33ms = 33333µs
```

**Si exposition > 1/FPS** :
```
FPS effectif = 1 / Temps_exposition
fps_param = ignoré

Exemple:
speed=50ms, fps=100 → FPS réel = 20 fps
```

**Recommandation planétaire** :
```python
# Calculer exposition optimale
optimal_exposure = (1.0 / target_fps) * 0.8  # 80% pour marge

# Exemple: 100 fps → 8ms max
sspeed = 8000  # µs
gain = 30-50   # Compenser exposition courte
```

### 4.2 Gain vidéo haute cadence

**Stratégie** :
1. Fixer FPS cible (ex: 100 fps)
2. Calculer exposition max (10ms)
3. Ajuster gain pour obtenir SNR acceptable

**Règle empirique** :
```
Gain requis ≈ Gain_base × (Exposition_base / Exposition_courte)

Exemple:
Exposition still : 1s, gain=10 → SNR acceptable
Exposition vidéo : 10ms (100× plus court)
Gain vidéo requis ≈ 10 × 10 = 100

(Calcul simplifié, ignorer bruit de lecture)
```

**Limites pratiques** :
- **IMX585** : Gain 50-150 pour 100 fps planétaire
- **Pi HQ** : Gain 40-80 pour 50 fps
- **Pi v3** : Gain 30-60 pour 50 fps

### 4.3 Balance des blancs vidéo

**Mode manuel (awb=0)** : Identique STILL
```python
red=15, blue=12  # Gains fixes toute la vidéo
```

**Mode auto (awb>0)** : Mise à jour dynamique
```
⚠ Attention: AWB auto peut causer variations couleur frame-to-frame
→ Recommandé: awb=0 (manuel) pour cohérence couleur
```

**Workflow calibration AWB vidéo** :
1. Capturer quelques frames preview (mode auto)
2. Observer gains ColorGains dans métadonnées
3. Fixer gains manuels (awb=0, red/blue ajustés)
4. Lancer enregistrement avec gains fixes

---

## 5. OPTIMISATION PERFORMANCE

### 5.1 Vitesse écriture carte SD/SSD

**Débits requis** :

| Résolution | FPS | Codec | Débit min requis |
|-----------|-----|-------|------------------|
| 800×600 | 178 | SER-YUV | ~127 MB/s |
| 1920×1080 | 100 | SER-YUV | ~295 MB/s |
| 1920×1080 | 30 | SER-RGB | ~178 MB/s |
| 3856×2180 | 51 | SER-RGB | ~1.25 GB/s |

**Types carte** :
- **Class 10** : 10 MB/s (insuffisant)
- **UHS-I U3** : 30 MB/s (minimum requis)
- **UHS-II V60** : 60 MB/s (recommandé)
- **UHS-II V90** : 90 MB/s (idéal)
- **SSD USB 3.x** : 200-500 MB/s (optimal)

**Test vitesse écriture** :
```bash
dd if=/dev/zero of=/home/admin/Videos/test.dat bs=1M count=1024
# Observer MB/s (doit être > débit requis)
rm /home/admin/Videos/test.dat
```

**Symptômes vitesse insuffisante** :
- Frames droppées (frame count < attendu)
- FPS réel < FPS configuré
- Blocages périodiques
- Fichiers corrompus (header frame count incorrect)

### 5.2 Synchronisation résolution/zoom

**Fonction automatique** : `sync_video_resolution_with_zoom()`

**Comportement IMX585** :
```python
# Zoom activé → vformat et fps adaptés automatiquement
zoom=2 (1920×1080, mode crop) → fps limité à 100
zoom=4 (800×600, mode crop) → fps limité à 178

# Zoom désactivé → restaure vformat/fps d'origine
zoom=0 → vformat et fps restaurés
```

**Backup automatique** :
```python
# Sauvegarde fps avant zoom
sync_video_resolution_with_zoom.fps_backup = fps

# Restauration après zoom=0
fps = sync_video_resolution_with_zoom.fps_backup
```

**Désactivation** :
```python
# Commenter appel dans code si résolution manuelle souhaitée
# sync_video_resolution_with_zoom()
```

### 5.3 Modes capteur natifs vidéo

**IMX585 - Modes hardware crop** :
```
Mode 0 (binning 2×2) : 1928×1090 @ 178 fps
Mode 1 (native)      : 3856×2180 @ 51 fps
Mode 2 (crop)        : 2880×2160 @ 51 fps
Mode 3 (crop)        : 1920×1080 @ 100 fps
Mode 4 (crop)        : 1280×720 @ 150 fps
Mode 5 (crop)        : 800×600 @ 178 fps
```

**Avantages crop hardware** :
- FPS max élevé (178 fps)
- Latence réduite (moins de données)
- Pas de perte qualité (vs binning logiciel)

**Configuration automatique** :
```python
sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode)
# Retourne tuple (width, height) optimal
```

### 5.4 Buffer et latence

**Configuration Picamera2** :
```python
video_config = picam2.create_video_configuration(
    main={"size": (width, height), "format": format_str},
    buffer_count=6,  # Nombre de buffers circulaires (défaut)
)
```

**Ajustement buffer** :
- `buffer_count=4` : Latence min, risque drop frames
- `buffer_count=6` : Équilibré (défaut)
- `buffer_count=8` : Sécurité max, latence accrue

**Monitoring frame drops** :
```python
metadata = picam2.capture_metadata()
dropped = metadata.get("FrameDrops", 0)
if dropped > 0:
    print(f"⚠ {dropped} frames perdues!")
```

---

## 6. POST-TRAITEMENT VIDÉO

### 6.1 Extraction frames SER

**Outils** :
- **PIPP** : Planetary Imaging PreProcessor (extraction, stabilisation)
- **SER Player** : Visualisation, extraction frames
- **AutoStakkert** : Stacking direct depuis SER

**Exemple PIPP** :
```
1. Charger fichier SER
2. Analyse qualité frames (sharpness)
3. Sélection top N% frames
4. Extraction → PNG/TIFF
5. Stabilisation (centrage cible)
```

### 6.2 Conversion formats

**SER → AVI** :
```bash
# Utiliser PIPP ou SER Player
# Export AVI non compressé ou MJPEG
```

**H.264 → Frames PNG** :
```bash
ffmpeg -i video.h264 -vf fps=30 frame_%04d.png
```

**YUV420 → RGB** :
```bash
ffmpeg -f rawvideo -pixel_format yuv420p -video_size 1920x1080 \
       -i video.yuv -pix_fmt rgb24 video.mp4
```

**MP4 → Timestamps corrigés** (Pi 5) :
```python
fix_video_timestamps("video.mp4", fps=30, quality_preset="veryfast")
# Sortie : video.mp4 (réécrit avec timestamps corrects)
```

### 6.3 Analyse qualité frames

**Calcul netteté (focus metric)** :
```python
for frame in ser_file:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    # Stocker sharpness pour tri
```

**Sélection meilleures frames** :
```python
# Tri par netteté décroissante
sorted_frames = sorted(frames, key=lambda f: f.sharpness, reverse=True)
# Garder top 10%
best_frames = sorted_frames[:int(len(sorted_frames) * 0.1)]
```

**Outils automatiques** :
- PIPP : Quality estimation automatique
- AutoStakkert : AP (Alignment Points) qualité
- Registax : Analyse gradient

---

## 7. WORKFLOW COMPLET

### 7.1 Planétaire Lucky Imaging

**Configuration** :
```python
mode = 0               # Manuel
speed = 5-15           # 5ms-15ms (67-200 fps possible)
gain = 50-150          # Élevé (compenser exposition courte)
fps = 100              # Cible 100 fps
vformat = 2            # 1920×1080 ou auto (zoom)
codec = 4              # SER-YUV (compromis qualité/taille)
vlen = 60-120          # 1-2 minutes
zoom = 2-3             # Selon taille planète (crop 1080p ou 720p)
awb = 0                # Manuel
red = 20, blue = 15    # Ajusté selon planète
denoise = 0            # Off (préserver détails)
```

**Étapes** :
1. Centrer objet (prévisualisation)
2. Ajuster focus (FWHM < 3-5 px)
3. Lancer VIDEO (60-120s)
4. Import PIPP → Analyser qualité
5. Sélectionner top 10-30% frames
6. AutoStakkert → Empilement
7. Registax/WinJupos → Ondelettes

**Checklist** :
- [ ] Carte SD/SSD ≥ UHS-II V60
- [ ] Gain adapté exposition courte
- [ ] AWB manuel (éviter variations)
- [ ] FPS réel = FPS cible (vérifier métadonnées)
- [ ] Focus optimal (FWHM stable)

### 7.2 Comètes / Astéroïdes

**Configuration** :
```python
mode = 0               # Manuel
speed = 30-50          # 30-50ms (20-33 fps)
gain = 20-40           # Moyen
fps = 25-30            # Standard
vformat = 3            # 1920×1080 (plein champ)
codec = 4              # SER-YUV
vlen = 300-600         # 5-10 minutes
zoom = 0               # Plein champ (contexte étoiles)
awb = 0                # Manuel
denoise = 0            # Off
```

**Traitement** :
1. Extraction frames (toutes, pas de sélection)
2. Alignement sur étoiles fixes (Siril, PIPP)
3. Médiane temporelle → Éliminer comète
4. Soustraction → Isoler comète
5. Stack frames comète → SNR élevé

### 7.3 Occultations stellaires

**Configuration** :
```python
mode = 0               # Manuel
speed = 5-10           # 5-10ms (100-200 fps)
gain = 60-100          # Élevé (précision temporelle)
fps = 100-150          # Max possible
vformat = 1            # 800×600 (vitesse max)
codec = 4              # SER-YUV (timestamps µs)
vlen = 30-60           # 30-60s (autour de l'évènement)
zoom = 4-5             # Crop max (FPS max)
```

**Analyse** :
1. Extraction timestamps µs (trailer SER v3)
2. Photométrie relative (intensité vs temps)
3. Détection chute luminosité (occultation)
4. Précision temporelle : ±1-10ms (selon FPS)

---

## 8. DÉPANNAGE

### 8.1 FPS réel inférieur à FPS configuré

**Causes** :
1. Exposition trop longue (sspeed > 1/fps)
2. Vitesse écriture insuffisante (carte lente)
3. Résolution trop élevée (bande passante)

**Solutions** :
1. Réduire sspeed (exposition ≤ 80% de 1/fps)
2. Augmenter gain (compenser exposition courte)
3. Carte SD/SSD plus rapide (UHS-II V90)
4. Réduire résolution (zoom crop)
5. Changer codec (H.264 si acceptable)

### 8.2 Frames droppées

**Symptômes** :
- Compteur frames < attendu
- Saccades lecture vidéo
- Gaps timestamps SER

**Solutions** :
1. Réduire FPS cible
2. Carte SD/SSD plus rapide
3. Augmenter buffer_count (8-10)
4. Réduire résolution
5. Vérifier charge CPU (fermer applications)

### 8.3 Fichier SER corrompu

**Symptômes** :
- Header frame count = 0
- Échec ouverture PIPP/AutoStakkert
- Taille fichier incorrecte

**Solutions** :
1. Vérifier espace disque disponible
2. Ne pas interrompre enregistrement (ESC)
3. Réparer header manuellement :
```python
# Calculer nombre frames
file_size = os.path.getsize("video.ser")
frame_size = width * height * 1.5  # YUV420
frame_count = (file_size - 178) // frame_size

# Mettre à jour header
update_ser_frame_count("video.ser", frame_count)
```

### 8.4 Couleurs incorrectes SER

**Symptômes** :
- Bleu → Rouge (ou inverse)
- Teinte uniforme incorrecte

**Causes** :
1. ColorID incorrect (header SER)
2. Pattern Bayer swap
3. AWB auto variables

**Solutions** :
1. Vérifier ColorID (100=RGB, 0=YUV)
2. Ajuster debayer swap_rb=True
3. Forcer AWB manuel (awb=0)
4. Correction post dans AutoStakkert (invert R/B)

### 8.5 Latence excessive

**Symptômes** :
- Délai preview → enregistrement
- Commandes lentes (start/stop)

**Solutions** :
1. Réduire buffer_count (4-6)
2. Désactiver preview pendant enregistrement
3. Mode binning (use_native_sensor_mode=0)
4. Réduire résolution

---

## 9. COMMANDES CLI

**Picamera2** (via Python) :
```python
encoder = H264Encoder(bitrate=10000000)
picam2.start_recording(encoder, "video.h264")
time.sleep(vlen)
picam2.stop_recording()
```

**rpicam-vid** (subprocess, fallback) :
```bash
rpicam-vid \
  --camera 0 \
  --width 1920 \
  --height 1080 \
  --framerate 30 \
  --codec h264 \
  --profile high \
  --level 4.2 \
  --quality 100 \
  --timeout 10000 \
  --denoise off \
  --awbgains 1.5,1.2 \
  --shutter 33333 \
  --gain 10 \
  --output video.h264
```

**Afficher commandes** :
```python
show_cmds = 1  # Menu configuration
# Affiche toutes commandes CLI générées
```

---

## 10. RÉSUMÉ RAPIDE

| Aspect | Paramètre | Recommandation astrophoto |
|--------|-----------|--------------------------|
| **Durée** | vlen | 60-120s (planétaire) |
| **FPS** | fps | 100 fps (Lune/planètes) |
| **Résolution** | vformat | Auto (sync zoom) |
| **Codec** | codec | 4 (SER-YUV) |
| **Exposition** | speed | ≤ 80% de 1/fps |
| **Gain** | gain | 50-150 (haute cadence) |
| **AWB** | awb | 0 (manuel) |
| **Stockage** | - | SSD USB 3.x ou UHS-II V90 |

**Workflow rapide** :
1. Configurer zoom → FPS auto-ajusté
2. Ajuster exposition (speed ≤ 1/fps × 0.8)
3. Augmenter gain pour compenser
4. Fixer AWB manuel (red/blue)
5. Lancer VIDEO (bouton ou touche v)
6. Traiter SER dans PIPP + AutoStakkert

---

**Fichier suivant** : `mode_emploi_03_TIMELAPSE.md`
