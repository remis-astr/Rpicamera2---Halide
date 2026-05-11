# RPiCamera2 — Halide

Interface de contrôle et d'imagerie astronomique avancée pour Raspberry Pi, optimisée pour le capteur **IMX585** et les sessions en ciel profond, planétaire et solaire.

Dérivé de [RPiCamera](https://github.com/Gordon999) de Gordon999, entièrement réécrit avec une interface graphique Pygame native, un pipeline de traitement RAW12 et des modules d'imagerie avancée compilés avec **Halide** pour les accélérations critiques.

---

## Matériel requis

| Composant | Détail |
|---|---|
| Raspberry Pi 5 | Architecture aarch64 — Debian 12 Bookworm |
| Capteur | IMX585 (nécessite libcamera custom) |
| Écran | 7″ HDMI tactile recommandé — [LCD Wiki](https://www.lcdwiki.com/7inch_HDMI_Display-H) |
| Stockage | NVMe SSD (recommandé, voire indispensable pour RAW12 haute cadence) |
| GPIO (optionnel) | Boutons externes mise au point / déclenchement |
| IMU (optionnel) | MPU6050 ou BNO055 — pour AstroHopper / mise en station polaire |

---

## Dépendances Python

```bash
pip3 install pygame opencv-python numpy matplotlib scipy scikit-image \
             picamera2 gpiozero Pillow websocket-client astropy
```

| Bibliothèque | Usage |
|---|---|
| `pygame` | Interface graphique, events, affichage preview |
| `opencv-python` | Traitement d'image, débayérisation, alignement |
| `numpy` | Calculs matriciels, pipeline RAW |
| `matplotlib` | Histogrammes RGB/luminance |
| `scipy` | Filtres, reconstruction d'axe polaire robuste (N≥4) |
| `scikit-image` | Détection de pics, imagerie planétaire |
| `picamera2` | Interface Raspberry Pi Camera |
| `gpiozero` | Boutons GPIO externes |
| `Pillow` | Chargement/sauvegarde d'images |
| `websocket-client` | Communication MiniCam (caméra secondaire) |
| `astropy` | Position précise de Polaris (optionnel) |

---

## Dépendances système

```bash
# Outils de base
sudo apt install ffmpeg libcamera-dev libcamera-apps

# ASTAP — plate solving (Finder, Push-To, Polar Align)
# Télécharger le binaire ARM64 et les catalogues sur https://www.hnsky.org/astap.htm
sudo cp astap /usr/bin/astap
sudo chmod +x /usr/bin/astap
sudo mkdir -p /opt/astap
# Copier les catalogues G05 ou D05 dans /opt/astap/
```

### libcamera custom pour IMX585

Le programme nécessite la libcamera modifiée par [will12753](https://github.com/will12753/libcamera-imx585) pour le support complet du capteur IMX585 (RAW12, tuning, contrôles avancés).

---

## Installation

```bash
git clone https://github.com/remis-astr/Rpicamera2---Halide.git
cd Rpicamera2---Halide
```

Aucune étape de compilation Python n'est nécessaire. Les pipelines Halide sont précompilés en `.so` dans `libastrostack/halide/`.

### Fichier de configuration

Copier et adapter `PiLCConfig104.txt` à la racine du projet. Ce fichier contient tous les paramètres persistants (résolution, gains, latitude/longitude observateur, chemins de sauvegarde, etc.).

### Chemins de sauvegarde par défaut

| Type | Répertoire |
|---|---|
| Photos / RAW | `~/Pictures/` (NVMe recommandé — la carte SD est trop lente pour le RAW12 haute cadence) |
| Vidéos | `~/Videos/` |
| Live Stacks | `~/stacks/live/` |
| Config | `~/PiLCConfig104.txt` |
| Logs | `~/rpicamera2.log` |

---

## Lancement

```bash
# Manuel
python3 RPiCamera2.py

# Via le script de démarrage (gère l'attente X11, ferme le splash)
bash start_rpicamera2.sh
```

Pour un démarrage automatique au boot, configurer `start_rpicamera2.sh` comme service systemd ou entrée autostart.

---

## Modes et fonctionnalités

### Imagerie

| Mode | Description |
|---|---|
| **Preview** | Aperçu temps réel avec histogramme, HFR/FWHM, SNR, AF |
| **Still / RAW** | Capture JPEG, PNG, RAW12 (DNG-compatible) |
| **Vidéo** | H.264, MJPEG, YUV420, RAW → SER (voir ci-dessous) |
| **Timelapse / AllSky** | Acquisition séquentielle, stacking timelapse |

#### Vidéo et zoom

Le mode **zoom** active un recadrage ROI sur le capteur, réduisant la résolution et augmentant significativement le framerate. Pour l'IMX585 :

| Résolution | FPS max |
|---|---|
| 640 × 480 | ~220 fps |
| 800 × 600 | ~178 fps |
| 1280 × 720 | ~150 fps |
| 1920 × 1080 | ~100 fps |
| 2880 × 2160 | ~51 fps |

Les vidéos YUV420 et RAW peuvent être converties automatiquement au format **SER** (avec timestamps, compatible RegiStax / AutoStakkert!3 / PIPP).

### Traitement avancé

| Mode | Description |
|---|---|
| **Live Stack DSO** | Empilement temps réel RAW12 natif — pipeline Halide + GHS stretch |
| **Lucky Stack RGB8** | Sélection des meilleures frames sur flux RGB8/YUV — score, alignement, stack, post-filtres |
| **Lucky Stack RAW** | Même pipeline mais en domaine Bayer RAW12 : score sur canal G1 linéaire, alignement sous-pixel, débayérisation unique en fin de stack — pas d'artefact ISP, 12 bits préservés |
| **JSK Live** | Pipeline HDR + débruitage pour ciel profond : RAW12 → stack (1–4 frames) → fusion HDR (médiane / moyenne / Mertens) → débayer → débruitage (bilatéral, Gaussian, médian, guided) → stretch. Accéléré ×16 via pipeline Halide AOT |
| **Galaxy** | Live stack avec enhancement galactique multi-échelle : réduction d'étoiles (ouverture morphologique) + filtre Frangi structurel modifié + USM multi-échelle. 3 presets : Elliptique, Spirale, Par la tranche |
| **Moon Mineral** | 6 presets scientifiques — saturation HSV, décorrélation PCA, faux-couleurs spectral NASA |
| **Solar** | Traitement dédié imagerie solaire |
| **Collimation** | Assistant de collimation par analyse de la tache de diffraction |

### MiniCam (caméra secondaire)

Contrôle d'une deuxième caméra (Pi Zero / Pi 5) via WiFi ou USB-gadget :

| Sous-mode | Description |
|---|---|
| **Live Stack** | DSO sur la MiniCam en parallèle |
| **Galaxy** | Stack galaxies RAW |
| **Lucky RAW** | Lucky imaging RAW depuis la MiniCam |
| **Solve (Push-To)** | Plate solving ASTAP + synchronisation Stellarium via protocole LX200 |
| **Finder** | Vue grand champ pour pointage |
| **Polar Align** | Mise en station polaire de la table équatoriale (voir ci-dessous) |

### Mise en station polaire

Accessible depuis le bouton **POLAR ALIGN** du panneau MiniCam.

1. **Capturer N positions** (3 à 6) — la table tourne sur ~14° pendant la session
2. **Plate solving ASTAP** automatique sur chaque capture
3. **Calcul de l'axe** : produit vectoriel (N=3) ou moindres carrés robuste scipy (N≥4)
4. **Résultat** : ΔAlt et ΔAz en arcminutes avec carte polaire Pygame (NCP, Polaris rotatif, flèche de correction)
5. **Guidage live** : phases LIVE_ALT et LIVE_AZ avec feedback IMU en temps réel

Précision typique avec ASTAP (~5") et 14° d'arc : ±2–10' selon le nombre de captures.

---

## Structure du projet

```
RPiCamera2.py              # Programme principal (~20 000 lignes)
polar_alignment.py         # Calculs purs mise en station (coordonnées, axe, IMU)
polar_widgets.py           # Widget carte polaire Pygame
polar_align_screen.py      # Interface mise en station (machine à états 5 phases)
start_rpicamera2.sh        # Script de lancement
PiLCConfig104.txt          # Configuration persistante
libastrostack/
  ├── session.py           # Pipeline Live Stack RAW12
  ├── lucky_imaging.py     # Lucky stack RGB/RAW
  ├── jsk_live.py          # Pipeline JSK planétaire
  ├── mineral_moon.py      # Traitement lune minérale
  ├── solar.py             # Traitement solaire
  ├── minicam.py           # Interface caméra secondaire
  ├── platesolve.py        # Plate solving ASTAP
  ├── stellarium_client.py # Protocole LX200 / Stellarium
  ├── aligner.py           # Alignement inter-frames
  ├── stretch.py           # GHS, Arcsinh, Log, MTF
  ├── finder/              # Mode Finder + Push-To + IMU AstroHopper
  └── halide/              # Pipelines Halide AOT (.so précompilés ARM64)
```

---

## Calibration IMU (AstroHopper / Polar Align)

Le module IMU utilise un **MPU6050** connecté à la MiniCam via I²C. La calibration des biais se fait depuis l'interface web de la MiniCam (`http://<ip>:8000`). Les axes sont remappés en convention Y-up avec zone morte configurable.

---

## Crédits

- Programme de base : [Gordon999/RPiCamera](https://github.com/Gordon999) — licence MIT
- libcamera IMX585 : [will12753/libcamera-imx585](https://github.com/will12753/libcamera-imx585)
- Plate solving : [ASTAP](https://www.hnsky.org/astap.htm) — Han Kleijn
- Capteur IMX585 : [SOHO Enterprise](https://soho-enterprise.com/)

---

## Licence

GPL-3.0 — voir fichier `LICENSE`.  
Attributions tierces dans `NOTICE`.
