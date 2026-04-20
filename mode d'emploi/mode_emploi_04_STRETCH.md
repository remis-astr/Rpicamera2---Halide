# MODE EMPLOI RPiCamera2 - STRETCH ASTRO

## 1. VUE D'ENSEMBLE

Le **stretch** (étirement d'histogramme) révèle les détails faibles dans les images astronomiques en redistribuant les niveaux de luminosité. Dans RPiCamera2, le stretch s'applique **uniquement au preview pygame** et aux PNG générés en mode LIVESTACK.

**Note importante** : Les captures STILL (JPEG/PNG/DNG) ne sont **PAS** stretched. Pour sauvegarder une image stretchée, utiliser le mode LIVESTACK.

---

## 2. MODES DE STRETCH

**Paramètre** : `stretch_preset` (0-2)

| Index | Mode | Usage |
|-------|------|-------|
| **0** | OFF | Désactivé (image brute) |
| **1** | GHS | Generalized Hyperbolic Stretch (avancé) |
| **2** | Arcsinh | Arc-sinus hyperbolique (simple) |

**Activation** : Menu → STRETCH SETTINGS

---

## 3. MODE ARCSINH (Simple et efficace)

### 3.1 Principe

**Formule** :
```
1. Normalisation: x_norm = (x - p_low) / (p_high - p_low)
2. Stretch: y = arcsinh(x_norm × factor) / arcsinh(factor)
3. Reconversion: output = y × 255
```

**Avantages** :
- Simple (1 paramètre principal)
- Doux sur les transitions
- Préserve détails faibles
- Rapide à calculer

### 3.2 Paramètres

#### Facteur d'étirement (stretch_factor)

**Valeurs** : 10-1000 (divisé par 10, stocké ×10)

**Correspondance** :
```
stretch_factor=100 (UI) → factor=10.0 (réel)
stretch_factor=300 (UI) → factor=30.0 (réel)
stretch_factor=600 (UI) → factor=60.0 (réel)
```

**Effet** :
```
factor=5   : Étirement léger (objets brillants)
factor=10  : Standard (défaut)
factor=30  : Fort (objets faibles, nébuleuses)
factor=60  : Très fort (détails extrêmes)
```

**Courbe** :
```
Arcsinh amplifie:
- Faibles valeurs ++  (fond de ciel, néb. diffuses)
- Moyennes valeurs +  (étoiles faibles)
- Hautes valeurs =    (étoiles brillantes, Lune)
```

#### Percentile bas (stretch_p_low)

**Valeurs** : 0-20 (divisé par 10, stocké ×10)

**Correspondance** :
```
stretch_p_low=0  → 0.0%  (noir absolu préservé)
stretch_p_low=5  → 0.5%  (clip ombres légères)
stretch_p_low=10 → 1.0%  (réduction bruit fond)
```

**Effet** : Fixe le point noir (valeur → 0)
- Faible (0-5) : Préserve ombres, bruit visible
- Moyen (5-15) : Bon compromis (défaut)
- Élevé (15-20) : Clip ombres, image claire

#### Percentile haut (stretch_p_high)

**Valeurs** : 9900-10000 (divisé par 100, stocké ×100)

**Correspondance** :
```
stretch_p_high=9950 → 99.50%
stretch_p_high=9980 → 99.80%
stretch_p_high=9998 → 99.98%  (défaut)
```

**Effet** : Fixe le point blanc (valeur → 255)
- 99.0-99.5% : Clip highlights, détails brillants perdus
- 99.5-99.9% : Équilibré
- 99.95-99.98% : Préserve détails highlights (défaut)
- 99.99-100% : Dynamique maximale

### 3.3 Réglages typiques

**Ciel profond (galaxies, nébuleuses)** :
```python
stretch_preset = 2         # Arcsinh
stretch_factor = 300-600   # Fort (30-60 réel)
stretch_p_low = 5-10       # 0.5-1.0%
stretch_p_high = 9998      # 99.98%
```

**Amas d'étoiles** :
```python
stretch_factor = 100-200   # Moyen (10-20)
stretch_p_low = 0-5        # Préserver étoiles faibles
stretch_p_high = 9995      # 99.95%
```

**Lune / Planètes** :
```python
stretch_preset = 0         # OFF (objets brillants)
# OU
stretch_factor = 50-100    # Léger (5-10)
```

---

## 4. MODE GHS (Avancé)

### 4.1 Principe

**Generalized Hyperbolic Stretch** : Transformation non-linéaire paramétrable avec 5 paramètres indépendants.

**Référence** : https://ghsastro.co.uk

**Zones de transformation** :
```
[0 ---- LP ---- SP ---- HP ---- 1]
 │       │       │       │       │
 Shadow  Stretch Mid  Stretch  Highlight
 Linear  Hyperbolic    Mirror   Linear
```

### 4.2 Paramètres GHS

#### D - Stretch factor (ghs_D)

**Valeurs** : 0-100 (divisé par 10, stocké ×10)

**Correspondance** :
```
ghs_D=10  → D=1.0  (léger)
ghs_D=31  → D=3.1  (défaut)
ghs_D=50  → D=5.0  (fort)
```

**Effet** : Force globale de l'étirement
- D < 2.0 : Doux (objets brillants)
- D = 2.0-4.0 : Standard (défaut 3.1)
- D > 4.0 : Agressif (objets très faibles)

#### b - Local intensity (ghs_b)

**Valeurs** : 1-199 (divisé par 10, stocké ×10, correspond à -5.0 à +19.9)

**Correspondance** :
```
ghs_b=1   → b=0.1   (défaut RPiCamera2)
ghs_b=10  → b=1.0   (harmonique)
ghs_b=60  → b=6.0   (forte concentration)
```

**Type de transformation selon b** :
```
b = -1.0 : Logarithmique (contraste élevé)
b < 0    : Intégrale (transition douce)
b = 0    : Exponentielle (équilibré)
b = 1.0  : Harmonique (classique 1/(1+Dx))
b > 1.0  : Hyperbolique (concentration contraste)
```

**Effet visuel** :
- b faible (0.1-1.0) : Transition douce, large plage tonale
- b moyen (1.0-6.0) : Contraste concentré
- b élevé (6.0-15.0) : Fort contraste local, détails fins

#### SP - Symmetry Point (ghs_SP)

**Valeurs** : 0-100 (divisé par 100, stocké ×100)

**Correspondance** :
```
ghs_SP=8   → SP=0.08
ghs_SP=19  → SP=0.19  (défaut)
ghs_SP=50  → SP=0.50
```

**Effet** : Point focal du contraste
- SP bas (0.05-0.15) : Contraste sur ombres/détails faibles
- SP moyen (0.15-0.30) : Équilibré (défaut 0.19)
- SP élevé (0.30-0.50) : Contraste sur moyennes lumières

**Règle** : Aligner SP sur luminosité moyenne de l'objet principal
```
Nébuleuse diffuse → SP=0.10-0.20 (sombre)
Galaxie brillante → SP=0.20-0.30 (moyen)
Amas ouvert       → SP=0.30-0.50 (clair)
```

#### LP - Shadow Protection (ghs_LP)

**Valeurs** : 0-SP (divisé par 100)

**Correspondance** :
```
ghs_LP=0   → LP=0.00  (pas de protection)
ghs_LP=5   → LP=0.05
ghs_LP=10  → LP=0.10
```

**Effet** : Zone linéaire dans les ombres (pas de stretch)
- LP=0 : Tout stretched (défaut, révèle fond de ciel)
- LP>0 : Préserve ombres profondes, réduit bruit

**Usage** : Activer (LP=0.05-0.10) si bruit de fond excessif

#### HP - Highlight Protection (ghs_HP)

**Valeurs** : SP-100 (divisé par 100)

**Correspondance** :
```
ghs_HP=0    → HP=0.00   (pas de protection)
ghs_HP=85   → HP=0.85
ghs_HP=100  → HP=1.00
```

**Effet** : Zone linéaire dans les highlights (pas de stretch)
- HP=0 ou HP≤SP : Tout stretched (défaut)
- HP>SP : Préserve highlights brillants

**Usage** : Activer (HP=0.85-0.95) pour Lune/planètes dans champ étoilé

### 4.3 Presets GHS

**Preset 0 - Étirement initial** (défaut) :
```python
ghs_D = 35   # D=3.5
ghs_b = 120  # b=12.0
ghs_SP = 8   # SP=0.08
ghs_LP = 0   # LP=0.00
ghs_HP = 100 # HP=1.00
```
Usage : Premier stretch stack brut, révèle maximum détails

**Preset 1 - Galaxies** :
```python
ghs_D = 20   # D=2.0
ghs_b = 60   # b=6.0
ghs_SP = 15  # SP=0.15
ghs_LP = 0
ghs_HP = 85  # HP=0.85 (protège noyau brillant)
```
Usage : Galaxies avec noyau brillant, bras spiraux faibles

**Preset 2 - Nébuleuses** :
```python
ghs_D = 15   # D=1.5
ghs_b = 30   # b=3.0
ghs_SP = 25  # SP=0.25
ghs_LP = 0
ghs_HP = 95  # HP=0.95
```
Usage : Nébuleuses en émission, détails diffus

**Présets menu** : `ghs_preset` (0-2) charge automatiquement les valeurs ci-dessus.

### 4.4 Réglage manuel GHS

**Workflow interactif** :
1. Activer preview stretch (stretch_preset=1)
2. Partir preset proche (0/1/2)
3. Ajuster D (force globale)
4. Ajuster b (concentration contraste)
5. Ajuster SP (point focal)
6. Optionnel : LP/HP si besoin

**Observation visuelle** :
- **D trop faible** : Image plate, détails invisibles
- **D trop élevé** : Bruit amplifié, halos
- **b trop faible** : Manque de contraste
- **b trop élevé** : Contraste excessif, artefacts
- **SP incorrect** : Objet principal trop sombre/clair

---

## 5. COMPARAISON GHS vs ARCSINH

| Aspect | Arcsinh | GHS |
|--------|---------|-----|
| **Simplicité** | ✓✓ (1 param) | ✗ (5 params) |
| **Rapidité réglage** | ✓✓ | ✗ |
| **Contrôle précis** | ✗ | ✓✓ |
| **Qualité maximale** | ✓ | ✓✓ |
| **Usage débutant** | ✓✓ Recommandé | ✗ |
| **Usage expert** | ✓ | ✓✓ Recommandé |

**Recommandation** :
- **Débutants / Preview rapide** : Arcsinh (stretch_preset=2)
- **Traitement final / Expert** : GHS (stretch_preset=1)

---

## 6. APPLICATION DANS RPICAMERA2

### 6.1 Preview pygame (temps réel)

**Fonction** : `astro_stretch(array)`

**Pipeline** :
```
Frame capture (RGB float32)
    ↓
astro_stretch() selon stretch_preset
    ↓
Conversion uint8
    ↓
Affichage pygame surface
```

**Latence** : ~5-20ms selon résolution
**Mise à jour** : Chaque frame preview

### 6.2 LIVESTACK PNG

**Sauvegarde automatique** : Stack cumulatif → PNG stretched

**Pipeline** :
```
Stack FITS linéaire (RAW, haute dynamique)
    ↓
apply_stretch() (libastrostack)
    ↓
PNG 8-bit ou 16-bit (selon dynamique)
```

**Intervalle** : Tous les N frames (`ls_preview_refresh`)

### 6.3 Allsky JPEG

**Option** : `allsky_apply_stretch=1`

**Pipeline** :
```
JPEG brut → astro_stretch() → JPEG strecthed → Vidéo MP4
```

**Usage** : Timelapse longue durée avec révélation détails automatique

### 6.4 STILL (Pas de stretch)

**Important** : Captures STILL **ne sont jamais** stretchées.

**Raison** : Préserver données brutes pour post-traitement

**Workaround** : Utiliser LIVESTACK avec 1 seule image
```python
# Capture 1 image → Stack → PNG stretched
# Plutôt que STILL → JPEG non stretched
```

---

## 7. POST-TRAITEMENT EXTERNE

### 7.1 Export paramètres vers Siril/PixInsight

**GHS** : Compatible Siril 1.2+ et PixInsight GHS script

**Correspondance paramètres** :
```
RPiCamera2        Siril/PixInsight
ghs_D    (×10)    D
ghs_b    (×10)    b
ghs_SP   (×100)   SP
ghs_LP   (×100)   LP
ghs_HP   (×100)   HP
```

**Exemple export** :
```
RPiCamera2: ghs_D=31, ghs_b=60, ghs_SP=19
Siril GHS : D=3.1, b=6.0, SP=0.19
```

### 7.2 Arcsinh externe

**Siril** : Asinh Transformation
```
Offset (shadows) = stretch_p_low / 10
Stretch factor   = stretch_factor / 10
```

**PixInsight** : ArcsinhStretch
```
Stretch = stretch_factor / 10
Black Point = stretch_p_low / 10
```

---

## 8. OPTIMISATION ET CONSEILS

### 8.1 Minimiser bruit amplifié

**Problème** : Stretch révèle bruit fond de ciel

**Solutions** :
1. Augmenter `stretch_p_low` (clip ombres)
2. Activer `ghs_LP` (protection shadows)
3. Réduire gain capture (moins de bruit source)
4. Empilement (LIVESTACK réduit bruit)

### 8.2 Éviter saturation highlights

**Problème** : Étoiles brillantes saturées (blanches)

**Solutions** :
1. Réduire `stretch_p_high` (99.95% → 99.80%)
2. Activer `ghs_HP` (protection highlights)
3. Réduire `D` ou `stretch_factor`

### 8.3 Ajustement interactif rapide

**Méthode** :
1. Activer preview stretch (stretch_preset>0)
2. Capturer 1 frame test (LIVESTACK mode, 1 frame)
3. Ajuster paramètres en live
4. Observer preview pygame temps réel
5. Valider réglages
6. Lancer stack complet

### 8.4 Stretch par canal (avancé)

**Modification code** : Appliquer stretch séparément R/G/B
```python
# Actuellement: stretch sur grayscale ou RGB ensemble
# Modification: stretch chaque canal indépendamment
stretched_r = astro_stretch(frame[:,:,0])
stretched_g = astro_stretch(frame[:,:,1])
stretched_b = astro_stretch(frame[:,:,2])
```

**Avantage** : Meilleur équilibre couleur nébuleuses Ha/OIII/SII

---

## 9. RÉSUMÉ RAPIDE

### Arcsinh (Recommandé débutants)

| Paramètre | Valeur défaut | Plage utile |
|-----------|---------------|-------------|
| stretch_factor | 60 (6.0) | 100-600 (10-60) |
| stretch_p_low | 0 | 0-10 (0-1%) |
| stretch_p_high | 9998 | 9990-9999 |

**Usage** : Ajuster `stretch_factor` uniquement, laisser percentiles par défaut.

### GHS (Recommandé experts)

| Paramètre | Galaxies | Nébuleuses | Défaut |
|-----------|----------|------------|--------|
| ghs_D | 20 (2.0) | 15 (1.5) | 31 (3.1) |
| ghs_b | 60 (6.0) | 30 (3.0) | 1 (0.1) |
| ghs_SP | 15 (0.15) | 25 (0.25) | 19 (0.19) |
| ghs_LP | 0 | 0 | 0 |
| ghs_HP | 85 (0.85) | 95 (0.95) | 0 |

**Usage** : Partir d'un preset (`ghs_preset`), ajuster D/b/SP si nécessaire.

---

**Fichier suivant** : `mode_emploi_05_LIVESTACK.md`
