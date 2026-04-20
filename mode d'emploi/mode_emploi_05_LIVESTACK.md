# MODE EMPLOI RPiCamera2 - LIVESTACK (Empilage temps réel)

## 1. VUE D'ENSEMBLE

Le mode **LIVESTACK** empile des images en temps réel avec alignement automatique et contrôle qualité. Optimisé pour l'astrophotographie ciel profond (galaxies, nébuleuses, amas).

**Déclenchement** : Menu principal → Bouton **LIVE STACK**

**Sorties** :
- **FITS linéaire** : Stack cumulatif haute dynamique (32-bit float)
- **PNG stretched** : Preview 8/16-bit (tous les N frames)
- **DNG individuels** : Images acceptées (optionnel)

---

## 2. PIPELINE DE TRAITEMENT

```
Capture RAW/RGB
    ↓
Débayérisation (si RAW Bayer)
    ↓
ISP optionnel (balance blancs, gamma)
    ↓
Contrôle Qualité ───┐
    ✓ ACCEPTÉE    ✗ REJETÉE
    ↓                ↓
Alignement       Log rejet
(Translation/      ↓
 Rotation/       Compteur
 Affine)         reject
    ↓
Empilage
(mean/median/
 kappa-sigma)
    ↓
Preview PNG stretched
(tous les N frames)
    ↓
Sauvegarde finale
FITS + PNG
```

---

## 3. PARAMÈTRES DE CONTRÔLE QUALITÉ

**Activation** : `ls_enable_qc` (0=OFF, 1=ON)

### 3.1 FWHM maximale (ls_max_fwhm)

**Valeurs** : 0-250 (divisé par 10, en pixels)

**Correspondance** :
```
ls_max_fwhm=0    → OFF (pas de test FWHM)
ls_max_fwhm=80   → 8.0 px (excellent seeing)
ls_max_fwhm=170  → 17.0 px (défaut, bon)
ls_max_fwhm=250  → 25.0 px (accepte seeing médiocre)
```

**Critère** : Image rejetée si FWHM > seuil
- FWHM faible = étoiles fines, bon seeing
- FWHM élevé = étoiles floues, turbulence

**Réglage** :
1. Observer FWHM preview (affichage temps réel)
2. Fixer seuil = FWHM_max_observé × 1.2
3. Rejette automatiquement images floues

### 3.2 Netteté minimale (ls_min_sharpness)

**Valeurs** : 0-150 (divisé par 1000)

**Correspondance** :
```
ls_min_sharpness=0   → OFF
ls_min_sharpness=70  → 0.070 (défaut)
ls_min_sharpness=150 → 0.150 (strict)
```

**Critère** : Laplacian variance > seuil
- Netteté faible = image floue (défocus, turbulence)
- Netteté élevée = détails fins visibles

### 3.3 Dérive maximale (ls_max_drift)

**Valeurs** : 0-5000 pixels

**Correspondance** :
```
ls_max_drift=0    → OFF
ls_max_drift=500  → 50 px (strict)
ls_max_drift=2500 → 250 px (défaut, tolérant)
```

**Critère** : Décalage (dx, dy) par rapport à référence < seuil
- Dérive faible = tracking précis
- Dérive élevée = monture mal alignée, guidage défaillant

**Usage** : Détecter erreurs guidage grossières

### 3.4 Étoiles minimales (ls_min_stars)

**Valeurs** : 0-20

**Correspondance** :
```
ls_min_stars=0   → OFF
ls_min_stars=10  → 10 étoiles min (défaut)
ls_min_stars=20  → 20 étoiles (dense)
```

**Critère** : Nombre d'étoiles détectées > seuil
- Insuffisant → Nuages, obstruction, défocus total

**Seuil recommandé** :
- Champ large ciel profond : 10-15
- Champ étroit galaxie : 5-10
- Milky Way : 20+

---

## 4. ALIGNEMENT

**Paramètre** : `ls_alignment_mode` (0-3)

| Mode | Description | Transforms | Usage |
|------|-------------|------------|-------|
| **0 - OFF** | Pas d'alignement | - | Monture parfaite (rare) |
| **1 - Translation** | X, Y | 2 DOF | Monture alt-az, dérive faible |
| **2 - Rotation** | X, Y, θ | 3 DOF | ✓ Défaut, monture équatoriale |
| **3 - Affine** | X, Y, θ, scale, shear | 6 DOF | Correction distorsion optique |

**Algorithme** :
1. Détection étoiles (peak_local_max)
2. Matching étoiles courante ↔ référence (kd-tree)
3. RANSAC (robustesse cosmiques, satellites)
4. Calcul matrice transformation
5. Warp image (interpolation cubique)

**Seuils rejection** :
- Rotation max : 5.0° (évite transformations aberrantes)
- Scale min/max : 0.95-1.05 (évite zoom artificiel)
- Min inliers : 30% (qualité matching minimum)

**Recommandation** : Mode 2 (Rotation) pour 99% des cas.

---

## 5. MÉTHODES D'EMPILAGE

**Paramètre** : `ls_stack_method` (0-4)

### 5.1 Mean (Moyenne)

**Index** : 0

**Formule** :
```
Stack(x,y) = Σ Images(x,y) / N
```

**Avantages** :
- Rapide
- Bruit blanc réduit √N
- Linéaire (préserve photométrie)

**Inconvénients** :
- Sensible cosmiques (pics isolés)
- Pas de rejection outliers

**Usage** : Défaut, bon compromis vitesse/qualité

### 5.2 Median (Médiane)

**Index** : 1

**Formule** :
```
Stack(x,y) = median([Image_1(x,y), ..., Image_N(x,y)])
```

**Avantages** :
- Robuste cosmiques (rejection automatique)
- Pas de paramètre à ajuster

**Inconvénients** :
- Plus lent (tri pixel par pixel)
- Perd photométrie stricte
- Nécessite ≥5 images pour efficacité

**Usage** : Cosmiques fréquents, pas de post-traitement photométrie

### 5.3 Kappa-Sigma (Recommandé)

**Index** : 2

**Paramètres** :
- `ls_stack_kappa` : 15-35 (divisé par 10, défaut 25=2.5σ)
- `ls_stack_iterations` : 1-5 (défaut 3)

**Algorithme** :
```
Itération 1:
  mean = moyenne pixels
  std = écart-type pixels
  Rejeter si |pixel - mean| > kappa × std

Itération 2:
  Recalculer mean/std sur pixels restants
  Rejeter outliers

Itération 3:
  Convergence → moyenne finale
```

**Avantages** :
- Rejette cosmiques (comme median)
- Préserve photométrie (comme mean)
- Réglable (kappa/iterations)

**Inconvénients** :
- Plus complexe (2 paramètres)
- Légèrement plus lent que mean

**Réglage kappa** :
```
kappa=15 (1.5σ) : Strict (rejette beaucoup)
kappa=25 (2.5σ) : ✓ Défaut équilibré
kappa=35 (3.5σ) : Tolérant (garde presque tout)
```

**Usage** : ✓ Défaut recommandé, meilleur compromis qualité/robustesse

### 5.4 Winsorized (Troncature)

**Index** : 3

**Principe** : Remplace extrêmes (min/max) par percentiles proches

**Usage** : Alternative median/kappa-sigma, intermédiaire

### 5.5 Weighted (Pondéré)

**Index** : 4

**Principe** : Pondération par qualité image (FWHM, SNR)

**Usage** : Avancé, favorise meilleures images

---

## 6. REFRESH PREVIEW

**Paramètre** : `ls_preview_refresh` (1-10 frames)

**Effet** : Sauvegarde PNG stretched tous les N frames acceptées

```
ls_preview_refresh=1 : Tous les frames (fichiers nombreux)
ls_preview_refresh=5 : ✓ Défaut (économie espace)
ls_preview_refresh=10: Réduit I/O (session longue)
```

**Sorties** :
```
/home/admin/stacks/
  └─ session_YYYYMMDD_HHMMSS/
       ├─ preview_001.png  (1 frame)
       ├─ preview_006.png  (6 frames)
       ├─ preview_011.png  (11 frames)
       ...
       └─ final.fits       (stack cumulatif)
```

**Usage pygame preview** : Affichage dernière PNG générée en temps réel

---

## 7. SAUVEGARDE

### 7.1 Options

- `ls_save_progress` : PNG intermédiaires (0=OFF, 1=ON)
- `ls_save_final` : FITS/PNG final (0=OFF, 1=ON)

### 7.2 Format FITS

**Type** : 32-bit float linéaire

**Contenu** : Stack cumulatif RAW (sans stretch)

**Usage** : Import Siril/PixInsight pour traitement avancé

**Header** :
```
NAXIS = 2/3 (grayscale/RGB)
BITPIX = -32 (float32)
STACKED = N (nombre frames)
EXPOSURE = Σ Temps_expo (secondes)
```

### 7.3 Format PNG

**Type** : 8-bit ou 16-bit (auto selon dynamique)

**Contenu** : Stack stretched (preview)

**Stretch** : Selon paramètres globaux (GHS/Arcsinh)

**Usage** : Visualisation rapide, partage

---

## 8. ISP (Image Signal Processor)

**Activation** : `isp_enable` (0=OFF, 1=ON)

**Fonction** : Correction balance blancs, gamma, contraste sur RAW

**Mode calibration** :
- **Manuel** : Charger config JSON (`isp_config_path`)
- **Auto** : Analyser pair RAW/YUV après N frames

**Paramètres** :
- `isp_auto_calibrate_after` : 10 frames (délai calibration)
- `isp_auto_update_only_wb` : True (maj gains seulement)

**Usage** : Correction dérive couleur longue pose, WB cohérent

---

## 9. WORKFLOW COMPLET

### 9.1 Configuration session

```python
# Exposition (comme STILL)
mode = 0
speed = 50-60  # 1-10s selon objet
gain = 5-15    # Faible à moyen
awb = 0        # Manuel
red = 15, blue = 12

# Format
extn = 3       # DNG (pour RAW stack)
raw_format = 1 # SRGGB16
zoom = 0       # Plein champ

# LiveStack
ls_alignment_mode = 2       # Rotation
ls_stack_method = 2         # Kappa-sigma
ls_stack_kappa = 25         # 2.5σ
ls_enable_qc = 1            # ✓ Contrôle qualité
ls_max_fwhm = 170           # 17 px
ls_min_sharpness = 70       # 0.070
ls_max_drift = 2500         # 250 px
ls_min_stars = 10           # 10 étoiles min
ls_preview_refresh = 5      # PNG tous les 5 frames
```

### 9.2 Lancement

1. Centrer objet (preview)
2. Focus optimal (FWHM < 5-10 px)
3. Appuyer **LIVE STACK**
4. Observer :
   - Compteurs acceptées/rejetées
   - FWHM temps réel
   - Preview PNG actualisé
5. Arrêt : Touche **ESC** ou bouton **STOP**

### 9.3 Suivi en temps réel

**Indicateurs OSD** :
```
LIVE STACK
Stacked: 45 / Total: 50 (90% acceptées)
FWHM: 8.2 px ✓ (vert si < seuil)
Exposure: 450s
```

**Fichiers générés** :
```
/home/admin/stacks/session_*/
  preview_005.png  ← Actualisé pygame
  preview_010.png
  ...
```

### 9.4 Fin session

**Sauvegarde automatique** :
```
session_YYYYMMDD_HHMMSS/
  ├─ final.fits           (stack cumulatif 32-bit)
  ├─ final_stretched.png  (preview 8/16-bit)
  ├─ rejected_list.txt    (log rejets)
  └─ config_session.json  (paramètres utilisés)
```

**Import Siril** :
```bash
cd /home/admin/stacks/session_*/
siril
> load final.fits
> autostretch
# Traiter (wavelets, deconvolution, etc.)
```

---

## 10. DÉPANNAGE

### 10.1 Toutes images rejetées

**Symptômes** : Compteur acceptées=0

**Causes** :
1. Seuils QC trop stricts
2. Défocus complet
3. Nuages

**Solutions** :
1. Désactiver QC (`ls_enable_qc=0`) temporairement
2. Ajuster focus (observer FWHM)
3. Augmenter seuils (`ls_max_fwhm`, `ls_min_sharpness`)

### 10.2 Alignement échoue

**Symptômes** : Images décalées dans stack

**Causes** :
1. Étoiles insuffisantes (< 5)
2. Champ trop uniforme (nébuleuse diffuse)
3. Dérive excessive (> max_drift)

**Solutions** :
1. Augmenter exposition (plus d'étoiles visibles)
2. Réduire zoom (champ plus large)
3. Améliorer guidage (autoguidage)
4. Mode alignment OFF si monture parfaite

### 10.3 Preview PNG non actualisé

**Symptômes** : Pygame affiche ancienne preview

**Causes** :
1. `ls_preview_refresh` trop élevé
2. Aucune image acceptée
3. Erreur sauvegarde PNG

**Solutions** :
1. Réduire `ls_preview_refresh` (1-3)
2. Vérifier permissions `/home/admin/stacks/`
3. Vérifier espace disque

### 10.4 Stack saturé (blanc)

**Symptômes** : PNG preview uniformément blanc

**Causes** :
1. Surexposition
2. Stretch trop agressif
3. Mauvaise normalisation RAW

**Solutions** :
1. Réduire exposition (speed, gain)
2. Ajuster stretch (réduire D ou factor)
3. Vérifier `raw_format` (12 vs 16-bit)

---

## 11. OPTIMISATIONS AVANCÉES

### 11.1 Correction pixels morts

**Paramètres** :
- `fix_bad_pixels_sigma` : 40 (4.0σ, divisé par 10)
- `fix_bad_pixels_min_adu` : 100 (10 ADU, divisé par 10)

**Activation** : Automatique si `fix_bad_pixels=True` dans débayérisation

**Algorithme** : Sigma-clipping local, remplacement par médiane voisins

### 11.2 Drizzle (Sur-échantillonnage)

**Non implémenté actuellement** : Prévu libastrostack v2.0

**Principe** : Combiner sous-pixels décalés → résolution augmentée

### 11.3 Lucky imaging intégré

**Voir mode LUCKY STACK** (mode séparé, buffer circulaire)

---

## 12. RÉSUMÉ RAPIDE

| Paramètre | Valeur défaut | Recommandation |
|-----------|---------------|----------------|
| **Alignement** | 2 (Rotation) | ✓ Garder |
| **Stack method** | 2 (Kappa-sigma) | ✓ Garder |
| **Kappa** | 25 (2.5σ) | ✓ Garder |
| **QC** | 1 (ON) | ✓ Activer |
| **FWHM max** | 170 (17 px) | Ajuster selon seeing |
| **Refresh** | 5 frames | ✓ Garder |

**Workflow minimal** :
1. Configurer exposition (mode=0, speed, gain)
2. Activer LIVE STACK
3. Observer compteurs
4. Arrêter après N frames acceptées (30-100+)
5. Récupérer `final.fits` pour traitement Siril

---

**Fichier suivant** : `mode_emploi_06_LUCKYSTACK.md`
