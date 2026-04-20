# MODE EMPLOI RPiCamera2 - LUCKY STACK (Lucky Imaging Planétaire)

## 1. VUE D'ENSEMBLE

Le mode **LUCKY STACK** est une variante du livestack optimisée pour l'imagerie planétaire haute cadence. Il capture un buffer circulaire, sélectionne les meilleures images (critère qualité), puis les empile.

**Déclenchement** : Menu → **LUCKY STACK** (sous-menu LIVE STACK)

**Différence avec LIVESTACK** :
- Buffer circulaire (10-200 frames)
- Sélection top N% meilleures images
- Haute cadence (50-178 fps)
- Alignement planétaire optionnel (disque)

**Usage** : Lune, planètes, Soleil

---

## 2. PRINCIPE LUCKY IMAGING

```
1. Remplir buffer (N frames)
   ├─ Frame 1 → Score qualité
   ├─ Frame 2 → Score qualité
   ...
   └─ Frame N → Score qualité

2. Trier par score (meilleurs → moins bons)

3. Sélectionner top X%
   (ex: 10% de 100 frames = 10 meilleures)

4. Aligner sélection (optionnel)

5. Empilement (mean/median/sigma-clip)

6. Afficher résultat
   ↓
7. Vider buffer, recommencer
```

**Avantage** : Rejette automatiquement frames floues (turbulence atmosphérique) en gardant seulement pics de seeing.

---

## 3. PARAMÈTRES BUFFER

### 3.1 Taille buffer (ls_lucky_buffer)

**Valeurs** : 10-200 frames

**Correspondance** :
```
ls_lucky_buffer=10   : 10 frames (test rapide)
ls_lucky_buffer=50   : 50 frames (standard)
ls_lucky_buffer=100  : 100 frames (optimal)
ls_lucky_buffer=200  : 200 frames (agressif)
```

**Impact** :
- Buffer petit (10-30) : Réactivité élevée, sélection limitée
- Buffer moyen (50-100) : ✓ Bon compromis
- Buffer grand (100-200) : Meilleure sélection, latence accrue

**Calcul latence** :
```
Latence = buffer_size / fps

Exemple:
Buffer=100, fps=100 → Latence=1s par stack
Buffer=100, fps=50  → Latence=2s par stack
```

### 3.2 Pourcentage à garder (ls_lucky_keep)

**Valeurs** : 1-50 (%)

**Correspondance** :
```
ls_lucky_keep=1   : Top 1% (ultra-sélectif)
ls_lucky_keep=10  : Top 10% (défaut, bon)
ls_lucky_keep=22  : Top 22% (tolérant)
ls_lucky_keep=50  : Top 50% (peu sélectif)
```

**Calcul frames gardées** :
```
keep_count = buffer_size × (ls_lucky_keep / 100)

Exemples:
Buffer=100, keep=10% → 10 frames empilées
Buffer=100, keep=1%  → 1 frame (meilleure seule)
Buffer=50,  keep=22% → 11 frames
```

**Stratégie** :
- **Seeing excellent** : 1-5% (garder seulement pics)
- **Seeing bon** : 10-15% (défaut)
- **Seeing médiocre** : 20-30% (SNR prioritaire)
- **Seeing catastrophique** : 40-50% (empile tout, inutile)

---

## 4. CRITÈRES DE QUALITÉ

**Paramètre** : `ls_lucky_score` (0-3)

### 4.1 Laplacian (0) - Défaut

**Principe** : Variance du Laplacien (détection contours)

**Formule** :
```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
score = laplacian.var()
```

**Avantages** :
- ✓ Rapide (<5ms)
- ✓ Robuste
- ✓ Recommandé défaut

**Score élevé** = contours nets = bonne image

### 4.2 Gradient (1)

**Principe** : Magnitude gradient (Sobel X/Y)

**Formule** :
```python
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
score = np.mean(np.sqrt(gx² + gy²))
```

**Avantages** :
- Sensible détails directionnels
- Bon pour planètes avec bandes (Jupiter)

### 4.3 Sobel (2)

**Principe** : Somme absolue Sobel X + Y

**Formule** :
```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
score = np.mean(|sobelx| + |sobely|)
```

**Similaire gradient**, légèrement différent

### 4.4 Tenengrad (3)

**Principe** : Sobel au carré (très sensible focus)

**Formule** :
```python
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
score = np.mean(gx² + gy²)
```

**Avantages** :
- Maximum sensibilité détails fins
- Meilleur pour planètes haute résolution (Mars détails)

**Inconvénients** :
- Plus sensible bruit
- Peut favoriser frames bruitées

**Recommandation** : **Laplacian (0)** pour 95% des cas.

---

## 5. ROI SCORING (Region of Interest)

**Paramètre** : `ls_lucky_roi` (20-100%)

**Effet** : Calcule score uniquement sur ROI centrale

**Correspondance** :
```
ls_lucky_roi=50  : ROI centrale 50% (défaut)
ls_lucky_roi=80  : ROI 80% (large)
ls_lucky_roi=100 : Frame complète
```

**Calcul ROI** :
```python
h, w = image.shape
margin_x = int(w * (100 - ls_lucky_roi) / 200)
margin_y = int(h * (100 - ls_lucky_roi) / 200)
roi = image[margin_y:h-margin_y, margin_x:w-margin_x]
```

**Usage** :
- **Planète petite** : 50-70% (focus sur cible, ignore fond)
- **Planète grande (Lune)** : 80-100% (tout le champ utile)

---

## 6. ALIGNEMENT PLANÉTAIRE

**Paramètre** : `ls_lucky_align` (0=OFF, 1=ON)

**Différence avec livestack** : Alignement spécialisé disque planétaire

### 6.1 Mode alignement planétaire

**Paramètre** : `ls_planetary_mode` (0-2)

| Mode | Description | Usage |
|------|-------------|-------|
| **0 - Disk** | Détection disque (Canny circles) | Lune pleine, Mars, Jupiter |
| **1 - Surface** | Corrélation surface | ✓ Défaut, universel |
| **2 - Hybrid** | Combinaison disk+surface | Expérimental |

**Recommandation** : Mode 1 (Surface) pour 90% des cas.

### 6.2 Paramètres détection disque

**Si mode=0 (Disk)** :

- `ls_planetary_disk_min` : 50 px (rayon min)
- `ls_planetary_disk_max` : 500 px (rayon max)
- `ls_planetary_threshold` : 30 (seuil Canny)
- `ls_planetary_ellipse` : 0 (cercle), 1 (ellipse)

**Usage** : Ajuster selon taille planète visible

### 6.3 Paramètres corrélation

- `ls_planetary_window` : 0=128px, 1=256px, 2=512px (taille fenêtre)
- `ls_planetary_upsample` : 10 (précision sub-pixel ×10)
- `ls_planetary_corr` : 30 (corrélation min ×100 = 0.30)
- `ls_planetary_max_shift` : 100 px (décalage max autorisé)

**Recommandation défaut** : Laisser valeurs par défaut sauf `window` (1=256px optimal).

---

## 7. MÉTHODES D'EMPILAGE

**Paramètre** : `ls_lucky_stack` (0-2)

| Index | Méthode | Usage |
|-------|---------|-------|
| **0** | Mean | ✓ Défaut, rapide |
| **1** | Median | Robuste cosmiques |
| **2** | Sigma-clip | Qualité max |

**Identique LIVESTACK** mais appliqué sur sélection buffer (pas cumulatif).

---

## 8. SAUVEGARDE

**Paramètre** : `ls_lucky_save_final` (0=OFF, 1=ON)

**Sorties** :
```
/home/admin/stacks/lucky_session_*/
  ├─ stack_001.fits      (Stack buffer 1)
  ├─ stack_001.png       (Preview stretched)
  ├─ stack_002.fits      (Stack buffer 2)
  ├─ stack_002.png
  ...
  └─ final_cumulative.fits  (Cumul tous stacks)
```

**Option progress** : `ls_lucky_save_progress=1` sauvegarde tous les 2 buffers.

---

## 9. DOUBLE NIVEAU STACKING

**Concept** : Lucky imaging implémente **2 niveaux d'empilage** :

### Niveau 1 - Intra-buffer (entre frames d'un buffer)

```
Buffer N frames → Sélection top X% → Stack 1
```

**Méthode** : `ls_lucky_stack` (mean/median/sigma-clip)

**Résultat** : 1 image stackée par buffer

### Niveau 2 - Inter-buffers (cumul entre stacks successifs)

```
Stack_1 + Stack_2 + ... + Stack_N → Stack cumulatif
```

**Méthode** : Empilage standard (kappa-sigma)

**Fonction** : `update_alignment_reference()` préserve cohérence entre buffers

**Résultat** : `final_cumulative.fits` (somme tous buffers)

---

## 10. WORKFLOW COMPLET

### 10.1 Configuration

```python
# Exposition haute cadence
mode = 0
speed = 5-15          # 5-15ms (67-200 fps)
gain = 50-150         # Élevé (compenser expo courte)
fps = 100             # Cible 100 fps
zoom = 2-3            # Crop hardware (selon taille planète)

# Lucky stack
ls_lucky_buffer = 100    # 100 frames/buffer
ls_lucky_keep = 10       # Top 10%
ls_lucky_score = 0       # Laplacian
ls_lucky_align = 1       # ON
ls_lucky_roi = 50        # ROI 50%
ls_lucky_stack = 0       # Mean
ls_planetary_mode = 1    # Surface
ls_planetary_window = 1  # 256px

# Autres
awb = 0                  # Manuel
red = 20, blue = 15      # Ajusté planète
```

### 10.2 Lancement

1. Centrer planète
2. Focus optimal (FWHM 3-7 px)
3. Activer **LUCKY STACK**
4. Observer :
   - Buffer fill % (0-100%)
   - Score seuil sélection
   - Preview actualisé chaque buffer
5. Laisser tourner 5-30 buffers (500-3000 frames)
6. Arrêter (ESC)

### 10.3 Post-traitement

**Import PIPP** : Si conversion SER vidéo souhaitée
```bash
# Alternative: Capturer SER directement (mode VIDEO)
# Lucky stack sur fichier SER externe
```

**Import Siril** :
```bash
siril
> load /home/admin/stacks/lucky_session_*/final_cumulative.fits
> autostretch
# Wavelets, deconvolution
```

**Registax/WinJupos** : Traiter stacks individuels ou cumul

---

## 11. COMPARAISON LIVESTACK vs LUCKYSTACK

| Aspect | LIVESTACK | LUCKYSTACK |
|--------|-----------|------------|
| **Cible** | Ciel profond | Planétaire |
| **Cadence** | 0.1-1 fps (poses longues) | 50-200 fps (courtes) |
| **Sélection** | Qualité absolue (seuils) | Top N% relatif (buffer) |
| **Alignement** | Étoiles (kd-tree) | Disque/surface planète |
| **Stack** | Cumulatif continu | Buffers successifs |
| **Durée** | Heures (30-500 frames) | Minutes (500-5000 frames) |
| **Objet** | Galaxie, nébuleuse | Lune, Mars, Jupiter |

---

## 12. OPTIMISATIONS PLANÉTAIRE

### 12.1 Calcul buffer optimal

**Règle** : Buffer doit couvrir ~1-2s de seeing

```
Temps_seeing = 0.5-2.0s (typique)
Buffer = fps × Temps_seeing

Exemple:
fps=100, seeing=1s → buffer=100 frames ✓
fps=50, seeing=2s  → buffer=100 frames ✓
```

### 12.2 Pourcentage adaptatif

**Seeing excellent (<1" FWHM)** :
```python
ls_lucky_keep = 1-5   # Très sélectif
# Garde seulement pics absolus
```

**Seeing moyen (1-2")** :
```python
ls_lucky_keep = 10-15  # ✓ Standard
```

**Seeing médiocre (2-3")** :
```python
ls_lucky_keep = 20-30  # Priorité SNR
```

**Seeing catastrophique (>3")** :
```python
# Lucky imaging peu efficace
# Préférer: Pose longue classique (LIVESTACK)
```

### 12.3 Drizzle planétaire

**Technique avancée** : Combiner sous-pixels décalés

**Prérequis** :
- Alignement sub-pixel (upsample=10+)
- Nombreux buffers (10+)
- Décalages suffisants (dérive légère OK)

**Non implémenté actuellement** : Prévu module drizzle avancé.

---

## 13. DÉPANNAGE

### 13.1 Toutes frames même score

**Symptômes** : Seuil sélection=score_min, pas de tri

**Causes** :
1. ROI trop petit (objet pas dedans)
2. Exposition incorrecte (saturation/noir)
3. Défocus total

**Solutions** :
1. Augmenter `ls_lucky_roi` (80-100%)
2. Ajuster exposition
3. Refaire focus

### 13.2 Alignement échoue

**Symptômes** : Stacks décalés, planète floue

**Causes** :
1. Dérive excessive (>max_shift)
2. Window trop petite
3. Mode disk inadapté

**Solutions** :
1. Augmenter `ls_planetary_max_shift` (200 px)
2. Augmenter `window` (2=512px)
3. Passer mode 1 (Surface)
4. Améliorer guidage

### 13.3 Buffer trop lent

**Symptômes** : FPS réel << FPS configuré

**Causes** :
1. Exposition trop longue (>1/fps)
2. Carte SD lente
3. Résolution trop élevée

**Solutions** :
1. Réduire `speed` (exposition)
2. Carte UHS-II V90
3. Augmenter zoom (crop résolution)

---

## 14. RÉSUMÉ RAPIDE

| Paramètre | Défaut | Optimal planétaire |
|-----------|--------|-------------------|
| **Buffer** | 10 | 100 frames |
| **Keep** | 10 | 10% (seeing bon) |
| **Score** | 0 (Laplacian) | ✓ Garder |
| **Align** | 1 (ON) | ✓ Garder |
| **ROI** | 50% | 50-70% |
| **Stack** | 0 (Mean) | ✓ Garder |
| **Mode** | 1 (Surface) | ✓ Garder |

**Workflow minimal** :
1. Haute cadence (speed=5-10ms, gain=50-150, fps=100)
2. Activer LUCKY STACK (buffer=100, keep=10%)
3. Laisser tourner 5-30 buffers
4. Récupérer `final_cumulative.fits`
5. Traiter Registax/Siril (wavelets)

---

**FIN DU MODE D'EMPLOI RPICAMERA2**

Les 6 fichiers complets couvrent :
1. **STILL** - Photographie
2. **VIDEO** - Enregistrement vidéo
3. **TIMELAPSE** - Timelapse et Allsky
4. **STRETCH** - Étirement d'histogramme
5. **LIVESTACK** - Empilage ciel profond
6. **LUCKYSTACK** - Lucky imaging planétaire
