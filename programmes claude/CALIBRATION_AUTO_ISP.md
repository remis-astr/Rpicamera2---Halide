# Calibration automatique de l'ISP par pics d'histogrammes

## 📋 Vue d'ensemble

Cette fonctionnalité permet d'automatiser l'alignement des couleurs dans l'ISP (Image Signal Processor) pour les modes de stack d'images RAW, en utilisant les pics des histogrammes RGB comme référence.

### ✨ Avantages

- **Automatique** : Plus besoin de calibrer manuellement les gains RGB
- **Adaptatif** : S'ajuste aux conditions d'observation et à l'objet
- **Robuste** : Basé sur les pics d'histogrammes (mode statistique), moins sensible aux outliers
- **Dynamique** : Peut recalibrer périodiquement pendant une session longue

### 🔧 Principe de fonctionnement

1. **Analyse** : Calcule l'histogramme de chaque canal RGB de l'image stackée
2. **Détection** : Identifie le pic (valeur la plus fréquente) de chaque canal
3. **Calcul** : Détermine les gains nécessaires pour aligner tous les pics
4. **Application** : Applique les gains au pipeline ISP

Le canal **vert** est utilisé comme référence (gain = 1.0), et les canaux rouge et bleu sont ajustés pour aligner leurs pics sur celui du vert.

---

## 🚀 Utilisation

### Option 1 : Configuration dans le code

```python
from libastrostack.config import StackingConfig
from libastrostack.session import LiveStackSession

# Créer la configuration
config = StackingConfig()

# Activer l'ISP
config.isp_enable = True
config.video_format = 'raw12'  # ou 'raw16'

# Configurer la calibration automatique
config.isp_auto_calibrate_method = 'histogram_peaks'  # Méthode recommandée
config.isp_auto_calibrate_after = 10                  # Calibrer après 10 frames
config.isp_recalibrate_interval = 50                  # Recalibrer tous les 50 frames

# Créer la session
session = LiveStackSession(config)
session.start()

# Le reste du code reste identique
# La calibration se fait automatiquement !
```

### Option 2 : Calibration manuelle ponctuelle

```python
from libastrostack.isp import ISPCalibrator, ISP

# Après avoir stacké quelques images...
stacked_image = session.stacker.get_stacked_image()

# Calibrer l'ISP
isp_config = ISPCalibrator.calibrate_from_stacked_image(
    stacked_image,
    method='histogram_peaks'  # ou 'gray_world'
)

# Appliquer la nouvelle configuration
session.isp = ISP(isp_config)

# Sauvegarder pour réutilisation
session.isp.save_config('my_isp_config.json')
```

---

## ⚙️ Paramètres de configuration

### `isp_auto_calibrate_method`

Méthode de calibration automatique :

- **`'none'`** (défaut) : Désactivé, utilise la config manuelle ou chargée
- **`'histogram_peaks'`** (recommandé) : Aligne les pics des histogrammes RGB
- **`'gray_world'`** : Suppose que la moyenne des couleurs doit être neutre

**Recommandation** : Utilisez `'histogram_peaks'` pour l'astrophotographie, car elle est plus robuste pour les images avec beaucoup de fond du ciel.

### `isp_auto_calibrate_after`

Nombre de frames stackées avant la calibration initiale.

- **Défaut** : `10`
- **Minimum recommandé** : `5` (pour avoir un signal suffisant)
- **Optimal** : `10-20` (bon compromis signal/délai)

**Pourquoi ?** Les premières frames peuvent avoir un SNR faible. Attendre quelques frames améliore la précision de la calibration.

### `isp_recalibrate_interval`

Nombre de frames entre chaque recalibration.

- **Défaut** : `0` (désactivé)
- **Valeur typique** : `50-100` frames
- **Sessions longues** : `25-50` frames (pour suivre les variations de conditions)

**Pourquoi recalibrer ?**
- Évolution des conditions atmosphériques
- Changement de l'objet dans le champ (ex: Voie Lactée qui se déplace)
- Drift thermique du capteur

**⚠️ Attention** : Ne pas recalibrer trop souvent (< 20 frames) car cela peut introduire des discontinuités.

---

## 📊 Exemple complet avec RPiCamera2

```python
from libastrostack.config import StackingConfig
from libastrostack.session import LiveStackSession

# Configuration optimale pour astrophotographie
config = StackingConfig()

# === ISP avec calibration automatique ===
config.isp_enable = True
config.video_format = 'raw12'
config.isp_auto_calibrate_method = 'histogram_peaks'
config.isp_auto_calibrate_after = 15      # Calibrer après 15 frames
config.isp_recalibrate_interval = 50      # Recalibrer tous les 50 frames

# === Contrôle qualité ===
config.quality.enable = True
config.quality.max_fwhm = 12.0
config.quality.min_stars = 10

# === Alignement ===
config.alignment_mode = 'rotation'

# === PNG ===
config.png_stretch_method = 'asinh'
config.png_stretch_factor = 10.0

# Créer et démarrer la session
session = LiveStackSession(config)
session.start()

# Stacker les images
# La calibration automatique se fera après 15 frames
# Et sera mise à jour tous les 50 frames
for frame in camera.capture_sequence():
    session.process_image_data(frame)

# Les gains RGB sont ajustés automatiquement !
```

---

## 🔍 Diagnostic et monitoring

### Vérifier les gains appliqués

Pendant l'exécution, les logs montrent :

```
[ISP] Calibration automatique initiale (méthode: histogram_peaks)...

=== Calibration par pics d'histogramme ===
  Rouge: pic=0.2150, gain=1.163
  Vert: pic=0.2500, gain=1.000
  Bleu: pic=0.3200, gain=0.781

✓ Luminosité moyenne: 0.245
✓ Gamma: 2.2

✓ Calibration terminée

[ISP] ✓ Calibration réussie! Gains RGB: R=1.163, G=1.000, B=0.781
```

### Visualiser les histogrammes

Utilisez le script de test fourni :

```bash
# Tester la calibration sur une image stackée existante
python test_isp_auto_color_balance.py stacked_image.fits

# Génère:
# - histogram_before_calibration.png
# - histogram_after_calibration.png
# - image_before_isp.png
# - image_after_isp.png
# - isp_config_auto_histogram.json
```

---

## 📈 Cas d'usage recommandés

### 1. **Sessions courtes (< 100 frames)**

```python
config.isp_auto_calibrate_method = 'histogram_peaks'
config.isp_auto_calibrate_after = 10
config.isp_recalibrate_interval = 0  # Pas de recalibration
```

**Pourquoi ?** Une calibration initiale suffit pour une courte session.

### 2. **Sessions longues (> 200 frames)**

```python
config.isp_auto_calibrate_method = 'histogram_peaks'
config.isp_auto_calibrate_after = 15
config.isp_recalibrate_interval = 50  # Recalibrer tous les 50 frames
```

**Pourquoi ?** Les conditions peuvent changer (dérive thermique, atmosphère, objet).

### 3. **All-sky ou grand champ avec Voie Lactée**

```python
config.isp_auto_calibrate_method = 'histogram_peaks'
config.isp_auto_calibrate_after = 20
config.isp_recalibrate_interval = 30  # Recalibration plus fréquente
```

**Pourquoi ?** La composition de l'image change beaucoup avec le mouvement du ciel.

### 4. **Objet ponctuel (planète, nébuleuse)**

```python
config.isp_auto_calibrate_method = 'histogram_peaks'
config.isp_auto_calibrate_after = 10
config.isp_recalibrate_interval = 100  # Recalibration rare
```

**Pourquoi ?** L'objet est stable, peu de variation dans les histogrammes.

---

## 🧪 Comparaison des méthodes

### `histogram_peaks` (Recommandé)

**Principe** : Aligne les pics (modes) des histogrammes RGB

**Avantages** :
- ✅ Très robuste aux outliers (étoiles brillantes, pixels chauds)
- ✅ Idéal pour images avec beaucoup de fond du ciel
- ✅ Représente la valeur la plus fréquente (fond du ciel en astro)

**Inconvénients** :
- ❌ Peut être moins précis si l'histogramme est multi-modal
- ❌ Nécessite un nombre suffisant de pixels (minimum 5 frames stackées)

### `gray_world`

**Principe** : Suppose que la moyenne des couleurs doit être grise

**Avantages** :
- ✅ Classique et bien établi
- ✅ Fonctionne avec peu de données

**Inconvénients** :
- ❌ Suppose que la scène contient une variété de couleurs (faux en astro)
- ❌ Sensible aux étoiles brillantes et objets colorés
- ❌ Pas idéal pour ciel profond

---

## 📝 Notes techniques

### Masquage des pixels extrêmes

La méthode `histogram_peaks` ignore automatiquement :
- Les pixels noirs (< 5% de la plage)
- Les pixels saturés (> 95% de la plage)

Cela améliore la robustesse en éliminant :
- Les pixels morts et hot pixels
- Les étoiles très brillantes saturées
- Le bruit de lecture dans les ombres

### Normalisation par le vert

Le canal **vert** est toujours la référence (gain = 1.0) car :
- C'est la convention standard (balance des blancs)
- Le capteur a généralement plus de photodiodes vertes (pattern Bayer)
- Le vert représente la luminance perçue

### Limites des gains

Les gains sont automatiquement clippés entre **0.5 et 2.0** pour éviter :
- Les valeurs extrêmes dues à des erreurs de mesure
- Les corrections excessives qui amplifient le bruit
- Les saturations ou pertes de données

---

## 🐛 Dépannage

### Problème : "Image trop sombre pour calibration"

**Cause** : L'image stackée a une luminosité moyenne trop faible (pic < 0.001)

**Solutions** :
1. Augmenter `isp_auto_calibrate_after` (ex: 20 au lieu de 10)
2. Vérifier l'exposition de la caméra
3. Vérifier les paramètres de gain de la caméra

### Problème : Les couleurs changent brusquement

**Cause** : Recalibration trop fréquente ou histogramme instable

**Solutions** :
1. Augmenter `isp_recalibrate_interval` (ex: 100 au lieu de 50)
2. Désactiver la recalibration (`isp_recalibrate_interval = 0`)
3. Utiliser une config ISP fixe après la calibration initiale

### Problème : Les couleurs sont toujours déséquilibrées

**Causes possibles** :
1. La méthode `gray_world` n'est pas adaptée à votre scène
2. Pas assez de frames pour une calibration fiable
3. L'image stackée est dominée par un objet coloré (nébuleuse rouge, etc.)

**Solutions** :
1. Utiliser `histogram_peaks` au lieu de `gray_world`
2. Augmenter `isp_auto_calibrate_after` (ex: 20-30 frames)
3. Utiliser une calibration manuelle avec une image de référence
4. Sauvegarder les gains d'une bonne calibration et les réutiliser

---

## 📚 Références

### Fichiers modifiés

- `libastrostack/isp.py` : Nouvelles méthodes de calibration
  - `_estimate_white_balance_from_histogram_peaks()` (ligne 194-269)
  - `calibrate_from_stacked_image()` (ligne 271-368)

- `libastrostack/config.py` : Nouveaux paramètres
  - `isp_auto_calibrate_method` (ligne 74)
  - `isp_auto_calibrate_after` (ligne 75)
  - `isp_recalibrate_interval` (ligne 76)

- `libastrostack/session.py` : Intégration dans le workflow
  - `_check_and_calibrate_isp_if_needed()` (ligne 506-583)

### Scripts de test

- `test_isp_auto_color_balance.py` : Test et visualisation

### Exemples de configuration ISP

- `isp_config_auto.json` : Config auto existante (calibration RAW vs YUV)
- `isp_config_neutral.json` : Config neutre pour FITS linéaire
- `isp_config_imx585.json` : Config spécifique capteur IMX585

---

## 🎯 Conclusion

L'alignement automatique des couleurs par pics d'histogrammes est une fonctionnalité puissante qui :

✅ Simplifie le workflow d'astrophotographie
✅ S'adapte automatiquement aux conditions
✅ Améliore la cohérence des couleurs
✅ Élimine les ajustements manuels fastidieux

**Recommandation** : Commencez avec les paramètres par défaut, puis ajustez selon vos besoins spécifiques.

**Contact** : Pour questions ou problèmes, voir le README principal du projet.
