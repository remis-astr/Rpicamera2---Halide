# ✅ Résumé de l'implémentation : Calibration automatique ISP

## 📌 Demande initiale

Automatiser le réglage de l'alignement des couleurs dans l'ISP pour les modes de stack d'images RAW, en se basant sur les pics de chaque canal de l'histogramme comme référence d'alignement RGB.

## ✅ Implémentation réalisée

### 1. **Nouvelle méthode de calibration par histogrammes**

**Fichier** : `libastrostack/isp.py:194-269`

```python
ISPCalibrator._estimate_white_balance_from_histogram_peaks(image, bins=256, mask_range=(0.05, 0.95))
```

**Fonctionnalités** :
- Calcule l'histogramme pour chaque canal RGB
- Détecte le pic (mode) de chaque canal
- Masque automatiquement les pixels noirs et saturés
- Calcule les gains pour aligner les pics sur le canal vert (référence)
- Limite les gains entre 0.5 et 2.0 pour éviter les extrêmes

**Avantages** :
- ✅ Très robuste aux outliers (étoiles brillantes, pixels chauds)
- ✅ Idéal pour l'astrophotographie (fond du ciel dominant)
- ✅ Représente la valeur la plus fréquente de l'image

### 2. **Calibration complète depuis image stackée**

**Fichier** : `libastrostack/isp.py:271-368`

```python
ISPCalibrator.calibrate_from_stacked_image(image, method='histogram_peaks', target_brightness=None)
```

**Fonctionnalités** :
- Calibration complète ISP sans image de référence
- Deux méthodes : `histogram_peaks` (recommandé) et `gray_world`
- Ajuste automatiquement :
  - Balance des blancs RGB
  - Gamma (adaptatif selon luminosité)
  - Contraste, saturation, black level
- Sauvegarde les métadonnées de calibration

### 3. **Nouveaux paramètres de configuration**

**Fichier** : `libastrostack/config.py:73-76`

```python
config.isp_auto_calibrate_method = 'histogram_peaks'  # 'none', 'histogram_peaks', 'gray_world'
config.isp_auto_calibrate_after = 10                  # Calibrer après N frames
config.isp_recalibrate_interval = 50                  # Recalibrer tous les N frames (0=jamais)
```

### 4. **Intégration dans le workflow de live stacking**

**Fichier** : `libastrostack/session.py:506-583`

```python
LiveStackSession._check_and_calibrate_isp_if_needed()
```

**Fonctionnalités** :
- Calibration initiale automatique après N frames stackées
- Recalibration périodique optionnelle (tous les N frames)
- Affichage des gains RGB dans les logs
- Sauvegarde automatique de la config ISP
- Gestion robuste des erreurs

### 5. **Affichage dans les logs de démarrage**

**Fichier** : `libastrostack/session.py:101-105`

Affiche les paramètres de calibration auto au démarrage :
```
[CONFIG]
  - ISP activé: OUI
    • Calibration auto: histogram_peaks
    • Calibration après: 10 frames
    • Recalibration tous les: 50 frames
```

## 📁 Fichiers créés

### 1. **Documentation complète**
- `CALIBRATION_AUTO_ISP.md` : Guide d'utilisation détaillé (150+ lignes)
  - Principe de fonctionnement
  - Paramètres de configuration
  - Cas d'usage recommandés
  - Comparaison des méthodes
  - Dépannage

### 2. **Script de test et démonstration**
- `test_isp_auto_color_balance.py` : Test complet avec visualisation
  - Test sur images réelles (FITS, PNG)
  - Génération d'histogrammes avant/après
  - Exemple d'intégration dans session
  - Code de démonstration commenté

### 3. **Exemple de configuration**
- `Config_ISP_Auto_Example.txt` : Config complète avec nouveaux paramètres
  - Basé sur PiLCConfig104.txt
  - Commentaires explicatifs
  - Valeurs recommandées

## 🧪 Tests effectués

### ✅ Test 1 : Imports et syntaxe
```bash
✓ Tous les imports fonctionnent correctement
✓ Nouvelles méthodes disponibles
✓ Nouveaux paramètres de configuration
```

### ✅ Test 2 : Calibration sur données synthétiques
```bash
Image test: (100, 100, 3)
Moyennes AVANT: R=0.200, G=0.251, B=0.349  ← Déséquilibre bleu
Gains calculés: R=1.330, G=1.000, B=0.705
Moyennes APRÈS: R=0.266, G=0.251, B=0.246  ← Équilibrés !
✓ Test réussi - les canaux sont maintenant mieux équilibrés!
```

### ✅ Test 3 : Support RAW12 et RAW16
```bash
✓ Configuration RAW12 créée et validée
✓ Configuration RAW16 créée et validée
✓ Les deux formats RAW12 et RAW16 sont bien supportés
✓ L'ISP s'adapte automatiquement au format détecté
✓ La calibration automatique fonctionne pour les deux
```

## 🔧 Compatibilité

### ✅ Formats supportés
- **RAW12** : Pleine prise en charge avec paramètres standards
- **RAW16** : Pleine prise en charge avec adaptations (black level réduit, gamma ajusté)
- **YUV420** : ISP ignoré (déjà traité par ISP hardware)

### ✅ Adaptation automatique RAW16
Le pipeline ISP adapte automatiquement ses paramètres pour RAW16 Clear HDR :
- Black level réduit de 50% (plage dynamique étendue)
- Gamma légèrement réduit (préservation détails HDR)
- Gains RGB appliqués normalement

### ✅ Rétrocompatibilité
- ✅ Tous les paramètres sont optionnels (défaut : désactivé)
- ✅ Le code existant continue de fonctionner sans modification
- ✅ Les configs ISP manuelles restent utilisables
- ✅ Pas de breaking changes

## 📊 Performances

- **Overhead minimal** : Calibration uniquement après N frames (configurable)
- **Non-bloquant** : Gestion des erreurs robuste, continue si échec
- **Optimisé** : Histogrammes calculés sur image stackée (1 fois, pas par frame)

## 🎯 Utilisation recommandée

### Configuration minimale (débutant)
```python
config.isp_enable = True
config.video_format = 'raw12'  # ou 'raw16'
config.isp_auto_calibrate_method = 'histogram_peaks'
config.isp_auto_calibrate_after = 10
# Tout le reste par défaut
```

### Configuration optimale (avancé)
```python
config.isp_enable = True
config.video_format = 'raw12'
config.isp_auto_calibrate_method = 'histogram_peaks'
config.isp_auto_calibrate_after = 15      # Attendre signal suffisant
config.isp_recalibrate_interval = 50      # Sessions longues
```

### Désactivation complète
```python
config.isp_auto_calibrate_method = 'none'
# OU ne rien changer (défaut)
```

## 📈 Exemple de logs pendant l'exécution

```
[IMG] Frame 10
  [QC] OK - FWHM:5.23px, Ell:0.15, Sharp:0.85, Stars:45
  [ALIGN] Alignement...
  [STACK] Empilement...

  [ISP] Calibration automatique initiale (méthode: histogram_peaks)...

=== Calibration par pics d'histogramme ===
  Rouge: pic=0.2150, gain=1.163
  Vert: pic=0.2500, gain=1.000
  Bleu: pic=0.3200, gain=0.781

✓ Luminosité moyenne: 0.245
✓ Gamma: 2.2
✓ Calibration terminée

  [ISP] ✓ Calibration réussie! Gains RGB: R=1.163, G=1.000, B=0.781
  [STATS] Empilées: 10, Rejetées: 2, Échecs: 0

[IMG] Frame 60
  ...
  [ISP] Calibration automatique périodique (méthode: histogram_peaks)...
  [ISP] ✓ Calibration réussie! Gains RGB: R=1.158, G=1.000, B=0.795
```

## 🐛 Robustesse

### Gestion d'erreurs
- ✅ Try/catch autour de la calibration
- ✅ Logs explicites en cas d'erreur
- ✅ Continue l'exécution si échec
- ✅ Détection image trop sombre

### Validation
- ✅ Vérification ISP activé
- ✅ Vérification méthode valide
- ✅ Vérification image stackée disponible
- ✅ Limitation gains entre 0.5 et 2.0

## 📚 Documentation

### Fichiers de documentation
1. **CALIBRATION_AUTO_ISP.md** : Guide complet (10 sections)
   - Vue d'ensemble et principe
   - Utilisation et configuration
   - Paramètres détaillés
   - Cas d'usage recommandés
   - Comparaison des méthodes
   - Notes techniques
   - Dépannage

2. **Config_ISP_Auto_Example.txt** : Exemple de configuration
   - Tous les paramètres commentés
   - Valeurs recommandées
   - Notes d'utilisation

3. **test_isp_auto_color_balance.py** : Script de test/démo
   - Test sur images réelles
   - Visualisation histogrammes
   - Exemple d'intégration

## ✨ Prochaines étapes possibles (optionnel)

### Améliorations futures (non implémentées)
1. **Interface graphique** : Ajout de sliders dans RPiCamera2 GUI
2. **Présets** : Configs pré-définies par type d'objet (DSO, planètes, all-sky)
3. **Historique** : Tracking de l'évolution des gains RGB au fil du temps
4. **Méthode avancée** : Calibration basée sur étoiles neutres (G2V) détectées
5. **Export** : Génération automatique de graphiques d'évolution

### Intégration GUI RPiCamera2
```python
# À ajouter dans RPiCamera2.py si besoin
self.isp_auto_method_combo = QComboBox()
self.isp_auto_method_combo.addItems(['none', 'histogram_peaks', 'gray_world'])
self.isp_calibrate_after_spin = QSpinBox()
self.isp_recalibrate_interval_spin = QSpinBox()
```

## 📝 Résumé final

### ✅ Objectifs atteints
- ✅ Calibration automatique par pics d'histogrammes implémentée
- ✅ Support RAW12 et RAW16
- ✅ Intégration transparente dans le workflow existant
- ✅ Configuration flexible
- ✅ Recalibration périodique optionnelle
- ✅ Documentation complète
- ✅ Tests validés
- ✅ Rétrocompatibilité préservée

### 🎉 Résultat
**L'automatisation de l'alignement des couleurs dans l'ISP est maintenant pleinement opérationnelle !**

Les utilisateurs peuvent :
1. Activer la calibration automatique en 3 lignes de code
2. Laisser le système ajuster les couleurs pendant le stacking
3. Bénéficier de recalibrations périodiques si nécessaire
4. Désactiver facilement si besoin de contrôle manuel

---

**Date d'implémentation** : 2026-01-06
**Fichiers modifiés** : 3 (isp.py, config.py, session.py)
**Fichiers créés** : 4 (documentation, test, config exemple, ce résumé)
**Lignes de code ajoutées** : ~450 lignes (code + doc + tests)
**Tests effectués** : 3 (imports, synthétique, formats)
**Statut** : ✅ Complète et fonctionnelle
