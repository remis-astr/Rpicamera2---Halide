# Guide d'activation ISP pour RPiCamera2

## ✅ Configuration complète !

Tous les paramètres sont maintenant en place et prêts à être utilisés.

## 🎯 Pour activer l'ISP

### Méthode simple : Éditer PiLCConfig104.txt

1. Ouvrez le fichier `PiLCConfig104.txt`
2. Trouvez la ligne `isp_enable : 0` (tout en bas, ligne 84)
3. Changez `0` en `1` :

```
isp_enable : 1
```

4. Sauvegardez et relancez RPiCamera2

**C'est tout !** 🎉

## 📋 Ce qui se passe automatiquement

Quand `isp_enable : 1` :

1. **Détection du format RAW** : SRGGB12 ou SRGGB16 détecté automatiquement
2. **Configuration ISP** : Le fichier `isp_config_imx585.json` est chargé
3. **Pipeline activé** :
   ```
   RAW12 → Debayer → Stack → ISP → Stretch → PNG 16-bit
                                            → FITS linéaire
   ```
4. **ISP appliqué** avec :
   - Black level: 3200
   - Color Correction Matrix (IMX585 calibrée)
   - White balance: R=0.568, G=1.0, B=0.574
   - Gamma: 3.0
   - Contraste: 1.1
   - Saturation: 1.2

## 🔍 Vérification dans les logs

Au démarrage du livestack, vous devriez voir :

```
[CONFIG] Format vidéo configuré: raw12 (source: SRGGB12)

============================================================
>>> LIBASTROSTACK SESSION
============================================================

[CONFIG]
  - Mode alignement: none
  - Contrôle qualité: NON
  - ISP activé: OUI  ← Doit être OUI
  - Format vidéo: raw12  ← Doit être raw12
  - PNG bit depth: Auto
  - Étirement PNG: off
  - FITS: Linéaire (RAW)
```

Et lors de la sauvegarde :

```
[DEBUG get_preview_png] Début
  • ISP activé: True  ← Doit être True
  • ISP instance: True  ← Doit être True
  • Format vidéo: raw12
  • PNG bit depth config: None
  → Application ISP...
  • Image après ISP: dtype=float32
  → Sélection bit depth:
     config.video_format = raw12
     → 16-bit (auto: 'raw' détecté dans format)
  ✓ Preview final: dtype=uint16, shape=(2180, 3872, 3)
[OK] PNG: /media/.../stack_SRGGB12_mean_...png
       Bit depth: 16-bit, Taille: 12000+ KB  ← ~12-16 MB
```

## 🎨 Configurations ISP disponibles

### Par défaut : Lumière du jour (5500K)
Fichier utilisé : `isp_config_imx585.json`
- Déjà créé et configuré
- Optimal pour observation diurne/crépuscule

### Créer d'autres températures de couleur

Si vous voulez tester d'autres températures :

```bash
# Tungstène (3000K) - lumière incandescente chaude
python3 convert_libcamera_isp.py "/home/admin/Rpicamera tests/imx585_lowlight.json" \
    -o isp_tungsten.json -t 3000

# Fluorescent (4000K)
python3 convert_libcamera_isp.py "/home/admin/Rpicamera tests/imx585_lowlight.json" \
    -o isp_fluorescent.json -t 4000

# Nuageux (6500K)
python3 convert_libcamera_isp.py "/home/admin/Rpicamera tests/imx585_lowlight.json" \
    -o isp_cloudy.json -t 6500
```

Puis dans `RPiCamera2.py` ligne 1422, changez :
```python
isp_config_path = "isp_tungsten.json"  # Au lieu de isp_config_imx585.json
```

## 🔧 Paramètres de configuration

### Dans PiLCConfig104.txt (ligne 84)
```
isp_enable : 0    # 0 = désactivé, 1 = activé
```

### Dans RPiCamera2.py (ligne 1422)
```python
isp_config_path = "isp_config_imx585.json"  # Fichier de config ISP
```

### Dans isp_config_imx585.json (ajustable manuellement)
```json
{
  "wb_red_gain": 0.568,      # Balance des blancs rouge
  "wb_green_gain": 1.0,      # Balance des blancs vert (référence)
  "wb_blue_gain": 0.574,     # Balance des blancs bleu
  "black_level": 3200,       # Niveau de noir du capteur
  "gamma": 3.0,              # Courbe gamma
  "contrast": 1.1,           # Contraste (1.0 = neutre)
  "saturation": 1.2,         # Saturation (1.0 = neutre)
  "ccm": [...]               # Color Correction Matrix 3x3
}
```

Vous pouvez modifier manuellement ces valeurs pour affiner le rendu.

## 📊 Comparaison avant/après

### Sans ISP (isp_enable : 0)
- Pipeline : RAW12 → Debayer → Stack → Stretch → PNG
- Résultat : Image débayérisée simple, couleurs RAW
- PNG : 16-bit (~12-16 MB)
- FITS : Linéaire (vrai RAW)

### Avec ISP (isp_enable : 1)
- Pipeline : RAW12 → Debayer → Stack → **ISP** → Stretch → PNG
- Résultat : Image avec corrections colorimétriques calibrées
- PNG : 16-bit (~12-16 MB)
- FITS : Linéaire (vrai RAW, AVANT ISP)
- Bonus : Balance des blancs, CCM, gamma optimisés pour IMX585

## ❓ FAQ

**Q: Le PNG sera-t-il toujours en 16-bit ?**
R: Oui, tant que vous êtes en mode RAW12 ou RAW16. En mode YUV420, le PNG sera en 8-bit.

**Q: L'ISP ralentit-il le traitement ?**
R: Non, l'ISP est appliqué 1x après le stack (pas sur chaque frame), donc impact minimal.

**Q: Puis-je désactiver temporairement l'ISP ?**
R: Oui, changez `isp_enable : 1` → `isp_enable : 0` et relancez le programme.

**Q: Le FITS est-il affecté par l'ISP ?**
R: Non ! Le FITS est toujours sauvegardé en **linéaire (avant ISP)** pour garder les vraies données RAW.

**Q: Puis-je ajuster les paramètres ISP en temps réel ?**
R: Pas directement, mais vous pouvez modifier `isp_config_imx585.json` et relancer le livestack.

## 🚀 Prochaines étapes

1. **Testez sans ISP d'abord** (isp_enable : 0)
   - Vérifiez que le PNG est bien en 16-bit
   - Confirmez que le format est détecté (raw12/raw16)

2. **Activez l'ISP** (isp_enable : 1)
   - Comparez les images avant/après
   - Vérifiez les logs pour confirmer l'application de l'ISP

3. **Ajustez si nécessaire**
   - Testez différentes températures de couleur
   - Modifiez contraste/saturation dans le JSON

---

**Tout est prêt ! Activez l'ISP et profitez des corrections colorimétriques calibrées pour votre IMX585 ! 🎯**
