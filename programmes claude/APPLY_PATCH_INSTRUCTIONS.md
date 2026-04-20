# Instructions d'application du patch ROI

## Modifications apportées

Les registres ROI (0x0344-0x034a) ont été **supprimés** de tous les 12 modes ROI du driver IMX585 :

- mode_3584x2016_regs_12bit
- mode_3200x1800_regs_12bit
- mode_3072x2048_regs_12bit
- mode_2560x1440_regs_12bit
- mode_2304x1728_regs_12bit
- mode_2048x2048_regs_12bit
- mode_1920x1080_regs_12bit
- mode_1600x1200_regs_12bit
- mode_1280x720_regs_12bit
- mode_1024x1024_regs_12bit
- mode_800x600_regs_12bit
- mode_640x480_regs_12bit

### Registres supprimés pour chaque mode

```c
/* ROI window configuration */
{ CCI_REG16(0x0344), XXX }, /* X_ADD_STA */
{ CCI_REG16(0x0346), XXX }, /* Y_ADD_STA */
{ CCI_REG16(0x0348), XXX }, /* X_ADD_END */
{ CCI_REG16(0x034a), XXX }, /* Y_ADD_END */
```

## Hypothèse

Le mode 4K (3856x2180) fonctionne correctement SANS programmer ces registres. L'hypothèse est que :

1. Ces registres (copiés du driver IMX477) ne sont pas appropriés pour l'IMX585
2. En les supprimant, le capteur fonctionne en mode full-frame (comme le 4K)
3. Libcamera/ISP effectue le cropping logiciel basé sur le champ `.crop` des modes

## État de la compilation

✅ Fichier source modifié : `/usr/src/imx585-0.0.1/imx585.c`
✅ Module compilé : `/usr/src/imx585-0.0.1/imx585.ko`
✅ Module installé : `/lib/modules/6.12.47+rpt-rpi-2712/kernel/drivers/media/i2c/imx585.ko`
✅ depmod exécuté

## Étapes suivantes

### 1. Redémarrer le système

```bash
sudo reboot
```

### 2. Vérifier le chargement du module

Après le redémarrage :

```bash
dmesg | grep imx585
lsmod | grep imx585
```

### 3. Tester les modes problématiques

Tester avec `rpicam-raw` (pas `rpicam-hello` qui applique des contraintes ISP) :

```bash
# Test 640x480
rpicam-raw --mode 640:480:12:P -t 100 -o test_640x480.raw

# Test 1024x1024
rpicam-raw --mode 1024:1024:12:P -t 100 -o test_1024x1024.raw

# Test 3584x2016
rpicam-raw --mode 3584:2016:12:P -t 100 -o test_3584x2016.raw
```

### 4. Vérifier le pattern Bayer

Les images doivent être visuellement correctes, sans pattern Bayer visible.

## Fichiers créés

- `imx585_modified.c` : Source modifié (dans Rpicamera2/)
- `APPLY_PATCH_INSTRUCTIONS.md` : Ce fichier

## Retour en arrière

Si les modifications ne fonctionnent pas :

```bash
cd /usr/src/imx585-0.0.1
sudo git checkout imx585.c  # Si le répertoire est un repo git
# OU restaurer depuis la sauvegarde
sudo make clean && sudo make
sudo cp imx585.ko /lib/modules/$(uname -r)/kernel/drivers/media/i2c/
sudo depmod -a
sudo reboot
```

## Résultats attendus

Si l'hypothèse est correcte :
- ✅ Les modes 640x480 et 1024x1024 auront un pattern Bayer correct
- ✅ Les images seront visuellement propres sans artefacts
- ⚠️ Le capteur capturera toujours en full-frame, libcamera effectuera le crop logiciel
- ⚠️ Les frame rates ne seront PAS améliorés (car le capteur lit toujours toute la surface)

Si l'hypothèse est incorrecte :
- ❌ Les images pourraient être incorrectes d'une autre manière
- Dans ce cas, il faudra trouver les vrais registres ROI de l'IMX585
