# Guide de démarrage automatique RPiCamera2

RPiCamera2 est maintenant configuré pour démarrer automatiquement au démarrage du Raspberry Pi 5.

## ✅ Configuration installée

Deux méthodes de démarrage automatique ont été configurées :

### Méthode 1 : Autostart Desktop (RECOMMANDÉE pour GUI)
- **Fichier** : `~/.config/autostart/rpicamera2.desktop`
- **Se lance** : Quand vous vous connectez à l'interface graphique
- **Avantage** : Simple, adapté aux applications GUI

### Méthode 2 : Service Systemd User
- **Fichier** : `~/.config/systemd/user/rpicamera2.service`
- **Se lance** : Avec la session utilisateur
- **Avantage** : Plus de contrôle, redémarrage automatique en cas d'erreur

## 📋 Commandes utiles

### Gérer l'autostart desktop

```bash
# Désactiver le démarrage automatique
rm ~/.config/autostart/rpicamera2.desktop

# Réactiver le démarrage automatique
cp /home/admin/Rpicamera\ tests/Rpicamera2/AUTOSTART_GUIDE.md ~/.config/autostart/rpicamera2.desktop
```

### Gérer le service systemd

```bash
# Voir le statut du service
systemctl --user status rpicamera2

# Démarrer le service maintenant
systemctl --user start rpicamera2

# Arrêter le service
systemctl --user stop rpicamera2

# Redémarrer le service
systemctl --user restart rpicamera2

# Désactiver le démarrage automatique
systemctl --user disable rpicamera2

# Réactiver le démarrage automatique
systemctl --user enable rpicamera2

# Voir les logs du service
journalctl --user -u rpicamera2 -f
```

## 📝 Logs

Les logs de l'application sont sauvegardés dans :
- **Fichier principal** : `/home/admin/rpicamera2.log`
- **Journal systemd** : `journalctl --user -u rpicamera2`

```bash
# Voir les derniers logs
tail -f ~/rpicamera2.log

# Effacer les anciens logs
> ~/rpicamera2.log
```

## ⚙️ Fichiers créés

1. **Script de lancement** : `/home/admin/Rpicamera tests/Rpicamera2/start_rpicamera2.sh`
   - Gère l'attente du système graphique
   - Configure l'environnement
   - Lance RPiCamera2.py avec logging

2. **Service systemd** : `~/.config/systemd/user/rpicamera2.service`
   - Définit le service systemd
   - Configure le redémarrage automatique

3. **Autostart desktop** : `~/.config/autostart/rpicamera2.desktop`
   - Fichier de démarrage automatique pour l'environnement de bureau

## 🔧 Dépannage

### L'application ne démarre pas au boot

1. Vérifier que le système graphique est actif :
   ```bash
   echo $DISPLAY
   # Devrait afficher :0 ou :1
   ```

2. Vérifier les permissions du script :
   ```bash
   ls -l "/home/admin/Rpicamera tests/Rpicamera2/start_rpicamera2.sh"
   # Devrait avoir les permissions -rwxr-xr-x
   ```

3. Tester le script manuellement :
   ```bash
   "/home/admin/Rpicamera tests/Rpicamera2/start_rpicamera2.sh"
   ```

4. Consulter les logs :
   ```bash
   cat ~/rpicamera2.log
   journalctl --user -u rpicamera2
   ```

### L'application démarre mais plante

Les logs dans `~/rpicamera2.log` contiendront les erreurs Python.

### Désactiver temporairement le démarrage automatique

```bash
# Méthode 1 (Autostart)
mv ~/.config/autostart/rpicamera2.desktop ~/.config/autostart/rpicamera2.desktop.disabled

# Méthode 2 (Systemd)
systemctl --user stop rpicamera2
systemctl --user disable rpicamera2
```

## 🚀 Test immédiat (sans reboot)

Pour tester le démarrage automatique sans redémarrer :

```bash
# Tester le script directement
"/home/admin/Rpicamera tests/Rpicamera2/start_rpicamera2.sh"

# OU tester via systemd
systemctl --user start rpicamera2
systemctl --user status rpicamera2
```

## ℹ️ Recommandation

**Utilisez l'autostart desktop** (Méthode 1) car :
- Plus simple pour les applications GUI
- Se lance automatiquement quand vous vous connectez
- Pas besoin de commandes systemd

Si vous préférez plus de contrôle et le redémarrage automatique en cas d'erreur, gardez aussi le service systemd activé.
