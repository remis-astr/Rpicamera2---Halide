#!/bin/bash
# Script de lancement automatique de RPiCamera2
# Gère l'environnement et attend que X11 soit prêt

# Attendre que le système graphique soit prêt (max 60 secondes)
echo "[RPiCamera2] Attente du système graphique..."
for i in {1..60}; do
    if [ -n "$DISPLAY" ] || [ -S /tmp/.X11-unix/X0 ]; then
        echo "[RPiCamera2] Système graphique détecté"
        break
    fi
    sleep 1
done

# Définir DISPLAY si nécessaire
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "[RPiCamera2] DISPLAY défini sur :0"
fi

# Aller dans le répertoire du script
cd "/home/admin/Rpicamera tests/Rpicamera2" || {
    echo "[RPiCamera2] ERREUR: Impossible d'accéder au répertoire"
    exit 1
}

# Attendre quelques secondes supplémentaires pour que tout soit stable
sleep 5

# Logger le démarrage
echo "[RPiCamera2] Démarrage de RPiCamera2.py à $(date)"

# Lancer RPiCamera2 avec logging
python3 RPiCamera2.py >> /home/admin/rpicamera2.log 2>&1

# Logger l'arrêt
echo "[RPiCamera2] Arrêt de RPiCamera2.py à $(date)" >> /home/admin/rpicamera2.log
