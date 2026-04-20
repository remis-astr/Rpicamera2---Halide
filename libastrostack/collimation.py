"""
Collimation Newton - Detection et classification des cercles optiques.

Detecte les 4 cercles concentriques visibles dans un telescope Newton
vu depuis le porte-oculaire :
  1. Porte-oculaire (focuser) -- le plus grand, cercle de reference
  2. Miroir secondaire -- contour du miroir plan diagonal
  3. Reflet du primaire -- reflet du miroir primaire dans le secondaire
  4. Reflet de la camera -- petit cercle sombre au centre

L'araignee (spider vanes) est ignoree (lignes droites, pas des cercles).

Detection optimisee :
- Downscale automatique pour images > 1080px (performance + gros cercles)
- Detection cercle par cercle avec plages de rayon configurables (% image)
- 1 seul appel HoughCircles par cercle (au lieu de 3 passes globales)
- Skip de frames configurable (detection tous les N frames)
"""

import cv2
import numpy as np
import time


# Couleurs d'affichage pour chaque cercle (R, G, B)
CIRCLE_COLORS = {
    'focuser':  (0, 255, 0),      # Vert
    'secondary': (0, 255, 255),    # Cyan
    'primary':  (255, 255, 0),     # Jaune
    'camera':   (255, 80, 80),     # Rouge
}

CIRCLE_LABELS = {
    'focuser':  'PO (Focuser)',
    'secondary': 'Secondaire',
    'primary':  'Reflet Primaire',
    'camera':   'Reflet Camera',
}

# Ordre de classification (du plus grand au plus petit rayon)
CIRCLE_ORDER = ['focuser', 'secondary', 'primary', 'camera']

# Plages de rayon par defaut (en % de min(h,w) de l'image de detection)
DEFAULT_CIRCLE_CONFIG = {
    'focuser':   {'enabled': False, 'min_pct': 35, 'max_pct': 49},
    'secondary': {'enabled': False, 'min_pct': 15, 'max_pct': 40},
    'primary':   {'enabled': False, 'min_pct': 8,  'max_pct': 25},
    'camera':    {'enabled': False, 'min_pct': 2,  'max_pct': 15},
}


class CollimationDetector:
    """Detecte et classifie les cercles de collimation Newton.

    Detection cercle par cercle optimisee :
    - Downscale automatique a 1080px max pour la detection
    - 1 appel HoughCircles par cercle avec plage de rayon specifique
    - Skip de frames configurable pour reduire la charge CPU
    - Lissage temporel pour stabiliser les resultats
    """

    def __init__(self):
        # Cercles detectes : {'focuser': (cx, cy, r), ...}
        self.circles = {}
        # Centre de reference (centre du focuser)
        self.focuser_center = None
        # Excentricite de chaque cercle par rapport au focuser (en pixels)
        self.eccentricities = {}

        # Sensibilite utilisateur (1-10)
        self._sensitivity = 5
        # Parametre accumulateur de base (ajuste par sensibilite)
        self._base_param2 = 30

        # Configuration par cercle (plages de rayon en % de min(h,w))
        self.circle_config = {}
        for name, cfg in DEFAULT_CIRCLE_CONFIG.items():
            self.circle_config[name] = dict(cfg)

        # Downscale : dimension max pour la detection (540 = rapide sur RPi)
        self.detection_max_dim = 540

        # Skip de frames : detection tous les N frames
        self.detect_interval = 4
        self._frame_count = 0

        # Protection temporelle : temps max de detection avant auto-adaptation
        self._last_detect_time = 0.0  # duree de la derniere detection (sec)
        self._detect_time_limit = 0.5  # si > 500ms, augmenter l'intervalle

        # Verrouillage (lock) : position figee en coordonnees image originale
        self._locked = {}  # {'focuser': (cx, cy, r), ...} ou absent si pas locke

        # Lissage temporel : moyenne glissante sur N frames
        self._history = {}  # par cercle : {'focuser': [(cx,cy,r), ...], ...}
        self._history_max = 5

    def set_sensitivity(self, value):
        """Ajuste la sensibilite (1=peu sensible, 10=tres sensible).
        Mappe sur param2 de HoughCircles : 1->50, 10->10."""
        value = max(1, min(10, value))
        self._sensitivity = value
        self._base_param2 = int(50 - (value - 1) * (40 / 9))

    def set_circle_enabled(self, name, enabled):
        """Active/desactive la detection d'un cercle."""
        if name in self.circle_config:
            self.circle_config[name]['enabled'] = bool(enabled)
            if not enabled:
                if name in self.circles:
                    del self.circles[name]
                if name in self._history:
                    del self._history[name]
                # Deverrouiller aussi si on desactive
                if name in self._locked:
                    del self._locked[name]

    def set_circle_locked(self, name, locked):
        """Verrouille/deverrouille un cercle a sa position actuelle.

        Quand un cercle est verrouille :
        - Sa position est figee (plus de HoughCircles pour lui)
        - Il sert de reference pour contraindre les cercles suivants
        - Le focuser verrouille resserre la zone de recherche des autres

        Args:
            name: nom du cercle ('focuser', 'secondary', 'primary', 'camera')
            locked: True pour verrouiller, False pour deverrouiller
        """
        if name not in self.circle_config:
            return
        if locked:
            if name in self.circles:
                self._locked[name] = self.circles[name]
                # Vider l'historique de lissage (position figee)
                if name in self._history:
                    del self._history[name]
        else:
            if name in self._locked:
                del self._locked[name]

    def is_circle_locked(self, name):
        """Retourne True si le cercle est verrouille."""
        return name in self._locked

    def get_locked_circles(self):
        """Retourne le dict des cercles verrouilles {name: (cx,cy,r)}."""
        return dict(self._locked)

    def set_circle_position(self, name, cx, cy, r):
        """Positionne un cercle manuellement (en coordonnees image originale).

        Contrairement a set_circle_locked() qui copie la position detectee,
        set_circle_position() permet :
        - Toute position, y compris partiellement hors image (utile pour le focuser)
        - Un rayon tres petit (min 5px pour camera, 10px pour les autres cercles)

        La position est stockee dans _locked (bypass la detection Hough).

        Args:
            name: nom du cercle ('focuser', 'secondary', 'primary', 'camera')
            cx, cy: centre en pixels de l'image originale (peut etre negatif ou > dim)
            r: rayon en pixels de l'image originale
        """
        if name not in self.circle_config:
            return
        min_r = 5 if name == 'camera' else 10
        r = max(min_r, int(r))
        self._locked[name] = (int(cx), int(cy), r)
        self.circles[name] = self._locked[name]
        # Vider l'historique pour eviter interference avec la position manuelle
        if name in self._history:
            del self._history[name]
        # Recalculer focuser_center si le focuser est repositionne
        if name == 'focuser':
            self.focuser_center = (int(cx), int(cy))
        # Recalculer les excentricites
        if self.focuser_center is not None:
            for n in ['secondary', 'primary', 'camera']:
                if n in self.circles:
                    ox, oy, _ = self.circles[n]
                    fcx, fcy = self.focuser_center
                    dist = np.sqrt((ox - fcx) ** 2 + (oy - fcy) ** 2)
                    self.eccentricities[n] = round(dist, 1)

    def set_circle_radius_range(self, name, min_pct, max_pct):
        """Change la plage de rayon d'un cercle (en % de min(h,w))."""
        if name in self.circle_config:
            self.circle_config[name]['min_pct'] = max(1, min(49, int(min_pct)))
            self.circle_config[name]['max_pct'] = max(2, min(50, int(max_pct)))
            # Garantir min < max
            if self.circle_config[name]['min_pct'] >= self.circle_config[name]['max_pct']:
                self.circle_config[name]['min_pct'] = self.circle_config[name]['max_pct'] - 1

    def set_detect_interval(self, n):
        """Change la frequence de detection (1=chaque frame, 10=tous les 10 frames)."""
        self.detect_interval = max(1, min(10, int(n)))

    def get_circle_config(self):
        """Retourne la configuration actuelle des cercles."""
        return {name: dict(cfg) for name, cfg in self.circle_config.items()}

    def detect(self, frame_rgb):
        """Detecte et classifie les cercles dans une frame RGB.

        Detection cercle par cercle avec downscale automatique :
        - L'image est reduite a max 1080px pour la detection
        - Chaque cercle active est detecte independamment avec sa plage de rayon
        - Les coordonnees sont remises a l'echelle de l'image originale
        - Un skip de frames reduit la charge CPU

        Args:
            frame_rgb: numpy array (H, W, 3) uint8 en RGB

        Returns:
            dict avec les cercles detectes {'focuser': (cx,cy,r), ...}
        """
        if frame_rgb is None or len(frame_rgb.shape) < 2:
            return self.circles

        # --- Skip de frames : retourner le cache ---
        self._frame_count += 1
        if self._frame_count % self.detect_interval != 1 and self._frame_count > 1:
            return self.circles

        # Mesurer le temps de detection
        _t_start = time.monotonic()

        # Reset pour cette detection
        self.circles = {}
        self.focuser_center = None
        self.eccentricities = {}

        # Convertir en niveaux de gris
        if len(frame_rgb.shape) == 3:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_rgb.copy()

        orig_h, orig_w = gray.shape[:2]

        # --- Downscale automatique (basé sur max(h,w) pour couvrir 1920x1080) ---
        scale_factor = 1.0
        if max(orig_h, orig_w) > self.detection_max_dim:
            scale_factor = self.detection_max_dim / max(orig_h, orig_w)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = gray.shape[:2]
        dim = min(h, w)

        # --- Pretraitement unique ---
        blur_size = max(3, int(5 * dim / 1080))
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        tile = max(4, min(16, int(8 * dim / 1080)))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile, tile))
        enhanced = clahe.apply(blurred)

        # Parametres communs adaptes a la taille
        scale = dim / 1080
        param1 = max(50, int(100 * max(scale, 0.5)))
        param2 = max(8, int(self._base_param2 * max(scale, 0.5)))
        min_dist = max(15, int(30 * scale))

        # --- Detection cercle par cercle ---
        # Injecter les cercles verrouilles comme references (en coords downscale)
        detected_circles = {}
        for name in CIRCLE_ORDER:
            if name in self._locked:
                lcx, lcy, lr = self._locked[name]
                detected_circles[name] = (
                    int(lcx * scale_factor),
                    int(lcy * scale_factor),
                    int(lr * scale_factor)
                )

        for name in CIRCLE_ORDER:
            cfg = self.circle_config[name]
            if not cfg['enabled']:
                continue

            # Cercle verrouille : pas de detection, on garde la position figee
            if name in self._locked:
                continue

            min_r = max(5, int(dim * cfg['min_pct'] / 100))
            max_r = max(min_r + 5, int(dim * cfg['max_pct'] / 100))

            # Adapter min_dist pour eviter de rater un cercle proche d'un autre
            local_min_dist = max(10, min_r // 2)

            try:
                circles_found = cv2.HoughCircles(
                    enhanced,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=local_min_dist,
                    param1=param1,
                    param2=param2,
                    minRadius=min_r,
                    maxRadius=max_r
                )
            except cv2.error:
                continue

            if circles_found is None:
                continue

            candidates = np.round(circles_found[0]).astype(int).tolist()

            # Filtrer : centre dans l'image (marge 5%)
            margin_x = w * 0.05
            margin_y = h * 0.05
            valid = []
            for c in candidates:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                if margin_x < cx < w - margin_x and margin_y < cy < h - margin_y:
                    valid.append((cx, cy, r))

            if not valid:
                continue

            # Choisir le meilleur candidat
            best = self._select_best_candidate(name, valid, detected_circles, dim)
            if best is not None:
                detected_circles[name] = best

        # --- Remettre a l'echelle originale (sauf cercles verrouilles) ---
        if scale_factor != 1.0:
            inv_scale = 1.0 / scale_factor
            for name in detected_circles:
                if name in self._locked:
                    # Cercle verrouille : remettre les coords originales directement
                    detected_circles[name] = self._locked[name]
                else:
                    cx, cy, r = detected_circles[name]
                    detected_circles[name] = (
                        int(cx * inv_scale),
                        int(cy * inv_scale),
                        int(r * inv_scale)
                    )
        else:
            # Pas de downscale : remettre les coords verrouillees originales
            for name in detected_circles:
                if name in self._locked:
                    detected_circles[name] = self._locked[name]

        # --- Lissage temporel par cercle (sauf verrouilles) ---
        for name in CIRCLE_ORDER:
            if name in self._locked:
                # Verrouille : position figee, pas de lissage
                self.circles[name] = self._locked[name]
                continue

            if name in detected_circles:
                if name not in self._history:
                    self._history[name] = []
                self._history[name].append(detected_circles[name])
                if len(self._history[name]) > self._history_max:
                    self._history[name].pop(0)

                # Moyenne glissante
                hist = self._history[name]
                avg_cx = int(np.mean([c[0] for c in hist]))
                avg_cy = int(np.mean([c[1] for c in hist]))
                avg_r = int(np.mean([c[2] for c in hist]))
                self.circles[name] = (avg_cx, avg_cy, avg_r)
            else:
                # Pas detecte cette frame : vider l'historique progressivement
                if name in self._history and len(self._history[name]) > 0:
                    self._history[name].pop(0)
                    if self._history[name]:
                        hist = self._history[name]
                        avg_cx = int(np.mean([c[0] for c in hist]))
                        avg_cy = int(np.mean([c[1] for c in hist]))
                        avg_r = int(np.mean([c[2] for c in hist]))
                        self.circles[name] = (avg_cx, avg_cy, avg_r)

        # --- Centre de reference et excentricites ---
        if 'focuser' in self.circles:
            cx, cy, _ = self.circles['focuser']
            self.focuser_center = (cx, cy)

            for name in ['secondary', 'primary', 'camera']:
                if name in self.circles:
                    ox, oy, _ = self.circles[name]
                    dist = np.sqrt((ox - cx) ** 2 + (oy - cy) ** 2)
                    self.eccentricities[name] = round(dist, 1)

        # --- Mesure et protection temporelle ---
        self._last_detect_time = time.monotonic() - _t_start
        if self._last_detect_time > self._detect_time_limit:
            # Detection trop lente : augmenter l'intervalle automatiquement
            old_interval = self.detect_interval
            self.detect_interval = min(10, self.detect_interval + 1)
            if self.detect_interval != old_interval:
                print(f"[COLIM] Detection lente ({self._last_detect_time:.2f}s) "
                      f"-> interval {old_interval} -> {self.detect_interval}")

        return self.circles

    def _select_best_candidate(self, name, candidates, already_detected, dim):
        """Selectionne le meilleur cercle parmi les candidats.

        Pour le focuser : le plus grand cercle.
        Pour les autres : le plus concentrique par rapport au focuser,
        en evitant les doublons avec les cercles deja detectes.
        """
        if not candidates:
            return None

        if name == 'focuser':
            # Le plus grand cercle
            return max(candidates, key=lambda c: c[2])

        # Pour les autres cercles, il faut un focuser de reference
        if 'focuser' not in already_detected:
            # Pas de focuser : prendre le plus concentrique par rapport au centre image
            ref_cx, ref_cy = dim // 2, dim // 2
        else:
            ref_cx, ref_cy, _ = already_detected['focuser']

        # Eliminer les candidats trop proches d'un cercle deja detecte
        filtered = []
        for c in candidates:
            too_close = False
            for det_name, det_circle in already_detected.items():
                r_diff = abs(c[2] - det_circle[2]) / max(c[2], det_circle[2], 1)
                center_dist = np.sqrt((c[0] - det_circle[0])**2 + (c[1] - det_circle[1])**2)
                if r_diff < 0.15 and center_dist < dim * 0.05:
                    too_close = True
                    break
            if not too_close:
                filtered.append(c)

        if not filtered:
            return None

        # Choisir le plus concentrique (distance min au centre de reference)
        best = min(filtered, key=lambda c: np.sqrt((c[0] - ref_cx)**2 + (c[1] - ref_cy)**2))

        # Verifier la concentricite
        # Tolerance resserree si le focuser est verrouille (position fiable)
        if 'focuser' in already_detected:
            focuser_r = already_detected['focuser'][2]
            dist = np.sqrt((best[0] - ref_cx)**2 + (best[1] - ref_cy)**2)
            tolerance = 0.25 if 'focuser' in self._locked else 0.40
            if dist > focuser_r * tolerance:
                return None

        return best

    def get_collimation_score(self):
        """Retourne un score de collimation (0-100).
        100 = parfaitement collimate.

        Chaque indicateur est normalise par le rayon de son cercle PARENT :
          - Secondaire  : ecc(secondaire vs focuser)  / focuser_r
          - Primaire    : ecc(primaire vs secondaire)  / secondary_r
          - Camera      : ecc(camera vs secondaire)    / secondary_r

        Seuil : 10% du rayon parent -> score 0.
        Score global = minimum des indicateurs actifs (maillon faible).
        """
        if not self.focuser_center or 'focuser' not in self.circles:
            return None

        focuser_r = self.circles['focuser'][2]
        if focuser_r <= 0:
            return None

        fcx, fcy = self.focuser_center
        TOLERANCE = 0.10  # 10% du rayon parent = score 0

        scores = []

        # --- Secondaire : doit etre centre dans le focuser ---
        if 'secondary' in self.circles:
            scx, scy, secondary_r = self.circles['secondary']
            ecc = np.sqrt((scx - fcx) ** 2 + (scy - fcy) ** 2)
            ratio = ecc / (focuser_r * TOLERANCE)
            scores.append(max(0, int(100 - ratio * 100)))

        # --- Primaire et camera : doivent etre centres dans le secondaire ---
        if 'secondary' in self.circles:
            scx, scy, secondary_r = self.circles['secondary']
            if secondary_r > 0:
                for name in ('primary', 'camera'):
                    if name in self.circles:
                        ox, oy, _ = self.circles[name]
                        ecc = np.sqrt((ox - scx) ** 2 + (oy - scy) ** 2)
                        ratio = ecc / (secondary_r * TOLERANCE)
                        scores.append(max(0, int(100 - ratio * 100)))

        if not scores:
            return None

        # Score = maillon faible (indicateur le plus degrade)
        return min(scores)
