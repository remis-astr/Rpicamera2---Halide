"""
galaxy_filter.py — Filtre d'enhancement galactique multi-échelle
Deux étapes complémentaires :

  1. STAR REDUCTION  : ouverture morphologique (supprime les sources ponctuelles)
  2. ENHANCEMENT     : deux couches combinables
       a) Frangi structurel modifié → carte de réponse des structures galactiques
          → combinaison gamma (halos sombres) + contraste local guidé (détails)
       b) USM multi-échelle (Unsharp Masking) → accentue les détails internes
          (bras spiraux, gradient bulbe, stries de poussière)

Frangi structurel modifié vs version originale :
  - ksize Sobel proportionnel à sigma → dérivées 2ndes précises aux grandes échelles
  - Composante LoG (Laplacien Gaussien) en plus du Frangi vessel :
      · Frangi vessel : structures allongées (bras spiraux, galaxie tranche)
      · LoG            : structures circulaires/ovales (halos, bulbe)
  - Normalisation robuste (percentile 95) → résistant aux étoiles/noyaux brillants
  - Accumulation pondérée par échelle (somme, pas max) → préserve l'info multi-échelle
  - Enhancement combiné : gamma (amplification halos sombres) +
                          contraste structurel local (révèle les détails fins)

Galaxy types et plages sigma par défaut :
    0 = Elliptique   : sigma 15–80 px  (larges structures, halo)  → 20% vessel / 80% LoG
    1 = Spirale      : sigma  5–30 px  (bras spiraux + bulbe)     → 65% vessel / 35% LoG
    2 = Par la tranche: sigma  3–20 px (structures fines, linéaires)→ 85% vessel / 15% LoG
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
GALAXY_PRESETS = [
    {
        'name':           'Elliptique',
        'galaxy_type':    0,
        'star_reduction': 0.70,
        'star_kernel':    9,
        'sigma_min':      15.0,
        'sigma_max':      80.0,
        'enhancement':    1.5,
        'usm_strength':   0.5,
        'n_scales':       4,
    },
    {
        'name':           'Spirale',
        'galaxy_type':    1,
        'star_reduction': 0.65,
        'star_kernel':    7,
        'sigma_min':      5.0,
        'sigma_max':      30.0,
        'enhancement':    2.0,
        'usm_strength':   1.0,
        'n_scales':       4,
    },
    {
        'name':           'Tranche',
        'galaxy_type':    2,
        'star_reduction': 0.80,
        'star_kernel':    5,
        'sigma_min':      3.0,
        'sigma_max':      20.0,
        'enhancement':    2.5,
        'usm_strength':   1.5,
        'n_scales':       4,
    },
]


class GalaxyEnhancer:
    """
    Filtre d'enhancement galactique.

    Utilisation minimale :
        ge = GalaxyEnhancer()
        ge.enabled = True
        ge.apply_preset(1)        # Spirale
        result_bgr = ge.process(img_bgr)

    Paramètres exposés (sliders UI) :
        galaxy_type    int    0=Elliptique, 1=Spirale, 2=Par la tranche
        star_reduction float  0=aucune réduction, 1=maximale (blend morpho open)
        star_kernel    int    Rayon kernel morpho (px, 3-15) — taille max des étoiles
        sigma_min      float  Echelle Frangi/USM minimum (px)
        sigma_max      float  Echelle Frangi/USM maximum (px)
        enhancement    float  Boost Frangi structurel (0=off, 0.5-3.0)
        usm_strength   float  Boost USM multi-échelle pour détails internes (0=off, 0.5-3.0)
        n_scales       int    Nombre d'échelles (2-6, compromis qualité/vitesse)
    """

    def __init__(self):
        self.enabled        = False
        self.galaxy_type    = 1
        self.star_reduction = 0.65
        self.star_kernel    = 7
        self.sigma_min      = 5.0
        self.sigma_max      = 30.0
        self.enhancement    = 2.0
        self.usm_strength   = 1.0
        self.n_scales       = 4
        self._morph_kernel  = None   # cache kernel morpho

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, **kwargs):
        """Met à jour un ou plusieurs paramètres."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._morph_kernel = None

    def apply_preset(self, idx: int):
        """Applique un preset prédéfini (0-2)."""
        if 0 <= idx < len(GALAXY_PRESETS):
            p = GALAXY_PRESETS[idx]
            for k, v in p.items():
                if k != 'name' and hasattr(self, k):
                    setattr(self, k, v)
            self._morph_kernel = None

    # ------------------------------------------------------------------
    # Etape 1 : Star reduction
    # ------------------------------------------------------------------

    def _get_morph_kernel(self):
        if self._morph_kernel is None:
            k = max(3, int(self.star_kernel))
            if k % 2 == 0:
                k += 1
            self._morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k, k))
        return self._morph_kernel

    def _star_reduce(self, L_f32: np.ndarray) -> np.ndarray:
        """
        Ouverture morphologique : supprime les sources < star_kernel px.
        La galaxie (bien plus grande que le kernel) est quasi intacte.
        Blend linéaire contrôlé par star_reduction (0=aucun effet, 1=maximal).
        """
        if self.star_reduction <= 0.0:
            return L_f32
        kernel   = self._get_morph_kernel()
        L_opened = cv2.morphologyEx(L_f32, cv2.MORPH_OPEN, kernel)
        alpha    = float(np.clip(self.star_reduction, 0.0, 1.0))
        return L_f32 * (1.0 - alpha) + L_opened * alpha

    # ------------------------------------------------------------------
    # Etape 2a : Frangi structurel modifié → carte de réponse
    # ------------------------------------------------------------------

    def _hessian_eigenvalues(self, L_f32: np.ndarray, sigma: float):
        """
        Hessienne scale-normalisée avec ksize Sobel proportionnel à sigma.

        Amélioration vs version originale :
          ksize_d = max(3, min(int(2*sigma+1)|1, 7))
          Au lieu de toujours 3 : donne des dérivées 2ndes bien plus précises
          aux grandes échelles (sigma > 3 px), notamment pour les halos elliptiques
          et les bras larges.

        λ1 ≤ λ2 (triées par valeur absolue croissante).
        Structure brillante allongée : λ1 ≈ 0, λ2 << 0.
        """
        ksize   = max(3, int(6.0 * sigma + 1.0) | 1)
        blurred = cv2.GaussianBlur(L_f32, (ksize, ksize), sigma)
        s2      = sigma * sigma
        # ksize_d proportionnel à sigma : 3 (σ≤1) → 5 (σ≈2) → 7 (σ≥3)
        # Limité à 7 car Sobel ksize > 7 est instable en CV_32F
        ksize_d = max(3, min(int(2.0 * sigma + 1.0) | 1, 7))
        Lxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=ksize_d) * s2
        Lyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=ksize_d) * s2
        Lxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=ksize_d) * s2
        disc = np.sqrt(np.maximum((Lxx - Lyy) ** 2 + 4.0 * Lxy * Lxy, 0.0))
        lam1 = ((Lxx + Lyy) - disc) * 0.5
        lam2 = ((Lxx + Lyy) + disc) * 0.5
        return lam1, lam2

    def _vessel_response_at_scale(self, lam1: np.ndarray, lam2: np.ndarray,
                                   beta: float = 0.5) -> np.ndarray:
        """
        Frangi vessel pour structures brillantes allongées.

        Amélioration vs version originale :
          c basé sur le percentile 95 de S (au lieu du max) :
          → résistant aux étoiles résiduelles et au noyau galactique brillant
          qui biaisaient c vers le haut et écrasaient la réponse sur le reste.

        beta proche de 0 : structures très allongées (tranche, bras fins)
        beta proche de 1 : répond aussi aux blobs (elliptiques)
        """
        mask = lam2 < 0
        Rb   = np.where(mask, lam1 / (lam2 - 1e-8), 1.0)
        S    = np.sqrt(lam1 ** 2 + lam2 ** 2)
        # Normalisation robuste : percentile 95 résiste aux pics brillants
        s_pos = S[S > 0]
        Sref  = float(np.percentile(s_pos, 95)) if s_pos.size > 0 else float(S.max())
        c     = 0.5 * Sref if Sref > 1e-10 else 1.0
        resp  = np.where(
            mask,
            np.exp(-Rb ** 2 / (2.0 * beta ** 2)) *
            (1.0 - np.exp(-S ** 2 / (2.0 * c ** 2))),
            0.0
        )
        return resp.astype(np.float32)

    def _log_response_at_scale(self, L_f32: np.ndarray, sigma: float) -> np.ndarray:
        """
        Réponse LoG (Laplacien de Gaussienne) scale-normalisée.

        Complémentaire au Frangi vessel : détecte les structures brillantes
        circulaires ou ovales que le filtre Frangi (conçu pour les tubes) manque :
          - Halos diffus des galaxies elliptiques
          - Bulbe central des spirales
          - Enveloppes ovales des lenticulaires

        LoG scale-normalisé < 0 pour structures brillantes sur fond sombre.
        On retient la partie négative (structures brillantes).

        Normalisation robuste : percentile 95 (résistant au noyau très brillant).
        """
        ksize   = max(3, int(6.0 * sigma + 1.0) | 1)
        blurred = cv2.GaussianBlur(L_f32, (ksize, ksize), sigma)
        # LoG scale-normalisé : σ² × ∇²G  (invariant d'échelle)
        lap  = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3) * (sigma ** 2)
        # Structures brillantes → LoG < 0 → prendre la partie négative
        resp = np.maximum(-lap, 0.0)
        # Normalisation robuste
        r_pos = resp[resp > 0]
        rref  = float(np.percentile(r_pos, 95)) if r_pos.size > 0 else float(resp.max())
        if rref > 1e-10:
            resp = np.minimum(resp / rref, 1.0)
        return resp.astype(np.float32)

    def _frangi_structural(self, L_f32: np.ndarray) -> np.ndarray:
        """
        Réponse structurelle galactique multi-échelle (Frangi modifié).

        Combine :
          - Frangi vessel (bras spiraux, disque, structures allongées)
          - LoG blob (halos diffus, bulbe, enveloppes elliptiques)

        Pondération vessel / LoG selon galaxy_type :
          0 Elliptique : 20% vessel + 80% LoG  → halo/enveloppe dominante
          1 Spirale    : 65% vessel + 35% LoG  → bras + bulbe
          2 Tranche    : 85% vessel + 15% LoG  → disque linéaire fin

        Accumulation pondérée par sigma (les grandes échelles ont plus de poids :
        les halos et bras larges sont les structures d'intérêt principal).
        La somme pondérée préserve l'information à toutes les échelles,
        contrairement au maximum qui écraserait les structures à échelles moyennes.
        """
        cfg = {0: (0.9, 0.20), 1: (0.6, 0.65), 2: (0.3, 0.85)}
        beta, w_vessel = cfg.get(int(self.galaxy_type), (0.6, 0.65))
        w_log = 1.0 - w_vessel

        sigma_min = max(1.0, float(self.sigma_min))
        sigma_max = max(sigma_min + 1.0, float(self.sigma_max))
        sigmas    = np.logspace(np.log10(sigma_min), np.log10(sigma_max),
                                max(2, int(self.n_scales)))

        vessel_acc = np.zeros_like(L_f32)
        log_acc    = np.zeros_like(L_f32)
        weight_sum = 0.0

        for sigma in sigmas:
            lam1, lam2 = self._hessian_eigenvalues(L_f32, sigma)
            r_v = self._vessel_response_at_scale(lam1, lam2, beta)
            r_l = self._log_response_at_scale(L_f32, sigma)
            # Pondération croissante avec sigma :
            # les grandes échelles (halos, bras larges) sont plus importantes
            w_sigma     = sigma / sigma_max
            vessel_acc += w_sigma * r_v
            log_acc    += w_sigma * r_l
            weight_sum += w_sigma

        if weight_sum > 1e-10:
            vessel_acc /= weight_sum
            log_acc    /= weight_sum

        # Normaliser chaque composante séparément (dynamiques très différentes)
        for arr in (vessel_acc, log_acc):
            amax = float(arr.max())
            if amax > 1e-10:
                arr /= amax

        # Combinaison pondérée vessel + LoG
        response = w_vessel * vessel_acc + w_log * log_acc
        rmax = float(response.max())
        if rmax > 1e-10:
            response /= rmax
        return response

    # ------------------------------------------------------------------
    # Etape 2b : USM multi-échelle progressif
    # ------------------------------------------------------------------

    def _usm_multiscale(self, L_f32: np.ndarray) -> np.ndarray:
        """
        Unsharp Masking multi-échelle progressif (décomposition ondelettes à trous).

        À chaque niveau i, le signal lissé précédent est soustrait du niveau
        actuel → détail à l'échelle sigma_i. Ce résidu est ajouté à L_f32 avec
        un poids strength/n_scales. La somme de tous les niveaux = boost total.

        Avantage vs USM simple : les structures à TOUTES les échelles sont
        amplifiées (du fin détail aux larges structures), indépendamment.
        """
        strength  = float(np.clip(self.usm_strength, 0.0, 5.0))
        sigma_min = max(1.0, float(self.sigma_min))
        sigma_max = max(sigma_min + 1.0, float(self.sigma_max))
        n         = max(2, int(self.n_scales))
        sigmas    = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n)
        weight    = strength / n

        result = L_f32.copy()
        prev   = L_f32.copy()
        for sigma in sigmas:
            ksize   = max(3, int(6.0 * sigma + 1.0) | 1)
            smooth  = cv2.GaussianBlur(prev, (ksize, ksize), sigma)
            detail  = prev - smooth   # positif là où localement plus brillant
            result  = result + weight * detail
            prev    = smooth          # itération progressive : chaque niveau soustrait du précédent

        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Etape 2c : Enhancement Frangi structurel (remplace le simple gamma)
    # ------------------------------------------------------------------

    def _structural_enhance(self, L: np.ndarray, response: np.ndarray,
                             boost: float) -> np.ndarray:
        """
        Enhancement structurel guidé par la réponse Frangi modifiée.

        Deux composantes combinées selon galaxy_type :

        [A] Gamma boost (amplification halos sombres) — inchangé vs original :
            L_gamma = L^(1/(1+boost×R))
            Très efficace pour révéler les halos diffus très sombres.
            Pour L=0.02, R=1, boost=2 : γ=1/3 → L_gamma = 0.27 (+13×)

        [B] Contraste structurel local (révèle les détails internes) :
            Fond local (grande échelle) = enveloppe lisse de la galaxie.
            Signal de structure = L - fond_local (transitions locales).
            L_local = clip(L + boost × 0.5 × R × signal_structure, 0, 1)
            → Amplifie bras spiraux, stries de poussière, asymétries
              là où Frangi/LoG répond, sans toucher le fond.

        Pondération A/B selon galaxy_type :
          0 Elliptique : 70% A + 30% B  → halo sombre est l'objectif principal
          1 Spirale    : 45% A + 55% B  → équilibre halo + détails bras
          2 Tranche    : 30% A + 70% B  → détails fins du disque prioritaires
        """
        # Composante A : gamma boost
        gamma_map = 1.0 / (1.0 + boost * response)
        L_gamma   = np.power(np.maximum(L, 1e-6), gamma_map)

        # Composante B : contraste structurel local
        # Fond local à grande échelle (enveloppe de la galaxie)
        sigma_bg  = max(1.0, float(self.sigma_max)) * 0.5
        ksize_bg  = max(3, int(6.0 * sigma_bg + 1.0) | 1)
        L_bg      = cv2.GaussianBlur(L, (ksize_bg, ksize_bg), sigma_bg)
        # Écart local à l'enveloppe : > 0 pour bras/détails, < 0 pour lacunes
        L_struct  = L - L_bg
        L_local   = np.clip(L + boost * 0.5 * response * L_struct, 0.0, 1.0)

        # Pondération selon type de galaxie
        mix_a = {0: 0.70, 1: 0.45, 2: 0.30}.get(int(self.galaxy_type), 0.45)
        mix_b = 1.0 - mix_a

        return np.clip(mix_a * L_gamma + mix_b * L_local, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Applique le filtre galaxy sur une image BGR uint8.

        Pipeline :
          1. BGR → LAB (isoler la luminance)
          2. Star reduction (ouverture morphologique)
          3. [si usm_strength > 0]   USM multi-échelle → accentue détails internes
          4. [si enhancement > 0]    Frangi structurel modifié → carte combinée
                                     vessel (bras) + LoG (halo/bulbe)
                                     → enhancement combiné gamma + contraste local
          5. LAB → BGR (couleur originale inchangée)

        Args:
            img_bgr : np.ndarray uint8 (H, W, 3) BGR
        Returns:
            np.ndarray uint8 (H, W, 3) BGR
        """
        if not self.enabled:
            return img_bgr
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            return img_bgr

        # --- Séparation luminance / couleur ---
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L_f = lab[:, :, 0].astype(np.float32) / 255.0

        # --- 1. Star reduction ---
        L = self._star_reduce(L_f)

        # --- 2. USM multi-échelle (détails internes) ---
        usm = float(np.clip(self.usm_strength, 0.0, 5.0))
        if usm > 0.0:
            L = self._usm_multiscale(L)

        # --- 3. Frangi structurel modifié → enhancement combiné ---
        boost = float(np.clip(self.enhancement, 0.0, 5.0))
        if boost > 0.0:
            response = self._frangi_structural(L)
            L = self._structural_enhance(L, response, boost)

        L = np.clip(L, 0.0, 1.0)

        # --- Recombinaison (canaux couleur A,B inchangés) ---
        lab_out          = lab.copy()
        lab_out[:, :, 0] = (L * 255.0).astype(np.uint8)
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
