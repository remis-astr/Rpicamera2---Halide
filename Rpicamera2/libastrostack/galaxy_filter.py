"""
galaxy_filter.py — Filtre d'enhancement galactique multi-échelle
Deux étapes complémentaires :

  1. STAR REDUCTION  : ouverture morphologique (supprime les sources ponctuelles)
  2. ENHANCEMENT     : deux couches combinables
       a) Frangi vessel multi-échelle → carte de réponse des structures galactiques
          → boost gamma (amplifie fortement les zones sombres/halo)
       b) USM multi-échelle (Unsharp Masking) → accentue les détails internes
          (bras spiraux, gradient bulbe, stries de poussière)

Pourquoi gamma et pas additif × L_proc ?
  L_enh = L^(1/(1+boost×R))
  Pour halo sombre (L=0.02) avec réponse R=1 et boost=2 :
    gamma = 1/3 → L_enh = 0.02^0.33 ≈ 0.27  (+13×)
  Pour centre brillant (L=0.8) :
    L_enh = 0.8^0.33 ≈ 0.93  (+16 %)
  → le halo faint bénéficie d'un boost bien plus fort que le centre saturé.

Galaxy types et plages sigma par défaut :
    0 = Elliptique   : sigma 15–80 px  (larges structures, halo)
    1 = Spirale      : sigma  5–30 px  (bras spiraux + bulbe)
    2 = Par la tranche: sigma  3–20 px (structures fines, linéaires)
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
        enhancement    float  Boost gamma via réponse Frangi vessel (0=off, 0.5-3.0)
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
    # Etape 2a : Frangi vessel multi-échelle → carte de réponse
    # ------------------------------------------------------------------

    def _hessian_eigenvalues(self, L_f32: np.ndarray, sigma: float):
        """
        Calcule les valeurs propres de la matrice Hessienne scale-normalisée.
        λ1 ≤ λ2 (valeurs propres triées par valeur absolue croissante).
        Pour une structure tubulaire brillante : λ1 ≈ 0, λ2 << 0.
        Pour un blob brillant : λ1 ≈ λ2 << 0.
        """
        ksize   = max(3, int(6.0 * sigma + 1.0) | 1)
        blurred = cv2.GaussianBlur(L_f32, (ksize, ksize), sigma)
        s2      = sigma * sigma
        Lxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * s2
        Lyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * s2
        Lxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * s2
        disc = np.sqrt(np.maximum((Lxx - Lyy) ** 2 + 4.0 * Lxy * Lxy, 0.0))
        lam1 = ((Lxx + Lyy) - disc) * 0.5
        lam2 = ((Lxx + Lyy) + disc) * 0.5
        return lam1, lam2

    def _vessel_response_at_scale(self, lam1: np.ndarray, lam2: np.ndarray,
                                   beta: float = 0.5) -> np.ndarray:
        """
        Filtre de Frangi vessel pour structures brillantes sur fond sombre.
        Répond aux structures allongées ET aux blobs selon beta.

        beta proche de 0 : favorise les structures très allongées (tranche)
        beta proche de 1 : répond aussi aux blobs circulaires (elliptique)
        """
        mask = lam2 < 0
        # Rb = rapport des valeurs propres (0 pour tube parfait, 1 pour blob)
        Rb   = np.where(mask, lam1 / (lam2 - 1e-8), 1.0)
        S    = np.sqrt(lam1 ** 2 + lam2 ** 2)
        Smax = float(S.max())
        c    = 0.5 * Smax if Smax > 1e-10 else 1.0
        resp = np.where(
            mask,
            np.exp(-Rb ** 2 / (2.0 * beta ** 2)) *
            (1.0 - np.exp(-S ** 2 / (2.0 * c ** 2))),
            0.0
        )
        return resp.astype(np.float32)

    def _frangi_multiscale(self, L_f32: np.ndarray) -> np.ndarray:
        """
        Réponse Frangi vessel maximum sur n_scales échelles en logspace.
        Retourne une carte normalisée 0–1 indiquant où se trouvent
        les structures galactiques à toutes les échelles.

        beta adapté au type de galaxie :
          type 0 (elliptique) : beta=0.9 → répond aux blobs larges
          type 1 (spirale)    : beta=0.6 → bras spiraux + bulbe
          type 2 (tranche)    : beta=0.3 → structures linéaires fines
        """
        beta_map = {0: 0.9, 1: 0.6, 2: 0.3}
        beta      = beta_map.get(int(self.galaxy_type), 0.6)
        sigma_min = max(1.0, float(self.sigma_min))
        sigma_max = max(sigma_min + 1.0, float(self.sigma_max))
        sigmas    = np.logspace(np.log10(sigma_min), np.log10(sigma_max),
                                max(2, int(self.n_scales)))
        response  = np.zeros_like(L_f32)
        for sigma in sigmas:
            lam1, lam2 = self._hessian_eigenvalues(L_f32, sigma)
            r          = self._vessel_response_at_scale(lam1, lam2, beta)
            response   = np.maximum(response, r)
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
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Applique le filtre galaxy sur une image BGR uint8.

        Pipeline :
          1. BGR → LAB (isoler la luminance)
          2. Star reduction (ouverture morphologique)
          3. [si usm_strength > 0]   USM multi-échelle → accentue détails internes
          4. [si enhancement > 0]    Frangi vessel → gamma boost → halo + structure
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

        # --- 3. Frangi vessel → gamma boost (halo + structures galactiques) ---
        boost = float(np.clip(self.enhancement, 0.0, 5.0))
        if boost > 0.0:
            response = self._frangi_multiscale(L)
            # Gamma map : gamma(x,y) = 1 / (1 + boost × response(x,y))
            # Zones haute réponse → gamma < 1 → brightening fort sur zones sombres
            gamma_map = 1.0 / (1.0 + boost * response)
            L = np.power(np.maximum(L, 1e-6), gamma_map)

        L = np.clip(L, 0.0, 1.0)

        # --- Recombinaison (canaux couleur A,B inchangés) ---
        lab_out          = lab.copy()
        lab_out[:, :, 0] = (L * 255.0).astype(np.uint8)
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
