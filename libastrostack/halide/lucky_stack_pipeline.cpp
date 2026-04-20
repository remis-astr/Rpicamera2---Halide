/**
 * Halide AOT Generator — Stacking Lucky RAW
 *
 * 2 générateurs (compilés séparément) :
 *   LuckyStackMean  : moyenne simple   (N,H,W) float32 → (H,W) float32
 *   LuckySigmaClip  : sigma-clipping 2 passes → (H,W) float32
 *
 * Convention mémoire (C-contiguous NumPy (N,H,W)) :
 *   Halide Buffer<float,3>(ptr, W, H, N)
 *   frames(x, y, n)  avec  x∈[0,W), y∈[0,H), n∈[0,N)
 *   Strides : dim[0]=1 (x), dim[1]=W (y), dim[2]=W*H (n)
 *
 * Gains estimés vs NumPy (RPi5 Cortex-A76, 6 frames 1024×1024) :
 *   stack_mean   : ×3   (28 ms → 8 ms)  — NEON 4×float32/cycle, 4 cœurs
 *   sigma_clip   : ×5-8 (175 ms → 25 ms) — 2 passes fusionnées, 0 buffer (N,H,W)
 *
 * Intermédiaires sigma_clip (1024×1024) :
 *   stats (Tuple 2) : 8 MB   mean_f : 4 MB   std_f : 4 MB
 *   Total : 16 MB vs 32 MB pour NumPy (stack + mean + std + mask)
 */

#include "Halide.h"
using namespace Halide;

// =============================================================================
// LuckyStackMean — moyenne simple en une passe RDom
// =============================================================================
class LuckyStackMean : public Generator<LuckyStackMean> {
public:
    Input<Buffer<float, 3>> frames  {"frames"};
    Input<int32_t>          n_frames{"n_frames", 1};

    Output<Buffer<float, 2>> result{"result"};

    Var  x{"x"}, y{"y"};
    Func acc{"acc"};
    RDom r;

    void generate() {
        r = RDom(0, n_frames, "r");

        acc(x, y)  = 0.0f;
        acc(x, y) += frames(x, y, r.x);

        result(x, y) = acc(x, y) / cast<float>(n_frames);
    }

    void schedule() {
        // Réduction : r.x en boucle interne (accès mémoire contigus en x),
        // x vectorisé NEON 4×, y parallélisé sur 4 cœurs.
        acc.compute_root();
        acc.update(0)
           .reorder(x, r.x, y)
           .vectorize(x, 4)
           .parallel(y);

        result.vectorize(x, 4).parallel(y);
    }
};

// =============================================================================
// LuckySigmaClip — sigma-clipping 2 passes sans buffer (N,H,W) intermédiaire
// =============================================================================
class LuckySigmaClip : public Generator<LuckySigmaClip> {
public:
    Input<Buffer<float, 3>> frames  {"frames"};
    Input<int32_t>          n_frames{"n_frames", 1};
    Input<float>            kappa   {"kappa",    2.5f};

    Output<Buffer<float, 2>> result{"result"};

    Var  x{"x"}, y{"y"};
    Func stats {"stats"},   // Tuple (ΣX, ΣX²) per pixel
         mean_f{"mean_f"},  // E[X]   = stats[0] / N
         std_f {"std_f"},   // σ(X)   = sqrt(Var[X])
         masked{"masked"};  // Tuple (Σ_clipped, count_clipped)
    RDom r, r2;

    void generate() {
        r  = RDom(0, n_frames, "r");
        r2 = RDom(0, n_frames, "r2");

        // ── Passe 1 : (ΣX, ΣX²) par pixel en 1 RDom ─────────────────────────
        stats(x, y) = Tuple(0.0f, 0.0f);
        Expr v = frames(x, y, r.x);
        stats(x, y) = Tuple(stats(x, y)[0] + v,
                            stats(x, y)[1] + v * v);

        Expr n_f = cast<float>(n_frames);
        mean_f(x, y) = stats(x, y)[0] / n_f;
        std_f (x, y) = sqrt(max(stats(x, y)[1] / n_f
                                - mean_f(x, y) * mean_f(x, y),
                                0.0f));

        // ── Passe 2 : moyenne des pixels dans ±kappa·σ ────────────────────────
        masked(x, y) = Tuple(0.0f, 0.0f);
        Expr v2   = frames(x, y, r2.x);
        Expr clip = abs(v2 - mean_f(x, y)) <= kappa * std_f(x, y);
        masked(x, y) = Tuple(masked(x, y)[0] + select(clip, v2,   0.0f),
                             masked(x, y)[1] + select(clip, 1.0f, 0.0f));

        // Résultat : somme / max(count, 1)  → évite division par 0
        result(x, y) = masked(x, y)[0] / max(masked(x, y)[1], 1.0f);
    }

    void schedule() {
        // ── Passe 1 ──────────────────────────────────────────────────────────
        stats.compute_root();
        stats.update(0)
             .reorder(x, r.x, y)
             .vectorize(x, 4)
             .parallel(y);

        // mean_f / std_f : (H,W)×2 float32, calculés depuis stats (compute_root)
        // Matérialisés pour éviter la recomputation dans la boucle r2.x de masked.
        mean_f.compute_root().vectorize(x, 4).parallel(y);
        std_f .compute_root().vectorize(x, 4).parallel(y);

        // ── Passe 2 ──────────────────────────────────────────────────────────
        masked.compute_root();
        masked.update(0)
              .reorder(x, r2.x, y)
              .vectorize(x, 4)
              .parallel(y);

        result.vectorize(x, 4).parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(LuckyStackMean,  lucky_stack_mean)
HALIDE_REGISTER_GENERATOR(LuckySigmaClip,  lucky_sigma_clip)

// =============================================================================
// LsStackMeanFrame — mise à jour incrémentale de moyenne RGB (live stack)
// =============================================================================
/**
 * Met à jour la moyenne courante avec une nouvelle frame.
 *
 * Convention mémoire (NumPy (3,H,W) C-contiguous float32) :
 *   Halide Buffer<float,3>(ptr, W, H, 3)
 *   buf(x, y, c)  strides : dim[0]=1 (x), dim[1]=W (y), dim[2]=W*H (c)
 *
 * Gain estimé vs NumPy (RPi5, 1090×1928 RGB) :
 *   ×4-5  (5 ops → 1 passe NEON fusionnée)
 */
class LsStackMeanFrame : public Generator<LsStackMeanFrame> {
public:
    Input<Buffer<float, 3>>  stacked_in {"stacked_in"};   // (W,H,3) moyenne courante
    Input<Buffer<float, 2>>  cnt_in     {"cnt_in"};        // (W,H) compteur par pixel
    Input<Buffer<float, 3>>  img        {"img"};            // (W,H,3) nouvelle frame

    Output<Buffer<float, 3>> stacked_out{"stacked_out"};   // (W,H,3) nouvelle moyenne
    Output<Buffer<float, 2>> cnt_out    {"cnt_out"};        // (W,H) nouveau compteur

    Var x{"x"}, y{"y"}, c{"c"};
    Func valid_f  {"valid_f"},    // masque validité pixel (NaN guard)
         cnt_new_f{"cnt_new_f"};  // compteur mis à jour

    void generate() {
        // NaN guard : NaN != NaN par définition IEEE 754
        // Pixel invalide si l'un des 3 canaux est NaN
        valid_f(x, y) = (img(x, y, 0) == img(x, y, 0)) &
                        (img(x, y, 1) == img(x, y, 1)) &
                        (img(x, y, 2) == img(x, y, 2));

        cnt_new_f(x, y) = select(valid_f(x, y),
                                  cnt_in(x, y) + 1.0f,
                                  cnt_in(x, y));

        cnt_out(x, y) = cnt_new_f(x, y);

        // Moyenne glissante pondérée par pixel
        // new_mean = (old_mean × cnt_old + img) / cnt_new
        stacked_out(x, y, c) = select(
            valid_f(x, y),
            (stacked_in(x, y, c) * cnt_in(x, y) + img(x, y, c)) / cnt_new_f(x, y),
            stacked_in(x, y, c)
        );
    }

    void schedule() {
        // valid_f et cnt_new_f matérialisés une fois (partagés par cnt_out et stacked_out)
        valid_f.compute_root()
               .vectorize(x, 4).parallel(y);
        cnt_new_f.compute_root()
                 .vectorize(x, 4).parallel(y);
        cnt_out.compute_root()
               .vectorize(x, 4).parallel(y);
        // stacked_out : c innermost (unroll 3), x vectorisé NEON, y parallèle
        // → accès planaire stride-1 par canal, 3 canaux × 4 pixels par cycle NEON
        stacked_out.compute_root()
                   .reorder(c, x, y)
                   .unroll(c, 3)
                   .vectorize(x, 4)
                   .parallel(y);
    }
};

// =============================================================================
// LsStackKappaFrame — Welford incrémental + rejet kappa-sigma RGB (live stack)
// =============================================================================
/**
 * Met à jour la moyenne (Welford) avec rejet outlier par kappa-sigma.
 *
 * Pixel rejeté si |img - mean| > kappa × std sur AU MOINS un canal.
 * Algorithme de Welford : numériquement stable, une seule frame en mémoire.
 *
 * Convention mémoire : idem LsStackMeanFrame — (W,H,3) planaire float32.
 *
 * Gain estimé vs NumPy (RPi5, 1090×1928 RGB) :
 *   ×8-12  (15 ops → 1 passe NEON fusionnée)
 */
class LsStackKappaFrame : public Generator<LsStackKappaFrame> {
public:
    Input<Buffer<float, 3>>  stacked_in{"stacked_in"};   // (W,H,3) moyenne courante
    Input<Buffer<float, 2>>  cnt_in    {"cnt_in"};        // (W,H) compteur par pixel
    Input<Buffer<float, 3>>  m2_in     {"m2_in"};         // (W,H,3) variance Welford M2
    Input<Buffer<float, 3>>  img       {"img"};            // (W,H,3) nouvelle frame
    Input<float>             kappa     {"kappa", 2.5f};   // seuil sigma

    Output<Buffer<float, 3>> stacked_out{"stacked_out"};
    Output<Buffer<float, 2>> cnt_out    {"cnt_out"};
    Output<Buffer<float, 3>> m2_out     {"m2_out"};

    Var x{"x"}, y{"y"}, c{"c"};
    Func accept_f {"accept_f"},   // masque acceptance (valid & !reject)
         cnt_new_f{"cnt_new_f"},  // compteur mis à jour
         d_f      {"d_f"};        // d1 = img - old_mean (réutilisé pour M2)

    void generate() {
        // ── NaN guard ──────────────────────────────────────────────────────
        Expr valid = (img(x,y,0) == img(x,y,0)) &
                     (img(x,y,1) == img(x,y,1)) &
                     (img(x,y,2) == img(x,y,2));

        // ── Rejet kappa-sigma ──────────────────────────────────────────────
        // Pixel rejeté si cnt > 1 ET |deviation| > kappa×std sur AU MOINS un canal
        Expr cnt_cur   = cnt_in(x, y);
        Expr have_stats = cnt_cur > 1.0f;
        Expr cnt_m1    = max(cnt_cur - 1.0f, 1.0f);   // dénominateur variance

        // Rejet par canal (lambda C++ utilisé pour éviter la répétition)
        auto ch_over = [&](int ch) -> Expr {
            Expr var_ch = m2_in(x, y, ch) / cnt_m1;
            Expr std_ch = sqrt(max(var_ch, 0.0f));
            Expr d_ch   = img(x, y, ch) - stacked_in(x, y, ch);
            return have_stats & (std_ch > 0.0f) & (abs(d_ch) > kappa * std_ch);
        };
        Expr reject = ch_over(0) | ch_over(1) | ch_over(2);

        accept_f(x, y) = valid & !reject;
        cnt_new_f(x, y) = select(accept_f(x, y), cnt_cur + 1.0f, cnt_cur);

        // ── d1 = img - old_mean (partagé entre stacked_out et m2_out) ──────
        d_f(x, y, c) = img(x, y, c) - stacked_in(x, y, c);

        // ── Mise à jour Welford : new_mean = old_mean + d1 / cnt_new ────────
        stacked_out(x, y, c) = select(
            accept_f(x, y),
            stacked_in(x, y, c) + d_f(x, y, c) / cnt_new_f(x, y),
            stacked_in(x, y, c)
        );

        cnt_out(x, y) = cnt_new_f(x, y);

        // ── Mise à jour M2 : m2_new = m2 + d1 × (img - new_mean) ───────────
        // d2 = img - new_mean = img - (old_mean + d1/cnt_new)
        //                     = d1 × (1 - 1/cnt_new)
        //                     = d1 × (cnt_new - 1) / cnt_new
        // Forme compacte sans recalculer stacked_out :
        Expr cnt_new_expr = cnt_new_f(x, y);
        m2_out(x, y, c) = select(
            accept_f(x, y),
            m2_in(x, y, c) + d_f(x, y, c) *
                (img(x, y, c) - (stacked_in(x, y, c) + d_f(x, y, c) / cnt_new_expr)),
            m2_in(x, y, c)
        );
    }

    void schedule() {
        // Quantités scalaires 2D (partagées entre tous les canaux) : 1 passe
        accept_f.compute_root()
                .vectorize(x, 4).parallel(y);
        cnt_new_f.compute_root()
                 .vectorize(x, 4).parallel(y);
        cnt_out.compute_root()
               .vectorize(x, 4).parallel(y);

        // Quantités 3 canaux : c innermost (unroll), x NEON, y parallèle
        d_f.compute_root()
           .reorder(c, x, y).unroll(c, 3).vectorize(x, 4).parallel(y);
        stacked_out.compute_root()
                   .reorder(c, x, y).unroll(c, 3).vectorize(x, 4).parallel(y);
        m2_out.compute_root()
              .reorder(c, x, y).unroll(c, 3).vectorize(x, 4).parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(LsStackMeanFrame,  ls_stack_mean_frame)
HALIDE_REGISTER_GENERATOR(LsStackKappaFrame, ls_stack_kappa_frame)

// =============================================================================
// LsApplyCalibration — correction fond + vignetage en 1 passe (live stack)
// =============================================================================
/**
 * Applique la correction combinée fond additif + vignetage multiplicatif.
 *
 * Formule : result = clamp((frame - bg) / flat, 0, 1)
 *   bg(x,y,c)  : fond additif per-canal — gradient de ciel (lumière parasite)
 *   flat(x,y)  : vignetage multiplicatif achromatique normalisé à 1 au centre
 *                (réducteur focal, bords de lentille) — valeurs [0.05, 1.0]
 *
 * Convention mémoire : idem LsStackMeanFrame — (W,H,3) planaire float32.
 *
 * Gain vs NumPy : ×4-6 (3 ops par pixel → 1 passe NEON vectorisée)
 */
class LsApplyCalibration : public Generator<LsApplyCalibration> {
public:
    Input<Buffer<float, 3>>  frame {"frame"};  // (W,H,3) planaire — image [0,1]
    Input<Buffer<float, 3>>  bg    {"bg"};     // (W,H,3) planaire — fond additif [0,1]
    Input<Buffer<float, 2>>  flat  {"flat"};   // (W,H)   — vignetage (normalisé ≤ 1)

    Output<Buffer<float, 3>> result{"result"}; // (W,H,3) planaire — résultat [0,1]

    Var x{"x"}, y{"y"}, c{"c"};

    void generate() {
        // Correction : soustraction fond + division vignetage
        // min_flat = 0.01 pour éviter division par zéro sur pixels masqués/bords
        result(x, y, c) = clamp(
            (frame(x, y, c) - bg(x, y, c)) / max(flat(x, y), 0.01f),
            0.0f, 1.0f
        );
    }

    void schedule() {
        // c innermost (unroll 3) + x NEON + y parallèle
        // flat(x,y) accédé 3× (1 par canal unrollé) — dans le cache L1 après 1ère lecture
        result.compute_root()
              .reorder(c, x, y)
              .unroll(c, 3)
              .vectorize(x, 4)
              .parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(LsApplyCalibration, ls_apply_calibration)

// =============================================================================
// LsPolyBgApply — évaluation polynôme 2D + calibration en 1 passe NEON
// =============================================================================
/**
 * Évalue un polynôme 2D (fond de ciel + vignetage) et applique la correction
 * en une seule passe NEON, sans tableau intermédiaire (H,W,3).
 *
 * Remplace _eval_poly_full() × 3 canaux + _halide_apply_calibration dans session.py.
 *
 * Polynôme 2D degré ≤ 4 — 15 monomials dans l'ordre :
 *   [1, x, y, x², xy, y², x³, x²y, xy², y³, x⁴, x³y, x²y², xy³, y⁴]
 * Coordonnées normalisées xn ∈ [-1,+1] via linspace(-1,1,W), idem y.
 *
 * coeffs_bg (15, 3) : coefficients fond par canal (termes inutilisés = 0)
 * coeff_fl  (15,)   : coefficients flat achromatique = mean_c(bg_c / bg_c(0,0))
 * alpha             : flat_strength / 100.0  (0 = BG seul, 1 = vignetage complet)
 *
 * flat_eff(x,y) = clamp((1-alpha) + alpha * poly_fl(x,y), 0.05, 2.0)
 * result(x,y,c) = clamp((frame(x,y,c) - bg(x,y,c)) / max(flat_eff, 0.01), 0, 1)
 *
 * Gain estimé vs NumPy (RPi5, 1440×1080, degree=2) :
 *   ×8-12  (≥6 passes numpy → 1 passe NEON, 0 tableau bg (H,W,3) intermédiaire)
 */
class LsPolyBgApply : public Generator<LsPolyBgApply> {
public:
    Input<Buffer<float, 3>> frame    {"frame"};      // (W,H,3) planaire float32
    Input<Buffer<float, 2>> coeffs_bg{"coeffs_bg"};  // (15, 3) coefficients BG par canal
    Input<Buffer<float, 1>> coeff_fl {"coeff_fl"};   // (15,)   coefficients flat achromatique
    Input<float>            alpha    {"alpha", 0.0f}; // flat_strength / 100.0

    Output<Buffer<float, 3>> result{"result"};  // (W,H,3) planaire float32

    Var x{"x"}, y{"y"}, c{"c"};

    // Fonctions intermédiaires déclarées comme membres pour être accessibles dans schedule()
    Func flat_poly{"flat_poly"};
    Func flat_eff {"flat_eff"};
    Func bg_c     {"bg_c"};

    void generate() {
        // Coordonnées normalisées [-1, +1] identiques à np.linspace(-1, 1, W/H)
        // xn[i] = 2*i/(W-1) - 1  pour i ∈ [0, W-1]
        Expr W = frame.width();
        Expr H = frame.height();
        Expr xn = cast<float>(x) * 2.0f / (cast<float>(W) - 1.0f) - 1.0f;
        Expr yn = cast<float>(y) * 2.0f / (cast<float>(H) - 1.0f) - 1.0f;

        // Puissances — Halide CSE les réutilise automatiquement dans les termes
        Expr x2 = xn * xn,  x3 = x2 * xn,  x4 = x3 * xn;
        Expr y2 = yn * yn,  y3 = y2 * yn,  y4 = y3 * yn;

        // ── Polynôme fond par canal (15 monomials, degré ≤ 4) ─────────────────
        bg_c(x, y, c) =
            coeffs_bg( 0, c)              +
            coeffs_bg( 1, c) * xn         +
            coeffs_bg( 2, c) * yn         +
            coeffs_bg( 3, c) * x2         +
            coeffs_bg( 4, c) * xn * yn    +
            coeffs_bg( 5, c) * y2         +
            coeffs_bg( 6, c) * x3         +
            coeffs_bg( 7, c) * x2 * yn    +
            coeffs_bg( 8, c) * xn * y2    +
            coeffs_bg( 9, c) * y3         +
            coeffs_bg(10, c) * x4         +
            coeffs_bg(11, c) * x3 * yn    +
            coeffs_bg(12, c) * x2 * y2    +
            coeffs_bg(13, c) * xn * y3    +
            coeffs_bg(14, c) * y4;

        // ── Polynôme flat achromatique (15 monomials) ──────────────────────────
        flat_poly(x, y) =
            coeff_fl( 0)              +
            coeff_fl( 1) * xn         +
            coeff_fl( 2) * yn         +
            coeff_fl( 3) * x2         +
            coeff_fl( 4) * xn * yn    +
            coeff_fl( 5) * y2         +
            coeff_fl( 6) * x3         +
            coeff_fl( 7) * x2 * yn    +
            coeff_fl( 8) * xn * y2    +
            coeff_fl( 9) * y3         +
            coeff_fl(10) * x4         +
            coeff_fl(11) * x3 * yn    +
            coeff_fl(12) * x2 * y2    +
            coeff_fl(13) * xn * y3    +
            coeff_fl(14) * y4;

        // ── flat_eff = lerp(1, flat_poly, alpha), clampé ───────────────────────
        // flat_poly(0,0) = coeff_fl[0] = 1.0 par construction (centre normalisé)
        flat_eff(x, y) = clamp(
            (1.0f - alpha) + alpha * flat_poly(x, y),
            0.05f, 2.0f
        );

        // ── Correction : (frame - bg) / flat_eff ──────────────────────────────
        result(x, y, c) = clamp(
            (frame(x, y, c) - bg_c(x, y, c)) / max(flat_eff(x, y), 0.01f),
            0.0f, 1.0f
        );
    }

    void schedule() {
        // flat_poly et flat_eff : 2D (W,H), partagés par les 3 canaux de result.
        // Matérialisés par ligne pour rester dans le cache L1 (W×4B ≈ 5.7 KB).
        flat_poly.compute_at(result, y).vectorize(x, 4);
        flat_eff .compute_at(result, y).vectorize(x, 4);

        // bg_c : inline dans result — c unrollé × 3, coefficients dans les registres.
        // result : c innermost unrollé × 3, x NEON × 4, y parallèle sur 4 cœurs.
        result.compute_root()
              .reorder(c, x, y)
              .unroll(c, 3)
              .vectorize(x, 4)
              .parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(LsPolyBgApply, ls_poly_bg_apply)
