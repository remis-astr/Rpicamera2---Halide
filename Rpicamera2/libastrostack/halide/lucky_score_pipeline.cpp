/**
 * Halide AOT Generator — Scoring de netteté per-frame (Lucky Stack RAW)
 *
 * 4 variantes compilées séparément (GeneratorParam<int> method) :
 *   0 = local_variance  : Var(ROI)            — 1 passe, Tuple 2 accumulateurs
 *   1 = laplacian       : Var(Lap(ROI))       — 1 passe, stencil 5 pts inline
 *   2 = tenengrad       : E[Gx²+Gy²]          — 1 passe, stencil Sobel inline
 *   3 = gradient        : E[√(Gx²+Gy²)]       — 1 passe, stencil Sobel + sqrt
 *
 * Avantage vs OpenCV/NumPy :
 *   Fusion complète : Gx, Gy calculés dans les registres NEON, pas de buffers
 *   intermédiaires. Trafic mémoire réduit de 4× pour tenengrad/gradient.
 *
 *   Vectorisation NEON de la boucle de réduction (4 float32/cycle via arm_dot_prod).
 *
 * Entrée  : float32 G1 half-res (extrait par LuckyPreprocess)
 * Params  : roi_x0, roi_y0, roi_w, roi_h (en coordonnées G1 half-res)
 * Sortie  : Buffer<float,1> de 1 élément = le score
 *
 * Noms générés dans le Makefile :
 *   lucky_score_var_hal, lucky_score_lap_hal,
 *   lucky_score_ten_hal, lucky_score_grad_hal
 */

#include "Halide.h"
using namespace Halide;

class LuckyScore : public Generator<LuckyScore> {
public:
    // ── Paramètre compile-time ────────────────────────────────────────────────
    GeneratorParam<int> method{"method", 1};  // 0=var 1=lap 2=tenengrad 3=gradient

    // ── Entrée ────────────────────────────────────────────────────────────────
    Input<Buffer<float, 2>> g1{"g1"};

    // ── ROI runtime (coordonnées dans G1 half-res) ────────────────────────────
    Input<int32_t> roi_x0{"roi_x0", 0};
    Input<int32_t> roi_y0{"roi_y0", 0};
    Input<int32_t> roi_w {"roi_w",  256};
    Input<int32_t> roi_h {"roi_h",  256};

    // ── Sortie : 1 élément = score ────────────────────────────────────────────
    Output<Buffer<float, 1>> score{"score"};

    // ── Membres accessibles depuis schedule() ────────────────────────────────
    Func g1c, acc;
    RDom r;
    Var  c0{"c0"};  // dimension de la sortie scalaire (taille = 1)

    void generate() {
        g1c = BoundaryConditions::repeat_edge(g1);
        r   = RDom(roi_x0, roi_w, roi_y0, roi_h, "r");

        if (method == 0) {
            // ── local_variance : Var(X) = E[X²] - E[X]² ─────────────────────
            // Tuple 2 accumulateurs : (ΣX, ΣX²) en une seule passe
            acc = Func("acc");
            acc() = Tuple(0.0f, 0.0f);
            Expr val = g1c(r.x, r.y);
            acc() = Tuple(acc()[0] + val,
                          acc()[1] + val * val);

            Expr n  = cast<float>(roi_w) * cast<float>(roi_h);
            Expr mn = acc()[0] / n;
            score(c0) = acc()[1] / n - mn * mn;

        } else if (method == 1) {
            // ── laplacian : Var(Lap(X)) ───────────────────────────────────────
            // Kernel OpenCV Laplacian ksize=3 : séparable [1,-2,1]⊗[1,2,1]
            // → noyau 2D = [2,0,2 ; 0,-8,0 ; 2,0,2]
            // (différent du Laplacien 5-pts mathématique [0,1,0;1,-4,1;0,1,0])
            // Centre : -8, diagonales : +2, croix : 0
            acc = Func("acc");
            acc() = Tuple(0.0f, 0.0f);
            Expr lap = -8.0f * g1c(r.x,   r.y  )
                       + 2.0f * g1c(r.x-1, r.y-1)
                       + 2.0f * g1c(r.x+1, r.y-1)
                       + 2.0f * g1c(r.x-1, r.y+1)
                       + 2.0f * g1c(r.x+1, r.y+1);
            acc() = Tuple(acc()[0] + lap,
                          acc()[1] + lap * lap);

            Expr n  = cast<float>(roi_w) * cast<float>(roi_h);
            Expr mn = acc()[0] / n;
            score(c0) = acc()[1] / n - mn * mn;

        } else if (method == 2) {
            // ── tenengrad : E[Gx²+Gy²] ───────────────────────────────────────
            // Sobel x :  [-1,0,+1 / -2,0,+2 / -1,0,+1]
            // Sobel y :  [-1,-2,-1 / 0,0,0 / +1,+2,+1]
            // Gx et Gy restent dans les registres — 0 buffer intermédiaire
            acc = Func("acc");
            acc() = 0.0f;
            Expr gx = - g1c(r.x-1, r.y-1) + g1c(r.x+1, r.y-1)
                      - 2.0f*g1c(r.x-1, r.y  ) + 2.0f*g1c(r.x+1, r.y  )
                      - g1c(r.x-1, r.y+1) + g1c(r.x+1, r.y+1);
            Expr gy = - g1c(r.x-1, r.y-1) - 2.0f*g1c(r.x, r.y-1) - g1c(r.x+1, r.y-1)
                      + g1c(r.x-1, r.y+1) + 2.0f*g1c(r.x, r.y+1) + g1c(r.x+1, r.y+1);
            acc() += gx*gx + gy*gy;

            Expr n = cast<float>(roi_w) * cast<float>(roi_h);
            score(c0) = acc() / n;

        } else {
            // ── gradient : E[√(Gx²+Gy²)] ─────────────────────────────────────
            acc = Func("acc");
            acc() = 0.0f;
            Expr gx = - g1c(r.x-1, r.y-1) + g1c(r.x+1, r.y-1)
                      - 2.0f*g1c(r.x-1, r.y  ) + 2.0f*g1c(r.x+1, r.y  )
                      - g1c(r.x-1, r.y+1) + g1c(r.x+1, r.y+1);
            Expr gy = - g1c(r.x-1, r.y-1) - 2.0f*g1c(r.x, r.y-1) - g1c(r.x+1, r.y-1)
                      + g1c(r.x-1, r.y+1) + 2.0f*g1c(r.x, r.y+1) + g1c(r.x+1, r.y+1);
            acc() += sqrt(gx*gx + gy*gy);

            Expr n = cast<float>(roi_w) * cast<float>(roi_h);
            score(c0) = acc() / n;
        }

        score.bound(c0, 0, 1);
    }

    void schedule() {
        // ── Schedule rfactor : vectorisation explicite de la réduction scalaire ──
        //
        // Problème précédent : acc() est scalaire (0 dimensions pures) → Halide
        // ne peut pas appliquer .vectorize() directement sur une RVar de réduction.
        // La vectorisation NEON était déléguée à l'auto-vectoriseur LLVM, qui ne
        // peut pas éliminer la dépendance de portée (acc[t+1] dépend de acc[t]).
        //
        // Solution rfactor : on découpe r.x en groupes de 4 (rxi ∈ [0,4)),
        // puis rfactor(rxi → lane) crée partial(lane) — 4 accumulateurs
        // indépendants qui peuvent être vectorisés en 1 registre NEON 128-bit.
        //
        // Avant : acc()     += expr(r.x, r.y)     [1 lane, dépendance séquentielle]
        // Après : partial(l) += expr(rxo*4+l, r.y) [4 lanes NEON simultanées]
        //         acc()       = Σ partial(l)        [réduction finale 4→1, déroulée]
        //
        // Gain attendu : ×1.5-2 sur la boucle de réduction (ROI 128×128, RPi5).
        // L'entrelacement des 4 accumulateurs supprime la dépendance de portée
        // qui limitait le pipeline FP du Cortex-A76.
        //
        // Buffer intermédiaire partial : 4 floats (ou 8 pour Tuple) = 16-32 B,
        // tient dans L1 → pas de coût mémoire significatif.

        RVar rxi, rxo;
        Var  lane{"lane"};

        // Découpage r.x → (rxo, rxi) avec rxi ∈ [0, 4)
        // rfactor sur le Stage update(0) → partial(lane)
        acc.update(0).split(r.x, rxo, rxi, 4);
        Func partial = acc.update(0).rfactor(rxi, lane);

        // Initialisation de partial : 4 zéros (ou Tuple(0,0)) en parallèle
        partial.compute_root()
               .vectorize(lane, 4);

        // Mise à jour : lane innermost (NEON), puis rxo, puis r.y
        partial.update(0)
               .reorder(lane, rxo, r.y)
               .vectorize(lane, 4);

        // Réduction finale acc() = Σ partial(lane) : 4 éléments, déroulé par LLVM
        acc.compute_root();
    }
};

HALIDE_REGISTER_GENERATOR(LuckyScore, lucky_score)
