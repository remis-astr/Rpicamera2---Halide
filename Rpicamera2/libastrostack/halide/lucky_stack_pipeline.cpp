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
