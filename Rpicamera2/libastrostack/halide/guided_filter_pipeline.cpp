/**
 * Halide AOT Generator — Guided Filter (He et al., 2013)
 *
 * Auto-guidage par canal : guide = input (chaque canal filtré indépendamment).
 * Box filter séparable H+V → O(N) indépendant du rayon.
 *
 * Algorithme (par canal, I normalisé [0,1]) :
 *   1. mean_I  = box(I,   r)
 *   2. mean_I2 = box(I²,  r)
 *   3. var_I   = mean_I2 - mean_I²
 *   4. a       = var_I / (var_I + eps_norm)      eps_norm = eps/255²
 *   5. b       = mean_I * (1 - a)
 *   6. mean_a  = box(a, r)
 *   7. mean_b  = box(b, r)
 *   8. output  = clamp((mean_a * I + mean_b) * 255, 0, 255)
 *
 * Input  : uint8 (W, H, 3) planaire — dim[0]=x stride=1, dim[2]=c stride=W*H
 * Output : uint8 (W, H, 3) planaire — même layout
 *
 * Paramètres :
 *   radius (compile-time) : demi-fenêtre du box filter (défaut 8 = fenêtre 17×17)
 *   eps    (runtime float) : régularisation, espace [0, 255²]
 *                            ex : 50=préservation forte, 200=modéré, 1000=fort lissage
 *
 * Perf cible RPi5 : ~30 ms @ 3840×2160×3 (vs ~500 ms bilateral d=10)
 */

#include "Halide.h"
using namespace Halide;

class GuidedFilterPipeline : public Generator<GuidedFilterPipeline> {
public:
    // ── Paramètre compile-time ────────────────────────────────────────────
    GeneratorParam<int> radius{"radius", 8};   // demi-fenêtre (window = 2r+1)

    // ── Entrée / sorties ──────────────────────────────────────────────────
    Input<Buffer<uint8_t, 3>>  input {"input"};   // (W, H, C=3) planaire
    Input<float>               eps   {"eps",  200.0f};  // régularisation [0, 255²]
    Output<Buffer<uint8_t, 3>> output{"output"};  // (W, H, C=3) planaire

    // ── Intermédiaires (membres pour schedule()) ──────────────────────────
    Func I    {"I"};
    Func I_bh {"I_bh"},  I_bv {"I_bv"},  mean_I {"mean_I"};
    Func I2   {"I2"};
    Func I2_bh{"I2_bh"}, I2_bv{"I2_bv"}, mean_I2{"mean_I2"};
    Func var_I{"var_I"}, ak{"ak"}, bk{"bk"};
    Func ak_bh{"ak_bh"}, ak_bv{"ak_bv"}, mean_ak{"mean_ak"};
    Func bk_bh{"bk_bh"}, bk_bv{"bk_bv"}, mean_bk{"mean_bk"};

    void generate() {
        Var x("x"), y("y"), c("c");

        const float sz = (float)((2*radius+1) * (2*radius+1));

        // Bords miroir sur les dimensions spatiales (x, y)
        Func clamped = BoundaryConditions::repeat_edge(
            input,
            {{0, input.dim(0).extent()}, {0, input.dim(1).extent()}}
        );

        // Normalisation → float [0, 1]
        I(x, y, c) = cast<float>(clamped(x, y, c)) / 255.0f;

        // ── Box filter de I : H puis V ────────────────────────────────────
        RDom rx_I (-(int)radius, 2*(int)radius+1, "rx_I");
        I_bh(x, y, c) = sum(I(x + rx_I, y, c));

        RDom ry_I (-(int)radius, 2*(int)radius+1, "ry_I");
        I_bv(x, y, c) = sum(I_bh(x, y + ry_I, c));

        mean_I(x, y, c) = I_bv(x, y, c) / sz;

        // ── Box filter de I² : H puis V ───────────────────────────────────
        I2(x, y, c) = I(x, y, c) * I(x, y, c);

        RDom rx_I2(-(int)radius, 2*(int)radius+1, "rx_I2");
        I2_bh(x, y, c) = sum(I2(x + rx_I2, y, c));

        RDom ry_I2(-(int)radius, 2*(int)radius+1, "ry_I2");
        I2_bv(x, y, c) = sum(I2_bh(x, y + ry_I2, c));

        mean_I2(x, y, c) = I2_bv(x, y, c) / sz;

        // ── Variance + coefficients a, b ──────────────────────────────────
        // eps normalisé de l'espace [0,255²] vers [0,1²]
        Expr eps_norm = eps / (255.0f * 255.0f);

        var_I(x, y, c) = mean_I2(x, y, c) - mean_I(x, y, c) * mean_I(x, y, c);
        ak   (x, y, c) = var_I(x, y, c) / (var_I(x, y, c) + eps_norm);
        bk   (x, y, c) = mean_I(x, y, c) * (1.0f - ak(x, y, c));

        // ── Box filter de a : H puis V ────────────────────────────────────
        RDom rx_a(-(int)radius, 2*(int)radius+1, "rx_a");
        ak_bh(x, y, c) = sum(ak(x + rx_a, y, c));

        RDom ry_a(-(int)radius, 2*(int)radius+1, "ry_a");
        ak_bv(x, y, c) = sum(ak_bh(x, y + ry_a, c));

        mean_ak(x, y, c) = ak_bv(x, y, c) / sz;

        // ── Box filter de b : H puis V ────────────────────────────────────
        RDom rx_b(-(int)radius, 2*(int)radius+1, "rx_b");
        bk_bh(x, y, c) = sum(bk(x + rx_b, y, c));

        RDom ry_b(-(int)radius, 2*(int)radius+1, "ry_b");
        bk_bv(x, y, c) = sum(bk_bh(x, y + ry_b, c));

        mean_bk(x, y, c) = bk_bv(x, y, c) / sz;

        // ── Sortie : q = mean_a * I + mean_b → uint8 ─────────────────────
        output(x, y, c) = cast<uint8_t>(clamp(
            (mean_ak(x, y, c) * I(x, y, c) + mean_bk(x, y, c)) * 255.0f,
            0.0f, 255.0f
        ));
    }

    void schedule() {
        Var x = output.args()[0];
        Var y = output.args()[1];

        // Les passes H doivent être matérialisées avant les passes V
        // (réductions, ne peuvent pas être inlinées dans la passe suivante)
        I_bh .compute_root().parallel(y).vectorize(x, 8);
        I2_bh.compute_root().parallel(y).vectorize(x, 8);
        ak_bh.compute_root().parallel(y).vectorize(x, 8);
        bk_bh.compute_root().parallel(y).vectorize(x, 8);

        // Passes V : matérialisées pour éviter recalcul dans les étapes suivantes
        I_bv .compute_root().parallel(y).vectorize(x, 8);
        I2_bv.compute_root().parallel(y).vectorize(x, 8);
        ak_bv.compute_root().parallel(y).vectorize(x, 8);
        bk_bv.compute_root().parallel(y).vectorize(x, 8);

        // Sortie parallélisée
        output.parallel(y).vectorize(x, 8);
    }
};

HALIDE_REGISTER_GENERATOR(GuidedFilterPipeline, guided_filter_pipeline)
