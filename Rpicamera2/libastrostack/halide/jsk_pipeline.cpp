/**
 * Halide AOT Generator — Pipeline JSK LIVE
 *
 * Fusionne en une seule passe :
 *   RAW12 uint16 (Bayer RGGB)
 *   → HDR multi-exposition virtuelle (2 ou 3 niveaux de clip)
 *   → Merge Median ou Mean
 *   → Débayérisation bilinéaire
 *   → Gains AWB (R, G, B)
 *   → uint8 RGB output
 *
 * Sans Halide : 4 buffers intermédiaires, 4 passes mémoire.
 * Avec Halide : 0 buffer intermédiaire, 1 passe mémoire fusionnée.
 *
 * Compilation AOT :
 *   g++ jsk_pipeline.cpp -o jsk_gen -lHalide -I$HALIDE/include -L$HALIDE/lib
 *       -Wl,-rpath,$HALIDE/lib -std=c++17
 *   ./jsk_gen -g jsk_pipeline -o . target=arm-64-linux-arm_dot_prod-arm_fp16
 *   → produit jsk_pipeline.a + jsk_pipeline.h
 */

#include "Halide.h"
#include <cstdio>

using namespace Halide;

// ─────────────────────────────────────────────────────────────────────────────
// Helper : médiane de 3 expressions scalaires
// ─────────────────────────────────────────────────────────────────────────────
Expr median3(Expr a, Expr b, Expr c) {
    return max(min(a, b), min(max(a, b), c));
}

// ─────────────────────────────────────────────────────────────────────────────
// Generator principal
// ─────────────────────────────────────────────────────────────────────────────
class JSKPipeline : public Generator<JSKPipeline> {
public:
    // ── Paramètres compile-time ───────────────────────────────────────────────
    // Nombre de niveaux HDR : 2 (11+12 bit) ou 3 (10+11+12 bit)
    GeneratorParam<int>  hdr_levels{"hdr_levels", 3};   // 2 ou 3
    // Méthode de merge : 0=Mean, 1=Median
    GeneratorParam<int>  hdr_method{"hdr_method", 1};   // 0=Mean 1=Median

    // ── Entrée ────────────────────────────────────────────────────────────────
    // RAW12 uint16, espace CSI-2 ×16 (valeurs 0-65520, signif. 0-4095)
    // Dimensions (W, H) — Bayer RGGB
    Input<Buffer<uint16_t, 2>> input{"input"};

    // ── Paramètres runtime ────────────────────────────────────────────────────
    // Gains AWB (multiplié par 256 pour travailler en entier)
    // r_gain=256 → ×1.0, r_gain=384 → ×1.5
    Input<int32_t> r_gain{"r_gain", 256};
    Input<int32_t> g_gain{"g_gain", 256};
    Input<int32_t> b_gain{"b_gain", 256};
    // Poids HDR par niveau (entiers 0-100)
    Input<int32_t> w0{"w0", 100};
    Input<int32_t> w1{"w1", 100};
    Input<int32_t> w2{"w2", 100};  // ignoré si hdr_levels==2

    // ── Sortie ────────────────────────────────────────────────────────────────
    // RGB uint8 (W, H, 3) — ch0=R, ch1=G, ch2=B
    Output<Buffer<uint8_t, 3>> output{"output"};

    // ── Pipeline ──────────────────────────────────────────────────────────────
    void generate() {
        Var x("x"), y("y"), c("c");

        // Entrée avec clamping aux bords (bords miroir)
        Func clamped = BoundaryConditions::mirror_image(input);

        // ── Étape 1 : normaliser le RAW12 en float [0, 1] ────────────────────
        // L'espace CSI-2 ×16 : pixel 12-bit réel = val_uint16 / 16
        // Max 12-bit natif = 4095, en CSI-2 = 65520
        const float max12 = 65520.0f;  // 4095 × 16

        // Niveau 0 : 12-bit complet (aucun clip)
        Expr raw_f = cast<float>(clamped(x, y));
        Expr lv0 = clamp(raw_f, 0.0f, max12) / max12;

        // Niveau 1 : clip à 11-bit (2047 × 16 = 32752)
        const float max11 = 32752.0f;
        Expr lv1 = clamp(raw_f, 0.0f, max11) / max11;

        // Niveau 2 : clip à 10-bit (1023 × 16 = 16368)
        const float max10 = 16368.0f;
        Expr lv2 = clamp(raw_f, 0.0f, max10) / max10;

        // ── Étape 2 : merge HDR en espace Bayer float ─────────────────────────
        Func merged("merged");
        if (hdr_method == 1) {
            // Median
            if (hdr_levels == 3) {
                merged(x, y) = median3(lv0, lv1, lv2);
            } else {
                // Median de 2 = moyenne
                merged(x, y) = (lv0 + lv1) * 0.5f;
            }
        } else {
            // Mean pondéré — total_w calculé selon hdr_levels (GeneratorParam connu à codegen)
            Expr total_w, safe_w;
            if (hdr_levels == 3) {
                total_w = cast<float>(w0 + w1 + w2);
                safe_w  = max(total_w, 1.0f);
                merged(x, y) = (lv0 * cast<float>(w0) +
                                lv1 * cast<float>(w1) +
                                lv2 * cast<float>(w2)) / safe_w;
            } else {
                total_w = cast<float>(w0 + w1);
                safe_w  = max(total_w, 1.0f);
                merged(x, y) = (lv0 * cast<float>(w0) +
                                lv1 * cast<float>(w1)) / safe_w;
            }
        }

        // ── Étape 3 : débayérisation bilinéaire RGGB ─────────────────────────
        // Pattern RGGB :
        //   (2k,   2l  ) → R
        //   (2k+1, 2l  ) → G1
        //   (2k,   2l+1) → G2
        //   (2k+1, 2l+1) → B
        //
        // Sélecteurs de parité
        Expr even_x = (x % 2) == 0;
        Expr even_y = (y % 2) == 0;

        // Voisins bilinéaires (lecture dans merged)
        Expr m   = merged(x,   y);
        Expr mN  = merged(x,   y-1);
        Expr mS  = merged(x,   y+1);
        Expr mW  = merged(x-1, y);
        Expr mE  = merged(x+1, y);
        Expr mNW = merged(x-1, y-1);
        Expr mNE = merged(x+1, y-1);
        Expr mSW = merged(x-1, y+1);
        Expr mSE = merged(x+1, y+1);
        Expr diag4 = (mNW + mNE + mSW + mSE) * 0.25f;
        Expr cross4 = (mN + mS + mW + mE) * 0.25f;
        Expr horiz2 = (mW + mE) * 0.5f;
        Expr vert2  = (mN + mS) * 0.5f;

        // Canal R
        Func R_ch("R_ch");
        R_ch(x, y) = select(
            even_x && even_y,  m,          // site R
            !even_x && even_y, horiz2,     // site G1 : R = moyenne H
            even_x && !even_y, vert2,      // site G2 : R = moyenne V
            diag4                          // site B  : R = moyenne diag
        );

        // Canal G
        Func G_ch("G_ch");
        G_ch(x, y) = select(
            even_x && even_y,  cross4,     // site R  : G = moyenne croix
            !even_x && even_y, m,          // site G1
            even_x && !even_y, m,          // site G2
            cross4                         // site B  : G = moyenne croix
        );

        // Canal B
        Func B_ch("B_ch");
        B_ch(x, y) = select(
            even_x && even_y,  diag4,      // site R  : B = moyenne diag
            !even_x && even_y, vert2,      // site G1 : B = moyenne V
            even_x && !even_y, horiz2,     // site G2 : B = moyenne H
            m                              // site B
        );

        // ── Étape 4 : gains AWB + conversion uint8 ───────────────────────────
        Func rgb("rgb");
        rgb(x, y, c) = cast<uint8_t>(clamp(
            select(
                c == 0, R_ch(x, y) * cast<float>(r_gain) * (1.0f/256.0f),
                c == 1, G_ch(x, y) * cast<float>(g_gain) * (1.0f/256.0f),
                        B_ch(x, y) * cast<float>(b_gain) * (1.0f/256.0f)
            ) * 255.0f,
            0.0f, 255.0f
        ));

        output(x, y, c) = rgb(x, y, c);
    }

    // ── Schedule ──────────────────────────────────────────────────────────────
    void schedule() {
        Var x = output.args()[0];
        Var y = output.args()[1];
        Var c = output.args()[2];

        // Layout planaire (défaut Halide) : dim[0]=x stride=1, dim[2]=c stride=W*H
        // Côté Python : sortie allouée en (3,H,W) numpy → transpose en (H,W,3)

        // Paralléliser sur y (4 cœurs A76) + vectoriser sur x (NEON 128-bit)
        // vectorize(x, 16) gère automatiquement le tail (bords d'image)
        output
            .reorder(x, y, c)           // x interne (stride=1 contigu)
            .vectorize(x, 16)           // NEON : 16 uint8 = 128-bit, tail automatique
            .parallel(y);              // 4 cœurs Cortex-A76
    }
};

// Enregistrement du generator
HALIDE_REGISTER_GENERATOR(JSKPipeline, jsk_pipeline)
