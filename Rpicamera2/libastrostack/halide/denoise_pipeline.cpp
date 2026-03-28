/**
 * Halide AOT Generator — Médiane 3×3 pour JSK LIVE
 *
 * Algorithme : réseau de tri par lignes, 19 comparateurs min/max.
 *   1. Sort chaque ligne de 3 pixels  (3×3 = 9 comparateurs)
 *   2. max des mins, min des maxes    (2+2  = 4 comparateurs)
 *   3. Sort des médianes de lignes    (3 comparateurs)
 *   4. Médiane globale = médiane de   (3 comparateurs)
 *      {max_min, med_meds, min_max}
 *
 * Input  : uint8 (W, H) canal unique, row-major C-contiguous
 * Output : uint8 (W, H) médiane 3×3
 *
 * Perf cible RPi5 Cortex-A76 : ~5 ms/canal @ 3840×2160
 * (NEON 16×uint8, 4 cœurs parallèles)
 *
 * Compilation AOT :
 *   g++ denoise_pipeline.cpp -o denoise_gen \
 *       $(CXXFLAGS) $(HALIDE_DIR)/lib/libHalide_GenGen.a $(LDFLAGS)
 *   ./denoise_gen -g denoise_pipeline -o generated \
 *       -f denoise_med3x3 target=$(TARGET)
 */

#include "Halide.h"
#include <utility>
#include <tuple>

using namespace Halide;

class DenoisePipeline : public Generator<DenoisePipeline> {
public:
    // Canal unique (H, W) uint8 — chaque canal est traité séparément côté Python
    Input<Buffer<uint8_t, 2>>  input{"input"};
    Output<Buffer<uint8_t, 2>> output{"output"};

    void generate() {
        Var x("x"), y("y");

        // Bords miroir pour éviter les artefacts aux coins/bords
        Func clamped = BoundaryConditions::repeat_edge(input);

        // ── Collecte des 9 voisins 3×3 en uint16 (évite overflow) ─────────────
        // Ligne 0 (y-1)
        Expr r0a = cast<uint16_t>(clamped(x-1, y-1));
        Expr r0b = cast<uint16_t>(clamped(x,   y-1));
        Expr r0c = cast<uint16_t>(clamped(x+1, y-1));
        // Ligne 1 (y)
        Expr r1a = cast<uint16_t>(clamped(x-1, y));
        Expr r1b = cast<uint16_t>(clamped(x,   y));
        Expr r1c = cast<uint16_t>(clamped(x+1, y));
        // Ligne 2 (y+1)
        Expr r2a = cast<uint16_t>(clamped(x-1, y+1));
        Expr r2b = cast<uint16_t>(clamped(x,   y+1));
        Expr r2c = cast<uint16_t>(clamped(x+1, y+1));

        // ── sort3 : trie 3 Expr → (lo, med, hi), 3 compare-swaps ──────────────
        // cs(a,b) → {min(a,b), max(a,b)}  (1 compare-swap = 1 paire min/max)
        auto cs = [](Expr a, Expr b) -> std::pair<Expr, Expr> {
            return {min(a, b), max(a, b)};
        };
        auto sort3 = [&cs](Expr a, Expr b, Expr c)
                -> std::tuple<Expr, Expr, Expr> {
            auto [lo1, hi1] = cs(a, b);      // cs1: trie a,b
            auto [lo2, hi2] = cs(hi1, c);    // cs2: hi2 = max des 3
            auto [lo3, med] = cs(lo1, lo2);  // cs3: lo3 = min des 3
            return {lo3, med, hi2};
        };

        // ── Tri de chaque ligne (9 comparateurs) ───────────────────────────────
        auto [r0_lo, r0_med, r0_hi] = sort3(r0a, r0b, r0c);
        auto [r1_lo, r1_med, r1_hi] = sort3(r1a, r1b, r1c);
        auto [r2_lo, r2_med, r2_hi] = sort3(r2a, r2b, r2c);

        // ── Max des mins, min des maxes (4 comparateurs) ───────────────────────
        Expr max_min = max(max(r0_lo, r1_lo), r2_lo);   // max des 3 mins de ligne
        Expr min_max = min(min(r0_hi, r1_hi), r2_hi);   // min des 3 maxes de ligne

        // ── Sort des médianes de lignes (3 comparateurs) ───────────────────────
        auto [mm_lo, mm_med, mm_hi] = sort3(r0_med, r1_med, r2_med);

        // ── Médiane globale = médiane de {max_min, mm_med, min_max} (3 cs) ─────
        // Propriété : le vrai médian des 9 valeurs = médiane de ces 3 bornes
        auto [g_lo, g_med, g_hi] = sort3(max_min, mm_med, min_max);

        output(x, y) = cast<uint8_t>(g_med);
    }

    void schedule() {
        Var x = output.args()[0];
        Var y = output.args()[1];

        output
            .vectorize(x, 16)   // NEON : 16 uint8 = 128-bit par cycle
            .parallel(y);       // 4 cœurs Cortex-A76
    }
};

HALIDE_REGISTER_GENERATOR(DenoisePipeline, denoise_pipeline)
