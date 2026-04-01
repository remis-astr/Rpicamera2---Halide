/**
 * Halide AOT Generator — Pipeline Lucky Stack RAW
 *
 * Deux générateurs :
 *
 * 1. LuckyPreprocess
 *    uint16 Bayer RGGB (CSI-2 ×16) + BL per-canal
 *    → float32 Bayer BL-soustrait (H, W)
 *    → float32 G1 demi-résolution (H/2, W/2)
 *
 *    Remplace dans lucky_raw.py :
 *      _apply_bl_per_channel()  → 4 ops NumPy strided
 *      _extract_g1()            → slice strided (fait deux fois : score + alignement)
 *
 *    Gain : 1 passe fusionnée NEON au lieu de 5 passes NumPy.
 *
 * 2. LuckyDebayerF32
 *    float32 Bayer RGGB (H, W) + gains AWB
 *    → float32 RGB (H, W, 3) planaire, ch0=R_phys ch1=G ch2=B_phys, valeurs [0-65535]
 *
 *    Remplace dans lucky_raw.py (debayer_bayer_stack) :
 *      np.clip + astype(uint16) + cv2.cvtColor + astype(float32) + gains × 2
 *
 *    Même convention de canaux que cv2.COLOR_BayerRG2BGR empirique (voir MEMORY.md).
 *
 * Pattern Bayer RGGB en coordonnées Halide (x=col, y=row) :
 *   R  : x%2==0, y%2==0
 *   G1 : x%2==1, y%2==0
 *   G2 : x%2==0, y%2==1
 *   B  : x%2==1, y%2==1
 *
 * Target : arm-64-linux-arm_dot_prod-arm_fp16 (RPi5 Cortex-A76)
 */

#include "Halide.h"
using namespace Halide;

// =============================================================================
// Generator 1 : LuckyPreprocess
// =============================================================================

class LuckyPreprocess : public Generator<LuckyPreprocess> {
public:
    // ── Entrée ────────────────────────────────────────────────────────────────
    // RAW12 uint16, espace CSI-2 ×16 (valeurs 0-65520)
    // Dimensions (W, H)
    Input<Buffer<uint16_t, 2>> input{"input"};

    // ── BL par canal, déjà en espace CSI-2 (ADU_12bit × 16.0) ───────────────
    // Passer 0.0 pour désactiver la correction d'un canal
    Input<float> bl_r {"bl_r",  0.0f};
    Input<float> bl_g1{"bl_g1", 0.0f};
    Input<float> bl_g2{"bl_g2", 0.0f};
    Input<float> bl_b {"bl_b",  0.0f};

    // ── Sorties ───────────────────────────────────────────────────────────────
    // Bayer BL-soustrait et clippé, float32 (W, H)
    Output<Buffer<float, 2>> bayer_out{"bayer_out"};
    // Canal G1 demi-résolution, float32 (W/2, H/2)
    Output<Buffer<float, 2>> g1_out{"g1_out"};

    void generate() {
        Var x("x"), y("y");

        // Bords miroir (évite les accès hors-buffer sur les bords impairs)
        Func clamped = BoundaryConditions::repeat_edge(input);

        // ── BL per-canal selon la parité du pixel ─────────────────────────────
        // Sélection compile-time-inlineable : select() produit un branchless NEON
        Expr even_x = (x % 2 == 0);
        Expr even_y = (y % 2 == 0);
        Expr bl = select(
             even_x &&  even_y, bl_r,    // R
            !even_x &&  even_y, bl_g1,   // G1
             even_x && !even_y, bl_g2,   // G2
            bl_b                          // B
        );

        // Conversion uint16 → float32, soustraction BL, clip à 0
        corrected(x, y) = max(cast<float>(clamped(x, y)) - bl, 0.0f);

        // ── Output 1 : Bayer complet ──────────────────────────────────────────
        bayer_out(x, y) = corrected(x, y);

        // ── Output 2 : G1 demi-résolution ────────────────────────────────────
        // G1 est à (x_odd, y_even) = (2*xh+1, 2*yh) en pleine résolution
        g1_out(x, y) = corrected(2*x + 1, 2*y);
    }

    void schedule() {
        // float32 NEON ARM64 : 4 floats par registre 128-bit
        // vectorize(x, 4) → tail automatique pour W non multiple de 4
        // parallel(y)     → 4 cœurs Cortex-A76

        // corrected.compute_root() : matérialise la conversion uint16→float32 + BL
        // en un seul buffer (H,W) float32. Sans cela, Halide l'inline dans chaque
        // sortie → les pixels G1 (x%2==1, y%2==0, soit 25% du total) sont calculés
        // deux fois : une fois pour bayer_out, une fois pour g1_out.
        corrected
            .compute_root()
            .vectorize(x, 4)
            .parallel(y);

        bayer_out
            .vectorize(x, 4)
            .parallel(y);

        g1_out
            .vectorize(x, 4)
            .parallel(y);
    }

private:
    Var  x{"x"}, y{"y"};
    Func corrected{"corrected"};   // membre pour accès depuis schedule()
};

// =============================================================================
// Generator 2 : LuckyDebayerF32
// =============================================================================

class LuckyDebayerF32 : public Generator<LuckyDebayerF32> {
public:
    // ── Entrée ────────────────────────────────────────────────────────────────
    // Stack Bayer float32 (W, H), BL déjà soustrait, valeurs [0, ~65535]
    Input<Buffer<float, 2>> input{"input"};

    // ── Gains AWB (multiplicateurs directs en float) ──────────────────────────
    // r_gain appliqué à ch0 (R_physique), b_gain à ch2 (B_physique)
    // g_gain = 1.0 (canal vert inchangé)
    Input<float> r_gain{"r_gain", 1.0f};
    Input<float> b_gain{"b_gain", 1.0f};

    // ── Sortie ────────────────────────────────────────────────────────────────
    // RGB float32 planaire (W, H, 3) :
    //   dim[0]=x stride=1, dim[1]=y stride=W, dim[2]=c stride=W*H
    // Côté Python : allouer (3, H, W) float32 C-contiguous → transpose(1,2,0) zero-copy
    // Convention canaux : ch0=R_phys, ch1=G, ch2=B_phys
    //   (identique au résultat de cv2.COLOR_BayerRG2BGR sur IMX585, voir MEMORY.md)
    Output<Buffer<float, 3>> output{"output"};

    void generate() {
        Var x("x"), y("y"), c("c");

        // Bords miroir pour éviter les artefacts aux bords de l'image
        Func clamped = BoundaryConditions::mirror_image(input);

        // ── Débayérisation bilinéaire RGGB ────────────────────────────────────
        // Pattern Bayer RGGB en coords Halide (x=col, y=row) :
        //   R  : even_x && even_y
        //   G1 : odd_x  && even_y
        //   G2 : even_x && odd_y
        //   B  : odd_x  && odd_y
        Expr even_x = (x % 2 == 0);
        Expr even_y = (y % 2 == 0);

        // Valeur centrale + voisins directs + diagonaux
        Expr m   = clamped(x,   y  );
        Expr mN  = clamped(x,   y-1);
        Expr mS  = clamped(x,   y+1);
        Expr mW  = clamped(x-1, y  );
        Expr mE  = clamped(x+1, y  );
        Expr mNW = clamped(x-1, y-1);
        Expr mNE = clamped(x+1, y-1);
        Expr mSW = clamped(x-1, y+1);
        Expr mSE = clamped(x+1, y+1);

        Expr diag4  = (mNW + mNE + mSW + mSE) * 0.25f;
        Expr cross4 = (mN  + mS  + mW  + mE ) * 0.25f;
        Expr horiz2 = (mW  + mE ) * 0.5f;
        Expr vert2  = (mN  + mS ) * 0.5f;

        // Canal R (ch0 = R_physique)
        Func R_ch("R_ch");
        R_ch(x, y) = select(
             even_x &&  even_y, m,       // site R  : valeur directe
            !even_x &&  even_y, horiz2,  // site G1 : interpolation horizontale
             even_x && !even_y, vert2,   // site G2 : interpolation verticale
            diag4                         // site B  : interpolation diagonale
        );

        // Canal G (ch1)
        Func G_ch("G_ch");
        G_ch(x, y) = select(
             even_x &&  even_y, cross4,  // site R  : interpolation en croix
            !even_x &&  even_y, m,       // site G1 : valeur directe
             even_x && !even_y, m,       // site G2 : valeur directe
            cross4                        // site B  : interpolation en croix
        );

        // Canal B (ch2 = B_physique)
        Func B_ch("B_ch");
        B_ch(x, y) = select(
             even_x &&  even_y, diag4,   // site R  : interpolation diagonale
            !even_x &&  even_y, vert2,   // site G1 : interpolation verticale
             even_x && !even_y, horiz2,  // site G2 : interpolation horizontale
            m                             // site B  : valeur directe
        );

        // ── Application gains AWB + clip [0, 65535] ───────────────────────────
        output(x, y, c) = clamp(
            select(
                c == 0, R_ch(x, y) * r_gain,  // R_phys × r_gain
                c == 1, G_ch(x, y),            // G inchangé
                        B_ch(x, y) * b_gain    // B_phys × b_gain
            ),
            0.0f, 65535.0f
        );
    }

    void schedule() {
        Var x = output.args()[0];
        Var y = output.args()[1];
        Var c = output.args()[2];

        // Layout planaire (défaut Halide) : dim[0]=x stride=1, dim[2]=c stride=W*H
        // reorder(x, y, c) → x interne (accès contigu), c externe
        output
            .reorder(x, y, c)
            .vectorize(x, 4)   // float32 NEON : 4 floats = 128-bit
            .parallel(y);      // 4 cœurs Cortex-A76
    }

private:
    Var x{"x"}, y{"y"}, c{"c"};
};

// =============================================================================
// Generator 3 : LsHotPixelRemoval
// =============================================================================
/**
 * Suppression des pixels chauds sur image Bayer float32 (live stack RAW).
 *
 * Algorithme :
 *   Pour chaque pixel (x, y), les 4 voisins de même couleur Bayer sont
 *   à (x±2, y) et (x, y±2) — même parité en x et y, donc même canal RGGB.
 *   On calcule la médiane exacte de ces 4 voisins via un réseau de tri 4 éléments.
 *   Si le pixel dépasse  threshold × max(med4, abs_floor)  il est remplacé par med4.
 *
 * Paramètres :
 *   threshold (float, défaut 5.0) — ratio de détection.
 *     Valeurs typiques :
 *       3.0  = aggressif (peut toucher des étoiles serrées en mauvais seeing)
 *       5.0  = standard  (sûr pour la plupart des cibles)
 *       10.0 = conservateur (uniquement pixels très chauds)
 *   abs_floor (float, défaut 100.0) — plancher absolu pour la référence en
 *     espace float32 BL-soustrait CSI-2 ×16 (≈ 6 ADU 12-bit).
 *     Empêche les faux positifs quand med4 ≈ 0 (fond noir).
 *
 * Convention mémoire : même que LuckyPreprocess — float32 (W, H) row-major.
 *
 * Gain vs NumPy (RPi5, 1090×1928) :
 *   ×4-6 — accès neighbourhood ±2 en NEON, 4 cœurs parallèles.
 */
class LsHotPixelRemoval : public Generator<LsHotPixelRemoval> {
public:
    // ── Entrée ────────────────────────────────────────────────────────────────
    // Bayer float32 BL-soustrait (W, H) — sortie de LuckyPreprocess
    Input<Buffer<float, 2>> input     {"input"};

    // ── Paramètres ────────────────────────────────────────────────────────────
    Input<float> threshold{"threshold", 5.0f};   // ratio de détection
    Input<float> abs_floor{"abs_floor", 100.0f}; // plancher absolu (CSI-2 ×16)

    // ── Sortie ────────────────────────────────────────────────────────────────
    Output<Buffer<float, 2>> output{"output"};

    void generate() {
        Var x("x"), y("y");

        // Bords miroir : répétition du bord pour éviter les accès hors-buffer
        // (les 4 pixels de bord par côté sont susceptibles d'avoir des voisins
        //  à ±2 en dehors du buffer)
        Func clamped = BoundaryConditions::repeat_edge(input);

        // ── 4 voisins de même couleur Bayer (distance ±2) ─────────────────────
        // Valides pour tout pixel (x,y) quel que soit son canal RGGB :
        //   R  (even_x, even_y) : voisins à (x±2,y) → even_x±2 ✓, (x,y±2) → even_y±2 ✓
        //   G1 (odd_x,  even_y) : idem avec odd_x±2 ✓
        //   G2 (even_x, odd_y)  : idem avec odd_y±2 ✓
        //   B  (odd_x,  odd_y)  : idem ✓
        Expr n_N = clamped(x,   y-2);
        Expr n_S = clamped(x,   y+2);
        Expr n_W = clamped(x-2, y  );
        Expr n_E = clamped(x+2, y  );

        // ── Médiane exacte de 4 valeurs — réseau de tri optimal ───────────────
        // Étape 1 : trier 2 paires
        Expr lo_NS = min(n_N, n_S),  hi_NS = max(n_N, n_S);
        Expr lo_WE = min(n_W, n_E),  hi_WE = max(n_W, n_E);
        // Étape 2 : médiane = (2e + 3e élément) / 2  après fusion des paires
        //   2e = max(lo_NS, lo_WE)   3e = min(hi_NS, hi_WE)
        Expr med4 = (max(lo_NS, lo_WE) + min(hi_NS, hi_WE)) * 0.5f;

        // ── Détection pixel chaud ──────────────────────────────────────────────
        // Référence = max(med4, abs_floor) pour éviter les faux positifs quand
        // med4 ≈ 0 (fond noir parfaitement soustrait) : sans plancher,
        // threshold × 0 = 0 flaguerait toute intensité non nulle.
        Expr ref    = max(med4, abs_floor);
        Expr is_hot = input(x, y) > threshold * ref;

        // Remplacement par la médiane des voisins si pixel chaud
        output(x, y) = select(is_hot, med4, input(x, y));
    }

    void schedule() {
        // Pas de Func intermédiaire à matérialiser : les accès ±2 en mémoire
        // contiguë sont vectorisables directement (gather NEON 4×float32).
        output
            .vectorize(x, 4)
            .parallel(y);
    }

private:
    Var x{"x"}, y{"y"};
};

// =============================================================================
HALIDE_REGISTER_GENERATOR(LuckyPreprocess,    lucky_preprocess)
HALIDE_REGISTER_GENERATOR(LuckyDebayerF32,   lucky_debayer_f32)
HALIDE_REGISTER_GENERATOR(LsHotPixelRemoval, ls_hot_pixel_removal)
