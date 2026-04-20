/**
 * Wrapper C pour les pipelines JSK et Lucky RAW Halide AOT.
 * Expose des fonctions appelables via Python ctypes.
 *
 * JSK (HDR + débayer uint8) :
 *   input  : uint16_t* (H × W, row-major)
 *   output : uint8_t*  (H × W × 3, row-major RGB)
 *   w, h   : dimensions
 *   r_gain, g_gain, b_gain : gains AWB × 256 (256 = ×1.0)
 *   w0, w1, w2 : poids HDR (0-100)
 *
 * Lucky RAW (BL-subtract + G1 + débayer float32) :
 *   lucky_raw_preprocess : uint16 → float32 Bayer + float32 G1 half-res
 *   lucky_debayer_f32    : float32 Bayer → float32 RGB planaire
 */

#include "HalideBuffer.h"
#include "generated/jsk_pipeline_med3.h"
#include "generated/jsk_pipeline_mean3.h"
#include "generated/jsk_pipeline_med2.h"
#include "generated/denoise_med3x3.h"
#include "generated/guided_filter.h"
#include "generated/lucky_preprocess_hal.h"
#include "generated/lucky_debayer_hal.h"
#include "generated/ls_hot_pixel_removal_hal.h"
#include "generated/lucky_score_var_hal.h"
#include "generated/lucky_score_lap_hal.h"
#include "generated/lucky_score_ten_hal.h"
#include "generated/lucky_score_grad_hal.h"
#include "generated/lucky_stack_mean_hal.h"
#include "generated/lucky_sigma_clip_hal.h"
#include "generated/ls_stack_mean_frame_hal.h"
#include "generated/ls_stack_kappa_frame_hal.h"
#include "generated/ls_apply_calibration_hal.h"
#include "generated/ls_poly_bg_apply_hal.h"

using HBuf16 = Halide::Runtime::Buffer<uint16_t, 2>;
using HBuf8  = Halide::Runtime::Buffer<uint8_t,  3>;
using HBuf8c = Halide::Runtime::Buffer<uint8_t,  2>;
using HBufF2 = Halide::Runtime::Buffer<float,    2>;
using HBufF3 = Halide::Runtime::Buffer<float,    3>;

extern "C" {

/**
 * Pipeline HDR Median 3 niveaux (12+11+10 bits).
 * Méthode la plus efficace pour le contraste solaire/lunaire.
 *
 * @param input   Bayer RAW12 uint16, shape (H, W), row-major
 * @param output  RGB uint8, shape (H, W, 3), row-major
 * @param w, h    Dimensions image (W=colonnes, H=lignes)
 * @param r_gain  Gain rouge  × 256 (ex: 384 = ×1.5)
 * @param g_gain  Gain vert   × 256
 * @param b_gain  Gain bleu   × 256
 * @param w0,w1,w2 Poids HDR niveaux 12/11/10 bits (0-100)
 * @return 0 si succès
 */
// ─── Helper layout ───────────────────────────────────────────────────────────
// Layout planaire (défaut Halide) : dim[0]=x stride=1, dim[1]=y stride=W, dim[2]=c stride=W*H
// Python alloue (3,H,W) uint8 C-contiguous → correspond exactement au planaire Halide
// Conversion en (H,W,3) : out.transpose(1,2,0) côté Python (zéro-copie metadata)
static HBuf8 make_out(uint8_t* ptr, int w, int h) {
    return HBuf8(ptr, w, h, 3);    // planaire : strides (1, W, W*H)
}

int jsk_hdr_median3(
    uint16_t* input, uint8_t* output,
    int w, int h,
    int r_gain, int g_gain, int b_gain,
    int w0, int w1, int w2)
{
    HBuf16 in_buf(input, w, h);
    HBuf8  out_buf = make_out(output, w, h);
    return jsk_pipeline_med3(
        in_buf, r_gain, g_gain, b_gain, w0, w1, w2, out_buf);
}

/**
 * Pipeline HDR Mean pondéré 3 niveaux.
 */
int jsk_hdr_mean3(
    uint16_t* input, uint8_t* output,
    int w, int h,
    int r_gain, int g_gain, int b_gain,
    int w0, int w1, int w2)
{
    HBuf16 in_buf(input, w, h);
    HBuf8  out_buf = make_out(output, w, h);
    return jsk_pipeline_mean3(
        in_buf, r_gain, g_gain, b_gain, w0, w1, w2, out_buf);
}

/**
 * Pipeline HDR Median 2 niveaux (12+11 bits).
 */
int jsk_hdr_median2(
    uint16_t* input, uint8_t* output,
    int w, int h,
    int r_gain, int g_gain, int b_gain,
    int w0, int w1)
{
    HBuf16 in_buf(input, w, h);
    HBuf8  out_buf = make_out(output, w, h);
    return jsk_pipeline_med2(
        in_buf, r_gain, g_gain, b_gain, w0, w1, 0, out_buf);
}

/**
 * Guided filter (He et al., 2013) sur image RGB planaire (W, H, 3) uint8.
 * Auto-guidage par canal (guide = input).
 * Box filter séparable H+V → O(N), radius=8 (fenêtre 17×17).
 *
 * @param input   uint8_t* planaire (3,H,W) C-contiguous — dim[0]=x stride=1
 * @param output  uint8_t* même layout
 * @param w, h    Dimensions image
 * @param eps     Régularisation, espace [0, 255²] (ex: 200.0 = modéré)
 * @return 0 si succès
 */
int jsk_denoise_guided(uint8_t* input, uint8_t* output, int w, int h, float eps)
{
    HBuf8 in_buf (input,  w, h, 3);
    HBuf8 out_buf(output, w, h, 3);
    return guided_filter(in_buf, eps, out_buf);
}

/**
 * Médiane 3×3 sur un canal uint8 2D.
 * Appelé 3× depuis Python (canaux R, G, B séparément).
 *
 * @param input   uint8_t* canal 2D (H, W), row-major C-contiguous
 * @param output  uint8_t* même dimensions, résultat médiane 3×3
 * @param w, h    Dimensions image
 * @return 0 si succès
 */
int jsk_denoise_median3x3(uint8_t* input, uint8_t* output, int w, int h)
{
    HBuf8c in_buf(input,  w, h);
    HBuf8c out_buf(output, w, h);
    return denoise_med3x3(in_buf, out_buf);
}

// =============================================================================
// Lucky Stack RAW — pipelines float32
// =============================================================================

/**
 * Prétraitement per-frame : BL subtract + G1 extraction.
 *
 * Remplace _apply_bl_per_channel() + _extract_g1() dans lucky_raw.py.
 * 1 passe Halide fusionnée (NEON vectorisé, 4 cœurs) au lieu de 5 passes NumPy.
 *
 * @param input     Bayer RAW12 uint16, shape (H, W), row-major
 * @param bayer_out float32, shape (H, W), row-major — Bayer BL-soustrait, clippé ≥ 0
 * @param g1_out    float32, shape (H/2, W/2), row-major — canal G1 demi-résolution
 * @param w, h      Dimensions image (W=colonnes, H=lignes)
 * @param bl_r      Black level R  en espace CSI-2 (ADU_12bit × 16.0)
 * @param bl_g1     Black level G1 en espace CSI-2
 * @param bl_g2     Black level G2 en espace CSI-2
 * @param bl_b      Black level B  en espace CSI-2
 * @return 0 si succès
 */
int lucky_raw_preprocess(
    uint16_t* input, float* bayer_out, float* g1_out,
    int w, int h,
    float bl_r, float bl_g1, float bl_g2, float bl_b)
{
    HBuf16 in_buf(input, w, h);
    HBufF2 bayer_buf(bayer_out, w, h);
    HBufF2 g1_buf(g1_out, w/2, h/2);
    return lucky_preprocess_hal(in_buf, bl_r, bl_g1, bl_g2, bl_b, bayer_buf, g1_buf);
}

/**
 * Suppression des pixels chauds sur image Bayer float32.
 *
 * Pour chaque pixel, calcule la médiane exacte des 4 voisins de même couleur
 * Bayer (à ±2 en x et y). Remplace le pixel si
 *   pixel > threshold × max(med4, abs_floor)
 *
 * Voisins à ±2 garantissent qu'on compare toujours R↔R, G1↔G1, G2↔G2, B↔B
 * sans mélanger les canaux Bayer.
 *
 * @param input      float32* (H, W) Bayer BL-soustrait (sortie de lucky_raw_preprocess)
 * @param output     float32* (H, W) même layout — pixels chauds remplacés
 * @param w, h       Dimensions image
 * @param threshold  Ratio de détection (5.0 = standard, 3.0 = agressif, 10.0 = conservateur)
 * @param abs_floor  Plancher absolu pour la référence en espace CSI-2 ×16 (défaut 100.0)
 *                   Empêche les faux positifs quand med4 ≈ 0 (fond parfaitement soustrait)
 * @return 0 si succès
 */
int ls_hot_pixel_removal(
    float* input, float* output,
    int w, int h,
    float threshold, float abs_floor)
{
    HBufF2 in_buf (input,  w, h);
    HBufF2 out_buf(output, w, h);
    return ls_hot_pixel_removal_hal(in_buf, threshold, abs_floor, out_buf);
}

/**
 * Débayérisation float32 du stack final + gains AWB.
 *
 * Remplace np.clip + astype(uint16) + cv2.cvtColor + astype(float32) + gains
 * dans debayer_bayer_stack() de lucky_raw.py.
 *
 * Convention canaux (identique à cv2.COLOR_BayerRG2BGR empirique sur IMX585) :
 *   ch0 = R_physique × r_gain
 *   ch1 = G (inchangé)
 *   ch2 = B_physique × b_gain
 *
 * @param input     Bayer float32, shape (H, W), row-major, valeurs [0, ~65535]
 * @param output    float32 planaire, forme (3, H, W) C-contiguous allouée côté Python
 *                  → transposer en (H, W, 3) zero-copy : out.transpose(1,2,0)
 * @param w, h      Dimensions image
 * @param r_gain    Gain rouge  (ex: 1.8 pour AWB)
 * @param b_gain    Gain bleu   (ex: 1.4 pour AWB)
 * @return 0 si succès
 */
int lucky_debayer_f32(
    float* input, float* output,
    int w, int h,
    float r_gain, float b_gain)
{
    HBufF2 in_buf(input, w, h);
    HBufF3 out_buf(output, w, h, 3);
    return lucky_debayer_hal(in_buf, r_gain, b_gain, out_buf);
}

// =============================================================================
// Lucky Stack RAW — Scoring (4 méthodes, chacune en passe unique fusionnée)
// =============================================================================

/**
 * Scorer générique : helper interne.
 * g1    : float32* (H/2, W/2) G1 half-res, row-major
 * score : float32* pointeur vers 1 élément (résultat)
 * w2,h2 : dimensions G1 (W/2, H/2)
 * roi_* : ROI en coordonnées G1
 */
static inline HBufF2 make_g1_buf(float* g1, int w2, int h2) {
    return HBufF2(g1, w2, h2);
}

using HBufF1 = Halide::Runtime::Buffer<float, 1>;
static inline HBufF1 make_score_buf(float* out) {
    return HBufF1(out, 1);
}

/**
 * Variance locale sur la ROI (1 passe, Tuple 2 accumulateurs).
 * @return 0 si succès
 */
int lucky_score_variance(float* g1, float* score_out,
                         int w2, int h2,
                         int roi_x0, int roi_y0, int roi_w, int roi_h)
{
    auto g1b  = make_g1_buf(g1, w2, h2);
    auto sb   = make_score_buf(score_out);
    return lucky_score_var_hal(g1b, roi_x0, roi_y0, roi_w, roi_h, sb);
}

/**
 * Variance du Laplacien 5-pts sur la ROI (1 passe fusionnée).
 */
int lucky_score_laplacian(float* g1, float* score_out,
                          int w2, int h2,
                          int roi_x0, int roi_y0, int roi_w, int roi_h)
{
    auto g1b  = make_g1_buf(g1, w2, h2);
    auto sb   = make_score_buf(score_out);
    return lucky_score_lap_hal(g1b, roi_x0, roi_y0, roi_w, roi_h, sb);
}

/**
 * Tenengrad E[Gx²+Gy²] sur la ROI (1 passe, Gx/Gy en registres).
 */
int lucky_score_tenengrad(float* g1, float* score_out,
                          int w2, int h2,
                          int roi_x0, int roi_y0, int roi_w, int roi_h)
{
    auto g1b  = make_g1_buf(g1, w2, h2);
    auto sb   = make_score_buf(score_out);
    return lucky_score_ten_hal(g1b, roi_x0, roi_y0, roi_w, roi_h, sb);
}

/**
 * Gradient moyen E[√(Gx²+Gy²)] sur la ROI (1 passe).
 */
int lucky_score_gradient(float* g1, float* score_out,
                         int w2, int h2,
                         int roi_x0, int roi_y0, int roi_w, int roi_h)
{
    auto g1b  = make_g1_buf(g1, w2, h2);
    auto sb   = make_score_buf(score_out);
    return lucky_score_grad_hal(g1b, roi_x0, roi_y0, roi_w, roi_h, sb);
}

// =============================================================================
// Lucky Stack RAW — Stack moyen et sigma-clipping (N,H,W) → (H,W)
// =============================================================================

/**
 * Moyenne simple de N frames float32.
 *
 * Convention mémoire : NumPy (N,H,W) C-contiguous float32
 *   → Halide Buffer<float,3>(ptr, W, H, N)  dim[0].stride=1 (x), dim[1].stride=W, dim[2].stride=W*H
 *
 * @param frames     float32* (N,H,W) C-contiguous — frames Bayer alignées
 * @param result_out float32* (H,W) C-contiguous  — stack résultat
 * @param w, h       Dimensions image
 * @param n_frames   Nombre de frames à empiler
 * @return 0 si succès
 */
int lucky_stack_mean(
    float* frames, float* result_out,
    int w, int h, int n_frames)
{
    HBufF3 frames_buf(frames, w, h, n_frames);
    HBufF2 result_buf(result_out, w, h);
    return lucky_stack_mean_hal(frames_buf, n_frames, result_buf);
}

/**
 * Sigma-clipping pixel-wise : 2 passes fusionnées, sans buffer (N,H,W) intermédiaire.
 *
 * Passe 1 : E[X] et σ(X) via Tuple (ΣX, ΣX²) en une seule RDom.
 * Passe 2 : moyenne des pixels dans |X - E[X]| ≤ kappa·σ(X).
 *
 * @param frames     float32* (N,H,W) C-contiguous
 * @param result_out float32* (H,W) C-contiguous
 * @param w, h       Dimensions image
 * @param n_frames   Nombre de frames (≥ 2, fallback mean si < 3 côté Python)
 * @param kappa      Seuil de rejet sigma (2.0–3.0 typique)
 * @return 0 si succès
 */
int lucky_sigma_clip(
    float* frames, float* result_out,
    int w, int h, int n_frames, float kappa)
{
    HBufF3 frames_buf(frames, w, h, n_frames);
    HBufF2 result_buf(result_out, w, h);
    return lucky_sigma_clip_hal(frames_buf, n_frames, kappa, result_buf);
}

// =============================================================================
// Live Stack — mise à jour incrémentale mean et kappa-sigma RGB (H,W,3)
// =============================================================================
// Convention mémoire : NumPy (3,H,W) C-contiguous float32
//   → Halide Buffer<float,3>(ptr, W, H, 3)
//   → buf(x,y,c) : dim[0] stride=1(x), dim[1] stride=W(y), dim[2] stride=W*H(c)
//
// Planar layout identique à lucky_debayer_hal output.

/**
 * Mise à jour incrémentale de moyenne RGB (une frame).
 *
 * Remplace _stack_mean() dans stacker.py (5 ops NumPy → 1 passe NEON).
 *
 * @param stacked_in   float32* (3,H,W) C-contiguous — moyenne courante
 * @param cnt_in       float32* (H,W)   C-contiguous — compteur per-pixel
 * @param img          float32* (3,H,W) C-contiguous — nouvelle frame
 * @param stacked_out  float32* (3,H,W) C-contiguous — nouvelle moyenne
 * @param cnt_out      float32* (H,W)   C-contiguous — nouveau compteur
 * @param w, h         Dimensions image
 * @return 0 si succès
 */
int ls_stack_mean_frame(
    float* stacked_in, float* cnt_in, float* img,
    float* stacked_out, float* cnt_out,
    int w, int h)
{
    HBufF3 si_buf(stacked_in,  w, h, 3);
    HBufF2 ci_buf(cnt_in,      w, h);
    HBufF3 im_buf(img,         w, h, 3);
    HBufF3 so_buf(stacked_out, w, h, 3);
    HBufF2 co_buf(cnt_out,     w, h);
    return ls_stack_mean_frame_hal(si_buf, ci_buf, im_buf, so_buf, co_buf);
}

/**
 * Mise à jour Welford + rejet kappa-sigma RGB (une frame).
 *
 * Remplace _stack_kappa() dans stacker.py (15 ops NumPy → 1 passe NEON).
 * Pixel rejeté si |img - mean| > kappa × std sur AU MOINS un canal.
 *
 * @param stacked_in   float32* (3,H,W) — moyenne courante
 * @param cnt_in       float32* (H,W)   — compteur per-pixel
 * @param m2_in        float32* (3,H,W) — variance M2 Welford
 * @param img          float32* (3,H,W) — nouvelle frame
 * @param kappa        seuil sigma (ex: 2.5)
 * @param stacked_out  float32* (3,H,W) — nouvelle moyenne
 * @param cnt_out      float32* (H,W)
 * @param m2_out       float32* (3,H,W) — nouveau M2
 * @param w, h         Dimensions image
 * @return 0 si succès
 */
int ls_stack_kappa_frame(
    float* stacked_in, float* cnt_in, float* m2_in, float* img,
    float  kappa,
    float* stacked_out, float* cnt_out, float* m2_out,
    int w, int h)
{
    HBufF3 si_buf(stacked_in,  w, h, 3);
    HBufF2 ci_buf(cnt_in,      w, h);
    HBufF3 m2i_buf(m2_in,      w, h, 3);
    HBufF3 im_buf(img,         w, h, 3);
    HBufF3 so_buf(stacked_out, w, h, 3);
    HBufF2 co_buf(cnt_out,     w, h);
    HBufF3 m2o_buf(m2_out,     w, h, 3);
    return ls_stack_kappa_frame_hal(si_buf, ci_buf, m2i_buf, im_buf, kappa,
                                    so_buf, co_buf, m2o_buf);
}

// =============================================================================
// Live Stack — correction calibration combinée (fond additif + vignetage)
// =============================================================================

/**
 * Applique fond additif et vignetage en une seule passe NEON.
 *
 * result = clamp((frame - bg) / max(flat, 0.01), 0, 1)
 *
 * @param frame   float32* (3,H,W) C-contiguous planaire — image normalisée [0,1]
 * @param bg      float32* (3,H,W) C-contiguous planaire — fond additif [0,1]
 * @param flat    float32* (H,W)   C-contiguous — vignetage achromatique (≤ 1 au centre)
 * @param result  float32* (3,H,W) C-contiguous planaire — image corrigée [0,1]
 * @param w, h    Dimensions image
 * @return 0 si succès
 */
int ls_apply_calibration(
    float* frame, float* bg, float* flat,
    float* result,
    int w, int h)
{
    HBufF3 frame_buf (frame,  w, h, 3);
    HBufF3 bg_buf    (bg,     w, h, 3);
    HBufF2 flat_buf  (flat,   w, h);
    HBufF3 result_buf(result, w, h, 3);
    return ls_apply_calibration_hal(frame_buf, bg_buf, flat_buf, result_buf);
}

// =============================================================================
// Live Stack — évaluation polynôme 2D + calibration fusionnées (1 passe NEON)
// =============================================================================

/**
 * Évalue un polynôme 2D fond (3 canaux) + flat achromatique, applique la
 * correction calibration en une seule passe NEON vectorisée.
 *
 * Remplace _eval_poly_full() × 3 + _halide_apply_calibration dans session.py.
 *
 * @param frame      float32* (3,H,W) C-contiguous planaire — image [0,1]
 * @param coeffs_bg  float32* (15,3)  C-contiguous — coefficients BG par canal
 *                   Ordre monomials : [1,x,y,x²,xy,y²,x³,x²y,xy²,y³,x⁴,x³y,x²y²,xy³,y⁴]
 *                   Termes inutilisés (degré < 4) = 0.0
 * @param coeff_fl   float32* (15,)   C-contiguous — coefficients flat achromatique
 *                   coeff_fl[0] = 1.0 au centre par construction (normalisé)
 * @param alpha      flat_strength / 100.0  (0 = BG seul, 1 = vignetage complet)
 * @param result     float32* (3,H,W) C-contiguous planaire — résultat [0,1]
 * @param w, h       Dimensions image
 * @return 0 si succès
 */
int ls_poly_bg_apply(
    float* frame, float* coeffs_bg, float* coeff_fl, float alpha,
    float* result,
    int w, int h)
{
    HBufF3 frame_buf  (frame,     w, h, 3);
    HBufF3 coeffs_buf (coeffs_bg, 15, 3, 1);  // (15,3) — dim[0]=term, dim[1]=channel
    HBufF2 coeff_fl_buf(coeff_fl, 15, 1);      // (15,1) → Buffer<float,2> dim[1]=1
    HBufF3 result_buf (result,    w, h, 3);

    // Halide attend Buffer<float,1> pour coeff_fl — wrappé en 2D avec extent[1]=1
    // On crée un Buffer<float,1> manuellement
    Halide::Runtime::Buffer<float, 1> fl1(coeff_fl, 15);
    Halide::Runtime::Buffer<float, 2> bg2(coeffs_bg, 15, 3);

    return ls_poly_bg_apply_hal(frame_buf, bg2, fl1, alpha, result_buf);
}

} // extern "C"
