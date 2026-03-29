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
#include "generated/lucky_score_var_hal.h"
#include "generated/lucky_score_lap_hal.h"
#include "generated/lucky_score_ten_hal.h"
#include "generated/lucky_score_grad_hal.h"
#include "generated/lucky_stack_mean_hal.h"
#include "generated/lucky_sigma_clip_hal.h"

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

} // extern "C"
