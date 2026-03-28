/**
 * Wrapper C pour les pipelines JSK Halide AOT.
 * Expose des fonctions appelables via Python ctypes.
 *
 * Signature Python-friendly :
 *   input  : uint16_t* (H × W, row-major)
 *   output : uint8_t*  (H × W × 3, row-major RGB)
 *   w, h   : dimensions
 *   r_gain, g_gain, b_gain : gains AWB × 256 (256 = ×1.0)
 *   w0, w1, w2 : poids HDR (0-100)
 */

#include "HalideBuffer.h"
#include "generated/jsk_pipeline_med3.h"
#include "generated/jsk_pipeline_mean3.h"
#include "generated/jsk_pipeline_med2.h"
#include "generated/denoise_med3x3.h"
#include "generated/guided_filter.h"

using HBuf16 = Halide::Runtime::Buffer<uint16_t, 2>;
using HBuf8  = Halide::Runtime::Buffer<uint8_t,  3>;
using HBuf8c = Halide::Runtime::Buffer<uint8_t,  2>;

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

} // extern "C"
