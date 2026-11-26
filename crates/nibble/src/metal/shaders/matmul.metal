// Matrix-matrix products for prompt batches: a transcription of
// ggml-metal's `kernel_mul_mm` (llama.cpp
// `ggml/src/ggml-metal/kernels/mul_mm.metal`, b12723) with its
// `dequantize_*` functions (`dequantize.h`), as llama.cpp runs it on GPUs
// without the Metal 4 tensor API: tiles of 64 weight rows by 32 inputs,
// weights dequantized and inputs rounded to `half` (`bfloat` for BF16
// weights) in threadgroup memory, products accumulated in f32 simdgroup
// matrices. ggml takes it for batches of more than 8 inputs; nibble uses
// it above 16 (`MATMUL_MIN_BATCH` in `metal/mod.rs`), so that verifying a
// draft stays bit-identical to decoding.
//
// nibble only takes it for matrices whose width is a multiple of 32, so
// ggml's bounds-checked input loads (`FC_mul_mm_bc_inp`) are left out; the
// bounds-checked output path is taken for partial tiles, which is where
// ggml's `FC_mul_mm_bc_out` would send them. With `MATVEC_ADD` every tile
// takes that path and adds its result to the output (llama.cpp's separate
// `ggml_add`, the same f32 sum).

// `MatmulArgs` in `metal/kernels.rs`: `dst = W · x` (or `+=` with
// `MATVEC_ADD`) for `n` inputs of `cols` values, `rows` rows of `W`
// `row_bytes` apart; each input's `rows` outputs follow one another.
struct MatmulArgs {
    uint cols;
    uint rows;
    uint row_bytes;
    uint n;
    uint mode;
};

// ggml-metal's bfloat matrices.
typedef matrix<bfloat, 4, 4> bfloat4x4;
typedef matrix<bfloat, 2, 4> bfloat2x4;

template <typename T> struct value_type;
template <> struct value_type<float4x4> { typedef float type; };
template <> struct value_type<half4x4> { typedef half type; };
template <> struct value_type<bfloat4x4> { typedef bfloat type; };

template <typename T> struct value_type_4x4;
template <> struct value_type_4x4<float> { typedef float4x4 type; };
template <> struct value_type_4x4<half> { typedef half4x4 type; };
template <> struct value_type_4x4<bfloat> { typedef bfloat4x4 type; };

template <typename T> struct value_type_2x4;
template <> struct value_type_2x4<float> { typedef float2x4 type; };
template <> struct value_type_2x4<half> { typedef half2x4 type; };
template <> struct value_type_2x4<bfloat> { typedef bfloat2x4 type; };

template <typename T> struct value_type_8x8;
template <> struct value_type_8x8<float> { typedef simdgroup_float8x8 type; };
template <> struct value_type_8x8<half> { typedef simdgroup_half8x8 type; };
template <> struct value_type_8x8<bfloat> { typedef simdgroup_bfloat8x8 type; };

#define QK_NL 16

// --- ggml's dequantize_* (16 values of a block, part `il`) ------------------

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = *((device const uint32_t *)xb->qh);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = *((device const uint32_t *)xb->qh);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }
    reg = (type4x4) reg_f;
}

// ggml multiplies `e8m0_to_fp32(e)` by its table of halved values; nibble's
// `e8m0_half` and doubled `kvalues_mxfp4_f` give the same products (powers
// of two times small integers, all exact).
template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;
    const float d = e8m0_half(xb->e);
    const uint8_t shr = il >= 1 ? 4 : 0;
    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        reg_f[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg_f[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg_f[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg_f[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];
    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;
    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * (q[i] & mask) - ml);
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;
    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : ((il/4)>0 ? 12 : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;
    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * (q[i] & mask) - (h[i] & m ? 0 : ml));
    }
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * (q[i] & mask) - ml);
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;
    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml);
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;
    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;
    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (typename value_type<type4x4>::type) (dl0 *  ((half)(q & 0xFF))       - ml);
        reg[i][1] = (typename value_type<type4x4>::type) (dl1 * ((float)(q & 0xFF00))     - ml);
        reg[i][2] = (typename value_type<type4x4>::type) (dl2 * ((float)(q & 0xFF0000))   - ml);
        reg[i][3] = (typename value_type<type4x4>::type) (dl3 * ((float)(q & 0xFF000000)) - ml);
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f));
        reg[1][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f));
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f));
        reg[3][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f));
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]));
        reg[1][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]));
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]));
        reg[3][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]));
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = (typename value_type<type4x4>::type) (dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]));
        reg[i/4+2][i%4] = (typename value_type<type4x4>::type) (dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]));
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] & 0xf) + ml);
        reg[1][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] >>  4) + ml);
        reg[2][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] & 0xf) + ml);
        reg[3][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] >>  4) + ml);
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;
    const float d = iq1_m_scale(*xb);
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;
    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] & 0xf) + ml1);
        reg[1][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] >>  4) + ml1);
        reg[2][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] & 0xf) + ml2);
        reg[3][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] >>  4) + ml2);
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[0]]);
        reg[i][1] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[1]]);
        reg[i][2] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[2]]);
        reg[i][3] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[3]]);
    }
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    const int ib32 = il/2;
    il = il%2;
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[0]]);
        reg[i][1] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[1]]);
        reg[i][2] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[2]]);
        reg[i][3] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[3]]);
    }
}

// --- kernel_mul_mm -----------------------------------------------------------

// Threadgroup memory of `matmul_*`: the weight and input tiles (64 × 32
// and 32 × 32 values of `S0`/`S1`, at most 2 bytes each), or the 64 × 32
// f32 results of a partial tile.
#define MATMUL_SMEM 8192

template <typename S0, typename S1, typename block_q, short nl,
          void (*dequantize_func)(device const block_q *, short, thread typename value_type_4x4<S0>::type &),
          typename T1>
void matmul_impl(
        constant MatmulArgs & args,
        device const char * src0,
        device const char * src1,
        device float * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiitg,
        ushort sgitg) {
    using S0_4x4 = typename value_type_4x4<S0>::type;
    using S0_8x8 = typename value_type_8x8<S0>::type;
    using S1_2x4 = typename value_type_2x4<S1>::type;
    using S1_8x8 = typename value_type_8x8<S1>::type;
    using T1_2x4 = typename value_type_2x4<T1>::type;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + NR0*NK*sizeof(S0));

    const int ne00 = args.cols;
    const int ne0 = args.rows;
    const int ne1 = args.n;

    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // A tile at the edge of the matrix or the batch may be smaller.
    const short nr0 = (ne0 - r0 < NR0) ? (ne0 - r0) : NR0;
    const short nr1 = (ne1 - r1 < NR1) ? (ne1 - r1) : NR1;

    // Threads past the edge load the last row instead.
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;
    const short offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + (uint64_t) args.row_bytes*(r0 + lr0)) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1 + (uint64_t) 4*ne00*(r1 + lr1) + 4*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        S0_4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (args.mode != MATVEC_ADD && r0 + NR0 <= ne0 && r1 + NR1 <= ne1) {
        device float * C = dst + (r0 + 32*(sgitg & 1)) + (uint64_t) (r1 + 16*(sgitg >> 1))*ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*ne0*(i/4), ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (args.mode == MATVEC_ADD) {
            // Every thread adds a share of the tile to the output.
            for (int e = tiitg; e < NR0*NR1; e += 4*N_SIMDWIDTH) {
                const int j = e / NR0;
                const int i = e % NR0;
                if (j < nr1 && i < nr0) {
                    device float * D = dst + r0 + i + (uint64_t) (r1 + j)*ne0;
                    *D = *D + ((threadgroup float *) shmem)[j*NR0 + i];
                }
            }
        } else if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = dst + r0 + (uint64_t) (r1 + j)*ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = ((threadgroup float *) shmem) + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

#define MATMUL_KERNEL(name, S, block_q, nl, deq) \
kernel void name( \
        constant MatmulArgs & args [[buffer(0)]], \
        device const char * w [[buffer(1)]], \
        device const char * x [[buffer(2)]], \
        device float * dst [[buffer(3)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        ushort tiitg [[thread_index_in_threadgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    matmul_impl<S, S, block_q, nl, deq<typename value_type_4x4<S>::type>, float>(args, w, x, dst, shmem, tgpig, tiitg, sgitg); \
}

MATMUL_KERNEL(matmul_f32,     half,   float4x4,      1,     dequantize_f32)
MATMUL_KERNEL(matmul_f16,     half,   half4x4,       1,     dequantize_f16)
MATMUL_KERNEL(matmul_bf16,    bfloat, bfloat4x4,     1,     dequantize_bf16)
MATMUL_KERNEL(matmul_q4_0,    half,   block_q4_0,    2,     dequantize_q4_0)
MATMUL_KERNEL(matmul_q4_1,    half,   block_q4_1,    2,     dequantize_q4_1)
MATMUL_KERNEL(matmul_q5_0,    half,   block_q5_0,    2,     dequantize_q5_0)
MATMUL_KERNEL(matmul_q5_1,    half,   block_q5_1,    2,     dequantize_q5_1)
MATMUL_KERNEL(matmul_q8_0,    half,   block_q8_0,    2,     dequantize_q8_0)
MATMUL_KERNEL(matmul_mxfp4,   half,   block_mxfp4,   2,     dequantize_mxfp4)
MATMUL_KERNEL(matmul_q2_K,    half,   block_q2_K,    QK_NL, dequantize_q2_K)
MATMUL_KERNEL(matmul_q3_K,    half,   block_q3_K,    QK_NL, dequantize_q3_K)
MATMUL_KERNEL(matmul_q4_K,    half,   block_q4_K,    QK_NL, dequantize_q4_K)
MATMUL_KERNEL(matmul_q5_K,    half,   block_q5_K,    QK_NL, dequantize_q5_K)
MATMUL_KERNEL(matmul_q6_K,    half,   block_q6_K,    QK_NL, dequantize_q6_K)
MATMUL_KERNEL(matmul_iq2_xxs, half,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs)
MATMUL_KERNEL(matmul_iq2_xs,  half,   block_iq2_xs,  QK_NL, dequantize_iq2_xs)
MATMUL_KERNEL(matmul_iq2_s,   half,   block_iq2_s,   QK_NL, dequantize_iq2_s)
MATMUL_KERNEL(matmul_iq3_xxs, half,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs)
MATMUL_KERNEL(matmul_iq3_s,   half,   block_iq3_s,   QK_NL, dequantize_iq3_s)
MATMUL_KERNEL(matmul_iq1_s,   half,   block_iq1_s,   QK_NL, dequantize_iq1_s)
MATMUL_KERNEL(matmul_iq1_m,   half,   block_iq1_m,   QK_NL, dequantize_iq1_m)
MATMUL_KERNEL(matmul_iq4_nl,  half,   block_iq4_nl,  2,     dequantize_iq4_nl)
MATMUL_KERNEL(matmul_iq4_xs,  half,   block_iq4_xs,  QK_NL, dequantize_iq4_xs)
