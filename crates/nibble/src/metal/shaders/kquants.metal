// Matrix-vector products and embedding lookups for the K-quants
// (Q3_K, Q4_K, Q5_K, Q6_K).
//
// Transcriptions of ggml-metal (llama.cpp `ggml/src/ggml-metal`): the matvecs
// are the `kernel_mul_mv_q{3,4,5,6}_K_f32_impl` of `kernels/mul_mv.metal` and
// the embedding lookups use the `dequantize_q*_K` of `kernels/dequantize.h`,
// both reduced to this backend's single activation row and argument structs:
//
// - ggml's `FC_mul_mv_nsg` function constant is `a.nsg`. With one activation
//   row, ggml's `r1`/`im`/`i12`/`i13` indices and the `nb02`/`nb03`/`nb1x`
//   strides are gone: `y` is the activation and `dst` the output row.
// - ggml lets a threadgroup read its rows past the end of the matrix (its
//   buffers are padded); ours are not, so every row comes from `matrix_row`,
//   which clamps to the last row, and only rows `< a.rows` are written, as
//   ggml does. The row pointers are computed per row instead of ggml's
//   `nb01` pointer advance, which loads the same bytes.
// - ggml's output guard is `first_row + row < args.ne0`; for a mul-mat the
//   destination's `ne0` is the weight row count, so it is `a.rows` here.
// - ggml's scale pointer named `a` (Q3_K) and its `y` local (Q6_K) collide
//   with our argument names and are `s` and `yb`.
//
// The K-quant reductions are a single `simd_sum` per row, so the `shmem`
// parameter of the shared matvec signature stays unused (ggml passes
// `nullptr` for these kernels and sets no threadgroup memory).

// ggml's `get_scale_min_k4_just2` (dequantize.h): the 6-bit scale and min of
// group `j + k` (0..8) from a 12-byte Q4_K/Q5_K scale table, where the scales
// and mins of the groups 4..7 hide in the high 2 bits of the earlier bytes.
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// ggml's `dequantize_q3_K`: the 16 values `[16*il, 16*il + 16)` of a block,
// written to `reg[j/4][j%4]`, so value `j` of the part ends up there.
static inline void dequantize_q3_K(device const block_q3_K *xb, short il, thread float4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
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
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

// ggml's `dequantize_q4_K`: as q3_K above. The high nibbles of a group keep
// their place in the byte, so their scale is `d / 16` instead of a shift.
// Note the half division: ggml rounds `d / 16` to f16 first.
static inline void dequantize_q4_K(device const block_q4_K * xb, short il, thread float4x4 & reg) {
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
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// ggml's `dequantize_q5_K`: as q4_K, with the fifth bit in `qh`.
static inline void dequantize_q5_K(device const block_q5_K *xb, short il, thread float4x4 & reg) {
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
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

// ggml's `dequantize_q6_K`: the 16 values `[16*il, 16*il + 16)` of a block,
// as 4 rows of 4. The quants are unsigned 6-bit values, the `- 32` comes in
// through `ml`; the scales are signed bytes.
static inline void dequantize_q6_K(device const block_q6_K *xb, short il, thread float4x4 & reg) {
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
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// ggml's `kernel_mul_mv_q3_K_f32_impl` (NR0 = 2): lanes 0..3 take the
// super-blocks (stride 4), the other dimension of the simdgroup splits each
// 32-value slice between its two scales, high bits in `hmask`.
template <short NR0>
static inline void matvec_q3_K_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        RmsCtx rms,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short nsg = a.nsg;

    const int nb = a.cols / QK_K;

    const int first_row = ((int) tgpig.x * nsg + sgitg) * NR0;

    device const block_q3_K * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q3_K *) matrix_row(w, w2, a, first_row + row);
    }

    float yl[32];

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;          // 0 or 1
    const short il  = 2*((tid%4)/2);  // 0 or 2
    const short ir  = tid%2;
    const short l0  = 8*ir;

    // The compiler does not see that ip and il have only four combinations,
    // so, as in ggml, the masks are looked up in tables instead of computed.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = y + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[NR0] = {0.f};
    float sumf2[NR0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        for (short row = 0; row < NR0; ++row) {
            device const uint16_t * q = (device const uint16_t *)(ax[row][i].qs + q_offset);
            device const uint16_t * h = (device const uint16_t *)(ax[row][i].hmask + l0);
            device const uint16_t * s = (device const uint16_t *)(ax[row][i].scales);
            const float d_all = (float) ax[row][i].d;

            scales16[0] = s[4];
            scales16[1] = s[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = s[il+0];
            scales16[1] = s[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);
        }

        y1 += 4 * QK_K;
    }

    for (short row = 0; row < NR0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    if (tiisg == 0) {
        store_rows<NR0>(a, dst, rms, first_row, sumf1);
    }
}

MATVEC_KERNEL(matvec_q3_K, matvec_q3_K_impl, 2)

// ggml's `kernel_mul_mv_q4_K_f32_impl` (NR0 = 2): lanes 0..3 take the
// super-blocks (stride 4), the other dimension of the simdgroup splits each
// 128-value half between its four scale groups.
template <short NR0>
static inline void matvec_q4_K_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        RmsCtx rms,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short nsg = a.nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = a.cols / QK_K;

    const int first_row = ((int) tgpig.x * nsg + sgitg) * NR0;

    device const block_q4_K * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q4_K *) matrix_row(w, w2, a, first_row + row);
    }

    float yl[16];
    float yh[16];

    float sumf[NR0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0; row++) {
            device const uint16_t * sc = (device const uint16_t *) ax[row][ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *) ax[row][ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &ax[row][ib].d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }
    if (tiisg == 0) {
        store_rows<NR0>(a, dst, rms, first_row, sumf);
    }
}

MATVEC_KERNEL(matvec_q4_K, matvec_q4_K_impl, 2)


// ggml's `kernel_mul_mv_q5_K_f32_impl`, with `NR0` rows per simdgroup
// (ggml takes one; a row's arithmetic is the same): lanes 0..3 take the super-blocks (stride 4), the fifth bits in `qh` are
// split per 32-value group.
template <short NR0>
static inline void matvec_q5_K_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        RmsCtx rms,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short nsg = a.nsg;

    const int nb = a.cols / QK_K;

    const int first_row = ((int) tgpig.x * nsg + sgitg) * NR0;

    device const block_q5_K * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q5_K *) matrix_row(w, w2, a, first_row + row);
    }

    float sumf[NR0]={0.f};

    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = y + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (short row = 0; row < NR0; ++row) {
            device const uint8_t * q1 = ax[row][i].qs + q_offset;
            device const uint8_t * qh = ax[row][i].qh + l0;
            device const half * dh = &ax[row][i].d;
            device const uint16_t * s = (device const uint16_t *) ax[row][i].scales + iq;

            device const uint8_t * q2 = q1 + 64;

            sc16[0] = s[0] & kmask1;
            sc16[1] = s[2] & kmask1;
            sc16[2] = ((s[4] >> 0) & kmask2) | ((s[0] & kmask3) >> 2);
            sc16[3] = ((s[4] >> 4) & kmask2) | ((s[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y1 += 4 * QK_K;
    }

    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }
    if (tiisg == 0) {
        store_rows<NR0>(a, dst, rms, first_row, sumf);
    }
}

MATVEC_KERNEL(matvec_q5_K, matvec_q5_K_impl, 2)


// ggml's `kernel_mul_mv_q6_K_f32_impl` (NR0 = 2): two lanes take the
// super-blocks (stride 2), the other dimension of the simdgroup splits each
// 128-value half between its eight scale groups.
template <short NR0>
static inline void matvec_q6_K_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        RmsCtx rms,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short nsg = a.nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = a.cols / QK_K;

    const int first_row = ((int) tgpig.x * nsg + sgitg) * NR0;

    device const block_q6_K * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q6_K *) matrix_row(w, w2, a, first_row + row);
    }

    float sumf[NR0] = { 0.f };

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;         // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const float * yb = y + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yb[l +  0];
            yl[4*l + 1] = yb[l + 32];
            yl[4*l + 2] = yb[l + 64];
            yl[4*l + 3] = yb[l + 96];
        }

        for (short row = 0; row < NR0; ++row) {
            device const uint8_t * q1 = ax[row][i].ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = ax[row][i].qh + q_offset_h;
            device const int8_t  * sc = ax[row][i].scales + is;
            device const half    * dh = &ax[row][i].d;

            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }
    if (tiisg == 0) {
        store_rows<NR0>(a, dst, rms, first_row, sumf);
    }
}

MATVEC_KERNEL(matvec_q6_K, matvec_q6_K_impl, 2)

// `matvec_q6_K_impl` for `nc` (at most `NC`) inputs at once: each row's
// weights are decoded once, and each input's arithmetic is the same as in a
// dispatch of its own. Only 2 inputs: with fast math the compiler may group
// sums differently in wider loops (`metal/tests.rs` checks this width, and
// the backend checks it again when it starts), and 4 inputs measured slower
// than two groups of 2. Q3_K, Q4_K, Q5_K and IQ4_XS have no such kernel:
// theirs regrouped sums or spilled registers, slower than the loop over
// inputs of `MATVEC_KERNEL`.
template <short NR0, short NC>
static inline void matvec_q6_K_nc(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        RmsCtx rms,
        short nc,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short nsg = a.nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = a.cols / QK_K;

    const int first_row = ((int) tgpig.x * nsg + sgitg) * NR0;

    device const block_q6_K * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q6_K *) matrix_row(w, w2, a, first_row + row);
    }

    float sumf[NC][NR0] = { { 0.f } };

    float yl[NC][16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        for (short c = 0; c < NC; ++c) {
            device const float * yb = y + (uint64_t) min(c, (short) (nc - 1)) * a.cols + i * QK_K + y_offset;
            for (short l = 0; l < 4; ++l) {
                yl[c][4*l + 0] = yb[l +  0];
                yl[c][4*l + 1] = yb[l + 32];
                yl[c][4*l + 2] = yb[l + 64];
                yl[c][4*l + 3] = yb[l + 96];
            }
        }

        for (short row = 0; row < NR0; ++row) {
            device const uint8_t * q1 = ax[row][i].ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = ax[row][i].qh + q_offset_h;
            device const int8_t  * sc = ax[row][i].scales + is;
            device const half    * dh = &ax[row][i].d;

            for (short c = 0; c < NC; ++c) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    sums[0] += yl[c][4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                    sums[1] += yl[c][4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                    sums[2] += yl[c][4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                    sums[3] += yl[c][4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
                }

                sumf[c][row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            }
        }
    }

    simd_write_nc<NR0, NC>(a, dst, rms, sumf, nc, first_row, tiisg);
}

MATVEC_KERNEL_NC(matvec_q6_K_c2, matvec_q6_K_nc, 2, 2)

// Embedding lookups for the K-quants: thread `i` decodes part `i % 16` of
// block `i / 16`, that is the 16 values `[16*i, 16*i + 16)` of the row, and
// does nothing past `cols / 16` parts (the dispatch rounds up to whole
// threadgroups).

// `reg` holds the part's 16 values as in the dequantize functions above:
// value `j` is `reg[j/4][j%4]`, so the part is four consecutive float4s.
static inline void embed_write16(device float * x, uint64_t at, thread const float4x4 & reg) {
    device float4 * out = (device float4 *) (x + at);
    out[0] = reg[0];
    out[1] = reg[1];
    out[2] = reg[2];
    out[3] = reg[3];
}

kernel void embed_q3_K(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint i = pos.x;
    const uint t = pos.y;
    if (i < a.cols / 16) {
        device const block_q3_K * b = (device const block_q3_K *) (w + (uint64_t) embed_row(a, slot, t) * a.row_bytes) + i / 16;
        float4x4 reg;
        dequantize_q3_K(b, (short) (i % 16), reg);
        embed_write16(x + (uint64_t) t * a.cols, 16 * i, reg);
    }
}

kernel void embed_q4_K(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint i = pos.x;
    const uint t = pos.y;
    if (i < a.cols / 16) {
        device const block_q4_K * b = (device const block_q4_K *) (w + (uint64_t) embed_row(a, slot, t) * a.row_bytes) + i / 16;
        float4x4 reg;
        dequantize_q4_K(b, (short) (i % 16), reg);
        embed_write16(x + (uint64_t) t * a.cols, 16 * i, reg);
    }
}

kernel void embed_q5_K(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint i = pos.x;
    const uint t = pos.y;
    if (i < a.cols / 16) {
        device const block_q5_K * b = (device const block_q5_K *) (w + (uint64_t) embed_row(a, slot, t) * a.row_bytes) + i / 16;
        float4x4 reg;
        dequantize_q5_K(b, (short) (i % 16), reg);
        embed_write16(x + (uint64_t) t * a.cols, 16 * i, reg);
    }
}

kernel void embed_q6_K(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint i = pos.x;
    const uint t = pos.y;
    if (i < a.cols / 16) {
        device const block_q6_K * b = (device const block_q6_K *) (w + (uint64_t) embed_row(a, slot, t) * a.row_bytes) + i / 16;
        float4x4 reg;
        dequantize_q6_K(b, (short) (i % 16), reg);
        embed_write16(x + (uint64_t) t * a.cols, 16 * i, reg);
    }
}

// IQ4_XS: 8 groups of 32 with 6-bit scales (bias 32); each nibble indexes
// `kvalues_iq4nl_f`. Byte `j` of a group's 16 holds value `j` in its low
// nibble and `j + 16` in its high nibble.
typedef struct {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
} block_iq4_xs;

constant float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f,
};

// ggml's `kernel_mul_mv_iq4_xs_f32_impl` (NR0 = 2): lanes 0..15 and 16..31
// take alternate super-blocks, each lane 8 values of a group; the value
// table is in threadgroup memory (32 floats).
template <short NR0>
static inline void matvec_iq4_xs_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        RmsCtx rms,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short nsg = a.nsg;
    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int first_row = ((int) tgpig.x * nsg + sgitg) * NR0;

    device const block_iq4_xs * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq4_xs *) matrix_row(w, w2, a, first_row + row);
    }

    const int nb = a.cols / QK_K;

    const short ix = tiisg/16;  // 0 or 1
    const short it = tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = ax[row][ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }
    if (tiisg == 0) {
        store_rows<NR0>(a, dst, rms, first_row, sumf);
    }
}

MATVEC_KERNEL(matvec_iq4_xs, matvec_iq4_xs_impl, 2)


// Thread `i` writes values `[16*i, 16*i + 16)`: half `i % 2` of group
// `(i % 16) / 2` of block `i / 16`, as `dequantize_row_iq4_xs` computes
// them (`(d * scale) * value`).
kernel void embed_iq4_xs(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint i = pos.x;
    const uint t = pos.y;
    if (i < a.cols / 16) {
        device const block_iq4_xs * b = (device const block_iq4_xs *) (w + (uint64_t) embed_row(a, slot, t) * a.row_bytes) + i / 16;
        const short g = (i % 16) / 2;
        const short shift = 4 * (i % 2);
        const int ls = (((b->scales_l[g/2] >> 4*(g%2)) & 0xf) | (((b->scales_h >> 2*g) & 3) << 4)) - 32;
        const float dl = (float) b->d * (float) ls;
        float4x4 reg;
        for (short j = 0; j < 16; ++j) {
            reg[j/4][j%4] = dl * kvalues_iq4nl_f[(b->qs[16*g + j] >> shift) & 0xf];
        }
        embed_write16(x + (uint64_t) t * a.cols, 16 * i, reg);
    }
}
