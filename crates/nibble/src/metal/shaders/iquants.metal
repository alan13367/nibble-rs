// The weight formats beyond the K-quants: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
// `IQ4_NL` and `MXFP4` (blocks of 32), `Q2_0` (blocks of 64), `Q2_K`, and
// the codebook I-quants.
//
// The matrix-vector kernels transcribe ggml-metal's
// (`kernel_mul_mv_*_f32_impl` in `mul_mv.metal`) with its simdgroup layout,
// rows per simdgroup and arithmetic, except that each row is addressed
// through `matrix_row` (for `MATVEC_SWIGLU`'s pairs), the I-quants always
// take ggml's layout for wide rows (no `FC_mul_mv_split`), and `MXFP4`
// decodes through a value table as `IQ4_NL` does. The codebooks
// (`iq2xxs_grid` and the rest) are generated from `quant::iq_grids` when the
// shaders are compiled. Every kernel is compared with the CPU reference in
// `metal/tests.rs`.

#define QK4_0 32
#define QK2_0 64

typedef struct { half d; uint8_t qs[16]; } block_q4_0;
typedef struct { half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
typedef struct { half d; half m; uint8_t qs[16]; } block_q4_1;
typedef struct { half d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;
typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1;
typedef struct { half d; uint8_t qs[16]; } block_iq4_nl;
typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;
typedef struct { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; half d; half dmin; } block_q2_K;
typedef struct { half d; uint16_t qs[QK_K / 8]; } block_iq2_xxs;
typedef struct { half d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; } block_iq2_xs;
typedef struct { half d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t scales[QK_K / 32]; } block_iq2_s;
typedef struct { half d; uint8_t qs[3 * QK_K / 8]; } block_iq3_xxs;
typedef struct { half d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t signs[QK_K / 8]; uint8_t scales[QK_K / 64]; } block_iq3_s;
typedef struct { half d; uint8_t qs[QK_K / 8]; uint16_t qh[QK_K / 32]; } block_iq1_s;
typedef struct { uint8_t qs[QK_K / 8]; uint8_t qh[QK_K / 16]; uint8_t scales[QK_K / 32]; } block_iq1_m;

#define IQ1_DELTA 0.125f

constant uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

constant float kvalues_mxfp4_f[16] = {
    0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f,
};

// `GGML_E8M0_TO_FP32_HALF`.
static inline float e8m0_half(uint8_t e) {
    return as_type<float>(e < 2 ? 0x00200000u << e : (uint) (e - 1) << 23);
}

// The simdgroup sums of a simdgroup's `NR0` rows, stored.
template <short NR0>
static inline void simd_write(constant MatvecArgs & a, device float * dst, thread float * sumf, int first_row, ushort tiisg, float scale) {
    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]) * scale;
    }
    if (tiisg == 0) {
        store_rows<NR0>(a, dst, first_row, sumf);
    }
}

// --- Q4_0, Q4_1, Q5_0, Q5_1 (ggml's `block_q_n_dot_y`) -----------------------

static inline float block_q_n_dot_y(device const block_q4_0 * qb, float sumy, thread float * yl, int il) {
    const float d = qb->d;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    device const uint16_t * qs = ((device const uint16_t *) qb + 1 + il / 2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

static inline float block_q_n_dot_y(device const block_q4_1 * qb, float sumy, thread float * yl, int il) {
    const float d = qb->d;
    const float m = qb->m;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    device const uint16_t * qs = ((device const uint16_t *) qb + 2 + il / 2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

static inline float block_q_n_dot_y(device const block_q5_0 * qb, float sumy, thread float * yl, int il) {
    const float d = qb->d;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    device const uint16_t * qs = ((device const uint16_t *) qb + 3 + il / 2);
    const uint32_t qh = qb->qh[0] | (qb->qh[1] << 8) | (qb->qh[2] << 16) | ((uint32_t) qb->qh[3] << 24);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i + 0 + il) << 4) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i + 1 + il) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i + 0 + il + 16) << 8) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i + 1 + il + 16) << 16) & 0x10000));
    }
    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

static inline float block_q_n_dot_y(device const block_q5_1 * qb, float sumy, thread float * yl, int il) {
    const float d = qb->d;
    const float m = qb->m;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    device const uint16_t * qs = ((device const uint16_t *) qb + 4 + il / 2);
    const uint32_t qh = qb->qh[0] | (qb->qh[1] << 8) | (qb->qh[2] << 16) | ((uint32_t) qb->qh[3] << 24);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i + 0 + il) << 4) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i + 1 + il) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i + 0 + il + 16) << 8) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i + 1 + il + 16) << 16) & 0x10000));
    }
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// ggml's `mul_vec_q_n_f32_impl`: each pair of lanes takes a block, a
// simdgroup 16 blocks at a time, for its `NR0` rows.
template <typename block_q, short NR0>
static inline void matvec_q_n_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NQ = 16;
    const int nb = a.cols / QK4_0;
    const int r0 = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_q * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q *) matrix_row(w, w2, a, r0 + row);
    }
    float sumf[NR0] = {0.f};
    const short ix = tiisg / 2;
    const short il = (tiisg % 2) * 8;
    float yl[16];
    device const float * yb = y + ix * QK4_0 + il;
    for (int ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = {0.f, 0.f};
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0] += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] / 256.f;
            sumy[1] += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.f;
            yl[i + 9] = yb[i + 17] / 4096.f;
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }
        yb += QK4_0 * 16;
    }
    simd_write<NR0>(a, dst, sumf, r0, tiisg, 1.0f);
}

template <short NR0> static inline void matvec_q4_0_impl(constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_q_n_impl<block_q4_0, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg);
}
template <short NR0> static inline void matvec_q4_1_impl(constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_q_n_impl<block_q4_1, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg);
}
template <short NR0> static inline void matvec_q5_0_impl(constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_q_n_impl<block_q5_0, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg);
}
template <short NR0> static inline void matvec_q5_1_impl(constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_q_n_impl<block_q5_1, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg);
}

// --- Q2_0 (ggml's `kernel_mul_mv_q2_0_f32_impl`) ----------------------------

// 16 weights of a block (from weight `il`) times `yl`: `d * (Σ_lo + 2 Σ_hi -
// Σ y)`, where `Σ_lo` and `Σ_hi` add the `y` whose code has its low or high
// bit set (code `c` stands for `c - 1`).
static inline float block_q2_0_dot_y(device const block_q2_0 * qb, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb->qs + il / 4;
    float acc_lo = 0.0f;
    float acc_hi = 0.0f;
    FOR_UNROLL (short k = 0; k < 4; ++k) {
        const uint8_t b = qs[k];
        acc_lo += select(0.0f, yl[4 * k + 0], bool(b & 0x01));
        acc_lo += select(0.0f, yl[4 * k + 1], bool(b & 0x04));
        acc_lo += select(0.0f, yl[4 * k + 2], bool(b & 0x10));
        acc_lo += select(0.0f, yl[4 * k + 3], bool(b & 0x40));
    }
    FOR_UNROLL (short k = 0; k < 4; ++k) {
        const uint8_t b = qs[k];
        acc_hi += select(0.0f, yl[4 * k + 0], bool(b & 0x02));
        acc_hi += select(0.0f, yl[4 * k + 1], bool(b & 0x08));
        acc_hi += select(0.0f, yl[4 * k + 2], bool(b & 0x20));
        acc_hi += select(0.0f, yl[4 * k + 3], bool(b & 0x80));
    }
    return qb->d * (acc_lo + 2.0f * acc_hi - sumy);
}

// Each group of 4 lanes takes a block, a lane 16 of its weights, a
// simdgroup 8 blocks at a time, for its `NR0` rows. No kernel for several
// inputs at once: the product is bound by its arithmetic, not by loading
// the codes, and one for two inputs measured no faster.
template <short NR0>
static inline void matvec_q2_0_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const int nb = a.cols / QK2_0;
    const int r0 = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_q2_0 * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q2_0 *) matrix_row(w, w2, a, r0 + row);
    }
    float yl[16];
    float sumf[NR0] = {0.f};
    const short ix = tiisg / 4;
    const short il = (tiisg % 4) * 16;
    device const float * yb = y + ix * QK2_0 + il;
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH / 4) {
        float sumy = 0.f;
        FOR_UNROLL (short i = 0; i < 16; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q2_0_dot_y(ax[row] + ib, sumy, yl, il);
        }
        yb += QK2_0 * (N_SIMDWIDTH / 4);
    }
    simd_write<NR0>(a, dst, sumf, r0, tiisg, 1.0f);
}

MATVEC_KERNEL(matvec_q2_0, matvec_q2_0_impl, 8)
MATVEC_KERNEL(matvec_q4_0, matvec_q4_0_impl, 4)
MATVEC_KERNEL(matvec_q4_1, matvec_q4_1_impl, 4)
MATVEC_KERNEL(matvec_q5_0, matvec_q5_0_impl, 4)
MATVEC_KERNEL(matvec_q5_1, matvec_q5_1_impl, 4)

// --- IQ4_NL and MXFP4 -----------------------------------------------------

static inline float block_scale(device const block_iq4_nl & b) {
    return (float) b.d;
}

static inline float block_scale(device const block_mxfp4 & b) {
    return e8m0_half(b.e);
}

// ggml's `kernel_mul_mv_iq4_nl_f32_impl`: lanes pair up on a block, each
// half of it through the value table in threadgroup memory (32 floats).
// `MXFP4` takes the same path with its own values and power-of-two scale.
template <typename block_q, short NR0>
static inline void matvec_nl_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg,
        constant float * values) {
    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_q * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q *) matrix_row(w, w2, a, first_row + row);
    }
    const int nb = a.cols / 32;
    const short ix = tiisg / 2;
    const short it = tiisg % 2;
    shmem_f32[tiisg] = values[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0] = {0.f};
    device const float * yb = y + ix * 32 + it * 8;
    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *) aux32;
    float4 qf1, qf2;
    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 * y4 = (device const float4 *) yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];
        for (short row = 0; row < NR0; row++) {
            device const block_q & xb = ax[row][ib];
            device const uint8_t * q4 = xb.qs + 8 * it;
            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 8) | (q4[2] << 16) | ((uint32_t) q4[3] << 24);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[4] | (q4[5] << 8) | (q4[6] << 16) | ((uint32_t) q4[7] << 24);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;
            sumf[row] += block_scale(xb) * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 16 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 1.0f);
}

template <short NR0> static inline void matvec_iq4_nl_impl(constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_nl_impl<block_iq4_nl, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg, kvalues_iq4nl_f);
}
template <short NR0> static inline void matvec_mxfp4_impl(constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_nl_impl<block_mxfp4, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg, kvalues_mxfp4_f);
}

MATVEC_KERNEL(matvec_iq4_nl, matvec_iq4_nl_impl, 2)
MATVEC_KERNEL(matvec_mxfp4, matvec_mxfp4_impl, 2)

// --- Q2_K -------------------------------------------------------------------

// ggml's `kernel_mul_mv_q2_K_f32_impl`.
template <short NR0>
static inline void matvec_q2_K_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const int nb = a.cols / QK_K;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_q2_K * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q2_K *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }
        for (short row = 0; row < NR0; row++) {
            device const block_q2_K & xb = ax[row][ib];
            device const uint8_t * sc = xb.scales + 8 * iq + is;
            device const uint16_t * qs = (device const uint16_t *) xb.qs + 16 * iq + 4 * ir;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
            }
            const float dall = xb.d;
            const float dmin = (float) xb.dmin * 1.f / 16.f;
            sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                 (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                 (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                 (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
        }
        y4 += 4 * QK_K;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 1.0f);
}

MATVEC_KERNEL(matvec_q2_K, matvec_q2_K_impl, 4)

// --- Codebook I-quants --------------------------------------------------------
//
// Lane `ix` takes groups of 32 weights `ix, ix + 32, ...` for each of the
// simdgroup's `NR0` rows (ggml's layout without `FC_mul_mv_split`).

template <short NR0>
static inline void matvec_iq2_xxs_impl(
        constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y,
        device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb32 = a.cols / 32;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_iq2_xxs * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq2_xxs *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
    threadgroup uint8_t * ssigns = (threadgroup uint8_t *) (svalues + 256);
    {
        int pos = (32 * sgitg + tiisg) * 4;
        for (int i = 0; i < 4; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        pos = (32 * sgitg + tiisg) * 2;
        for (int i = 0; i < 2; ++i) ssigns[pos + i] = ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const short ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        for (short row = 0; row < NR0; row++) {
            device const block_iq2_xxs & xr = ax[row][ibl];
            device const uint16_t * q2 = xr.qs + 4 * ib;
            const float db = xr.d;
            device const uint8_t * aux8 = (device const uint8_t *) q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));
            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *) (svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8 * l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;
        }
        y4 += 32 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 0.25f);
}

template <short NR0>
static inline void matvec_iq2_xs_impl(
        constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y,
        device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb32 = a.cols / 32;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_iq2_xs * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq2_xs *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
    threadgroup uint8_t * ssigns = (threadgroup uint8_t *) (svalues + 512);
    {
        int pos = (32 * sgitg + tiisg) * 8;
        for (int i = 0; i < 8; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
        pos = (32 * sgitg + tiisg) * 2;
        for (int i = 0; i < 2; ++i) ssigns[pos + i] = ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const short ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        for (short row = 0; row < NR0; row++) {
            device const block_iq2_xs & xr = ax[row][ibl];
            device const uint16_t * q2 = xr.qs + 4 * ib;
            const uint8_t sc = xr.scales[ib];
            const float db = xr.d;
            const float d1 = db * (0.5f + (sc & 0xf));
            const float d2 = db * (0.5f + (sc >> 4));
            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *) (svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8 * l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *) (svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8 * l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;
        }
        y4 += 32 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 0.25f);
}

template <short NR0>
static inline void matvec_iq2_s_impl(
        constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y,
        device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb32 = a.cols / 32;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_iq2_s * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq2_s *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    const short ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        for (short row = 0; row < NR0; row++) {
            device const block_iq2_s & xr = ax[row][ibl];
            device const uint8_t * qs = xr.qs + 4 * ib;
            device const uint8_t * signs = qs + QK_K / 8;
            const uint8_t qh = xr.qh[ib];
            const uint8_t sc = xr.scales[ib];
            const float db = xr.d;
            const float d1 = db * (0.5f + (sc & 0xf));
            const float d2 = db * (0.5f + (sc >> 4));
            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                constant uint8_t * grid1 = (constant uint8_t *) (iq2s_grid + (qs[l + 0] | ((qh << (8 - 2 * l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *) (iq2s_grid + (qs[l + 2] | ((qh << (4 - 2 * l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8 * l + j +  0] * grid1[j] * select(1, -1, signs[l + 0] & kmask_iq2xs[j]);
                    sum[1] += yl[8 * l + j + 16] * grid2[j] * select(1, -1, signs[l + 2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];
        }
        y4 += 32 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 0.25f);
}

template <short NR0>
static inline void matvec_iq3_xxs_impl(
        constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y,
        device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb32 = a.cols / 32;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_iq3_xxs * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq3_xxs *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    threadgroup uint8_t * ssigns = (threadgroup uint8_t *) (svalues + 256);
    {
        int pos = (32 * sgitg + tiisg) * 4;
        for (int i = 0; i < 4; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        pos = (32 * sgitg + tiisg) * 2;
        for (int i = 0; i < 2; ++i) ssigns[pos + i] = ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const short ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        for (short row = 0; row < NR0; row++) {
            device const block_iq3_xxs & xr = ax[row][ibl];
            device const uint8_t * q3 = xr.qs + 8 * ib;
            device const uint16_t * gas = (device const uint16_t *) (xr.qs + QK_K / 4) + 2 * ib;
            const float db = xr.d;
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));
            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *) (svalues + q3[2 * l + 0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *) (svalues + q3[2 * l + 1]);
                const uint8_t signs = ssigns[(aux32 >> 7 * l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8 * l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    sum[1] += yl[8 * l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);
        }
        y4 += 32 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 0.5f);
}

template <short NR0>
static inline void matvec_iq3_s_impl(
        constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y,
        device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb32 = a.cols / 32;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_iq3_s * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq3_s *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        int pos = (32 * sgitg + tiisg) * 8;
        for (int i = 0; i < 8; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const short ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        for (short row = 0; row < NR0; row++) {
            device const block_iq3_s & xr = ax[row][ibl];
            device const uint8_t * qs = xr.qs + 8 * ib;
            const uint8_t qh = xr.qh[ib];
            device const uint8_t * signs = xr.signs + 4 * ib;
            const float db = xr.d;
            const float d = db * (1 + 2 * ((xr.scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh & kmask_iq2xs[2 * l + 0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh & kmask_iq2xs[2 * l + 1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *) (table1 + qs[2 * l + 0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *) (table2 + qs[2 * l + 1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8 * l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j + 0]);
                    sum[1] += yl[8 * l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j + 4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);
        }
        y4 += 32 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 1.0f);
}

template <short NR0>
static inline void matvec_iq1_s_impl(
        constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y,
        device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb32 = a.cols / 32;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_iq1_s * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq1_s *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    const short ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float sumy = 0;
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        for (short row = 0; row < NR0; row++) {
            device const block_iq1_s & xr = ax[row][ibl];
            device const uint8_t * qs = xr.qs + 4 * ib;
            const uint16_t qh = xr.qh[ib];
            constant uint8_t * grid1 = (constant uint8_t *) (iq1s_grid_gpu + (qs[0] | ((qh << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *) (iq1s_grid_gpu + (qs[1] | ((qh << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *) (iq1s_grid_gpu + (qs[2] | ((qh << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *) (iq1s_grid_gpu + (qs[3] | ((qh >> 1) & 0x700)));
            float sum = 0;
            for (short j = 0; j < 4; ++j) {
                sum += yl[j +  0] * (grid1[j] & 0xf) + yl[j +  4] * (grid1[j] >> 4)
                     + yl[j +  8] * (grid2[j] & 0xf) + yl[j + 12] * (grid2[j] >> 4)
                     + yl[j + 16] * (grid3[j] & 0xf) + yl[j + 20] * (grid3[j] >> 4)
                     + yl[j + 24] * (grid4[j] & 0xf) + yl[j + 28] * (grid4[j] >> 4);
            }
            sumf[row] += (float) xr.d * (sum + sumy * (qh & 0x8000 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA)) * (2 * ((qh >> 12) & 7) + 1);
        }
        y4 += 32 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 1.0f);
}

// The f16 super-block scale of an `IQ1_M` block, spread over the top
// nibbles of its four scale words.
static inline float iq1_m_scale(device const block_iq1_m & b) {
    device const uint16_t * sc = (device const uint16_t *) b.scales;
    const ushort u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    return (float) as_type<half>(u);
}

template <short NR0>
static inline void matvec_iq1_m_impl(
        constant MatvecArgs & a, device const char * w, device const char * w2, device const float * y,
        device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb32 = a.cols / 32;
    const int first_row = (tgpig.x * a.nsg + sgitg) * NR0;
    device const block_iq1_m * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq1_m *) matrix_row(w, w2, a, first_row + row);
    }
    float yl[32];
    float sumf[NR0] = {0.f};
    const short ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i +  8]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 16]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 24]; sumy[3] += yl[i + 24];
        }
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        for (short row = 0; row < NR0; row++) {
            device const block_iq1_m & xr = ax[row][ibl];
            device const uint8_t * qs = xr.qs + 4 * ib;
            device const uint8_t * qh = xr.qh + 2 * ib;
            device const uint16_t * sc = (device const uint16_t *) xr.scales;
            constant uint8_t * grid1 = (constant uint8_t *) (iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *) (iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *) (iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *) (iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));
            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j +  0] * (grid1[j] & 0xf) + yl[j +  4] * (grid1[j] >> 4)
                        + yl[j +  8] * (grid2[j] & 0xf) + yl[j + 12] * (grid2[j] >> 4);
                sum[1] += yl[j + 16] * (grid3[j] & 0xf) + yl[j + 20] * (grid3[j] >> 4)
                        + yl[j + 24] * (grid4[j] & 0xf) + yl[j + 28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA);
            sumf[row] += iq1_m_scale(xr) * ((sum[0] + delta1) * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1) +
                                            (sum[1] + delta2) * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1));
        }
        y4 += 32 * 32;
    }
    simd_write<NR0>(a, dst, sumf, first_row, tiisg, 1.0f);
}

MATVEC_KERNEL(matvec_iq2_xxs, matvec_iq2_xxs_impl, 4)
MATVEC_KERNEL(matvec_iq2_xs, matvec_iq2_xs_impl, 4)
MATVEC_KERNEL(matvec_iq2_s, matvec_iq2_s_impl, 4)
MATVEC_KERNEL(matvec_iq3_xxs, matvec_iq3_xxs_impl, 4)
MATVEC_KERNEL(matvec_iq3_s, matvec_iq3_s_impl, 4)
MATVEC_KERNEL(matvec_iq1_s, matvec_iq1_s_impl, 4)
MATVEC_KERNEL(matvec_iq1_m, matvec_iq1_m_impl, 4)

// --- Embedding lookups --------------------------------------------------------
//
// Thread `i` decodes values `[n*i, n*i + n)` of the row, as the CPU's
// `dequantize_row_*` computes them: one block of 32 (`n = 32`) for the
// formats of blocks of 32, one group of 8 (`n = 8`) for the others.

static inline float2 decode_pair(device const block_q4_0 & b, short j) {
    const float d = b.d;
    return float2((float) ((b.qs[j] & 0xF) - 8) * d, (float) ((b.qs[j] >> 4) - 8) * d);
}
static inline float2 decode_pair(device const block_q4_1 & b, short j) {
    const float d = b.d, m = b.m;
    return float2(fma((float) (b.qs[j] & 0xF), d, m), fma((float) (b.qs[j] >> 4), d, m));
}
static inline uint32_t high_bits(device const uint8_t * qh) {
    return qh[0] | (qh[1] << 8) | (qh[2] << 16) | ((uint32_t) qh[3] << 24);
}
static inline float2 decode_pair(device const block_q5_0 & b, short j) {
    const float d = b.d;
    const uint32_t h = high_bits(b.qh);
    const int lo = (b.qs[j] & 0xF) | (((h >> j) & 1) << 4);
    const int hi = (b.qs[j] >> 4) | (((h >> (j + 16)) & 1) << 4);
    return float2((float) (lo - 16) * d, (float) (hi - 16) * d);
}
static inline float2 decode_pair(device const block_q5_1 & b, short j) {
    const float d = b.d, m = b.m;
    const uint32_t h = high_bits(b.qh);
    const int lo = (b.qs[j] & 0xF) | (((h >> j) & 1) << 4);
    const int hi = (b.qs[j] >> 4) | (((h >> (j + 16)) & 1) << 4);
    return float2(fma((float) lo, d, m), fma((float) hi, d, m));
}
static inline float2 decode_pair(device const block_iq4_nl & b, short j) {
    const float d = b.d;
    return float2(d * kvalues_iq4nl_f[b.qs[j] & 0xF], d * kvalues_iq4nl_f[b.qs[j] >> 4]);
}
static inline float2 decode_pair(device const block_mxfp4 & b, short j) {
    const float d = e8m0_half(b.e);
    return float2(kvalues_mxfp4_f[b.qs[j] & 0xF] * d, kvalues_mxfp4_f[b.qs[j] >> 4] * d);
}

template <typename block_q>
static inline void embed_q_n(constant EmbedArgs & a, device const char * w, device float * x, device const uint * slot, uint i) {
    if (i >= a.cols / 32) {
        return;
    }
    device const block_q & b = ((device const block_q *) (w + (uint64_t) embed_row(a, slot) * a.row_bytes))[i];
    for (short j = 0; j < 16; ++j) {
        const float2 v = decode_pair(b, j);
        x[32 * i + j] = v[0];
        x[32 * i + j + 16] = v[1];
    }
}

#define EMBED_Q_N(name, block_q) \
kernel void name(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]], \
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]], \
        uint i [[thread_position_in_grid]]) { \
    embed_q_n<block_q>(a, w, x, slot, i); \
}

EMBED_Q_N(embed_q4_0, block_q4_0)
EMBED_Q_N(embed_q4_1, block_q4_1)
EMBED_Q_N(embed_q5_0, block_q5_0)
EMBED_Q_N(embed_q5_1, block_q5_1)
EMBED_Q_N(embed_iq4_nl, block_iq4_nl)
EMBED_Q_N(embed_mxfp4, block_mxfp4)

// One block of 64 per thread, as `dequantize_row_q2_0`.
kernel void embed_q2_0(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= a.cols / QK2_0) {
        return;
    }
    device const block_q2_0 & b = ((device const block_q2_0 *) (w + (uint64_t) embed_row(a, slot) * a.row_bytes))[i];
    const float d = b.d;
    for (short j = 0; j < QK2_0; ++j) {
        x[QK2_0 * i + j] = (float) ((int) ((b.qs[j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
    }
}

// Group `g` (of 32) of 8 values of a 256-value block, into `out`.
static inline void decode8(device const block_q2_K & b, short g, thread float * out) {
    const float d = b.d, dmin = b.dmin;
    for (short l = 0; l < 8; ++l) {
        const short k = 8 * g + l;
        const uint8_t sc = b.scales[k / 16];
        const float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
        const uint8_t q = (b.qs[32 * (k / 128) + k % 32] >> (2 * ((k % 128) / 32))) & 3;
        out[l] = fma(dl, (float) q, -ml);
    }
}
static inline void decode8(device const block_iq2_xxs & b, short g, thread float * out) {
    const short ib32 = g / 4, l = g % 4;
    device const uint8_t * aux8 = (device const uint8_t *) (b.qs + 4 * ib32);
    const uint32_t aux1 = b.qs[4 * ib32 + 2] | (b.qs[4 * ib32 + 3] << 16);
    const float db = (float) b.d * (0.5f + (aux1 >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *) (iq2xxs_grid + aux8[l]);
    const uint8_t signs = ksigns_iq2xs[(aux1 >> 7 * l) & 127];
    for (short j = 0; j < 8; ++j) out[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
static inline void decode8(device const block_iq2_xs & b, short g, thread float * out) {
    const short ib32 = g / 4, l = g % 4;
    const float db = (float) b.d * (0.5f + ((b.scales[ib32] >> 4 * (l / 2)) & 0xf)) * 0.25f;
    const uint16_t q = b.qs[g];
    constant uint8_t * grid = (constant uint8_t *) (iq2xs_grid + (q & 511));
    const uint8_t signs = ksigns_iq2xs[q >> 9];
    for (short j = 0; j < 8; ++j) out[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
static inline void decode8(device const block_iq2_s & b, short g, thread float * out) {
    const short ib32 = g / 4, l = g % 4;
    const float db = (float) b.d * (0.5f + ((b.scales[ib32] >> 4 * (l / 2)) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *) (iq2s_grid + (b.qs[g] | ((b.qh[ib32] << (8 - 2 * l)) & 0x300)));
    const uint8_t signs = b.qs[QK_K / 8 + g];
    for (short j = 0; j < 8; ++j) out[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
static inline void decode8(device const block_iq3_xxs & b, short g, thread float * out) {
    const short ib32 = g / 4, l = g % 4;
    device const uint8_t * sas = b.qs + QK_K / 4 + 4 * ib32;
    const uint32_t aux = sas[0] | (sas[1] << 8) | (sas[2] << 16) | ((uint32_t) sas[3] << 24);
    const float db = (float) b.d * (0.5f + (aux >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux >> 7 * l) & 127];
    constant uint8_t * grid1 = (constant uint8_t *) (iq3xxs_grid + b.qs[8 * ib32 + 2 * l]);
    constant uint8_t * grid2 = (constant uint8_t *) (iq3xxs_grid + b.qs[8 * ib32 + 2 * l + 1]);
    for (short j = 0; j < 4; ++j) {
        out[j] = db * grid1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        out[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
}
static inline void decode8(device const block_iq3_s & b, short g, thread float * out) {
    const short ib32 = g / 4, l = g % 4;
    const float db = (float) b.d * (float) (1 + 2 * ((b.scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf));
    const uint16_t qh = b.qh[ib32];
    constant uint8_t * grid1 = (constant uint8_t *) (iq3s_grid + (b.qs[8 * ib32 + 2 * l] | ((qh << (8 - 2 * l)) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *) (iq3s_grid + (b.qs[8 * ib32 + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
    const uint8_t signs = b.signs[g];
    for (short j = 0; j < 4; ++j) {
        out[j] = db * grid1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        out[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
}
static inline void decode8(device const block_iq1_s & b, short g, thread float * out) {
    const short ib = g / 4, l = g % 4;
    const uint16_t qh = b.qh[ib];
    const float dl = (float) b.d * (float) (2 * ((qh >> 12) & 7) + 1);
    const float delta = qh & 0x8000 ? -IQ1_DELTA : IQ1_DELTA;
    constant uint8_t * grid = (constant uint8_t *) (iq1s_grid_gpu + (b.qs[g] | (((qh >> 3 * l) & 7) << 8)));
    for (short j = 0; j < 8; ++j) {
        const short v = (j < 4 ? grid[j] & 0xf : grid[j - 4] >> 4) - 1;
        out[j] = dl * ((float) v + delta);
    }
}
static inline void decode8(device const block_iq1_m & b, short g, thread float * out) {
    device const uint16_t * sc = (device const uint16_t *) b.scales;
    const short t = g / 2;
    const float dl = iq1_m_scale(b) * (float) (2 * ((sc[t / 4] >> (3 * (t % 4))) & 7) + 1);
    const uint8_t qh = b.qh[g / 2];
    const float delta = qh & (g % 2 == 0 ? 0x08 : 0x80) ? -IQ1_DELTA : IQ1_DELTA;
    constant uint8_t * grid = (constant uint8_t *) (iq1s_grid_gpu + (b.qs[g] | ((qh << (g % 2 == 0 ? 8 : 4)) & 0x700)));
    for (short j = 0; j < 8; ++j) {
        const short v = (j < 4 ? grid[j] & 0xf : grid[j - 4] >> 4) - 1;
        out[j] = dl * ((float) v + delta);
    }
}

#define EMBED_K8(name, block_q) \
kernel void name(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]], \
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]], \
        uint i [[thread_position_in_grid]]) { \
    if (i >= a.cols / 8) { \
        return; \
    } \
    device const block_q & b = ((device const block_q *) (w + (uint64_t) embed_row(a, slot) * a.row_bytes))[i / 32]; \
    float v[8]; \
    decode8(b, (short) (i % 32), v); \
    for (short j = 0; j < 8; ++j) { \
        x[8 * i + j] = v[j]; \
    } \
}

EMBED_K8(embed_q2_K, block_q2_K)
EMBED_K8(embed_iq2_xxs, block_iq2_xxs)
EMBED_K8(embed_iq2_xs, block_iq2_xs)
EMBED_K8(embed_iq2_s, block_iq2_s)
EMBED_K8(embed_iq3_xxs, block_iq3_xxs)
EMBED_K8(embed_iq3_s, block_iq3_s)
EMBED_K8(embed_iq1_s, block_iq1_s)
EMBED_K8(embed_iq1_m, block_iq1_m)
