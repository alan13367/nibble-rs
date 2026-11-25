// Matrix-vector products and embedding lookups for F32, F16, BF16 and Q8_0.

// ggml's `kernel_mul_mv_t_t_4_impl` (NR0 = 2): each simdgroup lane takes 16
// consecutive values of a 32-value block, blocks strided by the threadgroup.
template <typename T04, short NR0>
static inline void matvec_float4(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF / 4;
    const short nsg = a.nsg;

    const int nb = a.cols / NB;
    const int r0 = tgpig.x * NR0;

    device const float4 * y4 = (device const float4 *) y;

    device const T04 * ax4[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const T04 *) matrix_row(w, w2, a, r0 + row);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg / (N_SIMDWIDTH / NF);
    const short il = tiisg % (N_SIMDWIDTH / NF);

    const int ib0 = sgitg * NF + ix;

    float4 yl4[NF4];

    device const float4 * yb4 = y4 + (ib0 * NB + il * NF) / 4;

    for (int ib = ib0; ib < nb; ib += nsg * NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }
        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib * NB + il * NF) / 4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf[row] += sumq;
        }
        yb4 += nsg * NF * N_SIMDWIDTH / 4;
    }

    // The tail past the last whole block, one value per lane (four at a time:
    // `cols` is a multiple of 4).
    for (int i = nb * NB / 4 + sgitg * N_SIMDWIDTH + tiisg; i < (int) a.cols / 4; i += N_SIMDWIDTH * nsg) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += dot(float4(ax4[row][i]), y4[i]);
        }
    }

    reduce_and_write<NR0>(a, dst, sumf, r0, tiisg, sgitg, shmem);
}

template <short NR0>
static inline void matvec_f32_impl(constant MatvecArgs & a, device const char * w, device const char * w2,
        device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_float4<float4, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg);
}

template <short NR0>
static inline void matvec_f16_impl(constant MatvecArgs & a, device const char * w, device const char * w2,
        device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_float4<half4, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg);
}

template <short NR0>
static inline void matvec_bf16_impl(constant MatvecArgs & a, device const char * w, device const char * w2,
        device const float * y, device float * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_float4<bfloat4, NR0>(a, w, w2, y, dst, shmem, tgpig, tiisg, sgitg);
}

MATVEC_KERNEL(matvec_f32, matvec_f32_impl, 2)
MATVEC_KERNEL(matvec_f16, matvec_f16_impl, 2)
MATVEC_KERNEL(matvec_bf16, matvec_bf16_impl, 2)

// `matvec_float4` for `nc` (at most `NC`) inputs at once: each weight
// float4 is loaded once for all of them.
template <typename T04, short NR0, short NC>
static inline void matvec_float4_nc(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        short nc,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF / 4;
    const short nsg = a.nsg;

    const int nb = a.cols / NB;
    const int r0 = tgpig.x * NR0;

    device const T04 * ax4[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const T04 *) matrix_row(w, w2, a, r0 + row);
    }

    float sumf[NC][NR0] = { { 0.f } };

    const short ix = tiisg / (N_SIMDWIDTH / NF);
    const short il = tiisg % (N_SIMDWIDTH / NF);

    const int ib0 = sgitg * NF + ix;

    float4 yl4[NC][NF4];

    device const float4 * yb4[NC];
    for (short c = 0; c < NC; ++c) {
        yb4[c] = (device const float4 *) (y + (uint64_t) min(c, (short) (nc - 1)) * a.cols) + (ib0 * NB + il * NF) / 4;
    }

    for (int ib = ib0; ib < nb; ib += nsg * NF) {
        for (short c = 0; c < NC; ++c) {
            for (short i = 0; i < NF4; ++i) {
                yl4[c][i] = yb4[c][i];
            }
        }
        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib * NB + il * NF) / 4;
            for (short c = 0; c < NC; ++c) {
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NF4; ++i) {
                    sumq += dot(float4(xb4[i]), yl4[c][i]);
                }
                sumf[c][row] += sumq;
            }
        }
        for (short c = 0; c < NC; ++c) {
            yb4[c] += nsg * NF * N_SIMDWIDTH / 4;
        }
    }

    for (int i = nb * NB / 4 + sgitg * N_SIMDWIDTH + tiisg; i < (int) a.cols / 4; i += N_SIMDWIDTH * nsg) {
        for (short row = 0; row < NR0; row++) {
            for (short c = 0; c < NC; ++c) {
                device const float4 * y4 = (device const float4 *) (y + (uint64_t) min(c, (short) (nc - 1)) * a.cols);
                sumf[c][row] += dot(float4(ax4[row][i]), y4[i]);
            }
        }
    }

    reduce_and_write_nc<NR0, NC>(a, dst, sumf, nc, r0, tiisg, sgitg, shmem);
}

template <short NR0, short NC>
static inline void matvec_f32_nc(constant MatvecArgs & a, device const char * w, device const char * w2,
        device const float * y, device float * dst, short nc, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_float4_nc<float4, NR0, NC>(a, w, w2, y, dst, nc, shmem, tgpig, tiisg, sgitg);
}

template <short NR0, short NC>
static inline void matvec_f16_nc(constant MatvecArgs & a, device const char * w, device const char * w2,
        device const float * y, device float * dst, short nc, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_float4_nc<half4, NR0, NC>(a, w, w2, y, dst, nc, shmem, tgpig, tiisg, sgitg);
}

template <short NR0, short NC>
static inline void matvec_bf16_nc(constant MatvecArgs & a, device const char * w, device const char * w2,
        device const float * y, device float * dst, short nc, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    matvec_float4_nc<bfloat4, NR0, NC>(a, w, w2, y, dst, nc, shmem, tgpig, tiisg, sgitg);
}

MATVEC_KERNEL_NC(matvec_f32_c2, matvec_f32_nc, 2, 2)
MATVEC_KERNEL_NC(matvec_f16_c2, matvec_f16_nc, 2, 2)
MATVEC_KERNEL_NC(matvec_bf16_c2, matvec_bf16_nc, 2, 2)

// ggml's `kernel_mul_mv_q8_0_f32_impl` (NR0 = 2): each lane takes 8 values
// of a block, 4 lanes per block, blocks strided by the threadgroup; for
// `nc` inputs at once, each block's weights loaded once.
template <short NR0, short NC>
static inline void matvec_q8_0_impl(
        constant MatvecArgs & a,
        device const char * w,
        device const char * w2,
        device const float * y,
        device float * dst,
        short nc,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NQ = 8;
    const short nsg = a.nsg;

    const int nb = a.cols / QK8_0;
    const int r0 = tgpig.x * NR0;

    device const block_q8_0 * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q8_0 *) matrix_row(w, w2, a, r0 + row);
    }

    float sumf[NC][NR0] = { { 0.f } };

    const short ix = tiisg / (N_SIMDWIDTH / NQ);
    const short il = tiisg % (N_SIMDWIDTH / NQ);

    const int ib0 = sgitg * NQ + ix;

    device const float * yb = y + ib0 * QK8_0 + il * NQ;

    if (NC == 1) {
        float yl[NQ];
        for (int ib = ib0; ib < nb; ib += nsg * NQ) {
            for (short i = 0; i < NQ; ++i) {
                yl[i] = yb[i];
            }
            for (short row = 0; row < NR0; row++) {
                device const int8_t * qs = ax[row][ib].qs + il * NQ;
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
                sumf[0][row] += sumq * ax[row][ib].d;
            }
            yb += nsg * NQ * QK8_0;
        }
    } else {
        // Each block's weights are loaded once for every input; inputs past
        // `nc` repeat the last one (their sums are not written).
        float yl[NC][NQ];
        for (int ib = ib0; ib < nb; ib += nsg * NQ) {
            for (short c = 0; c < NC; ++c) {
                device const float * yc = yb + (uint64_t) min(c, (short) (nc - 1)) * a.cols;
                for (short i = 0; i < NQ; ++i) {
                    yl[c][i] = yc[i];
                }
            }
            for (short row = 0; row < NR0; row++) {
                device const int8_t * qs = ax[row][ib].qs + il * NQ;
                const half d = ax[row][ib].d;
                for (short c = 0; c < NC; ++c) {
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NQ; ++i) {
                        sumq += qs[i] * yl[c][i];
                    }
                    sumf[c][row] += sumq * d;
                }
            }
            yb += nsg * NQ * QK8_0;
        }
    }

    reduce_and_write_nc<NR0, NC>(a, dst, sumf, nc, r0, tiisg, sgitg, shmem);
}

MATVEC_KERNEL_NC(matvec_q8_0, matvec_q8_0_impl, 2, 1)
MATVEC_KERNEL_NC_ID(matvec_q8_0_id, matvec_q8_0_impl, 2)
MATVEC_KERNEL_NC(matvec_q8_0_c2, matvec_q8_0_impl, 2, 2)
MATVEC_KERNEL_NC(matvec_q8_0_c4, matvec_q8_0_impl, 2, 4)

// Embedding lookups: one thread per value.

template <typename T>
static inline void embed_float(constant EmbedArgs & a, device const char * w, device float * x, device const uint * slot, uint i) {
    if (i < a.cols) {
        x[i] = float(((device const T *) (w + (uint64_t) embed_row(a, slot) * a.row_bytes))[i]);
    }
}

kernel void embed_f32(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
    embed_float<float>(a, w, x, slot, i);
}

kernel void embed_f16(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
    embed_float<half>(a, w, x, slot, i);
}

kernel void embed_bf16(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
    embed_float<bfloat>(a, w, x, slot, i);
}

kernel void embed_q8_0(constant EmbedArgs & a [[buffer(0)]], device const char * w [[buffer(1)]],
        device float * x [[buffer(2)]], device const uint * slot [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
    if (i < a.cols) {
        device const block_q8_0 * b = (device const block_q8_0 *) (w + (uint64_t) embed_row(a, slot) * a.row_bytes) + i / QK8_0;
        x[i] = float(b->d) * float(b->qs[i % QK8_0]);
    }
}
