// Prompt-batch products of quantized weights: an int8 tensor-core GEMM.
//
// For batches of more than eight tokens, as llama.cpp switches from
// mul_mat_vec_q to mul_mat_q, and laid out as ggml-cuda's mul_mat_q is on
// Ampere: blocks of 256 threads compute 128 rows by J tokens, the weights of
// 256 values per row are unpacked to int8 in shared memory by whole warps
// reading consecutive words of a row, the activations are copied in as they
// are, and mma.sync multiplies them. Each group of 32 (16 for Q2_K, Q3_K and
// Q6_K) is scaled and accumulated in f32 on its own, so the rounding differs
// from the one-token kernels, as llama.cpp's batched products differ from its
// -ub 1 ones. Matrices with too few tiles to fill the GPU's waves split their
// stages between blocks (stream-K, reduced in a fixed order, so results do
// not depend on timing), and a feed-forward's gate and up products run as one
// kernel that applies SwiGLU, as ggml-cuda's fused mul_mat_q does.

#include "common.cuh"
#include "quant.cuh"

// Activations for the GEMM (quantize_mmq), like ggml's block_q8_1_mmq: per
// 128 values of a token, the scales d of its four blocks of 32 (quantized as
// quantize_q8_1 does), d * (the sum of the quants) of its eight halves of 16,
// then the 128 quants, 44 words in all. Chunks of 128 values are the outer
// index and tokens (padded to GEMM_TOK_PAD) the inner one, so the activations
// of a tile are one contiguous range, copied to shared memory as it is.
static constexpr int GEMM_YS = 44;
static constexpr int GEMM_TOK_PAD = 128;
static constexpr int GEMM_BM = 128;
static constexpr int GEMM_THREADS = 256;
// The weight loaders unroll their row loops fully, which keeps every load of
// a stage in flight at once: unrolling less measured slower, the loads'
// latency being what bounds the kernel.

// One block per token and chunk of 128 values; one warp per block of 32.
// Values past `cols` and tokens past `ncols` are zeros.
extern "C" __global__ void __launch_bounds__(128) quantize_mmq(
        const float * x, int * out, unsigned int cols, unsigned int ncols, unsigned int ystride) {
    const unsigned int tok = blockIdx.x;
    const unsigned int chunk = blockIdx.y;
    const unsigned int k = chunk * 128 + threadIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const float v = tok < ncols && k < cols ? x[(size_t) tok * cols + k] : 0.0f;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : (int) roundf(v / d);
    int s = q;
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    int * o = out + ((size_t) chunk * ystride + tok) * GEMM_YS;
    ((signed char *) (o + 12))[threadIdx.x] = (signed char) q;
    if (lane == 0) ((float *) o)[warp] = d;
    if (lane % 16 == 0) ((float *) o)[4 + 2 * warp + lane / 16] = d * (float) s;
}

// Types whose scales change every 16 values.
template<int TYPE> struct GemmSplit {
    static constexpr bool value = TYPE == 10 || TYPE == 11 || TYPE == 14;
};

// Types with a min term (value = scale * q - min).
template<int TYPE> struct GemmMin {
    static constexpr bool value = TYPE == 5 || TYPE == 7 || TYPE == 10 || TYPE == 12 || TYPE == 13;
};

// A weight row's words in shared memory: 64 of int8 quants (256 values), 16
// scales (per 32, or per 16 for split types), 16 mins for types that have
// them, and padding so that ldmatrix reads no two rows from the same banks.
template<int TYPE> struct GemmX {
    static constexpr int stride = GemmMin<TYPE>::value ? 100 : 84;
};

// Dynamic shared memory of a block.
template<int TYPE, int J> struct GemmShared {
    static constexpr int bytes = 4 * (GEMM_BM * GemmX<TYPE>::stride + 2 * J * GEMM_YS);
};

__device__ __forceinline__ float gemm_half(const unsigned char * p) {
    return __half2float(*(const half *) p);
}

// Unpacks the stage `st` (values 256 st to 256 st + 255) of rows row0 to
// row0 + 127 into `xs`, or with FUSE rows row0 to row0 + 63 of `w` (the
// gate) then of `w2` (up). Rows past the matrix repeat its last row (their
// results are not stored); groups past `ngroups` are zeros.
template<int TYPE, bool FUSE>
__device__ __forceinline__ void gemm_load_x(const unsigned char * w, const unsigned char * w2, int row0, MatvecArgs a,
        int st, int ngroups, int * xs) {
    constexpr int XS = GemmX<TYPE>::stride;
    float * xf = (float *) xs;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int last = (int) a.rows - 1;
    auto row_ptr = [&](int i) {
        if constexpr (FUSE) {
            return (i < GEMM_BM / 2 ? w : w2) + (size_t) min(row0 + i % (GEMM_BM / 2), last) * a.row_bytes;
        } else {
            return w + (size_t) min(row0 + i, last) * a.row_bytes;
        }
    };

    if constexpr (TYPE == 3) {  // Q8_0: two words of a 34-byte block per lane
        const int kb = lane / 8;
        const int kq = lane % 8;
        const int g = st * 8 + kb;
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 8) {
            const int i = i0 + warp;
            const unsigned char * rp = row_ptr(i);
            const int v0 = g < ngroups ? get_int_b2(rp + (size_t) g * 34 + 2, kq) : 0;
            const int v1 = g + 4 < ngroups ? get_int_b2(rp + (size_t) (g + 4) * 34 + 2, kq) : 0;
            xs[i * XS + kb * 8 + kq] = v0;
            xs[i * XS + (kb + 4) * 8 + kq] = v1;
        }
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 32) {
            const int i = i0 + warp * 4 + lane / 8;
            const int b = lane % 8;
            const int gb = st * 8 + b;
            xf[i * XS + 64 + b] = gb < ngroups ? gemm_half(row_ptr(i) + (size_t) gb * 34) : 0.0f;
        }
    } else if constexpr (TYPE >= 4 && TYPE <= 7) {  // Q4_0, Q4_1, Q5_0, Q5_1: one word per lane
        constexpr int bs = TYPE == 4 ? 18 : TYPE == 5 ? 20 : TYPE == 6 ? 22 : 24;
        constexpr int qs_at = TYPE == 4 ? 2 : TYPE == 5 ? 4 : TYPE == 6 ? 6 : 8;
        const int kb = lane / 4;
        const int kq = lane % 4;
        const int g = st * 8 + kb;
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 8) {
            const int i = i0 + warp;
            int lo = 0, hi = 0;
            if (g < ngroups) {
                const unsigned char * b = row_ptr(i) + (size_t) g * bs;
                const int ql = TYPE == 5 || TYPE == 7 ? get_int_b4(b + qs_at, kq) : get_int_b2(b + qs_at, kq);
                lo = (ql >> 0) & 0x0F0F0F0F;
                hi = (ql >> 4) & 0x0F0F0F0F;
                if constexpr (TYPE == 6 || TYPE == 7) {
                    const int qh = (TYPE == 7 ? get_int_b4(b + 4, 0) : get_int_b2(b + 2, 0)) >> (4 * kq);
                    lo |= (qh <<  4) & 0x00000010;
                    lo |= (qh << 11) & 0x00001000;
                    lo |= (qh << 18) & 0x00100000;
                    lo |= (qh << 25) & 0x10000000;
                    hi |= (qh >> 12) & 0x00000010;
                    hi |= (qh >>  5) & 0x00001000;
                    hi |= (qh <<  2) & 0x00100000;
                    hi |= (qh <<  9) & 0x10000000;
                }
                if constexpr (TYPE == 4) {
                    lo = __vsub4(lo, 0x08080808);
                    hi = __vsub4(hi, 0x08080808);
                } else if constexpr (TYPE == 6) {
                    lo = __vsub4(lo, 0x10101010);
                    hi = __vsub4(hi, 0x10101010);
                }
            }
            xs[i * XS + kb * 8 + kq] = lo;
            xs[i * XS + kb * 8 + kq + 4] = hi;
        }
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 32) {
            const int i = i0 + warp * 4 + lane / 8;
            const int b = lane % 8;
            const int gb = st * 8 + b;
            float d = 0.0f;
            [[maybe_unused]] float m = 0.0f;  // types with a min term
            if (gb < ngroups) {
                const unsigned char * p = row_ptr(i) + (size_t) gb * bs;
                d = gemm_half(p);
                if constexpr (TYPE == 5 || TYPE == 7) m = gemm_half(p + 2);
            }
            xf[i * XS + 64 + b] = d;
            if constexpr (TYPE == 5 || TYPE == 7) xf[i * XS + 80 + b] = -m;
        }
    } else if constexpr (TYPE == 12 || TYPE == 13) {  // Q4_K, Q5_K: one word of quants per lane
        constexpr int bs = TYPE == 12 ? sizeof(block_q4_K) : sizeof(block_q5_K);
        constexpr int qs_at = TYPE == 12 ? 16 : 48;
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 8) {
            const int i = i0 + warp;
            const unsigned char * b = row_ptr(i) + (size_t) st * bs;
            const int ql = get_int_b4(b + qs_at, lane);
            int lo = (ql >> 0) & 0x0F0F0F0F;
            int hi = (ql >> 4) & 0x0F0F0F0F;
            if constexpr (TYPE == 13) {
                const int qh = get_int_b4(b + 16, lane % 8);
                lo |= ((qh >> (2 * (lane / 8) + 0)) << 4) & 0x10101010;
                hi |= ((qh >> (2 * (lane / 8) + 1)) << 4) & 0x10101010;
            }
            xs[i * XS + 16 * (lane / 8) + lane % 8] = lo;
            xs[i * XS + 16 * (lane / 8) + lane % 8 + 8] = hi;
        }
        // Two lanes per row, four scales and mins each.
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 128) {
            const int i = i0 + warp * 16 + lane / 2;
            const int ksc = lane % 2;
            const unsigned char * b = row_ptr(i) + (size_t) st * bs;
            const int * scales = (const int *) (b + 4);
            const int sc32 = ((scales[ksc + (ksc != 0)] >> 0) & 0x0F0F0F0F) | ((scales[0] >> (2 * ksc)) & 0x30303030);
            const int m32 = ((scales[ksc + 1] >> (4 * ksc)) & 0x0F0F0F0F) | ((scales[1] >> (2 * ksc)) & 0x30303030);
            const float d = gemm_half(b);
            const float dmin = gemm_half(b + 2);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                xf[i * XS + 64 + 4 * ksc + l] = d * (float) ((sc32 >> (8 * l)) & 0xff);
                xf[i * XS + 80 + 4 * ksc + l] = dmin * (float) ((m32 >> (8 * l)) & 0xff);
            }
        }
    } else if constexpr (TYPE == 14) {  // Q6_K: one word of ql per lane
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 8) {
            const int i = i0 + warp;
            const unsigned char * b = row_ptr(i) + (size_t) st * sizeof(block_q6_K);
            const int ql = get_int_b2(b, lane);
            const int qh = get_int_b2(b + 128, 8 * (lane / 16) + lane % 8);
            const int qh0 = ((qh >> ((lane & 8) >> 2)) << 4) & 0x30303030;
            const int qh1 = (qh >> ((lane & 8) >> 2)) & 0x30303030;
            const int kq0 = 2 * lane - lane % 16;
            xs[i * XS + kq0] = __vsub4(((ql >> 0) & 0x0F0F0F0F) | qh0, 0x20202020);
            xs[i * XS + kq0 + 16] = __vsub4(((ql >> 4) & 0x0F0F0F0F) | qh1, 0x20202020);
        }
        // Two lanes per row, eight scales each.
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 128) {
            const int i = i0 + warp * 16 + lane / 2;
            const int h = lane % 2;
            const unsigned char * b = row_ptr(i) + (size_t) st * sizeof(block_q6_K);
            const float d = gemm_half(b + 208);
            const signed char * sc = (const signed char *) (b + 192) + 8 * h;
#pragma unroll
            for (int l = 0; l < 8; ++l) xf[i * XS + 64 + 8 * h + l] = d * (float) sc[l];
        }
    } else if constexpr (TYPE == 11 || TYPE == 10) {  // Q3_K, Q2_K: one word of 2-bit quants per lane, two rows per warp
        constexpr int bs = TYPE == 11 ? sizeof(block_q3_K) : sizeof(block_q2_K);
        const int kq = lane % 16;
#pragma unroll
        for (int i0 = 0; i0 < GEMM_BM; i0 += 16) {
            const int i = i0 + warp * 2 + lane / 16;
            const unsigned char * b = row_ptr(i) + (size_t) st * bs;
            if constexpr (TYPE == 11) {
                const int ql = get_int_b2(b + 32, kq);
                const int qh = get_int_b2(b, kq % 8) >> (4 * (kq / 8));
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int k = (kq / 8) * 32 + l * 8 + kq % 8;
                    const int lo = (ql >> (2 * l)) & 0x03030303;
                    const int hb = ((qh >> l) << 2) & 0x04040404;
                    xs[i * XS + k] = __vsub4(lo | hb, 0x04040404);
                }
            } else {
                const int ql = get_int_b4(b + 16, kq);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    xs[i * XS + (kq / 8) * 32 + l * 8 + kq % 8] = (ql >> (2 * l)) & 0x03030303;
                }
                const int sm = b[kq];
                xf[i * XS + 64 + kq] = gemm_half(b + 80) * (float) (sm & 15);
                xf[i * XS + 80 + kq] = gemm_half(b + 82) * (float) (sm >> 4);
            }
        }
        if constexpr (TYPE == 11) {
            // Four lanes per row, four scales each.
#pragma unroll
            for (int i0 = 0; i0 < GEMM_BM; i0 += 64) {
                const int i = i0 + warp * 8 + lane / 4;
                const int ksc = lane % 4;
                const unsigned char * b = row_ptr(i) + (size_t) st * bs;
                const unsigned char * scales = b + 96;
                const int sc_low = (get_int_b2(scales, ksc % 2) >> (4 * (ksc / 2))) & 0x0F0F0F0F;
                const int sc_high = ((get_int_b2(scales, 2) >> (2 * ksc)) << 4) & 0x30303030;
                const int sc = __vsub4(sc_low | sc_high, 0x20202020);
                const float d = gemm_half(b + 108);
#pragma unroll
                for (int l = 0; l < 4; ++l) xf[i * XS + 64 + 4 * ksc + l] = d * (float) (signed char) (sc >> (8 * l));
            }
        }
    }
}

// Copies the activations of chunk `chunk` (128 values) of tokens tok0 to
// tok0 + J - 1 into `ys` with cp.async.
template<int J>
__device__ __forceinline__ void gemm_load_y(const int * y, int * ys, unsigned int ystride, int tok0, int chunk) {
    const int * src = y + ((size_t) chunk * ystride + tok0) * GEMM_YS;
    constexpr int n = J * GEMM_YS / 4;
#pragma unroll
    for (int l0 = 0; l0 < n; l0 += GEMM_THREADS) {
        const int l = l0 + threadIdx.x;
        if (n % GEMM_THREADS == 0 || l < n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            const unsigned int dst = (unsigned int) __cvta_generic_to_shared(ys + 4 * l);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src + 4 * l));
#endif
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void gemm_wait_y() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
}

// A group's integer sum as a float, exactly (it is at most 32 * 127 * 127
// in magnitude). A plain conversion measured faster than the
// 0x4B400000 trick (two adds), which needs more registers.
__device__ __forceinline__ float gemm_i2f(int x) {
    return (float) x;
}

__device__ __forceinline__ void gemm_ldsm_x4(int (&r)[4], const int * p) {
    const unsigned int a = (unsigned int) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void gemm_mma_k32(int (&c)[4], const int (&a)[4], int b0, int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void gemm_mma_k16(int (&c)[4], int a0, int a1, int b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// Multiplies half `h` (groups 4h to 4h + 3) of the stage in `xs` with the
// activations in `ys`. Eight warps: four along the rows (32 each), two along
// the tokens (J / 2 each).
template<int TYPE, int J>
__device__ __forceinline__ void gemm_dot(const int * xs, const int * ys, int h, float (&acc)[2][J / 16][4]) {
    constexpr bool split = GemmSplit<TYPE>::value;
    constexpr bool has_min = GemmMin<TYPE>::value;
    constexpr int XS = GemmX<TYPE>::stride;
    constexpr int NT = J / 16;
    const float * xf = (const float *) xs;
    const float * yf = (const float *) ys;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int gid = lane / 4;
    const int tig = lane % 4;
    const int wm = (warp % 4) * 32;
    const int wn = (warp / 4) * (J / 2);
#pragma unroll
    for (int gg = 0; gg < 4; ++gg) {
        const int g = 4 * h + gg;
        int af[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            gemm_ldsm_x4(af[mt], xs + (wm + mt * 16 + lane % 16) * XS + g * 8 + (lane / 16) * 4);
        }
        // This thread's rows: gid and gid + 8 of each row tile.
        // (Types without split scales or mins leave some of these unused.)
        float sa[2][2];
        [[maybe_unused]] float sah[2][2], ma[2][2], mah[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int r = (wm + mt * 16 + gid + 8 * hh) * XS;
                if constexpr (split) {
                    sa[mt][hh] = xf[r + 64 + 2 * g];
                    sah[mt][hh] = xf[r + 64 + 2 * g + 1];
                    if constexpr (has_min) {
                        ma[mt][hh] = xf[r + 80 + 2 * g];
                        mah[mt][hh] = xf[r + 80 + 2 * g + 1];
                    }
                } else {
                    sa[mt][hh] = xf[r + 64 + g];
                    if constexpr (has_min) ma[mt][hh] = xf[r + 80 + g];
                }
            }
        }
#pragma unroll
        for (int np = 0; np < NT / 2; ++np) {
            int bf[4];
            gemm_ldsm_x4(bf, ys + (wn + np * 16 + lane % 8 + 8 * (lane / 16)) * GEMM_YS + 12 + gg * 8 + 4 * ((lane / 8) % 2));
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int nt = 2 * np + q;
                const int b0 = bf[2 * q];
                const int b1 = bf[2 * q + 1];
                // This thread's tokens: 2 tig and 2 tig + 1 of the token tile.
                float db[2];
                [[maybe_unused]] float sl[2], sh[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int t = (wn + nt * 8 + 2 * tig + e) * GEMM_YS;
                    db[e] = yf[t + gg];
                    if constexpr (has_min) {
                        sl[e] = yf[t + 4 + 2 * gg];
                        sh[e] = yf[t + 5 + 2 * gg];
                    }
                }
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if constexpr (split) {
                        int lo[4] = {0, 0, 0, 0};
                        int hi[4] = {0, 0, 0, 0};
                        gemm_mma_k16(lo, af[mt][0], af[mt][1], b0);
                        gemm_mma_k16(hi, af[mt][2], af[mt][3], b1);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float d = db[i % 2];
                            float v = gemm_i2f(lo[i]) * (sa[mt][i / 2] * d) + gemm_i2f(hi[i]) * (sah[mt][i / 2] * d);
                            if constexpr (has_min) v -= ma[mt][i / 2] * sl[i % 2] + mah[mt][i / 2] * sh[i % 2];
                            acc[mt][nt][i] += v;
                        }
                    } else {
                        int c[4] = {0, 0, 0, 0};
                        gemm_mma_k32(c, af[mt], b0, b1);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            acc[mt][nt][i] = fmaf(gemm_i2f(c[i]), sa[mt][i / 2] * db[i % 2], acc[mt][nt][i]);
                            if constexpr (has_min) acc[mt][nt][i] -= ma[mt][i / 2] * (sl[i % 2] + sh[i % 2]);
                        }
                    }
                }
            }
        }
    }
}

// Stages s0 to s1 - 1 of the tile at rows row0 and tokens tok0, added to
// `acc`. The activations of the next half stage are copied while the
// current one is multiplied. Ends with a barrier, so the shared tiles can
// be refilled at once.
template<int TYPE, int J, bool FUSE>
__device__ __forceinline__ void gemm_run(const unsigned char * w, const unsigned char * w2, const int * y, MatvecArgs a,
        unsigned int ystride, int row0, int tok0, int s0, int s1, int * xs, int * ys, float (&acc)[2][J / 16][4]) {
    const int ngroups = a.cols / 32;
    int * ys1 = ys + J * GEMM_YS;
    gemm_load_y<J>(y, ys, ystride, tok0, 2 * s0);
    for (int st = s0; st < s1; ++st) {
        gemm_load_x<TYPE, FUSE>(w, w2, row0, a, st, ngroups, xs);
        gemm_wait_y();
        __syncthreads();
        gemm_load_y<J>(y, ys1, ystride, tok0, 2 * st + 1);
        gemm_dot<TYPE, J>(xs, ys, 0, acc);
        gemm_wait_y();
        __syncthreads();
        if (st + 1 < s1) gemm_load_y<J>(y, ys, ystride, tok0, 2 * st + 2);
        gemm_dot<TYPE, J>(xs, ys1, 1, acc);
        __syncthreads();
    }
}

// The output row and token of accumulator (mt, nt, i) of thread `tid` in
// the tile at row0, tok0.
template<int J>
__device__ __forceinline__ void gemm_place(int tid, int mt, int nt, int i, int row0, int tok0, int & row, int & tok) {
    const int lane = tid % 32;
    const int warp = tid / 32;
    row = row0 + (warp % 4) * 32 + mt * 16 + lane / 4 + (i / 2) * 8;
    tok = tok0 + (warp / 4) * (J / 2) + nt * 8 + 2 * (lane % 4) + (i % 2);
}

template<int J>
__device__ __forceinline__ void gemm_store(float * dst, MatvecArgs a, int row0, int tok0, const float (&acc)[2][J / 16][4]) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < J / 16; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row, tok;
                gemm_place<J>(threadIdx.x, mt, nt, i, row0, tok0, row, tok);
                if (row < (int) a.rows && tok < (int) a.ncols) {
                    float * out = dst + (size_t) tok * a.rows + row;
                    if (a.mode == MATVEC_ADD) *out += acc[mt][nt][i]; else *out = acc[mt][nt][i];
                }
            }
        }
    }
}

// Stream-K partitioning, as ggml-cuda's mul_mat_q on Ampere but per token
// tile: for each token tile (blockIdx.x), `blocks` blocks (blockIdx.y)
// share the stages of the row tiles evenly, block b taking iterations
// [b * total / blocks, (b + 1) * total / blocks) in row-tile order. Blocks
// with the same b run the same weights at the same time, so the token tiles
// share them in L2. A block that runs all of a tile's stages stores it.
// Blocks that share a tile each keep their sums in `partial` (slot 0 for the
// tile a block starts in, 1 for the one it ends in) and count themselves in
// `counts`; the last to finish adds the sums in block order (so the result
// does not depend on which one that is), stores the tile and resets the
// count for the next GEMM.
struct GemmSplitK {
    unsigned int blocks;   // per token tile
    unsigned int tiles_n;  // token tiles
    unsigned int tiles;    // row tiles
    unsigned int stages;   // per tile
};

__device__ __forceinline__ unsigned int gemm_sk_begin(GemmSplitK sk, unsigned int b) {
    return (unsigned int) ((unsigned long long) b * sk.tiles * sk.stages / sk.blocks);
}

// The block whose iterations include `it`.
__device__ __forceinline__ unsigned int gemm_sk_block(GemmSplitK sk, unsigned int it) {
    unsigned int b = (unsigned int) ((unsigned long long) it * sk.blocks / ((unsigned long long) sk.tiles * sk.stages));
    while (b + 1 < sk.blocks && gemm_sk_begin(sk, b + 1) <= it) ++b;
    while (b > 0 && gemm_sk_begin(sk, b) > it) --b;
    return b;
}

// Where accumulator (mt, nt, i) of every thread is kept in a partial tile.
template<int J>
__device__ __forceinline__ int gemm_partial_at(int mt, int nt, int i) {
    return ((mt * (J / 16) + nt) * 4 + i) * GEMM_THREADS + threadIdx.x;
}

template<int J>
__device__ __forceinline__ float * gemm_partial_slot(float * partial, GemmSplitK sk, unsigned int b, int slot) {
    return partial + (((size_t) b * sk.tiles_n + blockIdx.x) * 2 + slot) * GEMM_BM * J;
}

// out[t][r] (=, +=) W[r] . x[t] for rows of `a.cols` values and `a.ncols`
// tokens; `y` holds the activations from quantize_mmq with `ystride` tokens.
// One block per tile when sk.blocks is 0, else stream-K.
template<int TYPE, int J>
__device__ __forceinline__ void gemm_impl(const unsigned char * w, const int * y, float * dst, MatvecArgs a,
        unsigned int ystride, GemmSplitK sk, float * partial, unsigned int * counts) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ int gemm_tiles[];
    __shared__ bool gemm_last;
    int * xs = gemm_tiles;
    int * ys = xs + GEMM_BM * GemmX<TYPE>::stride;
    if (sk.blocks == 0) {
        float acc[2][J / 16][4] = {};
        const int nst = (a.cols / 32 + 7) / 8;
        gemm_run<TYPE, J, false>(w, nullptr, y, a, ystride, blockIdx.y * GEMM_BM, blockIdx.x * J, 0, nst, xs, ys, acc);
        gemm_store<J>(dst, a, blockIdx.y * GEMM_BM, blockIdx.x * J, acc);
        return;
    }
    const unsigned int b = blockIdx.y;
    const unsigned int it0 = gemm_sk_begin(sk, b);
    const unsigned int it1 = gemm_sk_begin(sk, b + 1);
    const int tok0 = blockIdx.x * J;
    for (unsigned int it = it0; it < it1;) {
        const unsigned int tile = it / sk.stages;
        const int k0 = it - tile * sk.stages;
        const int k1 = min(sk.stages, k0 + (it1 - it));
        const int row0 = tile * GEMM_BM;
        float acc[2][J / 16][4] = {};
        gemm_run<TYPE, J, false>(w, nullptr, y, a, ystride, row0, tok0, k0, k1, xs, ys, acc);
        if (k0 == 0 && k1 == (int) sk.stages) {
            gemm_store<J>(dst, a, row0, tok0, acc);
        } else {
            float * p = gemm_partial_slot<J>(partial, sk, b, it == it0 ? 0 : 1);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int nt = 0; nt < J / 16; ++nt) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) p[gemm_partial_at<J>(mt, nt, i)] = acc[mt][nt][i];
                }
            }
            const unsigned int first = gemm_sk_block(sk, tile * sk.stages);
            const unsigned int last = gemm_sk_block(sk, tile * sk.stages + sk.stages - 1);
            // The partial is visible to the other blocks before they count it.
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned int * count = counts + (size_t) tile * sk.tiles_n + blockIdx.x;
                gemm_last = atomicAdd(count, 1) == last - first;
                if (gemm_last) *count = 0;
            }
            __syncthreads();
            if (gemm_last) {
                // The sums are added from memory, this block's too, so
                // that no second set of accumulators is live.
                __threadfence();
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                    for (int nt = 0; nt < J / 16; ++nt) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;
                    }
                }
                for (unsigned int c = first; c <= last; ++c) {
                    const float * q = gemm_partial_slot<J>(partial, sk, c, gemm_sk_begin(sk, c) / sk.stages == tile ? 0 : 1);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                        for (int nt = 0; nt < J / 16; ++nt) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) acc[mt][nt][i] += __ldcg(q + gemm_partial_at<J>(mt, nt, i));
                        }
                    }
                }
                gemm_store<J>(dst, a, row0, tok0, acc);
            }
        }
        it += k1 - k0;
    }
#endif
}

// out[t][r] = silu(G[r] . x[t]) * (U[r] . x[t]), as the GEMM above with one
// block per tile of 64 rows of each matrix (ggml-cuda's fused gate/up
// mul_mat_q; always tiled, since the activation needs whole sums). Warps 0,
// 1, 4 and 5 hold the gate rows and the others the same rows of up, which
// they pass through shared memory.
template<int TYPE, int J>
__device__ __forceinline__ void gemm_glu_impl(const unsigned char * gate, const unsigned char * up, const int * y,
        float * dst, MatvecArgs a, unsigned int ystride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ int gemm_tiles[];
    int * xs = gemm_tiles;
    int * ys = xs + GEMM_BM * GemmX<TYPE>::stride;
    constexpr int NA = 2 * (J / 16) * 4;  // accumulators per thread
    static_assert(4 * NA * 32 <= GEMM_BM * GemmX<TYPE>::stride, "the exchange fits the weight tile");
    const int row0 = blockIdx.y * (GEMM_BM / 2);
    const int tok0 = blockIdx.x * J;
    const int nst = (a.cols / 32 + 7) / 8;
    float acc[2][J / 16][4] = {};
    gemm_run<TYPE, J, true>(gate, up, y, a, ystride, row0, tok0, 0, nst, xs, ys, acc);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int pair = (warp / 4) * 2 + warp % 2;  // a gate warp and its up warp
    float * ex = (float *) xs + pair * NA * 32;
    if (warp % 4 >= 2) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < J / 16; ++nt) {
#pragma unroll
                for (int i = 0; i < 4; ++i) ex[((mt * (J / 16) + nt) * 4 + i) * 32 + lane] = acc[mt][nt][i];
            }
        }
    }
    __syncthreads();
    if (warp % 4 < 2) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < J / 16; ++nt) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int row, tok;
                    gemm_place<J>(threadIdx.x, mt, nt, i, row0, tok0, row, tok);
                    if (row < (int) a.rows && tok < (int) a.ncols) {
                        const float u = ex[((mt * (J / 16) + nt) * 4 + i) * 32 + lane];
                        dst[(size_t) tok * a.rows + row] = silu_op(acc[mt][nt][i]) * u;
                    }
                }
            }
        }
    }
#endif
}

#define EXPORT_GEMM_J(NAME, TYPE, J) \
    extern "C" __global__ void __launch_bounds__(GEMM_THREADS, 1) gemm_##NAME##_##J( \
            const unsigned char * w, const int * y, float * dst, MatvecArgs a, unsigned int ystride, \
            GemmSplitK sk, float * partial, unsigned int * counts) { \
        gemm_impl<TYPE, J>(w, y, dst, a, ystride, sk, partial, counts); \
    } \
    extern "C" __global__ void __launch_bounds__(GEMM_THREADS, 1) gemm_glu_##NAME##_##J( \
            const unsigned char * gate, const unsigned char * up, const int * y, float * dst, MatvecArgs a, \
            unsigned int ystride) { \
        gemm_glu_impl<TYPE, J>(gate, up, y, dst, a, ystride); \
    }

#define EXPORT_GEMM(NAME, TYPE) \
    EXPORT_GEMM_J(NAME, TYPE, 32) \
    EXPORT_GEMM_J(NAME, TYPE, 64) \
    EXPORT_GEMM_J(NAME, TYPE, 128)

EXPORT_GEMM(q8_0, 3)
EXPORT_GEMM(q4_0, 4)
EXPORT_GEMM(q4_1, 5)
EXPORT_GEMM(q5_0, 6)
EXPORT_GEMM(q5_1, 7)
EXPORT_GEMM(q2_K, 10)
EXPORT_GEMM(q3_K, 11)
EXPORT_GEMM(q4_K, 12)
EXPORT_GEMM(q5_K, 13)
EXPORT_GEMM(q6_K, 14)
