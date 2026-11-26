// nibble's Metal kernels: shared definitions.
//
// The shader files are concatenated in the order `metal/kernels.rs` lists
// them and compiled at startup with `newLibraryWithSource`.
//
// Matrix-vector kernels follow ggml-metal's (llama.cpp
// `ggml/src/ggml-metal/kernels/mul_mv.metal`): the same simdgroup layout,
// rows per simdgroup and reduction order, with f32 activations. Every kernel
// has a test against the CPU reference in `metal/tests.rs`.

#include <metal_stdlib>

using namespace metal;

#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define QK8_0 32
#define QK_K 256

typedef struct {
    half d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    half d;
} block_q3_K;

typedef struct {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K;

typedef struct {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half d;
} block_q6_K;

// What a matrix-vector kernel writes (`mode` of `MatvecArgs`).
// `dst = W · x`.
#define MATVEC_STORE 0
// `dst += W · x`.
#define MATVEC_ADD 1
// `dst = silu(W · x) * (W2 · x)`: virtual row `2r` is row `r` of `W`, and
// `2r + 1` row `r` of `W2`, so a simdgroup's (or threadgroup's) rows, an
// even number, are whole pairs.
#define MATVEC_SWIGLU 2
// `dst += W · x`, `xn = dst * norm`, and accumulates `dst^2` in 64-bit
// fixed-point (2^20 scale, exact commutative+associative integer addition),
// so the next product reads one 8-byte scalar and gets a bit-deterministic
// RMS scale without an `rmsnorm` dispatch or barrier.
#define MATVEC_RESIDUAL_NORM 3
#define RMS_FP_SCALE 1048576.0f
#define RMS_FP_INV (1.0f / 1048576.0f)


// `MatvecArgs` in `metal/kernels.rs`: `dst = W · x` (or as `mode` says) for
// `rows` rows of `cols` elements, rows `row_bytes` apart; `nsg` simdgroups
// per threadgroup; `ncols` inputs `x` of `cols` values one after another,
// their outputs (`rows` values each) one after another in `dst`.
struct MatvecArgs {
    uint cols;
    uint rows;
    uint row_bytes;
    uint nsg;
    uint mode;
    uint ncols;
    uint rms_in_len;
    uint rms_in_off;
    float rms_in_eps;
    uint rms_out_off;
};

struct RmsCtx {
    device const float * norm;
    device float * xn;
    device atomic_uint * sums;
    uint tok;
};

// Exact 64-bit unsigned atomic add via a `(lo, hi)` pair of 32-bit atomics.
static inline void rms_add_u64(device atomic_uint * pair, ulong val) {
    const uint lo = (uint) (val & 0xFFFFFFFFul);
    const uint hi = (uint) (val >> 32);
    const uint prev = atomic_fetch_add_explicit(pair, lo, memory_order_relaxed);
    const uint carry = (prev + lo) < prev ? 1u : 0u;
    if (hi + carry != 0u) {
        atomic_fetch_add_explicit(pair + 1, hi + carry, memory_order_relaxed);
    }
}

// Loads the deterministic fixed-point sum-of-squares for token `tok`.
static inline float rms_load_scale(constant MatvecArgs & a, device atomic_uint * sums, uint tok) {
    if (a.rms_in_len == 0) {
        return 1.0f;
    }
    device const uint * p = (device const uint *) (sums + 2 * (a.rms_in_off + tok));
    const ulong raw = (ulong) p[0] | ((ulong) p[1] << 32);
    return rsqrt((float(raw) * RMS_FP_INV) / float(a.rms_in_len) + a.rms_in_eps);
}

// `EmbedArgs`: decode row `row` (of `cols` elements) of a weight matrix, or
// with `from_slot` the row the token slot (`slot[0]`) holds.
struct EmbedArgs {
    uint cols;
    uint row;
    uint row_bytes;
    uint from_slot;
};

static inline uint embed_row(constant EmbedArgs & a, device const uint * slot, uint t) {
    return a.from_slot != 0 ? slot[t] : a.row;
}


static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Writes the totals of (virtual) rows `r0..r0 + NR0` as `a.mode` says,
// applying any deferred RMS scale from the previous fused residual.
template <short NR0>
static inline void store_rows(constant MatvecArgs & a, device float * dst, RmsCtx rms, int r0, thread const float * tot) {
    const float scale = rms_load_scale(a, rms.sums, rms.tok);
    if (a.mode == MATVEC_SWIGLU) {
        for (short p = 0; p < NR0 / 2 && r0 / 2 + p < (int) a.rows; ++p) {
            dst[r0 / 2 + p] = silu(tot[2 * p] * scale) * (tot[2 * p + 1] * scale);
        }
        return;
    }
    if (a.mode == MATVEC_RESIDUAL_NORM) {
        device float * xn = rms.xn + (uint64_t) rms.tok * a.rows;
        ulong sq_fp = 0;
        for (short row = 0; row < NR0 && r0 + row < (int) a.rows; ++row) {
            const float v = dst[r0 + row] + tot[row] * scale;
            dst[r0 + row] = v;
            xn[r0 + row] = v * rms.norm[r0 + row];
            sq_fp += (ulong) rint(v * v * RMS_FP_SCALE);
        }
        rms_add_u64(rms.sums + 2 * (a.rms_out_off + rms.tok), sq_fp);
        return;
    }
    for (short row = 0; row < NR0 && r0 + row < (int) a.rows; ++row) {
        const float v = tot[row] * scale;
        dst[r0 + row] = a.mode == MATVEC_ADD ? dst[r0 + row] + v : v;
    }
}

// ggml's `helper_mv_reduce_and_write`: sums each row's partial sums over
// the simdgroups of the threadgroup and writes rows `r0..r0 + NR0`.
template <short NR0>
static inline void reduce_and_write(
        constant MatvecArgs & a,
        device float * dst,
        RmsCtx rms,
        thread float * sumf,
        int r0,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + N_SIMDWIDTH * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot[NR0];
    for (short row = 0; row < NR0; ++row) {
        tot[row] = simd_sum(shmem_f32[row][tiisg]);
    }
    if (tiisg == 0 && sgitg == 0) {
        store_rows<NR0>(a, dst, rms, r0, tot);
    }
}

// Row `r` of the matrix (virtual row `r` of the pair `w`, `w2` for
// `MATVEC_SWIGLU`), clamped to the last row: a threadgroup's rows past the
// end are computed (as in ggml) but never written, and must not read past
// the buffer.
static inline device const char * matrix_row(device const char * w, device const char * w2, constant MatvecArgs & a, int r) {
    if (a.mode == MATVEC_SWIGLU) {
        w = (r & 1) ? w2 : w;
        r >>= 1;
    }
    return w + (uint64_t) min(r, (int) a.rows - 1) * a.row_bytes;
}

// The entry point of a matrix-vector kernel `name` whose body is
// `impl<NR0>(a, w, w2, x, dst, shmem, tgpig, tiisg, sgitg)`, run for each
// of the `ncols` inputs in turn: each input's output is computed exactly as
// in a dispatch of its own, and the threadgroup reads its weight rows from
// memory once, from its cache for the later inputs. The barrier keeps the
// next input's reduction from overwriting threadgroup memory that a
// simdgroup still reads.
// `MoeArgs` in `metal/kernels.rs`: the expert of slot `s` (of `k` per
// input row) is entry `s % k` of row `s / k` of the route (`2k` values a
// row: expert indices, then weights); experts are `expert_rows` rows apart,
// `count` of them. The slot's input is row `s` of `x` if `per_slot`, else
// row `s / k`; its output is row `s` of `dst`.
struct MoeArgs {
    uint k;
    uint per_slot;
    uint expert_rows;
    uint count;
};

static inline uint moe_expert(device const float * route, constant MoeArgs & m, uint s) {
    const uint e = (uint) route[(uint64_t) (s / m.k) * 2 * m.k + s % m.k];
    return min(e, m.count - 1);
}

// The arguments of a matrix-vector kernel `name_id`, which runs one
// mixture-of-experts slot (`tgpig.y`) per threadgroup row: expert `e`'s
// weights for that slot's input. `MATVEC_ID_PROLOGUE` defines `w_e`, `w2_e`,
// `x_s` and `dst_s` for the body.
#define MATVEC_ID_SIGNATURE(name) \
kernel void name( \
        constant MatvecArgs & a [[buffer(0)]], \
        device const char * w [[buffer(1)]], \
        device const float * x [[buffer(2)]], \
        device float * dst [[buffer(3)]], \
        device const char * w2 [[buffer(4)]], \
        device const float * route [[buffer(5)]], \
        constant MoeArgs & m [[buffer(6)]], \
        device atomic_uint * rms_sums [[buffer(7)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]])

#define MATVEC_ID_PROLOGUE \
    const uint s = tgpig.y; \
    const uint64_t off = (uint64_t) moe_expert(route, m, s) * m.expert_rows * a.row_bytes; \
    device const char * w_e = w + off; \
    device const char * w2_e = w2 + off; \
    const uint in_tok = m.per_slot != 0 ? s : s / m.k; \
    device const float * x_s = x + (uint64_t) in_tok * a.cols; \
    device float * dst_s = dst + (uint64_t) s * a.rows; \
    const RmsCtx rms = { nullptr, nullptr, rms_sums, in_tok };

// A one-input kernel `name` and its mixture-of-experts form `name_id`.
#define MATVEC_KERNEL(name, impl, NR0) \
MATVEC_ID_SIGNATURE(name##_id) { \
    MATVEC_ID_PROLOGUE \
    impl<NR0>(a, w_e, w2_e, x_s, dst_s, rms, shmem, tgpig, tiisg, sgitg); \
} \
MATVEC_KERNEL_LOOP(name, impl, NR0)

#define MATVEC_KERNEL_LOOP(name, impl, NR0) \
kernel void name( \
        constant MatvecArgs & a [[buffer(0)]], \
        device const char * w [[buffer(1)]], \
        device const float * x [[buffer(2)]], \
        device float * dst [[buffer(3)]], \
        device const char * w2 [[buffer(4)]], \
        device const float * rms_norm [[buffer(5)]], \
        device float * rms_xn [[buffer(6)]], \
        device atomic_uint * rms_sums [[buffer(7)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    for (uint c = 0; c < a.ncols; ++c) { \
        if (c > 0) { \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        const RmsCtx rms = { rms_norm, rms_xn, rms_sums, c }; \
        impl<NR0>(a, w, w2, x + (uint64_t) c * a.cols, dst + (uint64_t) c * a.rows, rms, shmem, tgpig, tiisg, sgitg); \
    } \
}

// The entry point of a matrix-vector kernel `name` whose body
// `impl<NR0, NC>(a, w, w2, x, dst, rms, nc, shmem, tgpig, tiisg, sgitg)` takes
// `nc` (at most `NC`) inputs at once, reading and decoding each weight
// once for all of them; each input's arithmetic is that of `NC = 1`, so its
// output is the same. The inputs go in groups of `NC`, group `tgpig.y` to
// each threadgroup.
#define MATVEC_KERNEL_NC(name, impl, NR0, NC) \
kernel void name( \
        constant MatvecArgs & a [[buffer(0)]], \
        device const char * w [[buffer(1)]], \
        device const float * x [[buffer(2)]], \
        device float * dst [[buffer(3)]], \
        device const char * w2 [[buffer(4)]], \
        device const float * rms_norm [[buffer(5)]], \
        device float * rms_xn [[buffer(6)]], \
        device atomic_uint * rms_sums [[buffer(7)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    const uint c = tgpig.y * NC; \
    if (c >= a.ncols) { \
        return; \
    } \
    const RmsCtx rms = { rms_norm, rms_xn, rms_sums, c }; \
    const short nc = (short) min((uint) NC, a.ncols - c); \
    impl<NR0, NC>(a, w, w2, x + (uint64_t) c * a.cols, dst + (uint64_t) c * a.rows, rms, nc, shmem, tgpig, tiisg, sgitg); \
}

// The mixture-of-experts form of a kernel `impl<NR0, NC>`, with one input.
#define MATVEC_KERNEL_NC_ID(name, impl, NR0) \
MATVEC_ID_SIGNATURE(name) { \
    MATVEC_ID_PROLOGUE \
    impl<NR0, 1>(a, w_e, w2_e, x_s, dst_s, rms, 1, shmem, tgpig, tiisg, sgitg); \
}

// `reduce_and_write` for each of `nc` inputs whose sums are in `sumf`,
// outputs `rows` apart.
template <short NR0, short NC>
static inline void reduce_and_write_nc(
        constant MatvecArgs & a,
        device float * dst,
        RmsCtx rms,
        thread float (*sumf)[NR0],
        short nc,
        int r0,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    for (short c = 0; c < NC; ++c) {
        if (c > 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (c < nc) {
            RmsCtx rc = rms;
            rc.tok = rms.tok + (uint) c;
            reduce_and_write<NR0>(a, dst + (uint64_t) c * a.rows, rc, sumf[c], r0, tiisg, sgitg, shmem);
        }
    }
}

// The simdgroup sums of each of `nc` inputs (one simdgroup's rows), stored.
template <short NR0, short NC>
static inline void simd_write_nc(
        constant MatvecArgs & a,
        device float * dst,
        RmsCtx rms,
        thread float (*sumf)[NR0],
        short nc,
        int r0,
        ushort tiisg) {
    for (short c = 0; c < NC; ++c) {
        if (c < nc) {
            for (short row = 0; row < NR0; ++row) {
                sumf[c][row] = simd_sum(sumf[c][row]);
            }
            RmsCtx rc = rms;
            rc.tok = rms.tok + (uint) c;
            if (tiisg == 0) {
                store_rows<NR0>(a, dst + (uint64_t) c * a.rows, rc, r0, sumf[c]);
            }
        }
    }
}

// The sum of `v` over the threadgroup's `nsg` simdgroups, returned to every
// thread. `buf` holds at least `N_SIMDWIDTH` floats.
static inline float threadgroup_sum(float v, threadgroup float * buf, ushort tiisg, ushort sgitg, ushort nsg) {
    v = simd_sum(v);
    if (sgitg == 0) {
        buf[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        buf[sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(buf[tiisg]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}
