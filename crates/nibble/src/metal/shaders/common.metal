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


// `MatvecArgs` in `metal/kernels.rs`: `dst = W · x` (or as `mode` says) for
// `rows` rows of `cols` elements, rows `row_bytes` apart; `nsg` simdgroups
// per threadgroup.
struct MatvecArgs {
    uint cols;
    uint rows;
    uint row_bytes;
    uint nsg;
    uint mode;
    uint pad;
};

// `EmbedArgs`: decode row `row` (of `cols` elements) of a weight matrix.
struct EmbedArgs {
    uint cols;
    uint row;
    uint row_bytes;
    uint pad;
};


static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Writes the totals of (virtual) rows `r0..r0 + NR0` as `a.mode` says.
template <short NR0>
static inline void store_rows(constant MatvecArgs & a, device float * dst, int r0, thread const float * tot) {
    if (a.mode == MATVEC_SWIGLU) {
        for (short p = 0; p < NR0 / 2 && r0 / 2 + p < (int) a.rows; ++p) {
            dst[r0 / 2 + p] = silu(tot[2 * p]) * tot[2 * p + 1];
        }
        return;
    }
    for (short row = 0; row < NR0 && r0 + row < (int) a.rows; ++row) {
        dst[r0 + row] = a.mode == MATVEC_ADD ? dst[r0 + row] + tot[row] : tot[row];
    }
}

// ggml's `helper_mv_reduce_and_write`: sums each row's partial sums over
// the simdgroups of the threadgroup and writes rows `r0..r0 + NR0`.
template <short NR0>
static inline void reduce_and_write(
        constant MatvecArgs & a,
        device float * dst,
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
        store_rows<NR0>(a, dst, r0, tot);
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
// `impl<NR0>(a, w, w2, x, dst, shmem, tgpig, tiisg, sgitg)`.
#define MATVEC_KERNEL(name, impl, NR0) \
kernel void name( \
        constant MatvecArgs & a [[buffer(0)]], \
        device const char * w [[buffer(1)]], \
        device const float * x [[buffer(2)]], \
        device float * dst [[buffer(3)]], \
        device const char * w2 [[buffer(4)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    impl<NR0>(a, w, w2, x, dst, shmem, tgpig, tiisg, sgitg); \
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
