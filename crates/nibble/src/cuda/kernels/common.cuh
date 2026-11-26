#pragma once

// CUDA-only subset of ggml b12675. Block layouts and QI/VDR constants are
// transcribed from ggml/src/ggml-common.h and ggml/src/ggml-cuda/vecdotq.cuh;
// reductions and scalar helpers come from ggml/src/ggml-cuda/common.cuh.
// The host uploads Codebooks with the corresponding tables from ggml-common.h.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

static constexpr int QK_K = 256;
static constexpr int K_SCALE_SIZE = 12;
static constexpr int WARP_SIZE = 32;

static constexpr int QK2_0 = 64;
static constexpr int QK4_0 = 32;
static constexpr int QK4_1 = 32;
static constexpr int QK5_0 = 32;
static constexpr int QK5_1 = 32;
static constexpr int QK8_0 = 32;
static constexpr int QK8_1 = 32;
static constexpr int QK4_NL = 32;
static constexpr int QK_MXFP4 = 32;

static constexpr int QR2_K = 4;
static constexpr int QR3_K = 4;
static constexpr int QR4_K = 2;
static constexpr int QR5_K = 2;
static constexpr int QR6_K = 2;
static constexpr int QR4_0 = 2;
static constexpr int QR4_1 = 2;
static constexpr int QR5_0 = 2;
static constexpr int QR5_1 = 2;
static constexpr int QR8_0 = 1;
static constexpr int QR8_1 = 1;
static constexpr int QR4_NL = 2;
static constexpr int QR_MXFP4 = 2;
static constexpr int QR1_S = 8;
static constexpr int QR1_M = 8;

static constexpr int QI2_0 = QK2_0 / 32;
static constexpr int QI4_0 = QK4_0 / (4 * QR4_0);
static constexpr int QI4_1 = QK4_1 / (4 * QR4_1);
static constexpr int QI5_0 = QK5_0 / (4 * QR5_0);
static constexpr int QI5_1 = QK5_1 / (4 * QR5_1);
static constexpr int QI8_0 = QK8_0 / (4 * QR8_0);
static constexpr int QI8_1 = QK8_1 / (4 * QR8_1);
static constexpr int QI4_NL = QK4_NL / (4 * QR4_NL);
static constexpr int QI_MXFP4 = QK_MXFP4 / (4 * QR_MXFP4);
static constexpr int QI2_K = QK_K / (4 * QR2_K);
static constexpr int QI3_K = QK_K / (4 * QR3_K);
static constexpr int QI4_K = QK_K / (4 * QR4_K);
static constexpr int QI5_K = QK_K / (4 * QR5_K);
static constexpr int QI6_K = QK_K / (4 * QR6_K);
static constexpr int QI2_XXS = QK_K / (4 * 4);
static constexpr int QI2_XS = QK_K / (4 * 4);
static constexpr int QI2_S = QK_K / (4 * 4);
static constexpr int QI3_XXS = QK_K / (4 * 4);
static constexpr int QI3_S = QK_K / (4 * 4);
static constexpr int QI4_XS = QK_K / (4 * 2);
static constexpr int QI1_S = QK_K / (4 * QR1_S);
static constexpr int QI1_M = QK_K / (4 * QR1_M);

// nibble's GGUF type numbers. This is deliberately not ggml's enum order.
enum GgmlType : int {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_BF16 = 2,
    GGML_TYPE_Q8_0 = 3,
    GGML_TYPE_Q4_0 = 4,
    GGML_TYPE_Q4_1 = 5,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_IQ4_NL = 8,
    GGML_TYPE_MXFP4 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_IQ4_XS = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ2_S = 18,
    GGML_TYPE_IQ3_XXS = 19,
    GGML_TYPE_IQ3_S = 20,
    GGML_TYPE_IQ1_S = 21,
    GGML_TYPE_IQ1_M = 22,
    GGML_TYPE_Q2_0 = 23,
};

struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};

struct block_q4_1 {
    union {
        struct { half d; half m; } data;
        half2 dm;
    };
    uint8_t qs[QK4_1 / 2];
};

struct block_q5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
};

struct block_q5_1 {
    union {
        struct { half d; half m; } data;
        half2 dm;
    };
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

struct block_q8_1 {
    union {
        struct { half d; half s; } data;
        half2 ds;
    };
    int8_t qs[QK8_1];
};

struct block_iq4_nl {
    half d;
    uint8_t qs[QK4_NL / 2];
};

struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2];
};

struct block_q2_K {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    union {
        struct { half d; half dmin; } data;
        half2 dm;
    };
};

struct block_q3_K {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[K_SCALE_SIZE];
    half d;
};

struct block_q4_K {
    union {
        struct { half d; half dmin; } data;
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

struct block_q5_K {
    union {
        struct { half d; half dmin; } data;
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half d;
};

struct block_iq4_xs {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
};

struct block_iq2_xxs {
    half d;
    uint16_t qs[QK_K / 8];
};

struct block_iq2_xs {
    half d;
    uint16_t qs[QK_K / 8];
    uint8_t scales[QK_K / 32];
};

struct block_iq2_s {
    half d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};

struct block_iq3_xxs {
    half d;
    uint8_t qs[3 * QK_K / 8];
};

struct block_iq3_s {
    half d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[QK_K / 64];
};

struct block_iq1_s {
    half d;
    uint8_t qs[QK_K / 8];
    uint16_t qh[QK_K / 32];
};

struct block_iq1_m {
    uint8_t qs[QK_K / 8];
    uint8_t qh[QK_K / 16];
    uint8_t scales[QK_K / 32];
};

union iq1m_scale_t {
    half f16;
    uint16_t u16;
};

struct block_q2_0 {
    half d;
    uint8_t qs[QK2_0 / 4];
};

static_assert(sizeof(block_q4_0) == 18, "wrong q4_0 block size");
static_assert(sizeof(block_q4_1) == 20, "wrong q4_1 block size");
static_assert(sizeof(block_q5_0) == 22, "wrong q5_0 block size");
static_assert(sizeof(block_q5_1) == 24, "wrong q5_1 block size");
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size");
static_assert(sizeof(block_q8_1) == 36, "wrong q8_1 block size");
static_assert(sizeof(block_iq4_nl) == 18, "wrong iq4_nl block size");
static_assert(sizeof(block_mxfp4) == 17, "wrong mxfp4 block size");
static_assert(sizeof(block_q2_K) == 84, "wrong q2_K block size");
static_assert(sizeof(block_q3_K) == 110, "wrong q3_K block size");
static_assert(sizeof(block_q4_K) == 144, "wrong q4_K block size");
static_assert(sizeof(block_q5_K) == 176, "wrong q5_K block size");
static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size");
static_assert(sizeof(block_iq4_xs) == 136, "wrong iq4_xs block size");
static_assert(sizeof(block_iq2_xxs) == 66, "wrong iq2_xxs block size");
static_assert(sizeof(block_iq2_xs) == 74, "wrong iq2_xs block size");
static_assert(sizeof(block_iq2_s) == 82, "wrong iq2_s block size");
static_assert(sizeof(block_iq3_xxs) == 98, "wrong iq3_xxs block size");
static_assert(sizeof(block_iq3_s) == 110, "wrong iq3_s block size");
static_assert(sizeof(block_iq1_s) == 50, "wrong iq1_s block size");
static_assert(sizeof(block_iq1_m) == 56, "wrong iq1_m block size");
static_assert(sizeof(block_q2_0) == 18, "wrong q2_0 block size");

struct Codebooks {
    uint64_t iq2xxs_grid[256];
    uint64_t iq2xs_grid[512];
    uint64_t iq2s_grid[1024];
    uint32_t iq3xxs_grid[256];
    uint32_t iq3s_grid[512];
    uint32_t iq1s_grid_gpu[2048];
    uint8_t ksigns_iq2xs[128];
};

static const __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// E2M1 values doubled, as in ggml's kvalues_fp4/kvalues_mxfp4.
static const __device__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static __device__ __forceinline__ float half_to_float(half x) {
    return __half2float(x);
}

static __device__ __forceinline__ float2 half2_to_float2(half2 x) {
    return __half22float2(x);
}

static __device__ __forceinline__ float bf16_to_float(nv_bfloat16 x) {
    return __bfloat162float(x);
}

static __device__ __forceinline__ float e8m0_to_fp32(uint8_t x) {
    const uint32_t bits = x == 0 ? 0x00400000u : uint32_t(x) << 23;
    return __uint_as_float(bits);
}

static __device__ __forceinline__ int get_int_b1(const void * x, int i32) {
    const uint8_t * x8 = static_cast<const uint8_t *>(x);
    int x32 = x8[4*i32 + 0] << 0;
    x32 |= x8[4*i32 + 1] << 8;
    x32 |= x8[4*i32 + 2] << 16;
    x32 |= x8[4*i32 + 3] << 24;
    return x32;
}

static __device__ __forceinline__ int get_int_b2(const void * x, int i32) {
    const uint16_t * x16 = static_cast<const uint16_t *>(x);
    return (x16[2*i32 + 0] << 0) | (x16[2*i32 + 1] << 16);
}

static __device__ __forceinline__ int get_int_b4(const void * x, int i32) {
    return static_cast<const int *>(x)[i32];
}

static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = reinterpret_cast<const int8_t *>(&a);
    const int8_t * b8 = reinterpret_cast<const int8_t *>(&b);
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (width == WARP_SIZE) {
        return __reduce_add_sync(0xffffffffu, x);
    }
#endif
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x.x += __shfl_xor_sync(0xffffffffu, x.x, offset, width);
        x.y += __shfl_xor_sync(0xffffffffu, x.y, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = __hadd2(x, __shfl_xor_sync(0xffffffffu, x, offset, width));
    }
    return x;
}

template<int width = WARP_SIZE, typename T>
static __device__ __forceinline__ T warp_sum(T x) {
    return warp_reduce_sum<width>(x);
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset, width));
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ unsigned int warp_min_u(unsigned int x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = min(x, __shfl_xor_sync(0xffffffffu, x, offset, width));
    }
    return x;
}

// The sum of `v` over a one-dimensional block, returned to every thread.
// The leading barrier lets a block call it again on the same `shared`: no
// thread overwrites a slot while another may still be reading the last result.
static __device__ float block_sum(float v, float * shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (int) ((blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

// The block size rms_norm_f32_dispatch (ggml-cuda b12675) picks for
// contiguous, aligned rows of `ncols` floats, and whether it loads float4s
// (`vpt` of them per thread).
struct RmsShape {
    int bs;
    int vpt;
    bool vec;
};

static __device__ __host__ __forceinline__ RmsShape rms_shape(int ncols) {
    const int n4 = ncols / 4;
    if (ncols % 4 == 0 && ncols > 0) {
        if (n4 <= 32) return {32, 1, true};
        if (n4 <= 128) return {128, 1, true};
        if (n4 <= 256) return {256, 1, true};
        if (n4 <= 512) return {512, 1, true};
        if (n4 <= 1024) return {1024, 1, true};
        if (n4 <= 4096) return {1024, 4, true};
        return {1024, 1, false};
    }
    return {ncols < 1024 ? 256 : 1024, 1, false};
}

// rms_norm_f32's scale `rsqrtf(mean(x^2) + eps)` for one row, with ggml's
// partition of the sum over its block, returned to every thread. The block
// needs at least rms_shape(ncols).bs threads; the rest only join barriers.
// Callers apply `scale * x[i] * w[i]` element by element, as ggml does.
static __device__ float rms_scale(const float * x, int ncols, float eps, float * s_sum) {
    const RmsShape sh = rms_shape(ncols);
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    float tmp = 0.0f;
    if (tid < sh.bs) {
        if (sh.vec) {
            const int ncols4 = ncols / 4;
            for (int k = 0; k < sh.vpt; k++) {
                const int i = tid + k*sh.bs;
                if (i < ncols4) {
                    const float4 xs = ((const float4 *) x)[i];
                    tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
                }
            }
        } else {
            for (int col = tid; col < ncols; col += sh.bs) {
                const float xi = x[col];
                tmp += xi * xi;
            }
        }
    }
    tmp = warp_sum(tmp);
    __syncthreads();
    if (sh.bs > 32) {
        if (lane == 0) s_sum[tid / 32] = tmp;
        __syncthreads();
        tmp = lane < sh.bs / 32 ? s_sum[lane] : 0.0f;
        tmp = warp_sum(tmp);
        __syncthreads();
    }
    float scale;
    if (sh.vec) {
        scale = rsqrtf(tmp / ncols + eps);
    } else {
        const float mean = tmp / ncols;
        scale = rsqrtf(mean + eps);
    }
    // Warp 0 holds the result when ggml's block is one warp.
    if (tid == 0) s_sum[0] = scale;
    __syncthreads();
    scale = s_sum[0];
    __syncthreads();
    return scale;
}

template<int QK, int BS, int QI, int VDR, int NWARPS = 4, int RPB = 1>
struct TypeTraitsBase {
    static constexpr int qk = QK;
    static constexpr int bs = BS;
    static constexpr int qi = QI;
    static constexpr int vdr = VDR;
    static constexpr int nwarps = NWARPS;
    static constexpr int rpb = RPB;
};

// MMVQ, ncols_dst=1, Turing and newer. The requested rpb values include the
// two-row low-bit types also used by ggml's generic table.
// mul_mat_vec_q's warps per block for one token (calc_nwarps): 4 for every
// type on Ampere and newer (its GENERIC table), 2 for the K-quants on Turing.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
static constexpr int KQUANT_NWARPS = 2;
#else
static constexpr int KQUANT_NWARPS = 4;
#endif

template<int TYPE> struct TypeTraits;
template<> struct TypeTraits<0> : TypeTraitsBase<1, 4, 1, 1> {};
template<> struct TypeTraits<1> : TypeTraitsBase<1, 2, 1, 1> {};
template<> struct TypeTraits<2> : TypeTraitsBase<1, 2, 1, 1> {};
template<> struct TypeTraits<3> : TypeTraitsBase<QK8_0, sizeof(block_q8_0), QI8_0, 2> {};
template<> struct TypeTraits<4> : TypeTraitsBase<QK4_0, sizeof(block_q4_0), QI4_0, 2> {};
template<> struct TypeTraits<5> : TypeTraitsBase<QK4_1, sizeof(block_q4_1), QI4_1, 2> {};
template<> struct TypeTraits<6> : TypeTraitsBase<QK5_0, sizeof(block_q5_0), QI5_0, 2> {};
template<> struct TypeTraits<7> : TypeTraitsBase<QK5_1, sizeof(block_q5_1), QI5_1, 2> {};
template<> struct TypeTraits<8> : TypeTraitsBase<QK4_NL, sizeof(block_iq4_nl), QI4_NL, 2> {};
template<> struct TypeTraits<9> : TypeTraitsBase<QK_MXFP4, sizeof(block_mxfp4), QI_MXFP4, 2> {};
template<> struct TypeTraits<10> : TypeTraitsBase<QK_K, sizeof(block_q2_K), QI2_K, 2, KQUANT_NWARPS, 2> {};
template<> struct TypeTraits<11> : TypeTraitsBase<QK_K, sizeof(block_q3_K), QI3_K, 2, KQUANT_NWARPS, 2> {};
template<> struct TypeTraits<12> : TypeTraitsBase<QK_K, sizeof(block_q4_K), QI4_K, 2, KQUANT_NWARPS> {};
template<> struct TypeTraits<13> : TypeTraitsBase<QK_K, sizeof(block_q5_K), QI5_K, 2, KQUANT_NWARPS> {};
template<> struct TypeTraits<14> : TypeTraitsBase<QK_K, sizeof(block_q6_K), QI6_K, 2, KQUANT_NWARPS> {};
template<> struct TypeTraits<15> : TypeTraitsBase<QK_K, sizeof(block_iq4_xs), QI4_XS, 4> {};
template<> struct TypeTraits<16> : TypeTraitsBase<QK_K, sizeof(block_iq2_xxs), QI2_XXS, 2> {};
template<> struct TypeTraits<17> : TypeTraitsBase<QK_K, sizeof(block_iq2_xs), QI2_XS, 2> {};
template<> struct TypeTraits<18> : TypeTraitsBase<QK_K, sizeof(block_iq2_s), QI2_S, 2> {};
template<> struct TypeTraits<19> : TypeTraitsBase<QK_K, sizeof(block_iq3_xxs), QI3_XXS, 2> {};
template<> struct TypeTraits<20> : TypeTraitsBase<QK_K, sizeof(block_iq3_s), QI3_S, 2> {};
template<> struct TypeTraits<21> : TypeTraitsBase<QK_K, sizeof(block_iq1_s), QI1_S, 1, 4, 2> {};
template<> struct TypeTraits<22> : TypeTraitsBase<QK_K, sizeof(block_iq1_m), QI1_M, 1, 4, 2> {};
template<> struct TypeTraits<23> : TypeTraitsBase<QK2_0, sizeof(block_q2_0), QI2_0, 1> {};
