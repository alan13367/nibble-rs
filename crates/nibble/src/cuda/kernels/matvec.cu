// Matrix-vector products, activation quantization and embedding lookups.
//
// The quantized products follow ggml-cuda's `mul_mat_vec_q` (b12675): one
// Q8_1 activation row per token, the same per-thread work partition and
// reduction order. NC > 1 only shares weight loads between independent token
// outputs; it does not change any token's arithmetic.

#include "common.cuh"
#include "quant.cuh"

struct MatvecArgs {
    unsigned int cols;
    unsigned int rows;
    unsigned int row_bytes;
    unsigned int ncols;
    unsigned int mode;
};

struct MoeArgs {
    unsigned int k;
    unsigned int per_slot;
    unsigned int expert_rows;
    unsigned int count;
};

#define MATVEC_STORE 0
#define MATVEC_ADD 1
#define MATVEC_SWIGLU 2

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ void store_result(
        float * dst, int row, int token, int rows, int mode, float sum, float up) {
    float * out = dst + (size_t) token * rows + row;
    if (mode == MATVEC_SWIGLU) {
        *out = silu(sum) * up;
    } else if (mode == MATVEC_ADD) {
        *out += sum;
    } else {
        *out = sum;
    }
}

template<int TYPE, int NC>
__device__ __forceinline__ void matvec_q_impl(
        const unsigned char * w, const unsigned char * w2, const block_q8_1 * x,
        float * dst, MatvecArgs a, const Codebooks * cb, int token0) {
    constexpr int qk = TypeTraits<TYPE>::qk;
    constexpr int qi = TypeTraits<TYPE>::qi;
    constexpr int vdr = TypeTraits<TYPE>::vdr;
    constexpr int nwarps = TypeTraits<TYPE>::nwarps;
    constexpr int rpb = TypeTraits<TYPE>::rpb;
    const int tid = 32 * threadIdx.y + threadIdx.x;
    const int row0 = rpb * blockIdx.x;
    const int blocks_per_row = a.cols / qk;
    constexpr int blocks_per_iter = vdr * nwarps * 32 / qi;
    const int tokens = min(NC, (int) a.ncols - token0);
    const int blocks_per_weight_row = a.row_bytes / TypeTraits<TYPE>::bs;

    float tmp[NC][rpb] = {};
    float up[NC][rpb] = {};
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / 32);
        const int iqs = vdr * (tid % (qi / vdr));
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j < tokens) {
                const block_q8_1 * y = x + (size_t) (token0 + j) * (a.cols / 32) + kby;
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    const int row = min(row0 + i, (int) a.rows - 1);
                    const int at = row * blocks_per_weight_row + kbx;
                    tmp[j][i] += vec_dot_q<TYPE>(w, y, at, iqs, cb);
                    if (a.mode == MATVEC_SWIGLU) {
                        up[j][i] += vec_dot_q<TYPE>(w2, y, at, iqs, cb);
                    }
                }
            }
        }
    }

    __shared__ float partial[nwarps > 1 ? nwarps - 1 : 1][NC][rpb][32];
    __shared__ float partial_up[nwarps > 1 ? nwarps - 1 : 1][NC][rpb][32];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
                partial_up[threadIdx.y - 1][j][i][threadIdx.x] = up[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        if (j < tokens) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps - 1; ++l) {
                    tmp[j][i] += partial[l][j][i][threadIdx.x];
                    up[j][i] += partial_up[l][j][i][threadIdx.x];
                }
                tmp[j][i] = warp_sum(tmp[j][i]);
                up[j][i] = warp_sum(up[j][i]);
                const int row = row0 + i;
                if (threadIdx.x == i && row < (int) a.rows) {
                    store_result(dst, row, token0 + j, a.rows, a.mode, tmp[j][i], up[j][i]);
                }
            }
        }
    }
}

// Two consecutive weights of a row as floats.
template<int TYPE>
__device__ __forceinline__ float2 float_pair(const unsigned char * row, int col2) {
    if constexpr (TYPE == 0) {
        return ((const float2 *) row)[col2];
    } else if constexpr (TYPE == 1) {
        return __half22float2(((const half2 *) row)[col2]);
    } else {
        const nv_bfloat162 v = ((const nv_bfloat162 *) row)[col2];
        return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
    }
}

// ggml-cuda's mul_mat_vec_f (b12675) for F32, F16 and BF16 weights: one
// block of mmvf_block_size(cols) threads per row (the launch's block), each
// thread taking pairs of columns, the activation used as it is (no rounding
// to the weight type). F16 accumulates in half2, as a product with no
// precision request does on GPUs with fast FP16, except when the SwiGLU gate
// is fused (ggml then accumulates in f32). Each of the NC tokens gets the
// arithmetic of a product of its own.
template<int TYPE, int NC>
__device__ __forceinline__ void matvec_float_impl(
        const unsigned char * w, const unsigned char * w2, const float * x,
        float * dst, MatvecArgs a, int token0) {
    const int row = blockIdx.x;
    const int tokens = min(NC, (int) a.ncols - token0);
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const bool gate = a.mode == MATVEC_SWIGLU;
    const int ncols2 = a.cols / 2;
    __shared__ float buf_iw[32];
    __shared__ float buf_iw_gate[32];
    if (bs > 32) {
        if (tid < 32) {
            buf_iw[tid] = 0.0f;
            buf_iw_gate[tid] = 0.0f;
        }
        __syncthreads();
    }
    const unsigned char * row_w = w + (size_t) row * a.row_bytes;
    const unsigned char * row_up = w2 + (size_t) row * a.row_bytes;
    float sumf[NC] = {};
    float sumf_gate[NC] = {};
    if (TYPE == 1 && !gate) {
        half2 sumh2[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) sumh2[j] = make_half2(0.0f, 0.0f);
        for (int col2 = tid; col2 < ncols2; col2 += bs) {
            const half2 tmpx = ((const half2 *) row_w)[col2];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                if (j < tokens) {
                    const float2 tmpy = ((const float2 *) (x + (size_t) (token0 + j) * a.cols))[col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
    } else {
        for (int col2 = tid; col2 < ncols2; col2 += bs) {
            // In SWIGLU mode `w` is the gate and `w2` the up projection:
            // ggml's main product is the up one, with the gate fused.
            const float2 tmpx = float_pair<TYPE>(gate ? row_up : row_w, col2);
            const float2 tmpx_gate = gate ? float_pair<TYPE>(row_w, col2) : make_float2(0.0f, 0.0f);
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                if (j < tokens) {
                    const float2 tmpy = ((const float2 *) (x + (size_t) (token0 + j) * a.cols))[col2];
                    sumf[j] += tmpx.x*tmpy.x;
                    sumf[j] += tmpx.y*tmpy.y;
                    if (gate) {
                        sumf_gate[j] += tmpx_gate.x*tmpy.x;
                        sumf_gate[j] += tmpx_gate.y*tmpy.y;
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        sumf[j] = warp_sum(sumf[j]);
        if (gate) sumf_gate[j] = warp_sum(sumf_gate[j]);
        if (bs > 32) {
            buf_iw[tid/32] = sumf[j];
            if (gate) buf_iw_gate[tid/32] = sumf_gate[j];
            __syncthreads();
            if (tid < 32) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_sum(sumf[j]);
                if (gate) {
                    sumf_gate[j] = buf_iw_gate[tid];
                    sumf_gate[j] = warp_sum(sumf_gate[j]);
                }
            }
            __syncthreads();
        }
    }
    if (tid != 0 || row >= (int) a.rows) {
        return;
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        if (j < tokens) {
            float * out = dst + (size_t) (token0 + j) * a.rows + row;
            if (gate) {
                *out = sumf[j] * silu(sumf_gate[j]);
            } else if (a.mode == MATVEC_ADD) {
                *out += sumf[j];
            } else {
                *out = sumf[j];
            }
        }
    }
}

// One warp quantizes each block of 32, as ggml-cuda's `quantize_q8_1`.
extern "C" __global__ void __launch_bounds__(128, 1) quantize_q8_1(
        const float * x, block_q8_1 * out, unsigned int n) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int block = 4 * blockIdx.x + warp;
    if (block >= (int) (n / 32)) {
        return;
    }
    const float v = x[32 * block + lane];
    float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : (int) roundf(v / d);
    const int sum = warp_sum(q);
    out[block].qs[lane] = (signed char) q;
    if (lane == 0) {
        out[block].ds = __floats2half2_rn(d, d * sum);
    }
}

struct EmbedArgs {
    unsigned int cols;
    unsigned int row;
    unsigned int row_bytes;
    unsigned int from_slot;
};

// Explicit exports give the host one unmangled function name per type.
// Embedding takes the same uploaded codebooks as the products.
#define EMBED_BODY(TYPE) \
    constexpr int qk = TypeTraits<TYPE>::qk; \
    __shared__ float values[qk]; \
    const int t = blockIdx.y; \
    const int block = blockIdx.x; \
    if (block >= (int) (a.cols / qk)) return; \
    const unsigned int row = a.from_slot ? tokens[t] : a.row; \
    if (threadIdx.x == 0) dequant_block<TYPE>(w + (size_t) row * a.row_bytes, block, values, cb); \
    __syncthreads(); \
    for (int i = threadIdx.x; i < qk; i += blockDim.x) out[(size_t) t * a.cols + block * qk + i] = values[i];

#define EXPORT_TYPE(NAME, TYPE) \
    extern "C" __global__ void matvec_##NAME(const unsigned char * w, const unsigned char * w2, const block_q8_1 * x, float * dst, MatvecArgs a, const Codebooks * cb) { matvec_q_impl<TYPE, 1>(w, w2, x, dst, a, cb, blockIdx.y); } \
    extern "C" __global__ void matvec_##NAME##_c4(const unsigned char * w, const unsigned char * w2, const block_q8_1 * x, float * dst, MatvecArgs a, const Codebooks * cb) { matvec_q_impl<TYPE, 4>(w, w2, x, dst, a, cb, 4 * blockIdx.y); } \
    extern "C" __global__ void matvec_##NAME##_id(const unsigned char * w, const unsigned char * w2, const block_q8_1 * x, float * dst, const float * route, MatvecArgs a, MoeArgs m, const Codebooks * cb) { \
        const int s = blockIdx.y; \
        const int e = min((unsigned int) route[(size_t) (s / m.k) * 2 * m.k + s % m.k], m.count - 1); \
        const size_t off = (size_t) e * m.expert_rows * a.row_bytes; \
        const int token = m.per_slot ? s : s / m.k; \
        matvec_q_impl<TYPE, 1>(w + off, w2 + off, x + (size_t) token * (a.cols / 32), dst + (size_t) s * a.rows, a, cb, 0); \
    } \
    extern "C" __global__ void embed_##NAME(const unsigned char * w, float * out, const unsigned int * tokens, EmbedArgs a, const Codebooks * cb) { EMBED_BODY(TYPE) }

EXPORT_TYPE(q8_0, 3)
EXPORT_TYPE(q4_0, 4)
EXPORT_TYPE(q4_1, 5)
EXPORT_TYPE(q5_0, 6)
EXPORT_TYPE(q5_1, 7)
EXPORT_TYPE(iq4_nl, 8)
EXPORT_TYPE(mxfp4, 9)
EXPORT_TYPE(q2_K, 10)
EXPORT_TYPE(q3_K, 11)
EXPORT_TYPE(q4_K, 12)
EXPORT_TYPE(q5_K, 13)
EXPORT_TYPE(q6_K, 14)
EXPORT_TYPE(iq4_xs, 15)
EXPORT_TYPE(iq2_xxs, 16)
EXPORT_TYPE(iq2_xs, 17)
EXPORT_TYPE(iq2_s, 18)
EXPORT_TYPE(iq3_xxs, 19)
EXPORT_TYPE(iq3_s, 20)
EXPORT_TYPE(iq1_s, 21)
EXPORT_TYPE(iq1_m, 22)
EXPORT_TYPE(q2_0, 23)

#define EXPORT_FLOAT(NAME, TYPE) \
    extern "C" __global__ void matvec_##NAME(const unsigned char * w, const unsigned char * w2, const float * x, float * dst, MatvecArgs a) { matvec_float_impl<TYPE, 1>(w, w2, x, dst, a, blockIdx.y); } \
    extern "C" __global__ void matvec_##NAME##_c4(const unsigned char * w, const unsigned char * w2, const float * x, float * dst, MatvecArgs a) { matvec_float_impl<TYPE, 4>(w, w2, x, dst, a, 4 * blockIdx.y); } \
    extern "C" __global__ void matvec_##NAME##_id(const unsigned char * w, const unsigned char * w2, const float * x, float * dst, const float * route, MatvecArgs a, MoeArgs m) { \
        const int s = blockIdx.y; \
        const int e = min((unsigned int) route[(size_t) (s / m.k) * 2 * m.k + s % m.k], m.count - 1); \
        const size_t off = (size_t) e * m.expert_rows * a.row_bytes; \
        const int token = m.per_slot ? s : s / m.k; \
        matvec_float_impl<TYPE, 1>(w + off, w2 + off, x + (size_t) token * a.cols, dst + (size_t) s * a.rows, a, 0); \
    } \
    extern "C" __global__ void embed_##NAME(const unsigned char * w, float * out, const unsigned int * tokens, EmbedArgs a) { \
        constexpr int qk = TypeTraits<TYPE>::qk; \
        __shared__ float values[qk]; \
        const int t = blockIdx.y; \
        const int block = blockIdx.x; \
        if (block >= (int) (a.cols / qk)) return; \
        const unsigned int row = a.from_slot ? tokens[t] : a.row; \
        if (threadIdx.x == 0) dequant_block<TYPE>(w + (size_t) row * a.row_bytes, block, values, nullptr); \
        __syncthreads(); \
        for (int i = threadIdx.x; i < qk; i += blockDim.x) out[(size_t) t * a.cols + block * qk + i] = values[i]; \
    }

EXPORT_FLOAT(f32, 0)
EXPORT_FLOAT(f16, 1)
EXPORT_FLOAT(bf16, 2)
