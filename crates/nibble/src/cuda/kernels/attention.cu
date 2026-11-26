// Q/K normalization and RoPE, cache append, and causal attention.
//
// F16, BF16 and F32 caches follow llama.cpp's attention without flash
// attention on ggml-cuda (b12675), one token at a time as with `-ub 1`:
// KQ = mul_mat_vec_f(K, q) with an f32 accumulator, soft_max_f32 over the
// row padded to llama.cpp's n_kv, and KQV = mul_mat_vec_f(V^T, probs), which
// accumulates in half2 for an F16 cache (KQV carries no precision request).
// Quantized caches, which llama.cpp only runs with flash attention, use one
// block per query head with an online softmax instead.

#include "common.cuh"
#include "kv.cuh"

struct AttentionArgs {
    unsigned int n_pos;
    unsigned int n;
    unsigned int head_dim;
    unsigned int group;
    unsigned int kv_dim;
    unsigned int q_stride;
    unsigned int gated;
    unsigned int n_rot;
    unsigned int norm;
    float scale;
    float base;
    float eps;
    unsigned int k_type;
    unsigned int v_type;
    unsigned int k_row;
    unsigned int v_row;
    unsigned int rot_k;
    unsigned int rot_v;
    unsigned int n_head;
    // ggml's attention path: the first token of this chunk of the batch, the
    // row stride of the score buffer, and the cache size rounded up to 256.
    unsigned int t0;
    unsigned int n_kv_max;
    unsigned int cap_pad;
    // llama.cpp's RoPE is multi-section (rope_multi, after a separate
    // rms_norm) rather than NEOX (fused with the norm).
    unsigned int multi_rope;
    // KQ runs as mul_mat_f (tensor cores) for tokens whose n_kv exceeds 512,
    // and the cache's allocated positions (reads past them load zeros).
    unsigned int kq_mma;
    unsigned int n_cells;
    // V is stored transposed, as llama.cpp stores it without flash attention:
    // value d of KV head kh at position p is element (kh * head_dim + d) *
    // n_cells + p.
    unsigned int v_trans;
};

// Value d of KV head kh at position p of a transposed F16, BF16 or F32 cache.
__device__ __forceinline__ size_t vt_index(const AttentionArgs & a, unsigned int kh, unsigned int d, unsigned int p) {
    return ((size_t) kh * a.head_dim + d) * a.n_cells + p;
}

__device__ __forceinline__ float sigmoid_att(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ void norm_rope_att(const AttentionArgs & a, const float * x, const float * w,
        unsigned int pos, float * tmp, float * partial) {
    const int hd = a.head_dim;
    // ggml-cuda fuses RMS_NORM + MUL + ROPE for these heads
    // (rms_norm_mul_rope_f32): a sum strided over its 256-thread block, which
    // is this launch's block, then `scale * x * w`.
    // For multi-section RoPE, ggml-cuda runs rms_norm_f32 (fused with the
    // MUL) and then rope_multi.
    float scale = 1.0f;
    if (a.norm && a.multi_rope) {
        scale = rms_scale(x, hd, a.eps, partial);
    } else if (a.norm) {
        float sum = 0.0f;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) sum += x[d] * x[d];
        scale = rsqrtf(block_sum(sum, partial) / hd + a.eps);
    }
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        tmp[d] = a.norm ? scale * x[d] * w[d] : x[d];
    }
    __syncthreads();
    // Each rotated value reads its partner's input, so all reads finish
    // before any thread writes back. One thread handles at most two values
    // (heads of up to 512 over 256 threads).
    const int half_rot = a.n_rot / 2;
    float rotated[2];
    for (int d = threadIdx.x, r = 0; d < hd; d += blockDim.x, ++r) {
        float v = tmp[d];
        if (d < (int) a.n_rot) {
            const int i = d < half_rot ? d : d - half_rot;
            const float theta_base = pos*powf(a.base, (2*i)/2.0f);
            const float cos_theta = cosf(theta_base) * 1.0f;
            const float sin_theta = sinf(theta_base) * 1.0f;
            const float x0 = tmp[i], x1 = tmp[i + half_rot];
            // The two kernels' compiled code fuses the second sum differently.
            if (d < half_rot) {
                v = x0 * cos_theta - x1 * sin_theta;
            } else if (a.multi_rope) {
                v = fmaf(x0, sin_theta, x1 * cos_theta);
            } else {
                v = fmaf(x1, cos_theta, x0 * sin_theta);
            }
        }
        rotated[r] = v;
    }
    __syncthreads();
    for (int d = threadIdx.x, r = 0; d < hd; d += blockDim.x, ++r) tmp[d] = rotated[r];
    __syncthreads();
}

extern "C" __global__ void attention_prep(float * q, const float * k_new, const float * v_new,
        const float * q_norm, const float * k_norm, unsigned char * k_cache,
        unsigned char * v_cache, AttentionArgs a) {
    extern __shared__ float tmp[];
    __shared__ float partial[32];
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int hd = a.head_dim;
    const int pos = a.n_pos + t;
    if (head < (int) a.n_head) {
        float * dst = q + (size_t) t * a.n_head * a.q_stride + head * a.q_stride;
        norm_rope_att(a, dst, q_norm, pos, tmp, partial);
        if (a.rot_k) kv_hadamard(tmp, hd, a.rot_k);
        for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = tmp[d];
    } else {
        const int kh = head - a.n_head;
        const int off = kh * hd;
        norm_rope_att(a, k_new + (size_t) t * a.kv_dim + off, k_norm, pos, tmp, partial);
        if (a.rot_k) kv_hadamard(tmp, hd, a.rot_k);
        kv_store_head(a.k_type, tmp, k_cache + (size_t) pos * a.k_row + kh * kv_bytes(a.k_type, hd), hd);
        if (a.v_trans) {
            for (int d = threadIdx.x; d < hd; d += blockDim.x) {
                const float v = v_new[(size_t) t * a.kv_dim + off + d];
                const size_t at = vt_index(a, kh, d, pos);
                if (a.v_type == KV_F16) ((half *) v_cache)[at] = __float2half_rn(v);
                else if (a.v_type == KV_BF16) ((nv_bfloat16 *) v_cache)[at] = __float2bfloat16_rn(v);
                else ((float *) v_cache)[at] = v;
            }
            return;
        }
        for (int d = threadIdx.x; d < hd; d += blockDim.x) tmp[d] = v_new[(size_t) t * a.kv_dim + off + d];
        if (a.rot_v) kv_hadamard(tmp, hd, a.rot_v);
        kv_store_head(a.v_type, tmp, v_cache + (size_t) pos * a.v_row + kh * kv_bytes(a.v_type, hd), hd);
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) attention(const float * q,
        const unsigned char * k_cache, const unsigned char * v_cache, float * out,
        AttentionArgs a) {
    constexpr int WARPS = 8;
    constexpr int MAX_CHUNKS = 16; // 32 lanes * 16 = heads of 512 values
    __shared__ float warp_m[WARPS];
    __shared__ float warp_l[WARPS];
    __shared__ float acc[WARPS * 512];
    __shared__ float result[512];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int hd = a.head_dim;
    const int end = a.n_pos + t + 1;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float * query = q + (size_t) t * a.n_head * a.q_stride + h * a.q_stride;
    const int kh = h / a.group;

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float local[MAX_CHUNKS] = {};
    for (int pos = warp; pos < end; pos += WARPS) {
        const unsigned char * krow = k_cache + (size_t) pos * a.k_row + kh * kv_bytes(a.k_type, hd);
        float score = 0.0f;
        for (int d = lane; d < hd; d += 32) score += query[d] * kv_load(a.k_type, krow, d);
        score = warp_sum(score) * a.scale;
        const float mn = fmaxf(m, score);
        const float old_scale = expf(m - mn);
        const float e = expf(score - mn);
        const unsigned char * vrow = v_cache + (size_t) pos * a.v_row + kh * kv_bytes(a.v_type, hd);
        l = l * old_scale + e;
#pragma unroll
        for (int j = 0; j < MAX_CHUNKS; ++j) {
            const int d = lane + 32 * j;
            if (d < hd) local[j] = local[j] * old_scale + e * kv_load(a.v_type, vrow, d);
        }
        m = mn;
    }
    if (lane == 0) {
        warp_m[warp] = m;
        warp_l[warp] = l;
    }
#pragma unroll
    for (int j = 0; j < MAX_CHUNKS; ++j) {
        const int d = lane + 32 * j;
        if (d < hd) acc[warp * hd + d] = local[j];
    }
    __syncthreads();
    float global_m = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) global_m = fmaxf(global_m, warp_m[w]);
    float global_l = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) global_l += warp_l[w] * expf(warp_m[w] - global_m);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) v += acc[w * hd + d] * expf(warp_m[w] - global_m);
        result[d] = v / global_l;
    }
    __syncthreads();
    if (a.rot_v) kv_hadamard(result, hd, a.rot_v);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = result[d];
        if (a.gated) v *= sigmoid_att(query[hd + d]);
        out[((size_t) t * a.n_head + h) * hd + d] = v;
    }
}

// llama.cpp's n_kv for the token at position `n_pos + t`: the used cells
// (including the token) rounded up to 256, at least 256, at most the cache.
__device__ __forceinline__ int att_n_kv(const AttentionArgs & a, unsigned int t) {
    const unsigned int used = a.n_pos + t + 1;
    return min(a.cap_pad, max(256u, (used + 255u) / 256u * 256u));
}

// mul_mat_vec_f's block size for rows of `ncols` values (launch_mul_mat_vec_f_cuda).
__device__ __forceinline__ int mmvf_block_size(int ncols) {
    int best = 32;
    int niter_best = (ncols + 2*32 - 1) / (2*32);
    for (int bs = 2*32; bs <= 256; bs += 32) {
        const int niter = (ncols + 2*bs - 1) / (2*bs);
        if (niter < niter_best) {
            niter_best = niter;
            best = bs;
        }
    }
    return best;
}

// Two consecutive cache values of one head as floats.
__device__ __forceinline__ float2 kv_pair(unsigned int ty, const unsigned char * head, int col2) {
    if (ty == KV_F16) return __half22float2(((const half2 *) head)[col2]);
    if (ty == KV_BF16) {
        const nv_bfloat162 v = ((const nv_bfloat162 *) head)[col2];
        return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
    }
    return ((const float2 *) head)[col2];
}

// The end of mul_mat_vec_f: a warp sum, then the warps' sums through shared
// memory when the block has more than one warp. `buf_iw` starts zeroed, so
// the slots of warps that hold no part of the row add zero, as in ggml.
__device__ __forceinline__ float mmvf_reduce(float sumf, float * buf_iw, int bs) {
    const int tid = threadIdx.x;
    sumf = warp_sum(sumf);
    if (bs > 32) {
        buf_iw[tid/32] = sumf;
        __syncthreads();
        if (tid < 32) {
            sumf = buf_iw[tid];
            sumf = warp_sum(sumf);
        }
    }
    return sumf;
}

// KQ: one block per (cache position, query head, token of the chunk), with
// mul_mat_vec_f's block size for a head (the launch's block). Masked
// positions of the padded row are written as -inf, which is what the
// softmax's `x*scale + mask` gives them.
extern "C" __global__ void attention_kq(const float * q, const unsigned char * k_cache,
        float * scores, AttentionArgs a) {
    __shared__ float buf_iw[32];
    const int p = blockIdx.x;
    const int h = blockIdx.y;
    const unsigned int t = a.t0 + blockIdx.z;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    float * row = scores + ((size_t) blockIdx.z * a.n_head + h) * a.n_kv_max;
    if (p >= att_n_kv(a, t) || (a.kq_mma && att_n_kv(a, t) > 512)) return;
    if ((unsigned int) p > a.n_pos + t) {
        if (tid == 0) row[p] = -CUDART_INF_F;
        return;
    }
    if (bs > 32) {
        if (tid < 32) buf_iw[tid] = 0.0f;
        __syncthreads();
    }
    const float2 * y2 = (const float2 *) (q + ((size_t) t * a.n_head + h) * a.q_stride);
    const unsigned char * krow = k_cache + (size_t) p * a.k_row + (h / a.group) * kv_bytes(a.k_type, a.head_dim);
    float sumf = 0.0f;
    for (int col2 = tid; col2 < (int) a.head_dim / 2; col2 += bs) {
        const float2 tmpx = kv_pair(a.k_type, krow, col2);
        const float2 tmpy = y2[col2];
        sumf += tmpx.x*tmpy.x;
        sumf += tmpx.y*tmpy.y;
    }
    sumf = mmvf_reduce(sumf, buf_iw, bs);
    if (tid == 0) row[p] = sumf;
}

// soft_max_f32 with ggml's block size for the token's row (the smallest
// power of two from 32 that covers it, at most 1024). The launch's block is
// the chunk's largest; threads past the token's size only take part in the
// reductions, where ggml reads the first bs/32 warp slots. The row is
// rewritten in place (ggml's choice when it is too long for shared memory;
// either way each value goes through the same operations).
extern "C" __global__ void attention_softmax(float * scores, AttentionArgs a) {
    __shared__ float buf_iw[32];
    const int h = blockIdx.x;
    const unsigned int t = a.t0 + blockIdx.y;
    const int tid = threadIdx.x;
    float * x = scores + ((size_t) blockIdx.y * a.n_head + h) * a.n_kv_max;
    float * vals = x;
    const int ncols = att_n_kv(a, t);
    int bs = 32;
    while (bs < ncols && bs < 1024) bs *= 2;
    const int lane = tid % 32;
    const int warp = tid / 32;

    float max_val = -CUDART_INF_F;
    if (tid < bs) {
        for (int col0 = 0; col0 < ncols; col0 += bs) {
            const int col = col0 + tid;
            if (col >= ncols) break;
            const float val = x[col]*a.scale + 0.0f;
            vals[col] = val;
            max_val = max(max_val, val);
        }
    }
    max_val = warp_max(max_val);
    if (bs > 32) {
        if (lane == 0) buf_iw[warp] = max_val;
        __syncthreads();
        max_val = lane < bs / 32 ? buf_iw[lane] : -CUDART_INF_F;
        max_val = warp_max(max_val);
    }

    float tmp = 0.0f;
    if (tid < bs) {
        for (int col0 = 0; col0 < ncols; col0 += bs) {
            const int col = col0 + tid;
            if (col >= ncols) break;
            const float val = expf(vals[col] - max_val);
            tmp += val;
            vals[col] = val;
        }
    }
    if (bs > 32) __syncthreads();
    tmp = warp_sum(tmp);
    if (bs > 32) {
        if (lane == 0) buf_iw[warp] = tmp;
        __syncthreads();
        tmp = lane < bs / 32 ? buf_iw[lane] : 0.0f;
        tmp = warp_sum(tmp);
    }
    const float inv_sum = 1.0f / tmp;
    if (tid < bs) {
        for (int col0 = 0; col0 < ncols; col0 += bs) {
            const int col = col0 + tid;
            if (col >= ncols) break;
            x[col] = vals[col] * inv_sum;
        }
    }
}

// KQV: one block per (value index d, query head, token of the chunk). The
// launch block has 256 threads (mul_mat_vec_f's largest), of which the
// token's block size for its padded row take part. Positions past the token
// have probability zero; their values are read as zero, giving the same
// zero products ggml adds.
extern "C" __global__ void attention_kqv(const float * probs, const unsigned char * v_cache,
        const float * q, float * out, AttentionArgs a) {
    __shared__ float buf_iw[32];
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const unsigned int t = a.t0 + blockIdx.z;
    const int tid = threadIdx.x;
    const int ncols = att_n_kv(a, t);
    const int bs = mmvf_block_size(ncols);
    const unsigned int last = a.n_pos + t;
    const float * y = probs + ((size_t) blockIdx.z * a.n_head + h) * a.n_kv_max;
    const unsigned int kh = h / a.group;
    if (tid < 32) buf_iw[tid] = 0.0f;
    __syncthreads();
    float sumf = 0.0f;
    if (a.v_type == KV_F16) {
        half2 sumh2 = {0.0f, 0.0f};
        if (tid < bs) {
            for (int col2 = tid; col2 < ncols / 2; col2 += bs) {
                const unsigned int p0 = 2 * col2;
                const half * vt = (const half *) v_cache + vt_index(a, kh, d, 0);
                const half x0 = p0 <= last ? vt[p0] : __float2half(0.0f);
                const half x1 = p0 + 1 <= last ? vt[p0 + 1] : __float2half(0.0f);
                const half2 tmpx = __halves2half2(x0, x1);
                const float2 tmpy = ((const float2 *) y)[col2];
                sumh2 += tmpx * make_half2(tmpy.x, tmpy.y);
            }
        }
        sumf = __low2float(sumh2) + __high2float(sumh2);
    } else if (tid < bs) {
        for (int col2 = tid; col2 < ncols / 2; col2 += bs) {
            const unsigned int p0 = 2 * col2;
            const size_t at = vt_index(a, kh, d, 0);
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (a.v_type == KV_BF16) {
                const nv_bfloat16 * vt = (const nv_bfloat16 *) v_cache + at;
                if (p0 <= last) x0 = __bfloat162float(vt[p0]);
                if (p0 + 1 <= last) x1 = __bfloat162float(vt[p0 + 1]);
            } else {
                const float * vt = (const float *) v_cache + at;
                if (p0 <= last) x0 = vt[p0];
                if (p0 + 1 <= last) x1 = vt[p0 + 1];
            }
            const float2 tmpy = ((const float2 *) y)[col2];
            sumf += x0*tmpy.x;
            sumf += x1*tmpy.y;
        }
    }
    sumf = mmvf_reduce(sumf, buf_iw, 256);
    if (tid == 0) {
        float v = sumf;
        const float * query = q + ((size_t) t * a.n_head + h) * a.q_stride;
        if (a.gated) v *= sigmoid_att(query[a.head_dim + d]);
        out[((size_t) t * a.n_head + h) * a.head_dim + d] = v;
    }
}

// ggml-cuda's mul_mat_f (mmf.cuh, b12675) for KQ with one token: ggml takes
// it over mul_mat_vec_f for an F16 (Turing+) or BF16 (Ampere+) cache once
// n_kv exceeds 512 with several KV heads. A block computes 32 positions of a
// head with blockDim.y warps splitting the head's columns; K rows and the
// query (rounded to the cache type) go through shared memory into ldmatrix
// fragments and m16n8k16 MMAs with f32 accumulators, in ggml's order. The
// tile helpers transcribe mma.cuh's NVIDIA layouts.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
static constexpr int MMF_ROWS = 32;
static constexpr int MMF_TILE_K = 32 + 4; // warp_size + mmf_get_padding()

// tile<16, 8, half2>: ldmatrix.x4 from rows of `stride` 32-bit words.
__device__ __forceinline__ void mmf_load_a(int (&xi)[4], const unsigned int * xs0, int stride) {
    const int * xs = (const int *) xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
}

// tile<8, 8, half2>: ldmatrix.x2.
__device__ __forceinline__ void mmf_load_b(int (&xi)[2], const unsigned int * xs0, int stride) {
    const int * xs = (const int *) xs0 + (threadIdx.x % 8) * stride + ((threadIdx.x / 8) * 4) % 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
}

template<bool BF16>
__device__ __forceinline__ void mmf_mma(float (&d)[4], const int (&a)[4], const int (&b)[2]) {
    int * di = (int *) d;
    if constexpr (BF16) {
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(di[0]), "+r"(di[1]), "+r"(di[2]), "+r"(di[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
    } else {
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(di[0]), "+r"(di[1]), "+r"(di[2]), "+r"(di[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
        // Turing has no m16n8k16: two m16n8k8, as ggml.
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(di[0]), "+r"(di[1]), "+r"(di[2]), "+r"(di[3])
            : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(di[0]), "+r"(di[1]), "+r"(di[2]), "+r"(di[3])
            : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#endif
    }
}

template<bool BF16>
__device__ __forceinline__ void attention_kq_mma_impl(const float * q, const unsigned char * k_cache,
        float * scores, const AttentionArgs & a) {
    extern __shared__ unsigned int data_mmf[];
    const int h = blockIdx.y;
    const unsigned int t = a.t0 + blockIdx.z;
    const int row0 = blockIdx.x * MMF_ROWS;
    const int lane = threadIdx.x;
    const int w = threadIdx.y;
    const int nwarps = blockDim.y;
    const int n_kv = att_n_kv(a, t);
    if (n_kv <= 512 || row0 >= n_kv) return;
    const int ncols = a.head_dim / 2;
    const int stride_row = a.k_row / 4;
    const unsigned int * x = (const unsigned int *) (k_cache + (h / a.group) * kv_bytes(a.k_type, a.head_dim));
    const float2 * y2 = (const float2 *) (q + ((size_t) t * a.n_head + h) * a.q_stride);
    unsigned int * tile_xy = data_mmf + w * (16 * MMF_TILE_K);

    float C[2][4] = {};
    for (int col = w * 32 + lane; col < ncols; col += nwarps * 32) {
        int A[2][4][4];
#pragma unroll
        for (int itA = 0; itA < 2; ++itA) {
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const unsigned int pos = row0 + itA * 16 + i;
                tile_xy[i * MMF_TILE_K + lane] = pos < a.n_cells ? x[(size_t) pos * stride_row + col] : 0u;
            }
            __syncwarp();
#pragma unroll
            for (int k0 = 0; k0 < 32; k0 += 8) mmf_load_a(A[itA][k0 / 8], tile_xy + k0, MMF_TILE_K);
        }
        __syncwarp();
#pragma unroll
        for (int j0 = 0; j0 < 8; ++j0) {
            const float2 tmp = j0 == 0 ? y2[col] : make_float2(0.0f, 0.0f);
            unsigned int bits;
            if constexpr (BF16) {
                const nv_bfloat162 v = __float22bfloat162_rn(tmp);
                bits = *(const unsigned int *) &v;
            } else {
                const half2 v = __float22half2_rn(tmp);
                bits = *(const unsigned int *) &v;
            }
            tile_xy[j0 * MMF_TILE_K + lane] = bits;
        }
        __syncwarp();
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 8) {
            int B[2];
            mmf_load_b(B, tile_xy + k0, MMF_TILE_K);
#pragma unroll
            for (int itA = 0; itA < 2; ++itA) mmf_mma<BF16>(C[itA], A[itA][k0 / 8], B);
        }
    }

    float * buf_iw = (float *) data_mmf;
    const int kiw = nwarps * MMF_ROWS + 4;
    if (nwarps > 1) __syncthreads(); else __syncwarp();
#pragma unroll
    for (int itA = 0; itA < 2; ++itA) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            // tile<16, 8, float>'s get_i / get_j.
            const int i = w * MMF_ROWS + itA * 16 + ((l / 2) * 8) + (lane / 4);
            const int j = ((lane % 4) * 2) + (l % 2);
            buf_iw[j * kiw + i] = C[itA][l];
        }
    }
    if (nwarps > 1) __syncthreads(); else __syncwarp();
    if (w > 0) return;
    float sum = 0.0f;
    for (int i0 = 0; i0 < nwarps * MMF_ROWS; i0 += MMF_ROWS) sum += buf_iw[i0 + lane];
    float * row = scores + ((size_t) blockIdx.z * a.n_head + h) * a.n_kv_max;
    const unsigned int pos = row0 + lane;
    row[pos] = pos > a.n_pos + t ? -CUDART_INF_F : sum;
}
#endif

extern "C" __global__ void attention_kq_mma(const float * q, const unsigned char * k_cache,
        float * scores, AttentionArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    if (a.k_type == KV_BF16) {
#if __CUDA_ARCH__ >= 800
        attention_kq_mma_impl<true>(q, k_cache, scores, a);
#endif
    } else {
        attention_kq_mma_impl<false>(q, k_cache, scores, a);
    }
#endif
}
