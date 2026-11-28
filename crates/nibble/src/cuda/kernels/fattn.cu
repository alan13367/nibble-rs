// Prompt-batch attention for an F16 cache: flash attention on tensor cores.
//
// For batches of more than eight tokens. One block takes 64 queries of a
// head, four warps of 16; each warp computes its scores for a block of keys
// with m16n8k16 MMAs (queries and keys in f16, f32 accumulators), applies the
// causal mask and an online softmax, and multiplies the probabilities (in
// f16) by V, which the cache stores transposed. The rounding differs from
// the one-token path (as llama.cpp's flash attention differs from its
// non-flash path).

#include "common.cuh"

__device__ __forceinline__ void fattn_mma(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16 bytes from global to shared memory without registers; zeros when `ok`
// is false (cells past the cache).
__device__ __forceinline__ void fattn_cp16(void * dst, const void * src, bool ok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned int d = (unsigned int) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
#endif
}

__device__ __forceinline__ void fattn_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;");
#endif
}

__device__ __forceinline__ void fattn_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ unsigned int fattn_pack(float lo, float hi) {
    const half2 v = __floats2half2_rn(lo, hi);
    return *(const unsigned int *) &v;
}

template<int HD>
struct FattnShape {
    static constexpr int KB = HD >= 256 ? 32 : 64;  // keys per block
    static constexpr int LDQ = HD + 8;              // halves per row of Q and K
    static constexpr int LDV = KB + 8;              // halves per row of V^T
    static constexpr int bytes = 2 * (64 * LDQ + KB * LDQ + HD * LDV);
};

template<int HD>
__device__ __forceinline__ void fattn_impl(const float * q, const unsigned char * k_cache,
        const unsigned char * v_cache, float * out, AttentionArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using S = FattnShape<HD>;
    constexpr int KB = S::KB;
    extern __shared__ half smem_fattn[];
    half * Qs = smem_fattn;
    half * Ks = Qs + 64 * S::LDQ;
    half * Vts = Ks + KB * S::LDQ;
    const int h = blockIdx.x;
    // The last queries see the most keys: their blocks start first.
    const int t0 = (gridDim.y - 1 - blockIdx.y) * 64;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int gid = lane / 4;
    const int tig = lane % 4;
    const int kh = h / a.group;
    const int n = a.n;

    for (int i = tid; i < 64 * HD; i += blockDim.x) {
        const int r = i / HD;
        const int d = i % HD;
        const int t = t0 + r;
        const float v = t < n ? q[((size_t) t * a.n_head + h) * a.q_stride + d] : 0.0f;
        Qs[r * S::LDQ + d] = __float2half_rn(v);
    }

    float o[HD / 8][4] = {};
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.0f, 0.0f};
    const float scale = a.scale * 1.4426950408889634f;  // exp2 instead of exp
    const int t_last = min(n, t0 + 64) - 1;
    const int kb_end = (a.n_pos + t_last) / KB + 1;
    const int qrow = warp * 16;
    const int qp0 = a.n_pos + t0 + qrow + gid;  // positions of the thread's two query rows
    const int qp1 = qp0 + 8;

    // The keys of a block are copied while the previous block's
    // probabilities meet V, and its V while its scores are computed: each
    // buffer is free just then. Rows of K and V^T are copied 16 bytes at a
    // time (the host checks that n_cells is a multiple of 8).
    auto load_k = [&](int p0) {
        for (int i = tid; i < KB * (HD / 8); i += blockDim.x) {
            const int r = i / (HD / 8);
            const int c = i % (HD / 8);
            const unsigned int p = p0 + r;
            const bool ok = p < a.n_cells;
            const unsigned char * src = k_cache + (size_t) (ok ? p : 0) * a.k_row + kh * 2 * HD + 16 * c;
            fattn_cp16(Ks + r * S::LDQ + 8 * c, src, ok);
        }
    };
    auto load_v = [&](int p0) {
        for (int i = tid; i < HD * (KB / 8); i += blockDim.x) {
            const int d = i / (KB / 8);
            const int c = i % (KB / 8);
            const unsigned int p = p0 + 8 * c;
            const bool ok = p < a.n_cells;
            const half * src = (const half *) v_cache + ((size_t) kh * HD + d) * a.n_cells + (ok ? p : 0);
            fattn_cp16(Vts + d * S::LDV + 8 * c, src, ok);
        }
        fattn_cp_commit();
    };
    load_k(0);
    fattn_cp_wait();
    __syncthreads();

    for (int kb = 0; kb < kb_end; ++kb) {
        const int p0 = kb * KB;
        load_v(p0);

        float s[KB / 8][4] = {};
#pragma unroll
        for (int ks = 0; ks < HD / 16; ++ks) {
            unsigned int af[4];
            const half * qa = Qs + (qrow + gid) * S::LDQ + ks * 16 + 2 * tig;
            af[0] = *(const unsigned int *) qa;
            af[1] = *(const unsigned int *) (qa + 8 * S::LDQ);
            af[2] = *(const unsigned int *) (qa + 8);
            af[3] = *(const unsigned int *) (qa + 8 * S::LDQ + 8);
#pragma unroll
            for (int nt = 0; nt < KB / 8; ++nt) {
                const half * kb_ = Ks + (nt * 8 + gid) * S::LDQ + ks * 16 + 2 * tig;
                fattn_mma(s[nt], af, *(const unsigned int *) kb_, *(const unsigned int *) (kb_ + 8));
            }
        }

        float mx[2] = {m[0], m[1]};
#pragma unroll
        for (int nt = 0; nt < KB / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kp = p0 + nt * 8 + 2 * tig + (e & 1);
                const int qp = e < 2 ? qp0 : qp1;
                const float v = kp > qp ? -CUDART_INF_F : s[nt][e] * scale;
                s[nt][e] = v;
                mx[e / 2] = fmaxf(mx[e / 2], v);
            }
        }
        // Every warp has its scores: K's buffer is free.
        fattn_cp_wait();
        __syncthreads();
        if (kb + 1 < kb_end) {
            load_k(p0 + KB);
            fattn_cp_commit();
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            alpha[r] = exp2f(m[r] - mx[r]);
            m[r] = mx[r];
            l[r] *= alpha[r];
        }
#pragma unroll
        for (int nt = 0; nt < KB / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float p = exp2f(s[nt][e] - m[e / 2]);
                s[nt][e] = p;
                l[e / 2] += p;
            }
        }
#pragma unroll
        for (int dt = 0; dt < HD / 8; ++dt) {
            o[dt][0] *= alpha[0];
            o[dt][1] *= alpha[0];
            o[dt][2] *= alpha[1];
            o[dt][3] *= alpha[1];
        }
#pragma unroll
        for (int j = 0; j < KB / 16; ++j) {
            unsigned int pf[4];
            pf[0] = fattn_pack(s[2 * j][0], s[2 * j][1]);
            pf[1] = fattn_pack(s[2 * j][2], s[2 * j][3]);
            pf[2] = fattn_pack(s[2 * j + 1][0], s[2 * j + 1][1]);
            pf[3] = fattn_pack(s[2 * j + 1][2], s[2 * j + 1][3]);
#pragma unroll
            for (int dt = 0; dt < HD / 8; ++dt) {
                const half * vb = Vts + (dt * 8 + gid) * S::LDV + j * 16 + 2 * tig;
                fattn_mma(o[dt], pf, *(const unsigned int *) vb, *(const unsigned int *) (vb + 8));
            }
        }
        // The next keys are in, and V's buffer is free.
        fattn_cp_wait();
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
    }
#pragma unroll
    for (int dt = 0; dt < HD / 8; ++dt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = t0 + qrow + gid + (e / 2) * 8;
            if (t >= n) continue;
            const int d = dt * 8 + 2 * tig + (e & 1);
            float v = o[dt][e] / l[e / 2];
            if (a.gated) {
                const float g = q[((size_t) t * a.n_head + h) * a.q_stride + HD + d];
                v *= 1.0f / (1.0f + expf(-g));
            }
            out[((size_t) t * a.n_head + h) * HD + d] = v;
        }
    }
#endif
}

#define EXPORT_FATTN(HD) \
    extern "C" __global__ void __launch_bounds__(128, 1) fattn_f16_##HD(const float * q, const unsigned char * k_cache, \
            const unsigned char * v_cache, float * out, AttentionArgs a) { \
        fattn_impl<HD>(q, k_cache, v_cache, out, a); \
    }

EXPORT_FATTN(64)
EXPORT_FATTN(128)
EXPORT_FATTN(256)
