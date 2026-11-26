// Element-wise operations, norms, routing and token selection.

#include "common.cuh"

struct CountArgs { unsigned int n; };
struct RmsArgs { unsigned int n; float eps; };
struct RouteArgs { unsigned int n_expert; unsigned int k; };
struct MoeSumArgs { unsigned int n; unsigned int k; float eps; };
struct RotateArgs {
    unsigned int len;
    unsigned int block;
    unsigned int signs;
    unsigned int head_dim;
    unsigned int n_k;
    unsigned int rep;
    unsigned int inverse;
};

__device__ __forceinline__ float silu_op(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_op(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void copy_f32(const float * src, float * dst, CountArgs a) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.n) dst[i] = src[i];
}

extern "C" __global__ void swiglu(float * gate, const float * up, CountArgs a) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.n) gate[i] = silu_op(gate[i]) * up[i];
}

extern "C" __global__ void rmsnorm(const float * x, const float * w, float * out, RmsArgs a) {
    __shared__ float partial[32];
    const int t = blockIdx.x;
    x += (size_t) t * a.n;
    out += (size_t) t * a.n;
    const float scale = rms_scale(x, a.n, a.eps, partial);
    for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) out[i] = scale * x[i] * w[i];
}

extern "C" __global__ void rotate(const float * x, const float * signs, float * out, RotateArgs a) {
    extern __shared__ float values[];
    const int n = a.block;
    const int first = blockIdx.x * n;
    const size_t row = (size_t) blockIdx.y * a.len;
    const float scale = rsqrtf((float) n);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int o = first + j;
        float v;
        if (a.inverse || a.head_dim == 0) {
            v = x[row + o];
        } else {
            const int d = o % a.head_dim;
            const int head = o / a.head_dim;
            v = x[row + d + a.head_dim * (head / a.rep + a.n_k * (head % a.rep))];
        }
        if (!a.inverse && a.signs) v *= signs[o];
        values[j] = v * scale;
    }
    for (int len = 1; len < n; len *= 2) {
        __syncthreads();
        for (int p = threadIdx.x; p < n / 2; p += blockDim.x) {
            const int i = (p / len) * 2 * len + p % len;
            const float u = values[i];
            const float v = values[i + len];
            values[i] = u + v;
            values[i + len] = u - v;
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int o = first + j;
        out[row + o] = a.inverse && a.signs ? values[j] * signs[o] : values[j];
    }
}

// One block selects the first maximum, including exact ties.
extern "C" __global__ void __launch_bounds__(1024, 1) argmax(const float * x, unsigned int * slots, CountArgs a) {
    __shared__ float values[1024];
    __shared__ unsigned int indices[1024];
    float best = -CUDART_INF_F;
    unsigned int at = 0xffffffffu;
    for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) {
        if (x[i] > best) {
            best = x[i];
            at = i;
        }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = at;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < (unsigned int) stride) {
            const float other = values[threadIdx.x + stride];
            const unsigned int oi = indices[threadIdx.x + stride];
            if (other > values[threadIdx.x] || (other == values[threadIdx.x] && oi < indices[threadIdx.x])) {
                values[threadIdx.x] = other;
                indices[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *slots = indices[0];
}

extern "C" __global__ void moe_route(const float * logits, float * route, RouteArgs a) {
    extern __shared__ float probs[];
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = a.n_expert;
    const int k = a.k;
    const float * l = logits + (size_t) t * n;
    float * r = route + (size_t) t * 2 * k;
    float mx = -CUDART_INF_F;
    for (int i = lane; i < n; i += 32) mx = fmaxf(mx, l[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
    for (int i = lane; i < n; i += 32) {
        const float p = expf(l[i] - mx);
        probs[i] = p;
        sum += p;
    }
    const float inv = 1.0f / warp_sum(sum);
    for (int i = lane; i < n; i += 32) probs[i] *= inv;
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float best = -CUDART_INF_F;
        unsigned int bi = 0xffffffffu;
        for (int i = lane; i < n; i += 32) {
            if (probs[i] > best) {
                best = probs[i];
                bi = i;
            }
        }
        const float m = warp_max(best);
        const unsigned int win = warp_min_u(best == m ? bi : 0xffffffffu);
        if (lane == 0) {
            const unsigned int e = win < (unsigned int) n ? win : 0;
            r[j] = (float) e;
            r[k + j] = m;
            probs[e] = -CUDART_INF_F;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float total = 0.0f;
        for (int j = 0; j < k; ++j) total += r[k + j];
        total = fmaxf(total, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) r[k + j] /= total;
    }
}

extern "C" __global__ void moe_combine(const float * products, const float * route, float * out, MoeSumArgs a) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= a.n) return;
    const float * weights = route + (size_t) t * 2 * a.k + a.k;
    float acc = 0.0f;
    for (unsigned int j = 0; j < a.k; ++j) {
        const float v = products[((size_t) t * a.k + j) * a.n + i] * weights[j];
        acc = j == 0 ? v : acc + v;
    }
    out[(size_t) t * a.n + i] = acc;
}

extern "C" __global__ void moe_residual(const float * moe, const float * shared, const float * gate,
        float * x, const float * norm, float * out, MoeSumArgs a) {
    __shared__ float partial[32];
    const int t = blockIdx.x;
    const size_t row = (size_t) t * a.n;
    const float g = sigmoid_op(gate[t]);
    for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) {
        x[row + i] = x[row + i] + (moe[row + i] + shared[row + i] * g);
    }
    __syncthreads();
    const float scale = rms_scale(x + row, a.n, a.eps, partial);
    for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) out[row + i] = scale * x[row + i] * norm[i];
}
