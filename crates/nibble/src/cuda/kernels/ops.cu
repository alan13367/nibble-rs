// Element-wise operations, norms, routing and token selection.

#include "common.cuh"

struct CountArgs { unsigned int n; };
struct RmsArgs { unsigned int n; float eps; };
struct RouteArgs { unsigned int n_expert; unsigned int k; unsigned int sigmoid; };
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

// The route of row `t` of the router logits: the gating's probabilities
// (a softmax over the experts, or LFM2's sigmoid of each with `bias` added
// to the selection scores, the weights unbiased), the `k` largest scores
// (the first of equal ones), and the probabilities of those divided by
// their sum. `probs` and `sel` hold `n_expert` floats each.
extern "C" __global__ void moe_route(const float * logits, const float * bias, float * route, RouteArgs a) {
    extern __shared__ float probs[];
    float * sel = probs + a.n_expert;
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = a.n_expert;
    const int k = a.k;
    const float * l = logits + (size_t) t * n;
    float * r = route + (size_t) t * 2 * k;
    if (a.sigmoid == 0) {
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
        for (int i = lane; i < n; i += 32) {
            probs[i] *= inv;
            sel[i] = probs[i];
        }
    } else {
        for (int i = lane; i < n; i += 32) {
            const float p = sigmoid_op(l[i]);
            probs[i] = p;
            sel[i] = p + bias[i];
        }
    }
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float best = -CUDART_INF_F;
        unsigned int bi = 0xffffffffu;
        for (int i = lane; i < n; i += 32) {
            if (sel[i] > best) {
                best = sel[i];
                bi = i;
            }
        }
        const float m = warp_max(best);
        const unsigned int win = warp_min_u(best == m ? bi : 0xffffffffu);
        if (lane == 0) {
            const unsigned int e = win < (unsigned int) n ? win : 0;
            r[j] = (float) e;
            r[k + j] = probs[e];
            sel[e] = -CUDART_INF_F;
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

// `x[t] += add[t]`, then `out[t] = rmsnorm(x[t]) * norm`: the end of a
// feed-forward whose output has no projection of its own (a mixture of
// experts' weighted sum).
extern "C" __global__ void add_norm(const float * add, float * x, const float * norm, float * out, RmsArgs a) {
    __shared__ float partial[32];
    const int t = blockIdx.x;
    const size_t row = (size_t) t * a.n;
    for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) {
        x[row + i] = x[row + i] + add[row + i];
    }
    __syncthreads();
    const float scale = rms_scale(x + row, a.n, a.eps, partial);
    for (unsigned int i = threadIdx.x; i < a.n; i += blockDim.x) out[row + i] = scale * x[row + i] * norm[i];
}

struct ShortConvArgs {
    unsigned int n_embd;
    unsigned int taps;
    unsigned int rows;   // taps - 1
    unsigned int oldest;
    unsigned int n;      // tokens of the batch
};

// One thread per channel of one token: the causal convolution of `b * x`
// over the taps values before the token (the ring's window before the
// batch, the batch's own `b` and `x` rows after), oldest tap first, one
// fused multiply-add per tap as ggml-cuda's `ssm_conv_f32` sums, gated by
// the token's own `c`.
extern "C" __global__ void lfm_short_conv(const float * bcx, const float * taps, const float * ring,
        float * out, ShortConvArgs a) {
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= (int) a.n_embd) return;
    const int e = a.n_embd;
    const int t = blockIdx.x;
    float sumf = 0.0f;
    for (unsigned int k = 0; k < a.taps; ++k) {
        const int p = t - (int) a.rows + (int) k;
        const float bx = p >= 0
            ? bcx[(size_t) p * 3 * e + c] * bcx[(size_t) p * 3 * e + 2 * e + c]
            : ring[(size_t) (((int) a.oldest + (int) a.rows + p) % (int) a.rows) * e + c];
        sumf = fmaf(bx, taps[k * e + c], sumf);
    }
    out[(size_t) t * e + c] = bcx[(size_t) t * 3 * e + e + c] * sumf;
}

// The last `gridDim.x` tokens' `b * x` values go back into the ring, each
// to the row after its window, as the serial semantics leaves them. Runs
// after `lfm_short_conv` (a stream orders the two), so no token reads a
// row this overwrites.
extern "C" __global__ void lfm_conv_store(const float * bcx, float * ring, ShortConvArgs a) {
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= (int) a.n_embd) return;
    const int e = a.n_embd;
    const unsigned int t = a.n - gridDim.x + blockIdx.x;
    const unsigned int row = (a.oldest + t) % a.rows;
    ring[(size_t) row * e + c] = bcx[(size_t) t * 3 * e + c] * bcx[(size_t) t * 3 * e + 2 * e + c];
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
