// Gated DeltaNet: causal convolution and the sequential delta rule.
//
// Each token's arithmetic follows llama.cpp's graph on ggml-cuda (b12675):
// ssm_conv_f32 with SiLU, the L2 norm as rms_norm_f32 (eps / d_k) scaled by
// 1 / sqrt(d_k), sigmoid and softplus (log1pf) for the gates, and
// gated_delta_net_cuda's update and partition (four state columns per warp,
// eight lanes per column). The batch form keeps each block's state columns
// in registers while replaying pending tokens and then evaluating the new
// batch.

#include "common.cuh"

struct DeltaBatchArgs {
    unsigned int n_k_heads;
    unsigned int d_k;
    unsigned int d_v;
    unsigned int key_dim;
    unsigned int conv_dim;
    unsigned int taps;
    unsigned int rows;
    unsigned int oldest;
    unsigned int replay;
    unsigned int n;
    unsigned int pending;
    unsigned int ba_read;
    unsigned int ba_write;
    float eps;
    float scale;
    // The L2 norm's rms_norm eps (eps / d_k) and its output scale (1 / sqrt(d_k)),
    // computed on the host as llama.cpp does.
    float l2_eps;
    float l2_scale;
    // The convolution outputs (q and k normalized) of every token were
    // computed by delta_prep into `prep`, conv_dim values per token.
    unsigned int prepped;
    // The last `tail` tokens of a prepped batch stay pending: the state is
    // committed before them, and delta_prep keeps their beta and alpha
    // projections from row ba_write of pending_ba.
    unsigned int tail;
};

__device__ __forceinline__ float silu_delta(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_delta(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ const float * delta_input(const DeltaBatchArgs & a,
        const float * ring, const float * x, unsigned int e) {
    const unsigned int in_ring = a.taps - 1 + a.replay;
    return e < in_ring ? ring + (size_t) ((a.oldest + e) % a.rows) * a.conv_dim
                       : x + (size_t) (e - in_ring) * a.conv_dim;
}

__device__ float conv_channel(const DeltaBatchArgs & a, const float * ring,
        const float * x, const float * taps, unsigned int u, unsigned int c) {
    const unsigned int len = a.taps - 1;
    float sum = 0.0f;
    for (unsigned int t = 0; t < len; ++t) {
        sum = fmaf(delta_input(a, ring, x, u + t)[c], taps[(size_t) t * a.conv_dim + c], sum);
    }
    sum = fmaf(delta_input(a, ring, x, u + len)[c], taps[(size_t) len * a.conv_dim + c], sum);
    // ssm_conv_f32 adds the (absent) bias before the SiLU.
    sum += 0.0f;
    return silu_delta(sum);
}

extern "C" __global__ void __launch_bounds__(128, 1) delta_net_batch(const float * x,
        const float * ring, const float * taps, const float * beta_in,
        const float * alpha_in, const float * dt_bias, const float * a_log,
        const float * state, float * out, float * state_out, float * pending_ba,
        float * commit, const float * prep, DeltaBatchArgs a) {
    extern __shared__ float shared[];
    __shared__ float partial[32];
    constexpr int WARPS = 4;
    // gated_delta_net_cuda: four columns per warp, eight lanes per column.
    constexpr int COLS = 4;
    constexpr int LANES = 8;
    constexpr int MAX_RPL = 64;
    const int h = blockIdx.x;
    const int dk = a.d_k;
    const int dv = a.d_v;
    const int value_dim = a.conv_dim - 2 * a.key_dim;
    const int nh = value_dim / dv;
    const int rows_per_group = WARPS * COLS;
    const int j0 = blockIdx.y * rows_per_group;
    const int kh = h % a.n_k_heads;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane_in_col = lane % LANES;
    const int j = j0 + warp * COLS + lane / LANES;
    const int jc = min(j, dv - 1);
    const int rpl = dk / LANES;
    float * qn = shared;
    float * kn = qn + dk;
    float * vv = kn + dk;
    // State column j (transposed layout: M[j][i] = S[i][j]), rows i = r * LANES + lane_in_col.
    float s_shard[MAX_RPL];
#pragma unroll
    for (int r = 0; r < MAX_RPL; ++r) {
        if (r < rpl) s_shard[r] = state[((size_t) h * dv + jc) * dk + r * LANES + lane_in_col];
    }

    for (unsigned int u = 0; u < a.replay + a.n; ++u) {
        const bool replayed = u < a.replay;
        const unsigned int t = u - a.replay;
        if (u > 0) __syncthreads();
        if (a.prepped) {
            const float * pu = prep + (size_t) u * a.conv_dim;
            for (int i = tid; i < dk; i += blockDim.x) {
                qn[i] = pu[kh * dk + i];
                kn[i] = pu[a.key_dim + kh * dk + i];
            }
            if (tid < rows_per_group && j0 + tid < dv) vv[tid] = pu[2 * a.key_dim + h * dv + j0 + tid];
            __syncthreads();
        } else {
            for (int i = tid; i < dk; i += blockDim.x) {
                qn[i] = conv_channel(a, ring, x, taps, u, kh * dk + i);
                kn[i] = conv_channel(a, ring, x, taps, u, a.key_dim + kh * dk + i);
            }
            if (tid < rows_per_group && j0 + tid < dv) {
                vv[tid] = conv_channel(a, ring, x, taps, u, 2 * a.key_dim + h * dv + j0 + tid);
            }
            __syncthreads();
            const float q_scale = rms_scale(qn, dk, a.l2_eps, partial);
            const float k_scale = rms_scale(kn, dk, a.l2_eps, partial);
            for (int i = tid; i < dk; i += blockDim.x) {
                qn[i] = q_scale * qn[i] * a.l2_scale;
                kn[i] = k_scale * kn[i] * a.l2_scale;
            }
            __syncthreads();
        }
        const float beta_raw = replayed ? pending_ba[(a.ba_read + u) * 2 * nh + h]
                                        : beta_in[(size_t) t * nh + h];
        const float alpha_raw = replayed ? pending_ba[(a.ba_read + u) * 2 * nh + nh + h]
                                         : alpha_in[(size_t) t * nh + h];
        if (!replayed && a.pending && blockIdx.y == 0 && tid == 0) {
            pending_ba[(a.ba_write + t) * 2 * nh + h] = beta_raw;
            pending_ba[(a.ba_write + t) * 2 * nh + nh + h] = alpha_raw;
        }
        const float beta_val = sigmoid_delta(beta_raw);
        const float biased = alpha_raw + dt_bias[h];
        const float softplus = biased > 20.0f ? biased : log1pf(expf(biased));
        const float gate = softplus * a_log[h];
        const float g_val = expf(gate);

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < MAX_RPL; ++r) {
            if (r < rpl) kv_shard += s_shard[r] * kn[r * LANES + lane_in_col];
        }
        const float kv_col = warp_sum<LANES>(kv_shard);
        const float delta_col = (vv[min(j, dv - 1) - j0] - g_val * kv_col) * beta_val;
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < MAX_RPL; ++r) {
            if (r < rpl) {
                const int i = r * LANES + lane_in_col;
                s_shard[r] = g_val * s_shard[r] + kn[i] * delta_col;
                attn_partial += s_shard[r] * qn[i];
            }
        }
        const float attn_col = warp_sum<LANES>(attn_partial);
        if (!replayed && lane_in_col == 0 && j < dv) {
            out[(size_t) t * value_dim + h * dv + j] = attn_col * a.scale;
        }
        if (a.pending && u + 1 == a.replay && j < dv) {
#pragma unroll
            for (int r = 0; r < MAX_RPL; ++r) {
                if (r < rpl) commit[((size_t) h * dv + j) * dk + r * LANES + lane_in_col] = s_shard[r];
            }
        }
    }
    if (j < dv) {
#pragma unroll
        for (int r = 0; r < MAX_RPL; ++r) {
            if (r < rpl) state_out[((size_t) h * dv + j) * dk + r * LANES + lane_in_col] = s_shard[r];
        }
    }
}

// The convolution (with SiLU) and the L2 norms of q and k for every token of
// the batch (replayed ones first) at once, with delta_net_batch's arithmetic,
// so that its sequential loop only applies the delta rule. Blocks of y below
// n_k_heads take a key head's q and k; the others 128 value channels each.
extern "C" __global__ void __launch_bounds__(128, 1) delta_prep(const float * x, const float * ring,
        const float * taps, const float * beta_in, const float * alpha_in, const float * dt_bias,
        const float * a_log, float * pending_ba, float * prep, float * gates, DeltaBatchArgs a) {
    extern __shared__ float shared[];
    __shared__ float partial[32];
    const unsigned int u = blockIdx.x;
    const int tid = threadIdx.x;
    const int dk = a.d_k;
    const int value_dim = a.conv_dim - 2 * a.key_dim;
    const int nh = value_dim / a.d_v;
    float * pu = prep + (size_t) u * a.conv_dim;
    if (blockIdx.y == gridDim.y - 1) {
        // The gates of every value head: sigmoid(beta) and exp(softplus(alpha + dt) * A).
        const bool replayed = u < a.replay;
        const unsigned int t = u - a.replay;
        for (int h = tid; h < nh; h += blockDim.x) {
            const float beta_raw = replayed ? pending_ba[(a.ba_read + u) * 2 * nh + h] : beta_in[(size_t) t * nh + h];
            const float alpha_raw = replayed ? pending_ba[(a.ba_read + u) * 2 * nh + nh + h] : alpha_in[(size_t) t * nh + h];
            const float biased = alpha_raw + dt_bias[h];
            const float softplus = biased > 20.0f ? biased : log1pf(expf(biased));
            const float gate = softplus * a_log[h];
            gates[(size_t) u * 2 * nh + h] = sigmoid_delta(beta_raw);
            gates[(size_t) u * 2 * nh + nh + h] = expf(gate);
            if (!replayed && t + a.tail >= a.n) {
                const unsigned int row = a.ba_write + t - (a.n - a.tail);
                pending_ba[row * 2 * nh + h] = beta_raw;
                pending_ba[row * 2 * nh + nh + h] = alpha_raw;
            }
        }
    } else if ((int) blockIdx.y < (int) a.n_k_heads) {
        const int kh = blockIdx.y;
        float * qn = shared;
        float * kn = qn + dk;
        for (int i = tid; i < dk; i += blockDim.x) {
            qn[i] = conv_channel(a, ring, x, taps, u, kh * dk + i);
            kn[i] = conv_channel(a, ring, x, taps, u, a.key_dim + kh * dk + i);
        }
        __syncthreads();
        const float q_scale = rms_scale(qn, dk, a.l2_eps, partial);
        const float k_scale = rms_scale(kn, dk, a.l2_eps, partial);
        for (int i = tid; i < dk; i += blockDim.x) {
            pu[kh * dk + i] = q_scale * qn[i] * a.l2_scale;
            pu[a.key_dim + kh * dk + i] = k_scale * kn[i] * a.l2_scale;
        }
    } else {
        const int c = (blockIdx.y - a.n_k_heads) * blockDim.x + tid;
        if (c < value_dim) pu[2 * a.key_dim + c] = conv_channel(a, ring, x, taps, u, 2 * a.key_dim + c);
    }
}

// The delta rule over a long batch from delta_prep's outputs, with
// delta_net_batch's partition and arithmetic: no shared memory or barriers,
// and each lane loads the next token's values while the current one is
// applied. RPL = d_k / 8 rows per lane. Replayed tokens update the state only;
// the last `a.tail` tokens do not (they stay pending).
template<int RPL>
__device__ __forceinline__ void delta_seq_impl(const float * prep, const float * gates, float * state,
        float * out, DeltaBatchArgs a) {
    constexpr int LANES = 8;
    const int h = blockIdx.x;
    const int dk = a.d_k;
    const int dv = a.d_v;
    const int value_dim = a.conv_dim - 2 * a.key_dim;
    const int nh = value_dim / dv;
    const int kh = h % a.n_k_heads;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_in_col = lane % LANES;
    const int j = blockIdx.y * 16 + warp * 4 + lane / LANES;
    const int jc = min(j, dv - 1);
    float s[RPL];
#pragma unroll
    for (int r = 0; r < RPL; ++r) s[r] = state[((size_t) h * dv + jc) * dk + r * LANES + lane_in_col];
    const unsigned int steps = a.replay + a.n;
    const float * pq = prep + kh * dk + lane_in_col;
    const float * pk = prep + a.key_dim + kh * dk + lane_in_col;
    const float * pv = prep + 2 * a.key_dim + h * dv + jc;
    float q[RPL], k[RPL], v, beta, g;
#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        q[r] = pq[r * LANES];
        k[r] = pk[r * LANES];
    }
    v = pv[0];
    beta = gates[h];
    g = gates[nh + h];
    for (unsigned int u = 0; u < steps; ++u) {
        float qc[RPL], kc[RPL];
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            qc[r] = q[r];
            kc[r] = k[r];
        }
        const float vc = v, betac = beta, gc = g;
        if (u + 1 < steps) {
            const size_t off = (size_t) (u + 1) * a.conv_dim;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                q[r] = pq[off + r * LANES];
                k[r] = pk[off + r * LANES];
            }
            v = pv[off];
            beta = gates[(size_t) (u + 1) * 2 * nh + h];
            g = gates[(size_t) (u + 1) * 2 * nh + nh + h];
        }
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; ++r) kv_shard += s[r] * kc[r];
        const float kv_col = warp_sum<LANES>(kv_shard);
        const float delta_col = (vc - gc * kv_col) * betac;
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            s[r] = gc * s[r] + kc[r] * delta_col;
            attn_partial += s[r] * qc[r];
        }
        const float attn_col = warp_sum<LANES>(attn_partial);
        if (u >= a.replay && lane_in_col == 0 && j < dv) {
            out[(size_t) (u - a.replay) * value_dim + h * dv + j] = attn_col * a.scale;
        }
        // The state is committed before the pending tail, if any.
        if (u + 1 + a.tail == steps && j < dv) {
#pragma unroll
            for (int r = 0; r < RPL; ++r) state[((size_t) h * dv + j) * dk + r * LANES + lane_in_col] = s[r];
        }
    }
}

#define EXPORT_DELTA_SEQ(DK) \
    extern "C" __global__ void __launch_bounds__(128) delta_seq_##DK(const float * prep, const float * gates, \
            float * state, float * out, DeltaBatchArgs a) { \
        delta_seq_impl<DK / 8>(prep, gates, state, out, a); \
    }

EXPORT_DELTA_SEQ(64)
EXPORT_DELTA_SEQ(128)
EXPORT_DELTA_SEQ(256)

extern "C" __global__ void delta_norm(float * out, const float * z, const float * norm, DeltaBatchArgs a) {
    extern __shared__ float values[];
    __shared__ float partial[32];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int dv = a.d_v;
    const int value_dim = a.conv_dim - 2 * a.key_dim;
    float * o = out + (size_t) t * value_dim + h * dv;
    const float * gate = z + (size_t) t * value_dim + h * dv;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) values[i] = o[i];
    const float scale = rms_scale(o, dv, a.eps, partial);
    for (int i = threadIdx.x; i < dv; i += blockDim.x) o[i] = scale * values[i] * norm[i] * silu_delta(gate[i]);
}
