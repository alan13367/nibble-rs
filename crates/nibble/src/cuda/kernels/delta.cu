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
        float * commit, DeltaBatchArgs a) {
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
