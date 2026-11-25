// Mixture-of-experts routing and the sums after the experts. The expert
// products themselves are the `matvec_*_id` kernels. Each `*Args` struct
// matches the one of the same name in `metal/kernels.rs`.

struct RouteArgs {
    uint n_expert;
    uint k;
};

// The route of row `t` of the router logits (one simdgroup per row): the
// softmax, the `k` largest probabilities (the first of equal ones), and
// those divided by their sum, as the CPU's `moe_route` computes them.
// `probs` holds `n_expert` floats.
kernel void moe_route(
        constant RouteArgs & a [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device float * route [[buffer(2)]],
        threadgroup float * probs [[threadgroup(0)]],
        uint t [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint n = a.n_expert, k = a.k;
    device const float * l = logits + (uint64_t) t * n;
    device float * r = route + (uint64_t) t * 2 * k;
    float mx = -INFINITY;
    for (uint i = lane; i < n; i += N_SIMDWIDTH) {
        mx = max(mx, l[i]);
    }
    mx = simd_max(mx);
    float sum = 0.0f;
    for (uint i = lane; i < n; i += N_SIMDWIDTH) {
        const float p = exp(l[i] - mx);
        probs[i] = p;
        sum += p;
    }
    const float inv = 1.0f / simd_sum(sum);
    for (uint i = lane; i < n; i += N_SIMDWIDTH) {
        probs[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < k; ++j) {
        float best = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint i = lane; i < n; i += N_SIMDWIDTH) {
            if (probs[i] > best) {
                best = probs[i];
                bi = i;
            }
        }
        const float m = simd_max(best);
        const uint win = simd_min(best == m ? bi : 0xFFFFFFFFu);
        if (lane == 0) {
            const uint e = win < n ? win : 0;
            r[j] = (float) e;
            r[k + j] = m;
            probs[e] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        float total = 0.0f;
        if (k == 8) {
            // `vDSP_sve`'s order on the CPU: four lanes, then pairwise.
            float s0 = r[8 + 0] + r[8 + 4], s1 = r[8 + 1] + r[8 + 5];
            float s2 = r[8 + 2] + r[8 + 6], s3 = r[8 + 3] + r[8 + 7];
            total = (s0 + s1) + (s2 + s3);
        } else {
            for (uint j = 0; j < k; ++j) {
                total += r[k + j];
            }
        }
        total = max(total, 6.103515625e-5f);
        for (uint j = 0; j < k; ++j) {
            r[k + j] = r[k + j] / total;
        }
    }
}

struct MoeSumArgs {
    uint n;
    uint k;
};

// `out[t][i] = Σ_j w_tj * e[t·k + j][i]`, from the first slot on, for rows
// of `n` values (one thread per value).
kernel void moe_combine(
        constant MoeSumArgs & a [[buffer(0)]],
        device const float * e [[buffer(1)]],
        device const float * route [[buffer(2)]],
        device float * out [[buffer(3)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint i = pos.x, t = pos.y;
    if (i >= a.n) {
        return;
    }
    device const float * w = route + (uint64_t) t * 2 * a.k + a.k;
    float acc = 0.0f;
    for (uint j = 0; j < a.k; ++j) {
        const float v = e[((uint64_t) t * a.k + j) * a.n + i] * w[j];
        acc = j == 0 ? v : acc + v;
    }
    out[(uint64_t) t * a.n + i] = acc;
}

// `x[t][i] += moe[t][i] + shared[t][i] * sigmoid(gate[t])`.
kernel void moe_residual(
        constant MoeSumArgs & a [[buffer(0)]],
        device const float * moe [[buffer(1)]],
        device const float * shared [[buffer(2)]],
        device const float * gate [[buffer(3)]],
        device float * x [[buffer(4)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint i = pos.x, t = pos.y;
    if (i >= a.n) {
        return;
    }
    const uint64_t at = (uint64_t) t * a.n + i;
    x[at] += moe[at] + shared[at] * sigmoid(gate[t]);
}
