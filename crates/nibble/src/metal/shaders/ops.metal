// Norms, RoPE, attention, the SwiGLU product and the Gated DeltaNet step.
// Each `*Args` struct matches the one of the same name in `metal/kernels.rs`.

struct RmsArgs {
    uint n;
    float eps;
};

// `out = rmsnorm(x) * w` for row `tgpig` of `n` values (`w` is shared by
// every row). `out` may be `x`.
kernel void rmsnorm(
        constant RmsArgs & a [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device const float * w [[buffer(2)]],
        device float * out [[buffer(3)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[N_SIMDWIDTH];
    x += (uint64_t) tgpig * a.n;
    out += (uint64_t) tgpig * a.n;
    float sum = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) {
        sum += x[i] * x[i];
    }
    sum = threadgroup_sum(sum, buf, tiisg, sgitg, nsg);
    const float scale = 1.0f / sqrt(sum / a.n + a.eps);
    for (uint i = tid; i < a.n; i += ntg) {
        out[i] = x[i] * scale * w[i];
    }
}

struct CountArgs {
    uint n;
};

// `g = silu(g) * u`.
kernel void swiglu(
        constant CountArgs & a [[buffer(0)]],
        device float * g [[buffer(1)]],
        device const float * u [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if (i < a.n) {
        g[i] = silu(g[i]) * u[i];
    }
}

// `dst = src`, `n` values.
kernel void copy_f32(
        constant CountArgs & a [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device float * dst [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if (i < a.n) {
        dst[i] = src[i];
    }
}

struct AttentionArgs {
    uint n_pos;
    uint head_dim;
    uint group;
    uint kv_dim;
    uint q_stride;
    uint gated;
    uint n_rot;
    uint norm;
    uint chunk;
    uint n_chunks;
    float scale;
    float base;
    float eps;
    uint pad;
};

// Threads per threadgroup of `attention`, and its simdgroups.
#define ATT_THREADS 256
#define ATT_SIMDGROUPS (ATT_THREADS / N_SIMDWIDTH)

// A head of `hd` values from `x`, optionally RMS-normalized with `w`, then
// NeoX RoPE at `pos` on the first `n_rot` values, into `out` (threadgroup
// memory; `tmp` is scratch of `hd` values).
static inline void norm_rope(constant AttentionArgs & a, device const float * x, device const float * w,
        uint pos, threadgroup float * tmp, threadgroup float * out, threadgroup float * buf,
        ushort tid, ushort tiisg, ushort sgitg) {
    const uint hd = a.head_dim;
    float sum = 0.0f;
    for (uint d = tid; d < hd; d += ATT_THREADS) {
        tmp[d] = x[d];
        sum += x[d] * x[d];
    }
    float scale = 1.0f;
    if (a.norm != 0) {
        sum = threadgroup_sum(sum, buf, tiisg, sgitg, ATT_SIMDGROUPS);
        scale = 1.0f / sqrt(sum / hd + a.eps);
    }
    for (uint d = tid; d < hd; d += ATT_THREADS) {
        tmp[d] = a.norm != 0 ? tmp[d] * scale * w[d] : tmp[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = a.n_rot / 2;
    for (uint d = tid; d < hd; d += ATT_THREADS) {
        float v = tmp[d];
        if (d < a.n_rot) {
            const uint i = d < half_rot ? d : d - half_rot;
            const float theta = float(pos) * pow(a.base, -2.0f * float(i) / float(a.n_rot));
            const float c = cos(theta);
            const float s = sin(theta);
            const float x0 = tmp[i];
            const float x1 = tmp[i + half_rot];
            v = d < half_rot ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        out[d] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline float threadgroup_max(float v, threadgroup float * buf, ushort tiisg, ushort sgitg) {
    v = simd_max(v);
    if (tiisg == 0) {
        buf[sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = buf[0];
    for (ushort g = 1; g < ATT_SIMDGROUPS; ++g) {
        m = max(m, buf[g]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

// Softmax attention of the `group` query heads of key head `tgpig.x` (`G`
// at most) over the cached positions of chunk `tgpig.y` (`chunk`
// positions each), so that each cached key and value is read once per
// token. The queries (and, in the chunk holding position `n_pos - 1`, the
// new key) get their RMSNorm and NeoX RoPE here, and that chunk stores the
// new key and value in the f16 cache. Simdgroups take positions in turn;
// lane `l` holds values `4l..4l+4` of each 128 (`D` groups of 128: head
// size up to `128 * D`). With one chunk the result is written to `out`
// (scaled by `sigmoid(gate)` for gated query heads, each followed by its
// gate); otherwise each head's unnormalized output, maximum score and sum
// of exponentials go to `part` for `attention_reduce`.
template <short G, short D>
static inline void attention_impl(
        constant AttentionArgs & a,
        device const float * q,
        device const float * k_new,
        device const float * v_new,
        device const float * q_norm,
        device const float * k_norm,
        device half * k_cache,
        device half * v_cache,
        device float * part,
        device float * out,
        threadgroup float * shared,
        threadgroup float * buf,
        uint2 tgpig,
        ushort tid,
        ushort tiisg,
        ushort sgitg) {
    const uint kh = tgpig.x;
    const uint c = tgpig.y;
    const uint hd = a.head_dim;
    const uint hd4 = hd / 4;
    const uint group = a.group;
    const uint kv_off = kh * hd;
    const uint pos = a.n_pos - 1;
    const uint t0 = c * a.chunk;
    const uint t1 = min(a.n_pos, t0 + a.chunk);
    const bool has_new = t1 == a.n_pos;

    threadgroup float * qs = shared;          // G heads of hd
    threadgroup float * ks = qs + G * hd;     // the new key
    threadgroup float * vs = ks + hd;         // the new value, as cached
    threadgroup float * tmp = vs + hd;
    threadgroup float * red = tmp + hd;       // ATT_SIMDGROUPS rows of hd
    threadgroup float * sc = red + ATT_SIMDGROUPS * hd; // G rows of chunk

    for (uint g = 0; g < group; ++g) {
        const uint h = kh * group + g;
        norm_rope(a, q + h * a.q_stride, q_norm, pos, tmp, qs + g * hd, buf, tid, tiisg, sgitg);
    }
    if (has_new) {
        norm_rope(a, k_new + kv_off, k_norm, pos, tmp, ks, buf, tid, tiisg, sgitg);
        // The new key and value as the cache holds them.
        for (uint d = tid; d < hd; d += ATT_THREADS) {
            const half k = half(ks[d]);
            const half v = half(v_new[kv_off + d]);
            ks[d] = float(k);
            vs[d] = float(v);
            const uint64_t at = (uint64_t) pos * a.kv_dim + kv_off + d;
            k_cache[at] = k;
            v_cache[at] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup const float4 * qs4 = (threadgroup const float4 *) qs;
    threadgroup const float4 * ks4 = (threadgroup const float4 *) ks;
    threadgroup const float4 * vs4 = (threadgroup const float4 *) vs;
    const uint chunk = a.chunk;

    for (uint t = t0 + sgitg; t < t1; t += ATT_SIMDGROUPS) {
        device const half4 * k4 = (device const half4 *) (k_cache + (uint64_t) t * a.kv_dim + kv_off);
        float4 kr[D];
        FOR_UNROLL (short j = 0; j < D; ++j) {
            const uint d4 = tiisg + N_SIMDWIDTH * j;
            kr[j] = d4 < hd4 ? (t == pos ? ks4[d4] : float4(k4[d4])) : float4(0.0f);
        }
        FOR_UNROLL (short g = 0; g < G; ++g) {
            if (g < (short) group) {
                float s = 0.0f;
                FOR_UNROLL (short j = 0; j < D; ++j) {
                    const uint d4 = tiisg + N_SIMDWIDTH * j;
                    if (d4 < hd4) {
                        s += dot(qs4[g * hd4 + d4], kr[j]);
                    }
                }
                s = simd_sum(s) * a.scale;
                if (tiisg == 0) {
                    sc[g * chunk + t - t0] = s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m[G];
    float l[G];
    FOR_UNROLL (short g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        if (g < (short) group) {
            threadgroup float * row = sc + g * chunk;
            float mg = -INFINITY;
            for (uint t = tid; t < t1 - t0; t += ATT_THREADS) {
                mg = max(mg, row[t]);
            }
            mg = threadgroup_max(mg, buf, tiisg, sgitg);
            float lg = 0.0f;
            for (uint t = tid; t < t1 - t0; t += ATT_THREADS) {
                const float e = exp(row[t] - mg);
                row[t] = e;
                lg += e;
            }
            m[g] = mg;
            l[g] = threadgroup_sum(lg, buf, tiisg, sgitg, ATT_SIMDGROUPS);
        }
    }

    float4 acc[G][D];
    FOR_UNROLL (short g = 0; g < G; ++g) {
        FOR_UNROLL (short j = 0; j < D; ++j) {
            acc[g][j] = 0.0f;
        }
    }
    for (uint t = t0 + sgitg; t < t1; t += ATT_SIMDGROUPS) {
        device const half4 * v4 = (device const half4 *) (v_cache + (uint64_t) t * a.kv_dim + kv_off);
        float4 vr[D];
        FOR_UNROLL (short j = 0; j < D; ++j) {
            const uint d4 = tiisg + N_SIMDWIDTH * j;
            vr[j] = d4 < hd4 ? (t == pos ? vs4[d4] : float4(v4[d4])) : float4(0.0f);
        }
        FOR_UNROLL (short g = 0; g < G; ++g) {
            if (g < (short) group) {
                const float e = sc[g * chunk + t - t0];
                FOR_UNROLL (short j = 0; j < D; ++j) {
                    acc[g][j] += e * vr[j];
                }
            }
        }
    }

    threadgroup float4 * red4 = (threadgroup float4 *) red;
    FOR_UNROLL (short g = 0; g < G; ++g) {
        if (g < (short) group) {
            const uint h = kh * group + g;
            FOR_UNROLL (short j = 0; j < D; ++j) {
                const uint d4 = tiisg + N_SIMDWIDTH * j;
                if (d4 < hd4) {
                    red4[sgitg * hd4 + d4] = acc[g][j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = tid; d < hd; d += ATT_THREADS) {
                float o = 0.0f;
                for (ushort s = 0; s < ATT_SIMDGROUPS; ++s) {
                    o += red[s * hd + d];
                }
                if (a.n_chunks == 1) {
                    o /= l[g];
                    if (a.gated != 0) {
                        o *= sigmoid(q[h * a.q_stride + hd + d]);
                    }
                    out[h * hd + d] = o;
                } else {
                    part[((uint64_t) h * a.n_chunks + c) * (hd + 2) + d] = o;
                }
            }
            if (a.n_chunks > 1 && tid == 0) {
                device float * p = part + ((uint64_t) h * a.n_chunks + c) * (hd + 2) + hd;
                p[0] = m[g];
                p[1] = l[g];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define ATTENTION_KERNEL(G, D) \
kernel void attention_g##G##_d##D( \
        constant AttentionArgs & a [[buffer(0)]], \
        device const float * q [[buffer(1)]], \
        device const float * k_new [[buffer(2)]], \
        device const float * v_new [[buffer(3)]], \
        device const float * q_norm [[buffer(4)]], \
        device const float * k_norm [[buffer(5)]], \
        device half * k_cache [[buffer(6)]], \
        device half * v_cache [[buffer(7)]], \
        device float * part [[buffer(8)]], \
        device float * out [[buffer(9)]], \
        threadgroup float * shared [[threadgroup(0)]], \
        uint2 tgpig [[threadgroup_position_in_grid]], \
        ushort2 tid2 [[thread_position_in_threadgroup]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float buf[N_SIMDWIDTH]; \
    attention_impl<G, D>(a, q, k_new, v_new, q_norm, k_norm, k_cache, v_cache, part, out, \
        shared, buf, tgpig, tid2.x, tiisg, sgitg); \
}

ATTENTION_KERNEL(1, 1)
ATTENTION_KERNEL(2, 1)
ATTENTION_KERNEL(4, 1)
ATTENTION_KERNEL(8, 1)
ATTENTION_KERNEL(1, 2)
ATTENTION_KERNEL(2, 2)
ATTENTION_KERNEL(4, 2)
ATTENTION_KERNEL(8, 2)
ATTENTION_KERNEL(1, 4)
ATTENTION_KERNEL(2, 4)
ATTENTION_KERNEL(4, 4)

// Combines the chunks of `attention` for query head `tgpig`, one thread per
// value.
kernel void attention_reduce(
        constant AttentionArgs & a [[buffer(0)]],
        device const float * q [[buffer(1)]],
        device const float * part [[buffer(2)]],
        device float * out [[buffer(3)]],
        uint h [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const uint hd = a.head_dim;
    device const float * p = part + (uint64_t) h * a.n_chunks * (hd + 2);
    float m = -INFINITY;
    for (uint c = 0; c < a.n_chunks; ++c) {
        m = max(m, p[c * (hd + 2) + hd]);
    }
    float l = 0.0f;
    for (uint c = 0; c < a.n_chunks; ++c) {
        l += p[c * (hd + 2) + hd + 1] * exp(p[c * (hd + 2) + hd] - m);
    }
    for (uint d = tid; d < hd; d += ntg) {
        float o = 0.0f;
        for (uint c = 0; c < a.n_chunks; ++c) {
            o += p[c * (hd + 2) + d] * exp(p[c * (hd + 2) + hd] - m);
        }
        o /= l;
        if (a.gated != 0) {
            o *= sigmoid(q[h * a.q_stride + hd + d]);
        }
        out[h * hd + d] = o;
    }
}

struct DeltaArgs {
    uint n_k_heads;
    uint d_k;
    uint d_v;
    uint key_dim;
    uint conv_dim;
    uint taps;
    uint rows;
    uint oldest;
    uint store;
    uint pending;
    float eps;
    float scale;
};

// The largest `d_k / 4 / N_SIMDWIDTH` a lane holds: `d_k` up to 512.
#define DELTA_MAX_CHUNKS 4

// Channel `c` of the causal convolution, with SiLU. The history is a ring
// of `rows` rows: the token's window is the `taps - 1` inputs from row
// `oldest` on, and its own input `x` goes to the row after them.
static inline float conv_channel(constant DeltaArgs & a, device const float * history,
        device const float * taps, device const float * x, uint c) {
    const uint len = a.taps - 1;
    float sum = 0.0f;
    for (uint t = 0; t < len; ++t) {
        const uint row = (a.oldest + t) % a.rows;
        sum = fma(history[(uint64_t) row * a.conv_dim + c], taps[(uint64_t) t * a.conv_dim + c], sum);
    }
    sum = fma(x[c], taps[(uint64_t) len * a.conv_dim + c], sum);
    return silu(sum);
}

// Rows of the state each simdgroup of `delta_net` updates.
#define DELTA_ROWS 4

// One token through a Gated DeltaNet layer for rows `tgpig.y * nsg *
// DELTA_ROWS` on of value head `tgpig.x`: the convolution of the head's q
// and k channels and of the rows' v channels (with `store`, each channel's
// new input is stored by one threadgroup, in the row after the window,
// which no threadgroup reads), L2-normalized q and k, and the gated delta
// rule on those rows of the head's state (`d_v` rows of `d_k`, ggml's
// transposed layout; each lane holds four values of a row at a time), read
// from `state` and written to `state_out` (the same buffer, or a work copy
// for a pending token, whose beta and alpha go to `pending_ba` with
// `pending`). Writes the rows' outputs, before the gated RMSNorm of
// `delta_norm`.
kernel void delta_net(
        constant DeltaArgs & a [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device float * history [[buffer(2)]],
        device const float * taps [[buffer(3)]],
        device const float * beta_in [[buffer(4)]],
        device const float * alpha [[buffer(5)]],
        device const float * dt_bias [[buffer(6)]],
        device const float * a_log [[buffer(7)]],
        device const float * state [[buffer(8)]],
        device float * out [[buffer(9)]],
        device float * state_out [[buffer(10)]],
        device float * pending_ba [[buffer(11)]],
        threadgroup float * shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort2 tid2 [[thread_position_in_threadgroup]],
        ushort2 ntg2 [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort ntg = ntg2.x;
    threadgroup float buf[N_SIMDWIDTH];
    const uint h = tgpig.x;
    const uint dk = a.d_k;
    const uint dv = a.d_v;
    const uint rows_per_group = nsg * DELTA_ROWS;
    const uint j0 = tgpig.y * rows_per_group;
    // ggml maps value head `h` to key head `h % n_k_heads`; value head `kh`
    // (its first threadgroup) stores the key head's new inputs.
    const uint kh = h % a.n_k_heads;
    const bool owns_qk = h == kh && tgpig.y == 0;
    threadgroup float * qn = shared;
    threadgroup float * kn = qn + dk;
    threadgroup float * vv = kn + dk;
    const uint spare = ((a.oldest + a.taps - 1) % a.rows) * a.conv_dim;

    const uint n4 = dk / 4;
    const uint jr = j0 + sgitg * DELTA_ROWS;
    // The state rows are loaded first, to overlap with the convolution.
    float4 s[DELTA_ROWS][DELTA_MAX_CHUNKS];
    FOR_UNROLL (short r = 0; r < DELTA_ROWS; ++r) {
        device const float4 * row = (device const float4 *) (state + ((uint64_t) h * dv + min(jr + r, dv - 1)) * dk);
        FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
            const uint i = tiisg + c * N_SIMDWIDTH;
            s[r][c] = i < n4 ? row[i] : 0.0f;
        }
    }
    float sq = 0.0f;
    float sk = 0.0f;
    for (uint i = tid; i < dk; i += ntg) {
        const uint cq = kh * dk + i;
        const uint ck = a.key_dim + kh * dk + i;
        const float qi = conv_channel(a, history, taps, x, cq);
        const float ki = conv_channel(a, history, taps, x, ck);
        qn[i] = qi;
        kn[i] = ki;
        sq += qi * qi;
        sk += ki * ki;
    }
    if (tid < rows_per_group && j0 + tid < dv) {
        const uint cv = 2 * a.key_dim + h * dv + j0 + tid;
        vv[tid] = conv_channel(a, history, taps, x, cv);
    }
    sq = threadgroup_sum(sq, buf, tiisg, sgitg, nsg);
    sk = threadgroup_sum(sk, buf, tiisg, sgitg, nsg);
    // The input's row is written after every read of the history above.
    for (uint i = tid; i < dk && owns_qk && a.store != 0; i += ntg) {
        history[spare + kh * dk + i] = x[kh * dk + i];
        history[spare + a.key_dim + kh * dk + i] = x[a.key_dim + kh * dk + i];
    }
    if (a.store != 0 && tid < rows_per_group && j0 + tid < dv) {
        const uint cv = 2 * a.key_dim + h * dv + j0 + tid;
        history[spare + cv] = x[cv];
    }
    if (a.pending != 0 && tgpig.y == 0 && tid == 0) {
        const uint nh = (a.conv_dim - 2 * a.key_dim) / dv;
        pending_ba[h] = beta_in[h];
        pending_ba[nh + h] = alpha[h];
    }
    // As llama.cpp: an RMSNorm with `eps / d_k`, then a scale by `1 / sqrt(d_k)`.
    const float inv_sqrt_n = 1.0f / sqrt(float(dk));
    const float q_scale = 1.0f / sqrt(sq / dk + a.eps / dk);
    const float k_scale = 1.0f / sqrt(sk / dk + a.eps / dk);
    for (uint i = tid; i < dk; i += ntg) {
        qn[i] = qn[i] * q_scale * inv_sqrt_n;
        kn[i] = kn[i] * k_scale * inv_sqrt_n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float beta = sigmoid(beta_in[h]);
    const float g = alpha[h] + dt_bias[h];
    const float softplus = g > 20.0f ? g : log(1.0f + exp(g));
    const float decay = exp(softplus * a_log[h]);

    threadgroup const float4 * q4 = (threadgroup const float4 *) qn;
    threadgroup const float4 * k4 = (threadgroup const float4 *) kn;
    FOR_UNROLL (short r = 0; r < DELTA_ROWS; ++r) {
        const uint j = jr + r;
        float kd = 0.0f;
        FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
            const uint i = tiisg + c * N_SIMDWIDTH;
            if (i < n4) {
                s[r][c] *= decay;
                kd += dot(s[r][c], k4[i]);
            }
        }
        const float delta = (vv[j - j0] - simd_sum(kd)) * beta;
        float o = 0.0f;
        device float4 * row = (device float4 *) (state_out + ((uint64_t) h * dv + j) * dk);
        FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
            const uint i = tiisg + c * N_SIMDWIDTH;
            if (i < n4) {
                s[r][c] += k4[i] * delta;
                if (j < dv) {
                    row[i] = s[r][c];
                }
                o += dot(s[r][c], q4[i]);
            }
        }
        o = simd_sum(o);
        if (tiisg == 0 && j < dv) {
            out[h * dv + j] = o * a.scale;
        }
    }
}

// `DeltaBatchArgs` in `metal/kernels.rs`: a DeltaNet layer over `replay`
// pending tokens (their inputs in the ring after the committed window from
// row `oldest`, their projections at row `ba_read` of `pending_ba`), then
// the `n` tokens of a batch.
struct DeltaBatchArgs {
    uint n_k_heads;
    uint d_k;
    uint d_v;
    uint key_dim;
    uint conv_dim;
    uint taps;
    uint rows;
    uint oldest;
    uint replay;
    uint n;
    uint pending;
    uint ba_read;
    uint ba_write;
    float eps;
    float scale;
};

// Input `e` of the sequence a batch's convolution runs over: the committed
// window, the replayed tokens' inputs (both in the ring), then the batch's.
static inline device const float * delta_input(constant DeltaBatchArgs & a,
        device const float * ring, device const float * x, uint e) {
    const uint in_ring = a.taps - 1 + a.replay;
    return e < in_ring ? ring + (uint64_t) ((a.oldest + e) % a.rows) * a.conv_dim
                       : x + (uint64_t) (e - in_ring) * a.conv_dim;
}

// Channel `c` of token `u`'s causal convolution, with SiLU: its window is
// inputs `u..u + taps - 1` of the sequence, its own input the next one. The
// sum runs as `conv_channel`'s.
static inline float conv_channel_batch(constant DeltaBatchArgs & a, device const float * ring,
        device const float * x, device const float * taps, uint u, uint c) {
    const uint len = a.taps - 1;
    float sum = 0.0f;
    for (uint t = 0; t < len; ++t) {
        sum = fma(delta_input(a, ring, x, u + t)[c], taps[(uint64_t) t * a.conv_dim + c], sum);
    }
    sum = fma(delta_input(a, ring, x, u + len)[c], taps[(uint64_t) len * a.conv_dim + c], sum);
    return silu(sum);
}

// `delta_net` for the replayed tokens and then every token of a batch, in
// order, in one dispatch: each threadgroup keeps its rows of the head's
// state in registers across the tokens and computes every token's
// convolution from the ring and the batch's inputs (it stores none: the
// ring is updated afterwards). Each token's arithmetic is that of
// `delta_net`. The state after the replayed tokens is written to `commit`
// when the batch stays pending (whose projections then go to row
// `ba_write` of `pending_ba`), and the final state to `state_out`.
kernel void delta_net_batch(
        constant DeltaBatchArgs & a [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device const float * ring [[buffer(2)]],
        device const float * taps [[buffer(3)]],
        device const float * beta_in [[buffer(4)]],
        device const float * alpha_in [[buffer(5)]],
        device const float * dt_bias [[buffer(6)]],
        device const float * a_log [[buffer(7)]],
        device const float * state [[buffer(8)]],
        device float * out [[buffer(9)]],
        device float * state_out [[buffer(10)]],
        device float * pending_ba [[buffer(11)]],
        device float * commit [[buffer(12)]],
        threadgroup float * shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort2 tid2 [[thread_position_in_threadgroup]],
        ushort2 ntg2 [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort ntg = ntg2.x;
    threadgroup float buf[N_SIMDWIDTH];
    const uint h = tgpig.x;
    const uint dk = a.d_k;
    const uint dv = a.d_v;
    const uint value_dim = a.conv_dim - 2 * a.key_dim;
    const uint nh = value_dim / dv;
    const uint rows_per_group = nsg * DELTA_ROWS;
    const uint j0 = tgpig.y * rows_per_group;
    const uint kh = h % a.n_k_heads;
    threadgroup float * qn = shared;
    threadgroup float * kn = qn + dk;
    threadgroup float * vv = kn + dk;

    const uint n4 = dk / 4;
    const uint jr = j0 + sgitg * DELTA_ROWS;
    float4 s[DELTA_ROWS][DELTA_MAX_CHUNKS];
    FOR_UNROLL (short r = 0; r < DELTA_ROWS; ++r) {
        device const float4 * row = (device const float4 *) (state + ((uint64_t) h * dv + min(jr + r, dv - 1)) * dk);
        FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
            const uint i = tiisg + c * N_SIMDWIDTH;
            s[r][c] = i < n4 ? row[i] : 0.0f;
        }
    }
    const float inv_sqrt_n = 1.0f / sqrt(float(dk));
    threadgroup const float4 * q4 = (threadgroup const float4 *) qn;
    threadgroup const float4 * k4 = (threadgroup const float4 *) kn;

    for (uint u = 0; u < a.replay + a.n; ++u) {
        const bool replayed = u < a.replay;
        const uint t = u - a.replay;
        if (u > 0) {
            // Every thread is done with the last token's q, k and v.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sq = 0.0f;
        float sk = 0.0f;
        for (uint i = tid; i < dk; i += ntg) {
            const float qi = conv_channel_batch(a, ring, x, taps, u, kh * dk + i);
            const float ki = conv_channel_batch(a, ring, x, taps, u, a.key_dim + kh * dk + i);
            qn[i] = qi;
            kn[i] = ki;
            sq += qi * qi;
            sk += ki * ki;
        }
        if (tid < rows_per_group && j0 + tid < dv) {
            vv[tid] = conv_channel_batch(a, ring, x, taps, u, 2 * a.key_dim + h * dv + j0 + tid);
        }
        sq = threadgroup_sum(sq, buf, tiisg, sgitg, nsg);
        sk = threadgroup_sum(sk, buf, tiisg, sgitg, nsg);
        const float q_scale = 1.0f / sqrt(sq / dk + a.eps / dk);
        const float k_scale = 1.0f / sqrt(sk / dk + a.eps / dk);
        for (uint i = tid; i < dk; i += ntg) {
            qn[i] = qn[i] * q_scale * inv_sqrt_n;
            kn[i] = kn[i] * k_scale * inv_sqrt_n;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float beta_raw = replayed ? pending_ba[(a.ba_read + u) * 2 * nh + h] : beta_in[t * nh + h];
        const float alpha_raw = replayed ? pending_ba[(a.ba_read + u) * 2 * nh + nh + h] : alpha_in[t * nh + h];
        if (!replayed && a.pending != 0 && tgpig.y == 0 && tid == 0) {
            pending_ba[(a.ba_write + t) * 2 * nh + h] = beta_raw;
            pending_ba[(a.ba_write + t) * 2 * nh + nh + h] = alpha_raw;
        }
        const float beta = sigmoid(beta_raw);
        const float g = alpha_raw + dt_bias[h];
        const float softplus = g > 20.0f ? g : log(1.0f + exp(g));
        const float decay = exp(softplus * a_log[h]);

        FOR_UNROLL (short r = 0; r < DELTA_ROWS; ++r) {
            const uint j = jr + r;
            float kd = 0.0f;
            FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
                const uint i = tiisg + c * N_SIMDWIDTH;
                if (i < n4) {
                    s[r][c] *= decay;
                    kd += dot(s[r][c], k4[i]);
                }
            }
            const float delta = (vv[j - j0] - simd_sum(kd)) * beta;
            float o = 0.0f;
            FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
                const uint i = tiisg + c * N_SIMDWIDTH;
                if (i < n4) {
                    s[r][c] += k4[i] * delta;
                    o += dot(s[r][c], q4[i]);
                }
            }
            o = simd_sum(o);
            if (!replayed && tiisg == 0 && j < dv) {
                out[(uint64_t) t * value_dim + h * dv + j] = o * a.scale;
            }
        }
        if (a.pending != 0 && u + 1 == a.replay) {
            FOR_UNROLL (short r = 0; r < DELTA_ROWS; ++r) {
                device float4 * row = (device float4 *) (commit + ((uint64_t) h * dv + jr + r) * dk);
                FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
                    const uint i = tiisg + c * N_SIMDWIDTH;
                    if (i < n4 && jr + r < dv) {
                        row[i] = s[r][c];
                    }
                }
            }
        }
    }
    FOR_UNROLL (short r = 0; r < DELTA_ROWS; ++r) {
        device float4 * row = (device float4 *) (state_out + ((uint64_t) h * dv + jr + r) * dk);
        FOR_UNROLL (short c = 0; c < DELTA_MAX_CHUNKS; ++c) {
            const uint i = tiisg + c * N_SIMDWIDTH;
            if (i < n4 && jr + r < dv) {
                row[i] = s[r][c];
            }
        }
    }
}

// The gated RMSNorm of value head `tgpig.x`'s DeltaNet output for token
// `tgpig.y`, in place: `rmsnorm(o) * norm * silu(z)`.
kernel void delta_norm(
        constant DeltaArgs & a [[buffer(0)]],
        device float * o [[buffer(1)]],
        device const float * z [[buffer(2)]],
        device const float * norm [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort2 tid2 [[thread_position_in_threadgroup]],
        ushort2 ntg2 [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort ntg = ntg2.x;
    threadgroup float buf[N_SIMDWIDTH];
    const uint dv = a.d_v;
    const uint value_dim = a.conv_dim - 2 * a.key_dim;
    o += (uint64_t) tgpig.y * value_dim + tgpig.x * dv;
    z += (uint64_t) tgpig.y * value_dim + tgpig.x * dv;
    float so = 0.0f;
    for (uint j = tid; j < dv; j += ntg) {
        so += o[j] * o[j];
    }
    so = threadgroup_sum(so, buf, tiisg, sgitg, nsg);
    const float o_scale = 1.0f / sqrt(so / dv + a.eps);
    for (uint j = tid; j < dv; j += ntg) {
        o[j] = o[j] * o_scale * norm[j] * silu(z[j]);
    }
}

// The index of the largest of `n` values (the first of equal ones, as the
// CPU's argmax), into `slot[0]`: one threadgroup.
kernel void argmax(
        constant CountArgs & a [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device uint * slot [[buffer(2)]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float best_v[N_SIMDWIDTH];
    threadgroup uint best_i[N_SIMDWIDTH];
    float v = -INFINITY;
    uint idx = 0xFFFFFFFFu;
    for (uint i = tid; i < a.n; i += ntg) {
        if (x[i] > v) {
            v = x[i];
            idx = i;
        }
    }
    // Lane by lane, keeping the larger value or, of equal ones, the first.
    for (ushort off = N_SIMDWIDTH / 2; off > 0; off /= 2) {
        const float ov = simd_shuffle_down(v, off);
        const uint oi = simd_shuffle_down(idx, off);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    if (tiisg == 0) {
        best_v[sgitg] = v;
        best_i[sgitg] = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (ushort s = 1; s < nsg; ++s) {
            if (best_v[s] > v || (best_v[s] == v && best_i[s] < idx)) {
                v = best_v[s];
                idx = best_i[s];
            }
        }
        slot[0] = idx == 0xFFFFFFFFu ? 0 : idx;
    }
}
