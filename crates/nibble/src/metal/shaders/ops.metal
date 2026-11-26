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
    // The cache types (`KV_*`), the bytes of a position's row, and the
    // rotation blocks of a quantized cache (0: none).
    uint k_type;
    uint v_type;
    uint k_row;
    uint v_row;
    uint rot_k;
    uint rot_v;
};

// Threads per threadgroup of `attention`, and its simdgroups.
#define ATT_THREADS 256
#define ATT_SIMDGROUPS (ATT_THREADS / N_SIMDWIDTH)

// A head of `hd` values from `x`, optionally RMS-normalized with `w`, then
// NeoX RoPE at `pos` on the first `n_rot` values, then the Walsh-Hadamard
// rotation in blocks of `rot` (if not 0), into `out` (threadgroup memory;
// `tmp` is scratch of `hd` values).
static inline void norm_rope(constant AttentionArgs & a, device const float * x, device const float * w,
        uint pos, uint rot, threadgroup float * tmp, threadgroup float * out, threadgroup float * buf,
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
    if (rot != 0) {
        kv_hadamard(out, hd, rot, tid, ATT_THREADS);
    }
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
        device uchar * k_cache,
        device uchar * v_cache,
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

    const uint k_head = kv_bytes(a.k_type, hd);
    const uint v_head = kv_bytes(a.v_type, hd);
    for (uint g = 0; g < group; ++g) {
        const uint h = kh * group + g;
        norm_rope(a, q + h * a.q_stride, q_norm, pos, a.rot_k, tmp, qs + g * hd, buf, tid, tiisg, sgitg);
    }
    if (has_new) {
        norm_rope(a, k_new + kv_off, k_norm, pos, a.rot_k, tmp, ks, buf, tid, tiisg, sgitg);
        for (uint d = tid; d < hd; d += ATT_THREADS) {
            vs[d] = v_new[kv_off + d];
        }
        if (a.rot_v != 0) {
            kv_hadamard(vs, hd, a.rot_v, tid, ATT_THREADS);
        }
        // The new key and value stored, then as the cache holds them.
        kv_store_head(a.k_type, ks, k_cache + (uint64_t) pos * a.k_row + kh * k_head, hd, tid, ATT_THREADS);
        kv_store_head(a.v_type, vs, v_cache + (uint64_t) pos * a.v_row + kh * v_head, hd, tid, ATT_THREADS);
    }

    threadgroup const float4 * qs4 = (threadgroup const float4 *) qs;
    threadgroup const float4 * ks4 = (threadgroup const float4 *) ks;
    threadgroup const float4 * vs4 = (threadgroup const float4 *) vs;
    const uint chunk = a.chunk;

    for (uint t = t0 + sgitg; t < t1; t += ATT_SIMDGROUPS) {
        device const uchar * kt = k_cache + (uint64_t) t * a.k_row + kh * k_head;
        float4 kr[D];
        FOR_UNROLL (short j = 0; j < D; ++j) {
            const uint d4 = tiisg + N_SIMDWIDTH * j;
            kr[j] = d4 < hd4 ? (t == pos ? ks4[d4] : kv_load4(a.k_type, kt, 4 * d4)) : float4(0.0f);
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
        device const uchar * vt = v_cache + (uint64_t) t * a.v_row + kh * v_head;
        float4 vr[D];
        FOR_UNROLL (short j = 0; j < D; ++j) {
            const uint d4 = tiisg + N_SIMDWIDTH * j;
            vr[j] = d4 < hd4 ? (t == pos ? vs4[d4] : kv_load4(a.v_type, vt, 4 * d4)) : float4(0.0f);
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
        device uchar * k_cache [[buffer(6)]], \
        device uchar * v_cache [[buffer(7)]], \
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

// --- Attention over a prompt batch -----------------------------------------
//
// A batch of more than 16 tokens (`MATMUL_MIN_BATCH` in `metal/mod.rs`)
// takes two dispatches instead of one per token: `attention_prep` gives
// every query and key head of every token its norm and RoPE (as
// `attention` does) and stores the keys and values in the cache, then
// `attention_prompt` computes every token's causal attention with
// simdgroup matrices. Like llama.cpp's batched attention, it rounds
// differently from decoding (queries in f16, sums in another order).

// Head `tgpig.x` of token `tgpig.y`, at position `n_pos + tgpig.y`: a query
// head (below `group * n_kv` heads) is normalized and rotated in place in
// `q`; a key head is normalized, rotated and stored in the cache with its
// value.
kernel void attention_prep(
        constant AttentionArgs & a [[buffer(0)]],
        device float * q [[buffer(1)]],
        device const float * k_new [[buffer(2)]],
        device const float * v_new [[buffer(3)]],
        device const float * q_norm [[buffer(4)]],
        device const float * k_norm [[buffer(5)]],
        device uchar * k_cache [[buffer(6)]],
        device uchar * v_cache [[buffer(7)]],
        threadgroup float * shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort2 tid2 [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float buf[N_SIMDWIDTH];
    const uint hd = a.head_dim;
    const uint n_head = a.group * (a.kv_dim / hd);
    const uint head = tgpig.x;
    const uint t = tgpig.y;
    const uint pos = a.n_pos + t;
    const ushort tid = tid2.x;
    threadgroup float * tmp = shared;
    threadgroup float * res = shared + hd;
    if (head < n_head) {
        device float * qh = q + (uint64_t) t * n_head * a.q_stride + head * a.q_stride;
        norm_rope(a, qh, q_norm, pos, a.rot_k, tmp, res, buf, tid, tiisg, sgitg);
        for (uint d = tid; d < hd; d += ATT_THREADS) {
            qh[d] = res[d];
        }
    } else {
        const uint kh = head - n_head;
        const uint off = kh * hd;
        norm_rope(a, k_new + (uint64_t) t * a.kv_dim + off, k_norm, pos, a.rot_k, tmp, res, buf, tid, tiisg, sgitg);
        kv_store_head(a.k_type, res, k_cache + (uint64_t) pos * a.k_row + kh * kv_bytes(a.k_type, hd), hd, tid, ATT_THREADS);
        for (uint d = tid; d < hd; d += ATT_THREADS) {
            res[d] = v_new[(uint64_t) t * a.kv_dim + off + d];
        }
        if (a.rot_v != 0) {
            kv_hadamard(res, hd, a.rot_v, tid, ATT_THREADS);
        }
        kv_store_head(a.v_type, res, v_cache + (uint64_t) pos * a.v_row + kh * kv_bytes(a.v_type, hd), hd, tid, ATT_THREADS);
    }
}

// `FlashArgs` in `metal/kernels.rs`.
struct FlashArgs {
    uint n;
    uint n_past;
    uint kv_dim;
    uint q_stride;
    uint gated;
    uint n_head;
    float scale;
    uint group;
    uint heads;
    uint tiles;
    uint k_type;
    uint v_type;
    uint k_row;
    uint v_row;
};

// Cached positions `attention_prompt` takes at a time (the KV cache holds a
// multiple of them).
#define FLASH_C 32

// Causal attention for 8 tokens per simdgroup: threadgroup `tgpig` takes
// `heads` of the `group` query heads of key head `tgpig.y / slices` (slice
// `tgpig.y % slices`) for `tiles` tiles of 8 tokens from `8 * tiles *
// tgpig.x`, one head and tile per simdgroup. A simdgroup keeps its queries
// in threadgroup memory (`half`) and its output in simdgroup matrices, and
// goes through the cache `FLASH_C` positions at a time with an online
// softmax (scores in f32); the keys and values are read from the f16 cache
// directly. Simdgroups share no threadgroup memory, so they need no
// threadgroup barriers.
template <short HD>
static inline void attention_prompt_impl(
        constant FlashArgs & a,
        device const float * q,
        device const half * k_cache,
        device const half * v_cache,
        device float * out,
        threadgroup char * shmem,
        uint2 tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short DT = HD / 8;
    constexpr short CT = FLASH_C / 8;
    const uint slices = a.group / a.heads;
    const uint kh = tgpig.y / slices;
    const uint g = (tgpig.y % slices) * a.heads + sgitg % a.heads;
    const uint t0 = (tgpig.x * a.tiles + sgitg / a.heads) * 8;
    if (t0 >= a.n) {
        return;
    }
    const uint h = kh * a.group + g;
    const uint q_row = a.n_head * a.q_stride;
    const uint nsg = a.heads * a.tiles;
    threadgroup half * qs = (threadgroup half *) shmem + sgitg * 8 * HD;
    threadgroup float * ss = (threadgroup float *) (shmem + nsg * 8 * HD * 2) + sgitg * (8 * FLASH_C + 64);
    threadgroup float * dg = ss + 8 * FLASH_C;

    for (short i = tiisg; i < 8 * HD; i += N_SIMDWIDTH) {
        const uint t = t0 + i / HD;
        qs[i] = t < a.n ? half(q[(uint64_t) t * q_row + h * a.q_stride + i % HD]) : half(0.0f);
    }
    for (short i = tiisg; i < 64; i += N_SIMDWIDTH) {
        dg[i] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 o[DT];
    FOR_UNROLL (short j = 0; j < DT; ++j) {
        o[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    float m[8];
    float l[8];
    FOR_UNROLL (short r = 0; r < 8; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    device const half * kp = k_cache + kh * HD;
    device const half * vp = v_cache + kh * HD;
    // Every position up to the tile's last token.
    const uint end = a.n_past + min(a.n, t0 + 8);
    for (uint p0 = 0; p0 < end; p0 += FLASH_C) {
        simdgroup_float8x8 s[CT];
        FOR_UNROLL (short c = 0; c < CT; ++c) {
            s[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        FOR_UNROLL (short k = 0; k < DT; ++k) {
            simdgroup_half8x8 mq;
            simdgroup_load(mq, qs + 8 * k, HD, 0, false);
            FOR_UNROLL (short c = 0; c < CT; ++c) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, kp + (uint64_t) (p0 + 8 * c) * a.kv_dim + 8 * k, a.kv_dim, 0, true);
                simdgroup_multiply_accumulate(s[c], mq, mk, s[c]);
            }
        }
        FOR_UNROLL (short c = 0; c < CT; ++c) {
            simdgroup_store(s[c], ss + 8 * c, FLASH_C, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Lane `tiisg` takes position `p0 + tiisg` of each row.
        const uint p = p0 + tiisg;
        FOR_UNROLL (short r = 0; r < 8; ++r) {
            const uint pos = a.n_past + t0 + r;
            float v = ss[r * FLASH_C + tiisg] * a.scale;
            v = p <= pos && p < end ? v : -INFINITY;
            const float mn = max(m[r], simd_max(v));
            const float e = exp(v - mn);
            const float ms = exp(m[r] - mn);
            l[r] = l[r] * ms + simd_sum(e);
            m[r] = mn;
            ss[r * FLASH_C + tiisg] = e;
            if (tiisg == 0) {
                dg[r * 9] = ms;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 md;
        simdgroup_load(md, dg, 8, 0, false);
        FOR_UNROLL (short j = 0; j < DT; ++j) {
            simdgroup_multiply(o[j], md, o[j]);
        }
        FOR_UNROLL (short c = 0; c < CT; ++c) {
            simdgroup_float8x8 mp;
            simdgroup_load(mp, ss + 8 * c, FLASH_C, 0, false);
            FOR_UNROLL (short j = 0; j < DT; ++j) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, vp + (uint64_t) (p0 + 8 * c) * a.kv_dim + 8 * j, a.kv_dim, 0, false);
                simdgroup_multiply_accumulate(o[j], mp, mv, o[j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Divide by the sums, then write 32 values of each row at a time.
    FOR_UNROLL (short r = 0; r < 8; ++r) {
        if (tiisg == r) {
            dg[r * 9] = 1.0f / l[r];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 md;
    simdgroup_load(md, dg, 8, 0, false);
    FOR_UNROLL (short j = 0; j < DT; ++j) {
        simdgroup_multiply(o[j], md, o[j]);
    }
    FOR_UNROLL (short j0 = 0; j0 < DT; j0 += 4) {
        FOR_UNROLL (short jj = 0; jj < 4; ++jj) {
            simdgroup_store(o[j0 + jj], ss + 8 * jj, FLASH_C, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = tiisg; i < 8 * FLASH_C; i += N_SIMDWIDTH) {
            const uint t = t0 + i / FLASH_C;
            const uint d = j0 * 8 + i % FLASH_C;
            if (t < a.n) {
                float v = ss[i];
                if (a.gated != 0) {
                    v *= sigmoid(q[(uint64_t) t * q_row + h * a.q_stride + HD + d]);
                }
                out[((uint64_t) t * a.n_head + h) * HD + d] = v;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define ATTENTION_PROMPT_KERNEL(HD) \
kernel void attention_prompt_d##HD( \
        constant FlashArgs & a [[buffer(0)]], \
        device const float * q [[buffer(1)]], \
        device const half * k_cache [[buffer(2)]], \
        device const half * v_cache [[buffer(3)]], \
        device float * out [[buffer(4)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint2 tgpig [[threadgroup_position_in_grid]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    attention_prompt_impl<HD>(a, q, k_cache, v_cache, out, shmem, tgpig, tiisg, sgitg); \
}

ATTENTION_PROMPT_KERNEL(64)
ATTENTION_PROMPT_KERNEL(128)
ATTENTION_PROMPT_KERNEL(256)

// `attention_prompt` for caches of any type: the threadgroup dequantizes
// each block of `FLASH_C` positions of the keys, then of the values, into
// threadgroup memory (`half`, as ggml-metal's flash attention holds them),
// which its simdgroups multiply from. The threadgroup goes through the
// positions its last token sees; a simdgroup masks those its own tokens do
// not.
template <short HD>
static inline void attention_prompt_staged_impl(
        constant FlashArgs & a,
        device const float * q,
        device const uchar * k_cache,
        device const uchar * v_cache,
        device float * out,
        threadgroup char * shmem,
        uint2 tgpig,
        ushort tid,
        ushort ntg,
        ushort tiisg,
        ushort sgitg) {
    constexpr short DT = HD / 8;
    constexpr short CT = FLASH_C / 8;
    const uint slices = a.group / a.heads;
    const uint kh = tgpig.y / slices;
    const uint g = (tgpig.y % slices) * a.heads + sgitg % a.heads;
    const uint t0 = (tgpig.x * a.tiles + sgitg / a.heads) * 8;
    const uint h = kh * a.group + g;
    const uint q_row = a.n_head * a.q_stride;
    const uint nsg = a.heads * a.tiles;
    threadgroup half * qs = (threadgroup half *) shmem + sgitg * 8 * HD;
    threadgroup float * ss = (threadgroup float *) (shmem + nsg * 8 * HD * 2) + sgitg * (8 * FLASH_C + 64);
    threadgroup float * dg = ss + 8 * FLASH_C;
    threadgroup half * kvs = (threadgroup half *) (shmem + nsg * (8 * HD * 2 + (8 * FLASH_C + 64) * 4));

    for (short i = tiisg; i < 8 * HD; i += N_SIMDWIDTH) {
        const uint t = t0 + i / HD;
        qs[i] = t < a.n ? half(q[(uint64_t) t * q_row + h * a.q_stride + i % HD]) : half(0.0f);
    }
    for (short i = tiisg; i < 64; i += N_SIMDWIDTH) {
        dg[i] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 o[DT];
    FOR_UNROLL (short j = 0; j < DT; ++j) {
        o[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    float m[8];
    float l[8];
    FOR_UNROLL (short r = 0; r < 8; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    const uint k_head = kv_bytes(a.k_type, HD);
    const uint v_head = kv_bytes(a.v_type, HD);
    const uint end = a.n_past + min(a.n, t0 + 8);
    const uint tg_end = a.n_past + min(a.n, (tgpig.x + 1) * a.tiles * 8);
    for (uint p0 = 0; p0 < tg_end; p0 += FLASH_C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < FLASH_C * HD / 4; i += ntg) {
            const uint p = p0 + i / (HD / 4);
            const uint e = 4 * (i % (HD / 4));
            const float4 v = p < tg_end ? kv_load4(a.k_type, k_cache + (uint64_t) p * a.k_row + kh * k_head, e) : float4(0.0f);
            *(threadgroup half4 *) (kvs + (i / (HD / 4)) * HD + e) = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s[CT];
        FOR_UNROLL (short c = 0; c < CT; ++c) {
            s[c] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        FOR_UNROLL (short k = 0; k < DT; ++k) {
            simdgroup_half8x8 mq;
            simdgroup_load(mq, qs + 8 * k, HD, 0, false);
            FOR_UNROLL (short c = 0; c < CT; ++c) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, kvs + 8 * c * HD + 8 * k, HD, 0, true);
                simdgroup_multiply_accumulate(s[c], mq, mk, s[c]);
            }
        }
        FOR_UNROLL (short c = 0; c < CT; ++c) {
            simdgroup_store(s[c], ss + 8 * c, FLASH_C, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint p = p0 + tiisg;
        FOR_UNROLL (short r = 0; r < 8; ++r) {
            const uint pos = a.n_past + t0 + r;
            float v = ss[r * FLASH_C + tiisg] * a.scale;
            v = p <= pos && p < end ? v : -INFINITY;
            const float mn = max(m[r], simd_max(v));
            // A tile past the batch sees no position: keep it finite.
            const float e = mn == -INFINITY ? 0.0f : exp(v - mn);
            const float ms = mn == -INFINITY ? 1.0f : exp(m[r] - mn);
            l[r] = l[r] * ms + simd_sum(e);
            m[r] = mn;
            ss[r * FLASH_C + tiisg] = e;
            if (tiisg == 0) {
                dg[r * 9] = ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < FLASH_C * HD / 4; i += ntg) {
            const uint p = p0 + i / (HD / 4);
            const uint e = 4 * (i % (HD / 4));
            const float4 v = p < tg_end ? kv_load4(a.v_type, v_cache + (uint64_t) p * a.v_row + kh * v_head, e) : float4(0.0f);
            *(threadgroup half4 *) (kvs + (i / (HD / 4)) * HD + e) = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 md;
        simdgroup_load(md, dg, 8, 0, false);
        FOR_UNROLL (short j = 0; j < DT; ++j) {
            simdgroup_multiply(o[j], md, o[j]);
        }
        FOR_UNROLL (short c = 0; c < CT; ++c) {
            simdgroup_float8x8 mp;
            simdgroup_load(mp, ss + 8 * c, FLASH_C, 0, false);
            FOR_UNROLL (short j = 0; j < DT; ++j) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, kvs + 8 * c * HD + 8 * j, HD, 0, false);
                simdgroup_multiply_accumulate(o[j], mp, mv, o[j]);
            }
        }
    }

    if (t0 >= a.n) {
        return;
    }
    FOR_UNROLL (short r = 0; r < 8; ++r) {
        if (tiisg == r) {
            dg[r * 9] = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 md;
    simdgroup_load(md, dg, 8, 0, false);
    FOR_UNROLL (short j = 0; j < DT; ++j) {
        simdgroup_multiply(o[j], md, o[j]);
    }
    FOR_UNROLL (short j0 = 0; j0 < DT; j0 += 4) {
        FOR_UNROLL (short jj = 0; jj < 4; ++jj) {
            simdgroup_store(o[j0 + jj], ss + 8 * jj, FLASH_C, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = tiisg; i < 8 * FLASH_C; i += N_SIMDWIDTH) {
            const uint t = t0 + i / FLASH_C;
            const uint d = j0 * 8 + i % FLASH_C;
            if (t < a.n) {
                float v = ss[i];
                if (a.gated != 0) {
                    v *= sigmoid(q[(uint64_t) t * q_row + h * a.q_stride + HD + d]);
                }
                out[((uint64_t) t * a.n_head + h) * HD + d] = v;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define ATTENTION_PROMPT_STAGED_KERNEL(HD) \
kernel void attention_prompt_staged_d##HD( \
        constant FlashArgs & a [[buffer(0)]], \
        device const float * q [[buffer(1)]], \
        device const uchar * k_cache [[buffer(2)]], \
        device const uchar * v_cache [[buffer(3)]], \
        device float * out [[buffer(4)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint2 tgpig [[threadgroup_position_in_grid]], \
        ushort2 tid2 [[thread_position_in_threadgroup]], \
        ushort2 ntg2 [[threads_per_threadgroup]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    attention_prompt_staged_impl<HD>(a, q, k_cache, v_cache, out, shmem, tgpig, tid2.x, ntg2.x, tiisg, sgitg); \
}

ATTENTION_PROMPT_STAGED_KERNEL(64)
ATTENTION_PROMPT_STAGED_KERNEL(128)
ATTENTION_PROMPT_STAGED_KERNEL(256)

// Head `tgpig.x` of token `tgpig.y` of an attention output, rotated back
// (llama.cpp rotates the output of attention over a quantized value cache
// with the values' rotation) in blocks of `rot_v`, then scaled by
// `sigmoid(gate)` for gated query heads, in place.
kernel void attention_unrotate(
        constant AttentionArgs & a [[buffer(0)]],
        device const float * q [[buffer(1)]],
        device float * out [[buffer(2)]],
        threadgroup float * shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort2 tid2 [[thread_position_in_threadgroup]]) {
    const uint hd = a.head_dim;
    const uint n_head = a.group * (a.kv_dim / hd);
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    const ushort tid = tid2.x;
    device float * o = out + ((uint64_t) t * n_head + h) * hd;
    for (uint d = tid; d < hd; d += ATT_THREADS) {
        shared[d] = o[d];
    }
    kv_hadamard(shared, hd, a.rot_v, tid, ATT_THREADS);
    for (uint d = tid; d < hd; d += ATT_THREADS) {
        float v = shared[d];
        if (a.gated != 0) {
            v *= sigmoid(q[(uint64_t) t * n_head * a.q_stride + h * a.q_stride + hd + d]);
        }
        o[d] = v;
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

// A prompt batch's DeltaNet layer runs as `delta_prep`, in parallel over
// the tokens, then `delta_scan`, sequential over them (after
// `delta_net_batch` has applied any replayed tokens). Together they do
// `delta_net_batch`'s arithmetic, with the same reductions over the same
// number of threads, so each output is the same; the scan no longer waits
// on threadgroup barriers and reductions between tokens.

// Token `tgpig.y` of the batch (input `replay + tgpig.y` of the sequence)
// for key head `tgpig.x`: the convolution of its q and k channels,
// L2-normalized, and of a share of the value channels, into `conv` (a row
// of `conv_dim` per token); threadgroup 0 also turns every value head's
// beta into `sigmoid(beta)` and alpha into the decay, in place.
kernel void delta_prep(
        constant DeltaBatchArgs & a [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device const float * ring [[buffer(2)]],
        device const float * taps [[buffer(3)]],
        device float * beta [[buffer(4)]],
        device float * alpha [[buffer(5)]],
        device const float * dt_bias [[buffer(6)]],
        device const float * a_log [[buffer(7)]],
        device float * conv [[buffer(8)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort2 tid2 [[thread_position_in_threadgroup]],
        ushort2 ntg2 [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const ushort tid = tid2.x;
    const ushort ntg = ntg2.x;
    threadgroup float buf[N_SIMDWIDTH];
    const uint kh = tgpig.x;
    const uint t = tgpig.y;
    const uint u = a.replay + t;
    const uint dk = a.d_k;
    const uint value_dim = a.conv_dim - 2 * a.key_dim;
    const uint nh = value_dim / a.d_v;
    device float * row = conv + (uint64_t) t * a.conv_dim;

    float sq = 0.0f;
    float sk = 0.0f;
    float qv[DELTA_MAX_CHUNKS * 4];
    float kv[DELTA_MAX_CHUNKS * 4];
    short m = 0;
    for (uint i = tid; i < dk; i += ntg, ++m) {
        const float qi = conv_channel_batch(a, ring, x, taps, u, kh * dk + i);
        const float ki = conv_channel_batch(a, ring, x, taps, u, a.key_dim + kh * dk + i);
        qv[m] = qi;
        kv[m] = ki;
        sq += qi * qi;
        sk += ki * ki;
    }
    sq = threadgroup_sum(sq, buf, tiisg, sgitg, nsg);
    sk = threadgroup_sum(sk, buf, tiisg, sgitg, nsg);
    const float inv_sqrt_n = 1.0f / sqrt(float(dk));
    const float q_scale = 1.0f / sqrt(sq / dk + a.eps / dk);
    const float k_scale = 1.0f / sqrt(sk / dk + a.eps / dk);
    m = 0;
    for (uint i = tid; i < dk; i += ntg, ++m) {
        row[kh * dk + i] = qv[m] * q_scale * inv_sqrt_n;
        row[a.key_dim + kh * dk + i] = kv[m] * k_scale * inv_sqrt_n;
    }

    const uint share = (value_dim + a.n_k_heads - 1) / a.n_k_heads;
    for (uint c = kh * share + tid; c < min(value_dim, (kh + 1) * share); c += ntg) {
        row[2 * a.key_dim + c] = conv_channel_batch(a, ring, x, taps, u, 2 * a.key_dim + c);
    }

    if (kh == 0) {
        for (uint h = tid; h < nh; h += ntg) {
            const float b = beta[t * nh + h];
            const float g = alpha[t * nh + h] + dt_bias[h];
            const float softplus = g > 20.0f ? g : log(1.0f + exp(g));
            beta[t * nh + h] = sigmoid(b);
            alpha[t * nh + h] = exp(softplus * a_log[h]);
        }
    }
}

// float4s of q and k each thread of `delta_scan` loads per block.
#define SCAN_LOADS 4

// The delta rule over the `n` tokens of `delta_prep`'s output for `R` rows
// per simdgroup of value head `tgpig.x` (heads of `32 * 4 * CH` keys at
// most), with the state in registers, read from and written to `state`.
// Each row's arithmetic is `delta_net_batch`'s. The threadgroup stages the
// inputs of `stage` tokens at a time in threadgroup memory, so a token's
// update waits on no device load (staging every token's own inputs from
// device memory was latency-bound: 6.5 µs a token on an M4 Max).
template <short R, short CH>
static inline void delta_scan_impl(
        constant DeltaBatchArgs & a,
        uint stage,
        device const float * conv,
        device const float * beta,
        device const float * decay,
        device float * state,
        device float * out,
        threadgroup float * shared,
        uint2 tgpig,
        ushort tid,
        ushort ntg,
        ushort tiisg,
        ushort sgitg,
        ushort nsg) {
    const uint h = tgpig.x;
    const uint dk = a.d_k;
    const uint dv = a.d_v;
    const uint value_dim = a.conv_dim - 2 * a.key_dim;
    const uint nh = value_dim / dv;
    const uint rows = nsg * R;
    const uint j0 = tgpig.y * rows;
    const uint jr = j0 + sgitg * R;
    const uint kh = h % a.n_k_heads;
    const uint n4 = dk / 4;
    // `stage` rows of q, k (dk each), v (`rows`), beta and decay.
    const uint stride = 2 * dk + rows + 2;
    float4 s[R][CH];
    FOR_UNROLL (short r = 0; r < R; ++r) {
        device const float4 * row = (device const float4 *) (state + ((uint64_t) h * dv + min(jr + r, dv - 1)) * dk);
        FOR_UNROLL (short c = 0; c < CH; ++c) {
            const uint i = tiisg + c * N_SIMDWIDTH;
            s[r][c] = i < n4 ? row[i] : 0.0f;
        }
    }
    // The next block's inputs are loaded into registers while the threads
    // work through the current one, then written to threadgroup memory:
    // `SCAN_LOADS` float4s of q and k and one of the other values per
    // thread (`stage` is chosen so that they cover a block).
    const uint dk4 = dk / 4;
    float4 pre[SCAN_LOADS];
    float pre_s = 0.0f;
    auto prefetch = [&](uint t0) {
        const uint nt = min(stage, a.n - t0);
        FOR_UNROLL (short m = 0; m < SCAN_LOADS; ++m) {
            const uint e = tid + m * ntg;
            const uint u = e / (2 * dk4);
            const uint i = e % (2 * dk4);
            if (u < nt) {
                device const float * row = conv + (uint64_t) (t0 + u) * a.conv_dim;
                pre[m] = *(device const float4 *) (i < dk4
                    ? row + kh * dk + 4 * i
                    : row + a.key_dim + kh * dk + 4 * (i - dk4));
            }
        }
        const uint u = tid / (rows + 2);
        const uint i = tid % (rows + 2);
        if (u < nt) {
            device const float * row = conv + (uint64_t) (t0 + u) * a.conv_dim;
            pre_s = i < rows ? row[2 * a.key_dim + h * dv + min(j0 + i, dv - 1)]
                : i == rows ? beta[(t0 + u) * nh + h] : decay[(t0 + u) * nh + h];
        }
    };
    if (a.n > 0) {
        prefetch(0);
    }
    for (uint t0 = 0; t0 < a.n; t0 += stage) {
        const uint nt = min(stage, a.n - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short m = 0; m < SCAN_LOADS; ++m) {
            const uint e = tid + m * ntg;
            const uint u = e / (2 * dk4);
            if (u < nt) {
                *(threadgroup float4 *) (shared + u * stride + 4 * (e % (2 * dk4))) = pre[m];
            }
        }
        if (tid / (rows + 2) < nt) {
            shared[(tid / (rows + 2)) * stride + 2 * dk + tid % (rows + 2)] = pre_s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + stage < a.n) {
            prefetch(t0 + stage);
        }
        for (uint u = 0; u < nt; ++u) {
            threadgroup const float * in = shared + u * stride;
            threadgroup const float4 * q4 = (threadgroup const float4 *) in;
            threadgroup const float4 * k4 = (threadgroup const float4 *) (in + dk);
            const float b = in[2 * dk + rows];
            const float dec = in[2 * dk + rows + 1];
            const uint t = t0 + u;
            FOR_UNROLL (short r = 0; r < R; ++r) {
                const uint j = jr + r;
                float kd = 0.0f;
                FOR_UNROLL (short c = 0; c < CH; ++c) {
                    const uint i = tiisg + c * N_SIMDWIDTH;
                    if (i < n4) {
                        s[r][c] *= dec;
                        kd += dot(s[r][c], k4[i]);
                    }
                }
                const float delta = (in[2 * dk + sgitg * R + r] - simd_sum(kd)) * b;
                float o = 0.0f;
                FOR_UNROLL (short c = 0; c < CH; ++c) {
                    const uint i = tiisg + c * N_SIMDWIDTH;
                    if (i < n4) {
                        s[r][c] += k4[i] * delta;
                        o += dot(s[r][c], q4[i]);
                    }
                }
                o = simd_sum(o);
                if (tiisg == 0 && j < dv) {
                    out[(uint64_t) t * value_dim + h * dv + j] = o * a.scale;
                }
            }
        }
    }
    FOR_UNROLL (short r = 0; r < R; ++r) {
        device float4 * row = (device float4 *) (state + ((uint64_t) h * dv + jr + r) * dk);
        FOR_UNROLL (short c = 0; c < CH; ++c) {
            const uint i = tiisg + c * N_SIMDWIDTH;
            if (i < n4 && jr + r < dv) {
                row[i] = s[r][c];
            }
        }
    }
}

#define DELTA_SCAN_KERNEL(R, CH) \
kernel void delta_scan_r##R##_c##CH( \
        constant DeltaBatchArgs & a [[buffer(0)]], \
        device const float * conv [[buffer(1)]], \
        device const float * beta [[buffer(2)]], \
        device const float * decay [[buffer(3)]], \
        device float * state [[buffer(4)]], \
        device float * out [[buffer(5)]], \
        constant CountArgs & stage [[buffer(6)]], \
        threadgroup float * shared [[threadgroup(0)]], \
        uint2 tgpig [[threadgroup_position_in_grid]], \
        ushort2 tid2 [[thread_position_in_threadgroup]], \
        ushort2 ntg2 [[threads_per_threadgroup]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]], \
        ushort nsg [[simdgroups_per_threadgroup]]) { \
    delta_scan_impl<R, CH>(a, stage.n, conv, beta, decay, state, out, shared, tgpig, tid2.x, ntg2.x, tiisg, sgitg, nsg); \
}

// One row per simdgroup and 8 simdgroups measured fastest on an M4 Max
// (Qwen3.5-4B pp512: 1208 tok/s, against 1100–1190 for 2 or 4 rows and 4,
// 16 or 32 simdgroups).
DELTA_SCAN_KERNEL(1, 1)
DELTA_SCAN_KERNEL(1, 2)
DELTA_SCAN_KERNEL(1, 4)

// --- Chunked delta rule for prompt batches ------------------------------
//
// The gated delta rule `M = g M + beta (v - g M k) k^T`, `o = M q` (state
// rows by value) over chunks of `DCHUNK` tokens at once, the WY form that
// flash-linear-attention computes (`chunk_gated_delta_rule`): with the
// cumulative decay `y_t` of a chunk (from the sum of the log-decays, so
// ratios never divide by an underflowed product), the strictly lower
// triangle `A[s][r] = beta_s (y_s / y_r) k_s.k_r` and `T = (I + A)^-1`,
// a chunk's corrections are `U = T (beta (V - y K M0^T))`, its outputs
// `O = y Q M0^T + P U` with `P[t][s] = (y_t / y_s) q_t.k_s` (`s <= t`),
// and its last state `M = y_C M0 + (diag(y_C / y) U)^T K`. `delta_chunk_prep`
// computes every chunk's `T`, `P` and log-decays in parallel;
// `delta_chunk_scan` goes through the chunks in order. The sums round
// differently from the token-by-token rule (the same in exact arithmetic).
// Heads of 128 keys and values.

#define DCHUNK 32
#define DHEAD 128
// Floats a chunk's prep leaves for the scan: T, P, the log-decays.
#define DCHUNK_PREP (2 * DCHUNK * DCHUNK + DCHUNK)

// Chunk `tgpig.x` of value head `tgpig.y`, 4 simdgroups: its `T`, `P` and
// cumulative log-decays into `prep`. Tokens past the batch count as
// `beta = 0`, `g = 1` (their rows of the batch buffers are not read for
// anything they could change).
kernel void delta_chunk_prep(
        constant DeltaBatchArgs & a [[buffer(0)]],
        device const float * conv [[buffer(1)]],
        device const float * beta [[buffer(2)]],
        device const float * decay [[buffer(3)]],
        device float * prep [[buffer(4)]],
        threadgroup float * shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint c = tgpig.x;
    const uint h = tgpig.y;
    const uint n_chunks = (a.n + DCHUNK - 1) / DCHUNK;
    const uint value_dim = a.conv_dim - 2 * a.key_dim;
    const uint nh = value_dim / a.d_v;
    const uint kh = h % a.n_k_heads;
    const uint t0 = c * DCHUNK;
    threadgroup float * g = shared;                  // G = K K^T, then A
    threadgroup float * qk = g + DCHUNK * DCHUNK;    // Q K^T
    threadgroup float * lg = qk + DCHUNK * DCHUNK;   // cumulative log-decays
    threadgroup float * bt = lg + DCHUNK;            // betas

    if (sgitg == 0) {
        const uint t = t0 + tiisg;
        const bool real = t < a.n;
        const float l = real ? log(max(decay[t * nh + h], FLT_MIN)) : 0.0f;
        lg[tiisg] = simd_prefix_inclusive_sum(l);
        bt[tiisg] = real ? beta[t * nh + h] : 0.0f;
    }

    // Simdgroup `sgitg` takes the 8 rows from `8 * sgitg` of G and QK.
    device const float * kp = conv + (uint64_t) t0 * a.conv_dim + a.key_dim + kh * DHEAD;
    device const float * qp = conv + (uint64_t) t0 * a.conv_dim + kh * DHEAD;
    simdgroup_float8x8 mg[4];
    simdgroup_float8x8 mq[4];
    for (short j = 0; j < 4; ++j) {
        mg[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mq[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    for (short kk = 0; kk < DHEAD / 8; ++kk) {
        simdgroup_float8x8 ka;
        simdgroup_float8x8 qa;
        simdgroup_load(ka, kp + (uint64_t) 8 * sgitg * a.conv_dim + 8 * kk, a.conv_dim, 0, false);
        simdgroup_load(qa, qp + (uint64_t) 8 * sgitg * a.conv_dim + 8 * kk, a.conv_dim, 0, false);
        for (short j = 0; j < 4; ++j) {
            simdgroup_float8x8 kb;
            simdgroup_load(kb, kp + (uint64_t) 8 * j * a.conv_dim + 8 * kk, a.conv_dim, 0, true);
            simdgroup_multiply_accumulate(mg[j], ka, kb, mg[j]);
            simdgroup_multiply_accumulate(mq[j], qa, kb, mq[j]);
        }
    }
    for (short j = 0; j < 4; ++j) {
        simdgroup_store(mg[j], g + 8 * sgitg * DCHUNK + 8 * j, DCHUNK, 0, false);
        simdgroup_store(mq[j], qk + 8 * sgitg * DCHUNK + 8 * j, DCHUNK, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * out = prep + ((uint64_t) h * n_chunks + c) * DCHUNK_PREP;
    device float * tm = out;                        // T
    device float * pm = out + DCHUNK * DCHUNK;      // P
    // A in place of G, and P, for the elements of this thread.
    for (uint e = tid; e < DCHUNK * DCHUNK; e += 4 * N_SIMDWIDTH) {
        const uint s = e / DCHUNK;
        const uint r = e % DCHUNK;
        const float ratio = exp(lg[s] - lg[r]);
        g[e] = r < s ? bt[s] * ratio * g[e] : 0.0f;
        pm[e] = r <= s ? ratio * qk[e] : 0.0f;
    }
    if (tid < DCHUNK) {
        out[2 * DCHUNK * DCHUNK + tid] = lg[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // T = (I + A)^-1 by forward substitution, column `tiisg` per lane of
    // simdgroup 0 (T's rows overwrite QK, no longer needed).
    if (sgitg == 0) {
        threadgroup float * tt = qk;
        const uint col = tiisg;
        for (uint s = 0; s < DCHUNK; ++s) {
            float v = s == col ? 1.0f : 0.0f;
            for (uint r = col; r < s; ++r) {
                v -= g[s * DCHUNK + r] * tt[r * DCHUNK + col];
            }
            tt[s * DCHUNK + col] = s >= col ? v : 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint s = 0; s < DCHUNK; ++s) {
            tm[s * DCHUNK + col] = tt[s * DCHUNK + col];
        }
    }
}

// Rows `32 * tgpig.y..` of value head `tgpig.x`'s state through every chunk
// of the batch in order (4 simdgroups), the state in threadgroup memory,
// writing each token's output (before the gated norm) to `out`.
kernel void delta_chunk_scan(
        constant DeltaBatchArgs & a [[buffer(0)]],
        device const float * conv [[buffer(1)]],
        device const float * beta [[buffer(2)]],
        device const float * prep [[buffer(3)]],
        device float * state [[buffer(4)]],
        device float * out [[buffer(5)]],
        threadgroup float * shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NJ = 32;  // state rows per threadgroup
    const uint h = tgpig.x;
    const uint j0 = tgpig.y * NJ;
    const uint n_chunks = (a.n + DCHUNK - 1) / DCHUNK;
    const uint value_dim = a.conv_dim - 2 * a.key_dim;
    const uint nh = value_dim / a.d_v;
    const uint kh = h % a.n_k_heads;
    threadgroup float * m = shared;                    // NJ rows of DHEAD
    threadgroup float * rr = m + NJ * DHEAD;           // C x NJ: R, then O
    threadgroup float * uu = rr + DCHUNK * NJ;         // C x NJ: U, then W
    threadgroup float * dg = uu + DCHUNK * NJ;         // 4 diagonal 8x8 per simdgroup
    threadgroup float * sc = dg + 4 * 64;              // per token: y, beta, log y

    device float * st = state + ((uint64_t) h * a.d_v + j0) * DHEAD;
    for (uint e = tid; e < NJ * DHEAD; e += 4 * N_SIMDWIDTH) {
        m[e] = st[e];
    }
    for (uint e = tid; e < 4 * 64; e += 4 * N_SIMDWIDTH) {
        dg[e] = 0.0f;
    }

    for (uint c = 0; c < n_chunks; ++c) {
        const uint t0 = c * DCHUNK;
        device const float * pc = prep + ((uint64_t) h * n_chunks + c) * DCHUNK_PREP;
        device const float * tm = pc;
        device const float * pm = pc + DCHUNK * DCHUNK;
        device const float * lg = pc + 2 * DCHUNK * DCHUNK;
        device const float * kp = conv + (uint64_t) t0 * a.conv_dim + a.key_dim + kh * DHEAD;
        device const float * qp = conv + (uint64_t) t0 * a.conv_dim + kh * DHEAD;
        if (tid < DCHUNK) {
            const uint t = t0 + tid;
            sc[3 * tid] = exp(lg[tid]);
            sc[3 * tid + 1] = t < a.n ? beta[t * nh + h] : 0.0f;
            sc[3 * tid + 2] = lg[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // KM = K M^T and QM = Q M^T: tokens `8 * sgitg..` by the NJ rows.
        simdgroup_float8x8 km[4];
        simdgroup_float8x8 qm[4];
        for (short j = 0; j < 4; ++j) {
            km[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            qm[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        for (short kk = 0; kk < DHEAD / 8; ++kk) {
            simdgroup_float8x8 ka;
            simdgroup_float8x8 qa;
            simdgroup_load(ka, kp + (uint64_t) 8 * sgitg * a.conv_dim + 8 * kk, a.conv_dim, 0, false);
            simdgroup_load(qa, qp + (uint64_t) 8 * sgitg * a.conv_dim + 8 * kk, a.conv_dim, 0, false);
            for (short j = 0; j < 4; ++j) {
                simdgroup_float8x8 mt;
                simdgroup_load(mt, m + 8 * j * DHEAD + 8 * kk, DHEAD, 0, true);
                simdgroup_multiply_accumulate(km[j], ka, mt, km[j]);
                simdgroup_multiply_accumulate(qm[j], qa, mt, qm[j]);
            }
        }
        for (short j = 0; j < 4; ++j) {
            simdgroup_store(km[j], rr + 8 * sgitg * NJ + 8 * j, NJ, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // R = beta (V - y KM).
        for (uint e = tid; e < DCHUNK * NJ; e += 4 * N_SIMDWIDTH) {
            const uint s = e / NJ;
            const uint j = e % NJ;
            const float v = conv[(uint64_t) (t0 + s) * a.conv_dim + 2 * a.key_dim + h * a.d_v + j0 + j];
            rr[e] = sc[3 * s + 1] * (v - sc[3 * s] * rr[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // U = T R (T is lower triangular).
        simdgroup_float8x8 um[4];
        for (short j = 0; j < 4; ++j) {
            um[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        for (short sb = 0; sb <= sgitg; ++sb) {
            simdgroup_float8x8 ta;
            simdgroup_load(ta, tm + 8 * sgitg * DCHUNK + 8 * sb, DCHUNK, 0, false);
            for (short j = 0; j < 4; ++j) {
                simdgroup_float8x8 rb;
                simdgroup_load(rb, rr + 8 * sb * NJ + 8 * j, NJ, 0, false);
                simdgroup_multiply_accumulate(um[j], ta, rb, um[j]);
            }
        }
        for (short j = 0; j < 4; ++j) {
            simdgroup_store(um[j], uu + 8 * sgitg * NJ + 8 * j, NJ, 0, false);
        }
        // This simdgroup's tokens' decays on a diagonal.
        threadgroup float * d = dg + 64 * sgitg;
        if (tiisg < 8) {
            d[tiisg * 9] = sc[3 * (8 * sgitg + tiisg)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = y QM + P U.
        simdgroup_float8x8 dm;
        simdgroup_load(dm, d, 8, 0, false);
        for (short j = 0; j < 4; ++j) {
            simdgroup_multiply(qm[j], dm, qm[j]);
        }
        for (short sb = 0; sb <= sgitg; ++sb) {
            simdgroup_float8x8 pa;
            simdgroup_load(pa, pm + 8 * sgitg * DCHUNK + 8 * sb, DCHUNK, 0, false);
            for (short j = 0; j < 4; ++j) {
                simdgroup_float8x8 ub;
                simdgroup_load(ub, uu + 8 * sb * NJ + 8 * j, NJ, 0, false);
                simdgroup_multiply_accumulate(qm[j], pa, ub, qm[j]);
            }
        }
        for (short j = 0; j < 4; ++j) {
            simdgroup_store(qm[j], rr + 8 * sgitg * NJ + 8 * j, NJ, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float yc = sc[3 * (DCHUNK - 1)];
        const float lc = sc[3 * (DCHUNK - 1) + 2];
        for (uint e = tid; e < DCHUNK * NJ; e += 4 * N_SIMDWIDTH) {
            const uint s = e / NJ;
            const uint j = e % NJ;
            if (t0 + s < a.n) {
                out[(uint64_t) (t0 + s) * value_dim + h * a.d_v + j0 + j] = rr[e] * a.scale;
            }
            // W = diag(y_C / y) U.
            uu[e] *= exp(lc - sc[3 * s + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // M = y_C M + W^T K: this simdgroup's 32 columns of every row.
        for (short jb = 0; jb < NJ / 8; ++jb) {
            for (short ib = 0; ib < 4; ++ib) {
                const short col = 32 * sgitg + 8 * ib;
                simdgroup_float8x8 acc;
                simdgroup_load(acc, m + 8 * jb * DHEAD + col, DHEAD, 0, false);
                simdgroup_float8x8 ycm = make_filled_simdgroup_matrix<float, 8>(0.0f);
                // A scalar times the tile, as a product with y_C I.
                if (tiisg < 8) {
                    d[tiisg * 9] = yc;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(ycm, d, 8, 0, false);
                simdgroup_multiply(acc, ycm, acc);
                for (short sb = 0; sb < DCHUNK / 8; ++sb) {
                    simdgroup_float8x8 wt;
                    simdgroup_float8x8 kb;
                    simdgroup_load(wt, uu + 8 * sb * NJ + 8 * jb, NJ, 0, true);
                    simdgroup_load(kb, kp + (uint64_t) 8 * sb * a.conv_dim + col, a.conv_dim, 0, false);
                    simdgroup_multiply_accumulate(acc, wt, kb, acc);
                }
                simdgroup_store(acc, m + 8 * jb * DHEAD + col, DHEAD, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = tid; e < NJ * DHEAD; e += 4 * N_SIMDWIDTH) {
        st[e] = m[e];
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
