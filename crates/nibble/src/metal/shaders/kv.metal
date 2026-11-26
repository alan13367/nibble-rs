// The KV cache types (llama.cpp's `--cache-type-k` / `-v`): reading four
// values of a cached head, storing a head, and the Walsh-Hadamard rotation
// llama.cpp applies to what a quantized cache holds.
//
// A cache holds one row per position: every KV head's values, each head
// starting on a block boundary (heads are a multiple of 32 values for the
// quantized types). The type is an argument of the attention kernels
// (`KV_*`, `KvType::code` in `metal/mod.rs`), so one pipeline serves every
// pair of types; the branch on it is uniform.

#define KV_F16 0
#define KV_F32 1
#define KV_BF16 2
#define KV_Q8_0 3
#define KV_Q4_0 4
#define KV_Q4_1 5
#define KV_IQ4_NL 6
#define KV_Q5_0 7
#define KV_Q5_1 8

// Bytes per block of 32 of a quantized type.
static inline uint kv_block_bytes(uint ty) {
    switch (ty) {
        case KV_Q8_0: return 34;
        case KV_Q4_1: return 20;
        case KV_Q5_0: return 22;
        case KV_Q5_1: return 24;
        default: return 18;
    }
}

static inline float kv_f16(device const uchar * p) {
    return float(as_type<half>(ushort(p[0] | (p[1] << 8))));
}

// Values `j..j + 4` (`j` a multiple of 4) of a quantized block, as ggml's
// `dequantize_row_*` computes them.
static inline float4 kv_dequant4(uint ty, device const uchar * b, uint j) {
    float4 r;
    const float d = kv_f16(b);
    switch (ty) {
        case KV_Q8_0: {
            device const char * qs = (device const char *) (b + 2);
            r = float4(qs[j], qs[j + 1], qs[j + 2], qs[j + 3]) * d;
            break;
        }
        case KV_Q4_0:
        case KV_IQ4_NL: {
            device const uchar * qs = b + 2;
            const uint shift = j < 16 ? 0 : 4;
            const uint at = j % 16;
            for (short i = 0; i < 4; ++i) {
                const uint q = (qs[at + i] >> shift) & 0x0F;
                r[i] = ty == KV_Q4_0 ? float(int(q) - 8) * d : d * kvalues_iq4nl_f[q];
            }
            break;
        }
        case KV_Q4_1: {
            const float m = kv_f16(b + 2);
            device const uchar * qs = b + 4;
            const uint shift = j < 16 ? 0 : 4;
            for (short i = 0; i < 4; ++i) {
                r[i] = fma(float((qs[j % 16 + i] >> shift) & 0x0F), d, m);
            }
            break;
        }
        default: {
            // Q5_0, Q5_1: the fifth bit of value `e` is bit `e` of `qh`.
            const bool one = ty == KV_Q5_1;
            const float m = one ? kv_f16(b + 2) : 0.0f;
            device const uchar * h = b + (one ? 4 : 2);
            const uint qh = h[0] | (h[1] << 8) | (h[2] << 16) | (uint(h[3]) << 24);
            device const uchar * qs = h + 4;
            const uint shift = j < 16 ? 0 : 4;
            for (short i = 0; i < 4; ++i) {
                const uint e = j + i;
                const uint q = ((qs[e % 16] >> shift) & 0x0F) | (((qh >> e) & 1) << 4);
                r[i] = one ? fma(float(q), d, m) : float(int(q) - 16) * d;
            }
            break;
        }
    }
    return r;
}

// Values `e..e + 4` of a cached head starting at `head` (`e` a multiple of 4).
static inline float4 kv_load4(uint ty, device const uchar * head, uint e) {
    switch (ty) {
        case KV_F16: return float4(*(device const half4 *) (head + 2 * e));
        case KV_F32: return *(device const float4 *) (head + 4 * e);
        case KV_BF16: return float4(*(device const bfloat4 *) (head + 2 * e));
        default: return kv_dequant4(ty, head + (e / 32) * kv_block_bytes(ty), e % 32);
    }
}

// Bytes of `n` values of type `ty`.
static inline uint kv_bytes(uint ty, uint n) {
    switch (ty) {
        case KV_F32: return 4 * n;
        case KV_F16:
        case KV_BF16: return 2 * n;
        default: return n / 32 * kv_block_bytes(ty);
    }
}

static inline void kv_put_f16(device uchar * p, float v) {
    const ushort h = as_type<ushort>(half(v));
    p[0] = h & 0xFF;
    p[1] = h >> 8;
}

// ggml-metal's `best_index_int8`: the index of the value of
// `kvalues_iq4nl_f` nearest `x`.
static inline uchar kv_best_iq4nl(float x) {
    if (x <= kvalues_iq4nl_f[0]) {
        return 0;
    }
    if (x >= kvalues_iq4nl_f[15]) {
        return 15;
    }
    int ml = 0;
    int mu = 15;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < kvalues_iq4nl_f[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    return x - kvalues_iq4nl_f[mu - 1] < kvalues_iq4nl_f[mu] - x ? mu - 1 : mu;
}

// Stores one block of 32 values `x` as quantized type `ty` at `b`, as
// ggml-metal's `quantize_*` (`kernels/quantize.h`, which its `set_rows`
// stores a cache with) compute it, and replaces `x` by what the block
// holds.
static inline void kv_quantize_block(uint ty, threadgroup float * x, device uchar * b) {
#pragma METAL fp math_mode(safe)
    uchar q[32];
    uint qh = 0;
    float dh = 0.0f;
    float mh = 0.0f;
    if (ty == KV_Q8_0) {
        float amax = 0.0f;
        for (short i = 0; i < 32; ++i) {
            amax = max(amax, fabs(x[i]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        kv_put_f16(b, d);
        dh = float(half(d));
        device char * qs = (device char *) (b + 2);
        for (short i = 0; i < 32; ++i) {
            const char v = char(round(x[i] * id));
            qs[i] = v;
            x[i] = float(v) * dh;
        }
        return;
    }
    if (ty == KV_Q4_1 || ty == KV_Q5_1) {
        const bool five = ty == KV_Q5_1;
        float lo = five ? x[0] : FLT_MAX;
        float hi = five ? x[0] : -FLT_MAX;
        for (short i = five ? 1 : 0; i < 32; ++i) {
            lo = min(lo, x[i]);
            hi = max(hi, x[i]);
        }
        const float d = (hi - lo) / (five ? 31.0f : 15.0f);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        kv_put_f16(b, d);
        kv_put_f16(b + 2, lo);
        dh = float(half(d));
        mh = float(half(lo));
        for (short i = 0; i < 32; ++i) {
            const float v = (x[i] - lo) * id;
            q[i] = five ? uchar(v + 0.5f) : min(uchar(15), uchar(char(v + 0.5f)));
        }
    } else if (ty == KV_IQ4_NL) {
        float amax = 0.0f;
        float vmax = 0.0f;
        for (short i = 0; i < 32; ++i) {
            if (amax < fabs(x[i])) {
                amax = fabs(x[i]);
                vmax = x[i];
            }
        }
        const float d = vmax / kvalues_iq4nl_f[0];
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        float sumqx = 0.0f;
        float sumq2 = 0.0f;
        for (short j = 0; j < 16; ++j) {
            const float x0 = x[j];
            const float x1 = x[j + 16];
            q[j] = kv_best_iq4nl(x0 * id);
            q[j + 16] = kv_best_iq4nl(x1 * id);
            const float v0 = kvalues_iq4nl_f[q[j]];
            const float v1 = kvalues_iq4nl_f[q[j + 16]];
            const float w0 = x0 * x0;
            const float w1 = x1 * x1;
            sumqx += w0 * v0 * x0 + w1 * v1 * x1;
            sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
        }
        const float ds = sumq2 > 0.0f ? sumqx / sumq2 : d;
        kv_put_f16(b, ds);
        dh = float(half(ds));
    } else {
        // Q4_0, Q5_0: a scale from the value of largest magnitude.
        const bool five = ty == KV_Q5_0;
        float amax = 0.0f;
        float vmax = 0.0f;
        for (short i = 0; i < 32; ++i) {
            if (amax < fabs(x[i])) {
                amax = fabs(x[i]);
                vmax = x[i];
            }
        }
        const float d = vmax / (five ? -16.0f : -8.0f);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        kv_put_f16(b, d);
        dh = float(half(d));
        for (short i = 0; i < 32; ++i) {
            q[i] = min(uchar(five ? 31 : 15), uchar(char(x[i] * id + (five ? 16.5f : 8.5f))));
        }
    }
    for (short i = 0; i < 32; ++i) {
        qh |= uint((q[i] >> 4) & 1) << i;
        switch (ty) {
            case KV_Q4_0: x[i] = float(int(q[i]) - 8) * dh; break;
            case KV_Q5_0: x[i] = float(int(q[i]) - 16) * dh; break;
            case KV_IQ4_NL: x[i] = dh * kvalues_iq4nl_f[q[i]]; break;
            default: x[i] = fma(float(q[i]), dh, mh); break;
        }
    }
    const uint qs_at = ty == KV_Q5_0 ? 6 : ty == KV_Q5_1 ? 8 : ty == KV_Q4_1 ? 4 : 2;
    for (short i = 0; i < 16; ++i) {
        b[qs_at + i] = (q[i] & 0x0F) | ((q[i + 16] & 0x0F) << 4);
    }
    if (ty == KV_Q5_0 || ty == KV_Q5_1) {
        device uchar * h = b + (ty == KV_Q5_0 ? 2 : 4);
        for (short i = 0; i < 4; ++i) {
            h[i] = (qh >> (8 * i)) & 0xFF;
        }
    }
}

// Stores the `hd` values `x` of a head as type `ty` at `head`, `nt`
// threads from `tid` sharing the work, and replaces `x` by what the cache
// holds. Ends with a threadgroup barrier.
static inline void kv_store_head(uint ty, threadgroup float * x, device uchar * head, uint hd,
        ushort tid, ushort nt) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    switch (ty) {
        case KV_F16:
            for (uint d = tid; d < hd; d += nt) {
                const half v = half(x[d]);
                ((device half *) head)[d] = v;
                x[d] = float(v);
            }
            break;
        case KV_F32:
            for (uint d = tid; d < hd; d += nt) {
                ((device float *) head)[d] = x[d];
            }
            break;
        case KV_BF16:
            for (uint d = tid; d < hd; d += nt) {
                const bfloat v = bfloat(x[d]);
                ((device bfloat *) head)[d] = v;
                x[d] = float(v);
            }
            break;
        default:
            for (uint blk = tid; blk < hd / 32; blk += nt) {
                kv_quantize_block(ty, x + 32 * blk, head + blk * kv_block_bytes(ty));
            }
            break;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The orthonormal Walsh-Hadamard transform of each block of `n` (a power
// of two) of the `hd` values `x`, `nt` threads from `tid`: each value
// scaled by `1 / sqrt(n)`, then the butterflies (as ggml's CPU
// `ggml_compute_forward_fwht`; ggml-metal's kernel forms its sums
// differently). Starts and ends with a threadgroup barrier.
static inline void kv_hadamard(threadgroup float * x, uint hd, uint n, ushort tid, ushort nt) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(float(n));
    for (uint d = tid; d < hd; d += nt) {
        x[d] *= scale;
    }
    for (uint len = 1; len < n; len *= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = tid; p < hd / 2; p += nt) {
            const uint i = (p / len) * 2 * len + p % len;
            const float u = x[i];
            const float v = x[i + len];
            x[i] = u + v;
            x[i + len] = u - v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
