// KV-cache encodings and llama.cpp's Hadamard rotation of quantized caches.
#pragma once
#include "common.cuh"

#define KV_F16 0
#define KV_F32 1
#define KV_BF16 2
#define KV_Q8_0 3
#define KV_Q4_0 4
#define KV_Q4_1 5
#define KV_IQ4_NL 6
#define KV_Q5_0 7
#define KV_Q5_1 8

__device__ __constant__ signed char kv_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ unsigned int kv_block_bytes(unsigned int ty) {
    switch (ty) {
        case KV_Q8_0: return 34;
        case KV_Q4_1: return 20;
        case KV_Q5_0: return 22;
        case KV_Q5_1: return 24;
        default: return 18;
    }
}

__device__ __forceinline__ unsigned int kv_bytes(unsigned int ty, unsigned int n) {
    switch (ty) {
        case KV_F32: return 4 * n;
        case KV_F16:
        case KV_BF16: return 2 * n;
        default: return n / 32 * kv_block_bytes(ty);
    }
}

__device__ __forceinline__ float kv_f16(const unsigned char * p) {
    return __half2float(__ushort_as_half((unsigned short) (p[0] | (p[1] << 8))));
}

__device__ __forceinline__ float kv_load(unsigned int ty, const unsigned char * head, unsigned int e) {
    if (ty == KV_F32) return ((const float *) head)[e];
    if (ty == KV_F16) return __half2float(((const half *) head)[e]);
    if (ty == KV_BF16) return __bfloat162float(((const nv_bfloat16 *) head)[e]);
    const unsigned char * b = head + (e / 32) * kv_block_bytes(ty);
    const unsigned int j = e % 32;
    const float d = kv_f16(b);
    if (ty == KV_Q8_0) return d * ((const signed char *) (b + 2))[j];
    if (ty == KV_Q4_0 || ty == KV_IQ4_NL) {
        const unsigned int q = (b[2 + j % 16] >> (j < 16 ? 0 : 4)) & 15;
        return ty == KV_Q4_0 ? d * (int(q) - 8) : d * kv_iq4nl[q];
    }
    if (ty == KV_Q4_1) {
        const float m = kv_f16(b + 2);
        const unsigned int q = (b[4 + j % 16] >> (j < 16 ? 0 : 4)) & 15;
        return fmaf((float) q, d, m);
    }
    const bool one = ty == KV_Q5_1;
    const float m = one ? kv_f16(b + 2) : 0.0f;
    const unsigned char * h = b + (one ? 4 : 2);
    const unsigned int qh = h[0] | (h[1] << 8) | (h[2] << 16) | ((unsigned int) h[3] << 24);
    const unsigned char * qs = h + 4;
    const unsigned int q = ((qs[j % 16] >> (j < 16 ? 0 : 4)) & 15) | (((qh >> j) & 1) << 4);
    return one ? fmaf((float) q, d, m) : d * (int(q) - 16);
}

__device__ __forceinline__ void kv_put_f16(unsigned char * p, float v) {
    const unsigned short h = __half_as_ushort(__float2half_rn(v));
    p[0] = h;
    p[1] = h >> 8;
}

__device__ __forceinline__ unsigned char kv_best_iq4nl(float x) {
    if (x <= kv_iq4nl[0]) return 0;
    if (x >= kv_iq4nl[15]) return 15;
    int lo = 0, hi = 15;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (x < kv_iq4nl[mid]) hi = mid; else lo = mid;
    }
    return x - kv_iq4nl[hi - 1] < kv_iq4nl[hi] - x ? hi - 1 : hi;
}

// One thread quantizes a block and replaces its inputs by their cached values.
__device__ void kv_quantize_block(unsigned int ty, float * x, unsigned char * b) {
    unsigned char q[32];
    unsigned int qh = 0;
    float dh = 0.0f, mh = 0.0f;
    if (ty == KV_Q8_0) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        kv_put_f16(b, d);
        dh = __half2float(__float2half_rn(d));
        for (int i = 0; i < 32; ++i) {
            const signed char v = (signed char) roundf(x[i] * id);
            ((signed char *) (b + 2))[i] = v;
            x[i] = v * dh;
        }
        return;
    }
    if (ty == KV_Q4_1 || ty == KV_Q5_1) {
        const bool five = ty == KV_Q5_1;
        float lo = x[0], hi = x[0];
        for (int i = 1; i < 32; ++i) {
            lo = fminf(lo, x[i]);
            hi = fmaxf(hi, x[i]);
        }
        const float d = (hi - lo) / (five ? 31.0f : 15.0f);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        kv_put_f16(b, d);
        kv_put_f16(b + 2, lo);
        dh = __half2float(__float2half_rn(d));
        mh = __half2float(__float2half_rn(lo));
        for (int i = 0; i < 32; ++i) q[i] = (unsigned char) (int) ((x[i] - lo) * id + 0.5f);
    } else if (ty == KV_IQ4_NL) {
        float amax = 0.0f, vmax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            if (amax < fabsf(x[i])) {
                amax = fabsf(x[i]);
                vmax = x[i];
            }
        }
        const float d = vmax / kv_iq4nl[0];
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        float sumqx = 0.0f, sumq2 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float x0 = x[j], x1 = x[j + 16];
            q[j] = kv_best_iq4nl(x0 * id);
            q[j + 16] = kv_best_iq4nl(x1 * id);
            const float v0 = kv_iq4nl[q[j]], v1 = kv_iq4nl[q[j + 16]];
            const float w0 = x0 * x0, w1 = x1 * x1;
            sumqx += w0 * v0 * x0 + w1 * v1 * x1;
            sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
        }
        const float ds = sumq2 > 0.0f ? sumqx / sumq2 : d;
        kv_put_f16(b, ds);
        dh = __half2float(__float2half_rn(ds));
    } else {
        const bool five = ty == KV_Q5_0;
        float amax = 0.0f, vmax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            if (amax < fabsf(x[i])) {
                amax = fabsf(x[i]);
                vmax = x[i];
            }
        }
        const float d = vmax / (five ? -16.0f : -8.0f);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        kv_put_f16(b, d);
        dh = __half2float(__float2half_rn(d));
        for (int i = 0; i < 32; ++i) q[i] = (unsigned char) (int) (x[i] * id + (five ? 16.5f : 8.5f));
    }
    for (int i = 0; i < 32; ++i) {
        qh |= ((q[i] >> 4) & 1) << i;
        if (ty == KV_Q4_0) x[i] = (int(q[i]) - 8) * dh;
        else if (ty == KV_Q5_0) x[i] = (int(q[i]) - 16) * dh;
        else if (ty == KV_IQ4_NL) x[i] = dh * kv_iq4nl[q[i]];
        else x[i] = fmaf((float) q[i], dh, mh);
    }
    const unsigned int at = ty == KV_Q5_0 ? 6 : ty == KV_Q5_1 ? 8 : ty == KV_Q4_1 ? 4 : 2;
    for (int i = 0; i < 16; ++i) b[at + i] = (q[i] & 15) | ((q[i + 16] & 15) << 4);
    if (ty == KV_Q5_0 || ty == KV_Q5_1) {
        unsigned char * h = b + (ty == KV_Q5_0 ? 2 : 4);
        for (int i = 0; i < 4; ++i) h[i] = qh >> (8 * i);
    }
}

__device__ void kv_hadamard(float * x, unsigned int hd, unsigned int n) {
    __syncthreads();
    const float scale = rsqrtf((float) n);
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) x[d] *= scale;
    for (unsigned int len = 1; len < n; len *= 2) {
        __syncthreads();
        for (unsigned int p = threadIdx.x; p < hd / 2; p += blockDim.x) {
            const unsigned int i = (p / len) * 2 * len + p % len;
            const float u = x[i], v = x[i + len];
            x[i] = u + v;
            x[i + len] = u - v;
        }
    }
    __syncthreads();
}

__device__ void kv_store_head(unsigned int ty, float * x, unsigned char * head, unsigned int hd) {
    __syncthreads();
    if (ty == KV_F16) {
        for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
            const half v = __float2half_rn(x[d]);
            ((half *) head)[d] = v;
            x[d] = __half2float(v);
        }
    } else if (ty == KV_F32) {
        for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) ((float *) head)[d] = x[d];
    } else if (ty == KV_BF16) {
        for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
            const nv_bfloat16 v = __float2bfloat16_rn(x[d]);
            ((nv_bfloat16 *) head)[d] = v;
            x[d] = __bfloat162float(v);
        }
    } else {
        for (unsigned int blk = threadIdx.x; blk < hd / 32; blk += blockDim.x) {
            kv_quantize_block(ty, x + 32 * blk, head + blk * kv_block_bytes(ty));
        }
    }
    __syncthreads();
}
