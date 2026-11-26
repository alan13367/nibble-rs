#pragma once

// Quantized MMVQ primitives transcribed from ggml b12675
// ggml/src/ggml-cuda/vecdotq.cuh. The integer work, dp4a calls, half2
// arithmetic, indexing and floating-point update order are kept unchanged;
// ggml's global I-quant tables are reached through the caller's Codebooks.
// dequant_block is a one-thread/one-block adaptation of dequantize.cuh and
// dequantize_row_* in ggml-quants.c for embedding lookups.

#include "common.cuh"

namespace quant_detail {

static constexpr int VDR_Q2_0 = 1;
static constexpr int VDR_Q4_0 = 2;
static constexpr int VDR_Q4_1 = 2;
static constexpr int VDR_Q5_0 = 2;
static constexpr int VDR_Q5_1 = 2;
static constexpr int VDR_Q8_0 = 2;
static constexpr int VDR_IQ4_NL = 2;
static constexpr int VDR_MXFP4 = 2;
static constexpr int VDR_Q2_K = 2;
static constexpr int VDR_Q3_K = 2;
static constexpr int VDR_Q4_K = 2;
static constexpr int VDR_Q5_K = 2;
static constexpr int VDR_Q6_K = 2;
static constexpr int VDR_IQ4_XS = 4;
static constexpr int VDR_IQ2_XXS = 2;
static constexpr int VDR_IQ2_XS = 2;
static constexpr int VDR_IQ2_S = 2;
static constexpr int VDR_IQ3_XXS = 2;
static constexpr int VDR_IQ3_S = 2;
static constexpr int VDR_IQ1_S = 1;
static constexpr int VDR_IQ1_M = 1;

static constexpr float IQ1S_DELTA = 0.125f;
static constexpr float IQ1M_DELTA = 0.125f;

static __device__ __forceinline__ int2 get_int_from_table_16(int q4, const int8_t * table) {
    // CUDA's byte selector has three-bit indices. Select both table halves,
    // then use each index's fourth bit to select between them.
    const uint32_t * table32 = reinterpret_cast<const uint32_t *>(table);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

static __device__ __forceinline__ uint32_t unpack_ksigns(uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101;
}

template<int vdr>
static __device__ __forceinline__ float dot_q4_0(const int * v, const int * u, float d4, half2 ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

template<int vdr>
static __device__ __forceinline__ float dot_q4_1(const int * v, const int * u, half2 dm4, half2 ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ != 610
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

template<int vdr>
static __device__ __forceinline__ float dot_q5_0(const int * vl, const int * vh, const int * u, float d5, half2 ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
        vi0 |= (vh[i] << 4) & 0x00000010;
        vi0 |= (vh[i] << 11) & 0x00001000;
        vi0 |= (vh[i] << 18) & 0x00100000;
        vi0 |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);

        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
        vi1 |= (vh[i] >> 12) & 0x00000010;
        vi1 |= (vh[i] >> 5) & 0x00001000;
        vi1 |= (vh[i] << 2) & 0x00100000;
        vi1 |= (vh[i] << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

template<int vdr>
static __device__ __forceinline__ float dot_q5_1(const int * vl, const int * vh, const int * u, half2 dm5, half2 ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
        vi0 |= (vh[i] << 4) & 0x00000010;
        vi0 |= (vh[i] << 11) & 0x00001000;
        vi0 |= (vh[i] << 18) & 0x00100000;
        vi0 |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);

        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
        vi1 |= (vh[i] >> 12) & 0x00000010;
        vi1 |= (vh[i] >> 5) & 0x00001000;
        vi1 |= (vh[i] << 2) & 0x00100000;
        vi1 |= (vh[i] << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ != 610
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

template<int vdr>
static __device__ __forceinline__ float dot_q8_0(const int * v, const int * u, float d8_0, float d8_1) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }
    return d8_0*d8_1 * float(sumi);
}

static __device__ __forceinline__ float dot_q2_K_impl(
        const int * v, const int * u, const int * scales, half2 dm2, const float * d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[i/2] >> (16*(i%2));
        int sumi_d = 0;
        int sumi_m = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q2_K; ++j) {
            const int vi = (v[j] >> (2*i)) & 0x03030303;
            sumi_d = ggml_cuda_dp4a(vi, u[VDR_Q2_K*i + j], sumi_d);
            sumi_m = ggml_cuda_dp4a(0x01010101, u[VDR_Q2_K*i + j], sumi_m);
        }
        sumf_d += d8[i] * (sumi_d * (sc & 0xF));
        sumf_m += d8[i] * (sumi_m * ((sc >> 4) & 0xF));
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

static __device__ __forceinline__ float dot_q3_K_impl(
        const int * vl, const int * vh, const int * u, const int * scales, float d3, const float * d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int sc_low = (scales[i/2] >> (16*(i%2))) & 0x0F;
        const int sc_high = (scales[2] >> (16*(i%2) + 2*(i/2))) & 0x03;
        const int sc = (sc_low | (sc_high << 4)) - 32;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q3_K; ++j) {
            const int vil = (vl[j] >> (2*i)) & 0x03030303;
            const int vih = ((vh[j] >> i) << 2) & 0x04040404;
            sumi = ggml_cuda_dp4a(vil | vih, u[VDR_Q3_K*i + j], sumi);
            sumi = ggml_cuda_dp4a(0xFCFCFCFC, u[VDR_Q3_K*i + j], sumi);
        }
        sumf += d8[i] * (sumi * sc);
    }
    return d3 * sumf;
}

static __device__ __forceinline__ float dot_q4_K_impl(
        const int * v, const int * u, const uint8_t * sc, const uint8_t * m, half2 dm4, const float * d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float dot_q5_K_impl(
        const int * vl, const int * vh, const int * u, const uint8_t * sc, const uint8_t * m, half2 dm5, const float * d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

static __device__ __forceinline__ float dot_q6_K_impl(
        const int * vl, const int * vh, const int * u, const int8_t * scales, float d, const float * d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q6_K; ++j) {
            const int vil = (vl[j] >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh[j] >> (4*i)) << 4) & 0x30303030;
            sumi = ggml_cuda_dp4a(vil | vih, u[VDR_Q6_K*i + j], sumi);
            sumi = ggml_cuda_dp4a(0xE0E0E0E0, u[VDR_Q6_K*i + j], sumi);
        }
        sumf += d8[i] * (sumi * sc);
    }
    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q8_0(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q8_0 * bq8_0 = static_cast<const block_q8_0 *>(vbq) + kbx;
    int v[VDR_Q8_0];
    int u[VDR_Q8_0];
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }
    return dot_q8_0<VDR_Q8_0>(v, u, half_to_float(bq8_0->d), __low2float(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q4_0(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q4_0 * bq4_0 = static_cast<const block_q4_0 *>(vbq) + kbx;
    int v[VDR_Q4_0];
    int u[2*VDR_Q4_0];
#pragma unroll
    for (int i = 0; i < VDR_Q4_0; ++i) {
        v[i] = get_int_b2(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }
    return dot_q4_0<VDR_Q4_0>(v, u, half_to_float(bq4_0->d), bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q4_1(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q4_1 * bq4_1 = static_cast<const block_q4_1 *>(vbq) + kbx;
    int v[VDR_Q4_1];
    int u[2*VDR_Q4_1];
#pragma unroll
    for (int i = 0; i < VDR_Q4_1; ++i) {
        v[i] = get_int_b4(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }
    return dot_q4_1<VDR_Q4_1>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q5_0 * bq5_0 = static_cast<const block_q5_0 *>(vbq) + kbx;
    int vl[VDR_Q5_0];
    int vh[VDR_Q5_0];
    int u[2*VDR_Q5_0];
#pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        vl[i] = get_int_b2(bq5_0->qs, iqs + i);
        vh[i] = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }
    return dot_q5_0<VDR_Q5_0>(vl, vh, u, half_to_float(bq5_0->d), bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q5_1 * bq5_1 = static_cast<const block_q5_1 *>(vbq) + kbx;
    int vl[VDR_Q5_1];
    int vh[VDR_Q5_1];
    int u[2*VDR_Q5_1];
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        vl[i] = get_int_b4(bq5_1->qs, iqs + i);
        vh[i] = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }
    return dot_q5_1<VDR_Q5_1>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_iq4_nl(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_iq4_nl * bq4 = static_cast<const block_iq4_nl *>(vbq) + kbx;
    const int * q8 = reinterpret_cast<const int *>(bq8_1->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ4_NL; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_mxfp4(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_mxfp4 * bq4 = static_cast<const block_mxfp4 *>(vbq) + kbx;
    const int * q8 = reinterpret_cast<const int *>(bq8_1->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_q2_K(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q2_K * bq2_K = static_cast<const block_q2_K *>(vbq) + kbx;
    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int iqs8 = iqs % QI8_1;
    int v[VDR_Q2_K];
#pragma unroll
    for (int j = 0; j < VDR_Q2_K; ++j) {
        v[j] = get_int_b4(bq2_K->qs, iqs + j);
    }
    const int sc_shift = 8 * (iqs8 / (QI8_1/2));
    int scales[2];
    scales[0] = get_int_b4(bq2_K->scales, 2*(iqs / QI8_1) + 0) >> sc_shift;
    scales[1] = get_int_b4(bq2_K->scales, 2*(iqs / QI8_1) + 1) >> sc_shift;
    int u[VDR_Q2_K*QR2_K];
    float d8[QR2_K];
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q2_K; ++j) {
            u[VDR_Q2_K*i + j] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs8 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return dot_q2_K_impl(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q3_K * bq3_K = static_cast<const block_q3_K *>(vbq) + kbx;
    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int iqs8 = iqs % QI8_1;
    const float d = half_to_float(bq3_K->d);
    int vl[VDR_Q3_K];
    int vh[VDR_Q3_K];
#pragma unroll
    for (int j = 0; j < VDR_Q3_K; ++j) {
        vl[j] = get_int_b2(bq3_K->qs, iqs + j);
        vh[j] = get_int_b2(bq3_K->hmask, iqs8 + j) >> bq8_offset;
    }
    const int sc_shift = 8 * (iqs8 / (QI8_1/2)) + 4 * (iqs / (QI3_K/2));
    int scales[3];
#pragma unroll
    for (int l = 0; l < 3; ++l) {
        scales[l] = get_int_b2(bq3_K->scales, l) >> sc_shift;
    }
    int u[VDR_Q3_K*QR3_K];
    float d8[QR3_K];
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q3_K; ++j) {
            u[VDR_Q3_K*i + j] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs8 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return dot_q3_K_impl(vl, vh, u, scales, d, d8);
}

static __device__ __forceinline__ void get_k45_scales(
        const uint8_t * scales_bytes, int bq8_offset, uint8_t * sc, uint8_t * m) {
    const uint16_t * scales = reinterpret_cast<const uint16_t *>(scales_bytes);
    const int j = bq8_offset/2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    sc[0] = reinterpret_cast<const uint8_t *>(aux)[0];
    sc[1] = reinterpret_cast<const uint8_t *>(aux)[1];
    m[0] = reinterpret_cast<const uint8_t *>(aux)[2];
    m[1] = reinterpret_cast<const uint8_t *>(aux)[3];
}

static __device__ __forceinline__ float vec_dot_q4_K(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q4_K * bq4_K = static_cast<const block_q4_K *>(vbq) + kbx;
    int v[2];
    int u[2*QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int * q4 = reinterpret_cast<const int *>(bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint8_t sc[2];
    uint8_t m[2];
    get_k45_scales(bq4_K->scales, bq8_offset, sc, m);
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }
    return dot_q4_K_impl(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q5_K(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q5_K * bq5_K = static_cast<const block_q5_K *>(vbq) + kbx;
    int vl[2];
    int vh[2];
    int u[2*QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = reinterpret_cast<const int *>(bq5_K->qs + 16*bq8_offset + 4*((iqs/2)%4));
    const int * qh = reinterpret_cast<const int *>(bq5_K->qh + 4*((iqs/2)%4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint8_t sc[2];
    uint8_t m[2];
    get_k45_scales(bq5_K->scales, bq8_offset, sc, m);
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }
    return dot_q5_K_impl(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q6_K * bq6_K = static_cast<const block_q6_K *>(vbq) + kbx;
    const int bq8_offset = 2*QR6_K*(iqs/(QI6_K/2)) + (iqs%(QI6_K/2))/(QI6_K/4);
    const int scale_offset = (QI6_K/4)*(iqs/(QI6_K/2)) + (iqs%(QI6_K/2))/(QI6_K/8);
    const int vh_shift = 2*((iqs%(QI6_K/2))/(QI6_K/4));
    int vl[VDR_Q6_K];
    int vh[VDR_Q6_K];
#pragma unroll
    for (int j = 0; j < VDR_Q6_K; ++j) {
        vl[j] = get_int_b2(bq6_K->ql, iqs + j);
        vh[j] = get_int_b2(bq6_K->qh, (QI6_K/4)*(iqs/(QI6_K/2)) + iqs%(QI6_K/4) + j) >> vh_shift;
    }
    const int8_t * scales = bq6_K->scales + scale_offset;
    int u[VDR_Q6_K*QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q6_K; ++j) {
            u[VDR_Q6_K*i + j] = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs%QI8_1 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }
    return dot_q6_K_impl(vl, vh, u, scales, half_to_float(bq6_K->d), d8);
}

static __device__ __forceinline__ float vec_dot_iq4_xs(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_iq4_xs * bq4 = static_cast<const block_iq4_xs *>(vbq) + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_xxs(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks * cb) {
    const block_iq2_xxs * bq2 = static_cast<const block_iq2_xxs *>(vbq) + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = reinterpret_cast<const uint8_t *>(&q2);
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = reinterpret_cast<const uint2 *>(cb->iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7*k0/2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_xs(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks * cb) {
    const block_iq2_xs * bq2 = static_cast<const block_iq2_xs *>(vbq) + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = reinterpret_cast<const uint16_t *>(&q2_packed);
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = reinterpret_cast<const uint2 *>(cb->iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_s(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks * cb) {
    const block_iq2_s * bq2 = static_cast<const block_iq2_s *>(vbq) + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
    const int qh = bq2->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8 = reinterpret_cast<const uint8_t *>(&signs_packed_32);
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = reinterpret_cast<const int *>(cb->iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_xxs(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks * cb) {
    const block_iq3_xxs * bq3 = static_cast<const block_iq3_xxs *>(vbq) + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = reinterpret_cast<const uint8_t *>(&q3_packed);
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(cb->iq3xxs_grid[q3[l0 + 0]], cb->iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_s(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks * cb) {
    const block_iq3_s * bq3 = static_cast<const block_iq3_s *>(vbq) + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
    const int qh = bq3->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8 = reinterpret_cast<const uint8_t *>(&signs_packed_32);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            cb->iq3s_grid[qs[l0 + 0] | ((qh << (8-l0)) & 0x100)],
            cb->iq3s_grid[qs[l0 + 1] | ((qh << (7-l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq1_s(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks * cb) {
    const block_iq1_s * bq1 = static_cast<const block_iq1_s *>(vbq) + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = cb->iq1s_grid_gpu[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000)*(2.0f*IQ1S_DELTA/0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x*sumi + ds.y*delta);
}

static __device__ __forceinline__ float vec_dot_iq1_m(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks * cb) {
    const block_iq1_m * bq1 = static_cast<const block_iq1_m *>(vbq) + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
    int sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2*iqs + l0/4] >> (4*((l0/2)%2));
        const int grid = cb->iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0/4] = ggml_cuda_dp4a(grid0, u0, sumi[l0/4]);
        sumi[l0/4] = ggml_cuda_dp4a(grid1, u1, sumi[l0/4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08)*(2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }
    const uint16_t * sc = reinterpret_cast<const uint16_t *>(bq1->scales);
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs/2] >> (6*(iqs%2));
    const int sc0 = 2*((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2*((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0])*sc0 + (sumi[1] + sumf[1])*sc1);
}

static __device__ __forceinline__ float vec_dot_q2_0(
        const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs, const Codebooks *) {
    const block_q2_0 * bq2_0 = static_cast<const block_q2_0 *>(vbq) + kbx;
    const float d2 = half_to_float(bq2_0->d);
    const int16_t * qs = reinterpret_cast<const int16_t *>(bq2_0->qs) + iqs*4;
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j*2+0);
        const int v = get_int_b4(bq8_1_chunk->qs, j*2+1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

template<typename T>
static __device__ __forceinline__ const T * block_at(const void * row, int block) {
    return static_cast<const T *>(row) + block;
}

static __device__ __forceinline__ void dequant_f32(const void * row, int block, float * out, const Codebooks *) {
    out[0] = block_at<float>(row, block)[0];
}

static __device__ __forceinline__ void dequant_f16(const void * row, int block, float * out, const Codebooks *) {
    out[0] = half_to_float(block_at<half>(row, block)[0]);
}

static __device__ __forceinline__ void dequant_bf16(const void * row, int block, float * out, const Codebooks *) {
    out[0] = bf16_to_float(block_at<nv_bfloat16>(row, block)[0]);
}

static __device__ __forceinline__ void dequant_q8_0(const void * row, int block, float * out, const Codebooks *) {
    const block_q8_0 * x = block_at<block_q8_0>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        out[j] = x->qs[j]*d;
    }
}

static __device__ __forceinline__ void dequant_q4_0(const void * row, int block, float * out, const Codebooks *) {
    const block_q4_0 * x = block_at<block_q4_0>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int j = 0; j < QK4_0/2; ++j) {
        const int x0 = (x->qs[j] & 0x0F) - 8;
        const int x1 = (x->qs[j] >> 4) - 8;
        out[j] = x0*d;
        out[j + QK4_0/2] = x1*d;
    }
}

static __device__ __forceinline__ void dequant_q4_1(const void * row, int block, float * out, const Codebooks *) {
    const block_q4_1 * x = block_at<block_q4_1>(row, block);
    const float2 dm = __half22float2(x->dm);
#pragma unroll
    for (int j = 0; j < QK4_1/2; ++j) {
        const int x0 = x->qs[j] & 0x0F;
        const int x1 = x->qs[j] >> 4;
        out[j] = x0*dm.x + dm.y;
        out[j + QK4_1/2] = x1*dm.x + dm.y;
    }
}

static __device__ __forceinline__ uint32_t load_u32(const uint8_t * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

static __device__ __forceinline__ void dequant_q5_0(const void * row, int block, float * out, const Codebooks *) {
    const block_q5_0 * x = block_at<block_q5_0>(row, block);
    const float d = half_to_float(x->d);
    const uint32_t qh = load_u32(x->qh);
#pragma unroll
    for (int j = 0; j < QK5_0/2; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh_1 = (qh >> (j + 12)) & 0x10;
        const int x0 = ((x->qs[j] & 0x0F) | xh_0) - 16;
        const int x1 = ((x->qs[j] >> 4) | xh_1) - 16;
        out[j] = x0*d;
        out[j + QK5_0/2] = x1*d;
    }
}

static __device__ __forceinline__ void dequant_q5_1(const void * row, int block, float * out, const Codebooks *) {
    const block_q5_1 * x = block_at<block_q5_1>(row, block);
    const float2 dm = __half22float2(x->dm);
    const uint32_t qh = load_u32(x->qh);
#pragma unroll
    for (int j = 0; j < QK5_1/2; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh_1 = (qh >> (j + 12)) & 0x10;
        const int x0 = (x->qs[j] & 0x0F) | xh_0;
        const int x1 = (x->qs[j] >> 4) | xh_1;
        out[j] = x0*dm.x + dm.y;
        out[j + QK5_1/2] = x1*dm.x + dm.y;
    }
}

static __device__ __forceinline__ void dequant_iq4_nl(const void * row, int block, float * out, const Codebooks *) {
    const block_iq4_nl * x = block_at<block_iq4_nl>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int j = 0; j < QK4_NL/2; ++j) {
        out[j] = d*kvalues_iq4nl[x->qs[j] & 0x0F];
        out[j + QK4_NL/2] = d*kvalues_iq4nl[x->qs[j] >> 4];
    }
}

static __device__ __forceinline__ void dequant_mxfp4(const void * row, int block, float * out, const Codebooks *) {
    const block_mxfp4 * x = block_at<block_mxfp4>(row, block);
    const float d = e8m0_to_fp32(x->e)*0.5f;
#pragma unroll
    for (int j = 0; j < QK_MXFP4/2; ++j) {
        out[j] = kvalues_mxfp4[x->qs[j] & 0x0F]*d;
        out[j + QK_MXFP4/2] = kvalues_mxfp4[x->qs[j] >> 4]*d;
    }
}

static __device__ __forceinline__ void dequant_q2_K(const void * row, int block, float * out, const Codebooks *) {
    const block_q2_K * x = block_at<block_q2_K>(row, block);
    const float d = __low2float(x->dm);
    const float min = __high2float(x->dm);
    const uint8_t * q = x->qs;
    float * y = out;
    int is = 0;
#pragma unroll
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = x->scales[is++];
            float dl = d*(sc & 0xF);
            float ml = min*(sc >> 4);
#pragma unroll
            for (int l = 0; l < 16; ++l) *y++ = dl*int8_t((q[l] >> shift) & 3) - ml;
            sc = x->scales[is++];
            dl = d*(sc & 0xF);
            ml = min*(sc >> 4);
#pragma unroll
            for (int l = 0; l < 16; ++l) *y++ = dl*int8_t((q[l+16] >> shift) & 3) - ml;
            shift += 2;
        }
        q += 32;
    }
}

static __device__ __forceinline__ void dequant_q3_K(const void * row, int block, float * out, const Codebooks *) {
    const block_q3_K * x = block_at<block_q3_K>(row, block);
    const float d_all = half_to_float(x->d);
    const uint8_t * q = x->qs;
    const uint8_t * hm = x->hmask;
    uint8_t m = 1;
    uint32_t aux[4] = {};
    uint8_t * aux_bytes = reinterpret_cast<uint8_t *>(aux);
#pragma unroll
    for (int j = 0; j < 12; ++j) aux_bytes[j] = x->scales[j];
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
    const int8_t * scales = reinterpret_cast<const int8_t *>(aux);
    float * y = out;
    int is = 0;
#pragma unroll
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float dl = d_all*(scales[is++] - 32);
#pragma unroll
            for (int l = 0; l < 16; ++l) {
                *y++ = dl*(int8_t((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            }
            dl = d_all*(scales[is++] - 32);
#pragma unroll
            for (int l = 0; l < 16; ++l) {
                *y++ = dl*(int8_t((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j+4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ void dequant_q4_K(const void * row, int block, float * out, const Codebooks *) {
    const block_q4_K * x = block_at<block_q4_K>(row, block);
    const uint8_t * q = x->qs;
    const float d = __low2float(x->dm);
    const float min = __high2float(x->dm);
    float * y = out;
    int is = 0;
#pragma unroll
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x->scales, sc, m);
        const float d1 = d*sc;
        const float m1 = min*m;
        get_scale_min_k4(is + 1, x->scales, sc, m);
        const float d2 = d*sc;
        const float m2 = min*m;
#pragma unroll
        for (int l = 0; l < 32; ++l) *y++ = d1*(q[l] & 0xF) - m1;
#pragma unroll
        for (int l = 0; l < 32; ++l) *y++ = d2*(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

static __device__ __forceinline__ void dequant_q5_K(const void * row, int block, float * out, const Codebooks *) {
    const block_q5_K * x = block_at<block_q5_K>(row, block);
    const uint8_t * ql = x->qs;
    const uint8_t * qh = x->qh;
    const float d = __low2float(x->dm);
    const float min = __high2float(x->dm);
    float * y = out;
    int is = 0;
    uint8_t u1 = 1;
    uint8_t u2 = 2;
#pragma unroll
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x->scales, sc, m);
        const float d1 = d*sc;
        const float m1 = min*m;
        get_scale_min_k4(is + 1, x->scales, sc, m);
        const float d2 = d*sc;
        const float m2 = min*m;
#pragma unroll
        for (int l = 0; l < 32; ++l) *y++ = d1*((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
#pragma unroll
        for (int l = 0; l < 32; ++l) *y++ = d2*((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

static __device__ __forceinline__ void dequant_q6_K(const void * row, int block, float * out, const Codebooks *) {
    const block_q6_K * x = block_at<block_q6_K>(row, block);
    const float d = half_to_float(x->d);
    const uint8_t * ql = x->ql;
    const uint8_t * qh = x->qh;
    const int8_t * sc = x->scales;
    float * y = out;
#pragma unroll
    for (int n = 0; n < QK_K; n += 128) {
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int is = l/16;
            const int8_t q1 = int8_t((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = int8_t((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = int8_t((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = int8_t((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d*sc[is+0]*q1;
            y[l+32] = d*sc[is+2]*q2;
            y[l+64] = d*sc[is+4]*q3;
            y[l+96] = d*sc[is+6]*q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static __device__ __forceinline__ void dequant_iq4_xs(const void * row, int block, float * out, const Codebooks *) {
    const block_iq4_xs * x = block_at<block_iq4_xs>(row, block);
    const uint8_t * qs = x->qs;
    const float d = half_to_float(x->d);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        const int ls = ((x->scales_l[ib/2] >> 4*(ib%2)) & 0x0F) | (((x->scales_h >> 2*ib) & 3) << 4);
        const float dl = d*(ls - 32);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            out[32*ib + j] = dl*kvalues_iq4nl[qs[j] & 0x0F];
            out[32*ib + j + 16] = dl*kvalues_iq4nl[qs[j] >> 4];
        }
        qs += 16;
    }
}

static __device__ __forceinline__ uint8_t grid_byte(uint64_t grid, int j) {
    return uint8_t(grid >> (8*j));
}

static __device__ __forceinline__ uint8_t grid_byte(uint32_t grid, int j) {
    return uint8_t(grid >> (8*j));
}

static __device__ __forceinline__ void dequant_iq2_xxs(const void * row, int block, float * out, const Codebooks * cb) {
    const block_iq2_xxs * x = block_at<block_iq2_xxs>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        const uint16_t * q2 = x->qs + 4*ib;
        const uint8_t * aux8 = reinterpret_cast<const uint8_t *>(q2);
        const uint32_t aux32 = uint32_t(q2[2]) | (uint32_t(q2[3]) << 16);
        const float db = d*(0.5f + (aux32 >> 28))*0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint64_t grid = cb->iq2xxs_grid[aux8[l]];
            const uint8_t signs = cb->ksigns_iq2xs[(aux32 >> 7*l) & 127];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[32*ib + 8*l + j] = db*grid_byte(grid, j)*(signs & (1 << j) ? -1.0f : 1.0f);
            }
        }
    }
}

static __device__ __forceinline__ void dequant_iq2_xs(const void * row, int block, float * out, const Codebooks * cb) {
    const block_iq2_xs * x = block_at<block_iq2_xs>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        float db[2];
        db[0] = d*(0.5f + (x->scales[ib] & 0x0F))*0.25f;
        db[1] = d*(0.5f + (x->scales[ib] >> 4))*0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint16_t q2 = x->qs[4*ib + l];
            const uint64_t grid = cb->iq2xs_grid[q2 & 511];
            const uint8_t signs = cb->ksigns_iq2xs[q2 >> 9];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[32*ib + 8*l + j] = db[l/2]*grid_byte(grid, j)*(signs & (1 << j) ? -1.0f : 1.0f);
            }
        }
    }
}

static __device__ __forceinline__ void dequant_iq2_s(const void * row, int block, float * out, const Codebooks * cb) {
    const block_iq2_s * x = block_at<block_iq2_s>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        float db[2];
        db[0] = d*(0.5f + (x->scales[ib] & 0x0F))*0.25f;
        db[1] = d*(0.5f + (x->scales[ib] >> 4))*0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int index = x->qs[4*ib+l] | ((x->qh[ib] << (8-2*l)) & 0x300);
            const uint64_t grid = cb->iq2s_grid[index];
            const uint8_t signs = x->qs[QK_K/8 + 4*ib+l];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[32*ib + 8*l + j] = db[l/2]*grid_byte(grid, j)*(signs & (1 << j) ? -1.0f : 1.0f);
            }
        }
    }
}

static __device__ __forceinline__ void dequant_iq3_xxs(const void * row, int block, float * out, const Codebooks * cb) {
    const block_iq3_xxs * x = block_at<block_iq3_xxs>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        const uint32_t aux32 = load_u32(x->qs + QK_K/4 + 4*ib);
        const float db = d*(0.5f + (aux32 >> 28))*0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = cb->ksigns_iq2xs[(aux32 >> 7*l) & 127];
            const uint32_t grid1 = cb->iq3xxs_grid[x->qs[8*ib + 2*l+0]];
            const uint32_t grid2 = cb->iq3xxs_grid[x->qs[8*ib + 2*l+1]];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[32*ib + 8*l + j] = db*grid_byte(grid1, j)*(signs & (1 << j) ? -1.0f : 1.0f);
                out[32*ib + 8*l + j+4] = db*grid_byte(grid2, j)*(signs & (1 << (j+4)) ? -1.0f : 1.0f);
            }
        }
    }
}

static __device__ __forceinline__ void dequant_iq3_s(const void * row, int block, float * out, const Codebooks * cb) {
    const block_iq3_s * x = block_at<block_iq3_s>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        const float db = d*(1 + 2*((x->scales[ib/2] >> 4*(ib%2)) & 0x0F));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = x->signs[4*ib+l];
            const int index1 = x->qs[8*ib+2*l+0] | ((x->qh[ib] << (8-2*l)) & 256);
            const int index2 = x->qs[8*ib+2*l+1] | ((x->qh[ib] << (7-2*l)) & 256);
            const uint32_t grid1 = cb->iq3s_grid[index1];
            const uint32_t grid2 = cb->iq3s_grid[index2];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[32*ib + 8*l + j] = db*grid_byte(grid1, j)*(signs & (1 << j) ? -1.0f : 1.0f);
                out[32*ib + 8*l + j+4] = db*grid_byte(grid2, j)*(signs & (1 << (j+4)) ? -1.0f : 1.0f);
            }
        }
    }
}

static __device__ __forceinline__ void dequant_iq1_s(const void * row, int block, float * out, const Codebooks * cb) {
    const block_iq1_s * x = block_at<block_iq1_s>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        const uint16_t qh = x->qh[ib];
        const float dl = d*(2*((qh >> 12) & 7) + 1);
        const float delta = qh & 0x8000 ? -1.0f - IQ1S_DELTA : -1.0f + IQ1S_DELTA;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            uint32_t grid32[2];
            grid32[0] = cb->iq1s_grid_gpu[x->qs[4*ib+l] | (((qh >> 3*l) & 7) << 8)];
            grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
            grid32[0] &= 0x0f0f0f0f;
            const int8_t * q = reinterpret_cast<const int8_t *>(grid32);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[32*ib + 8*l + j] = dl*(q[j] + delta);
            }
        }
    }
}

static __device__ __forceinline__ void dequant_iq1_m(const void * row, int block, float * out, const Codebooks * cb) {
    const block_iq1_m * x = block_at<block_iq1_m>(row, block);
    const uint16_t * sc = reinterpret_cast<const uint16_t *>(x->scales);
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = half_to_float(scale.f16);
#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int ib16 = 2*ib + l/2;
            const float dl = d*(2*((sc[ib16/4] >> 3*(ib16%4)) & 7) + 1);
            const float delta = x->qh[2*ib+l/2] & (0x08 << 4*(l%2)) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
            uint32_t grid32[2];
            grid32[0] = cb->iq1s_grid_gpu[x->qs[4*ib+l] | (((x->qh[2*ib+l/2] >> 4*(l%2)) & 7) << 8)];
            grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
            grid32[0] &= 0x0f0f0f0f;
            const int8_t * q = reinterpret_cast<const int8_t *>(grid32);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[32*ib + 8*l + j] = dl*(q[j] + delta);
            }
        }
    }
}

static __device__ __forceinline__ void dequant_q2_0(const void * row, int block, float * out, const Codebooks *) {
    const block_q2_0 * x = block_at<block_q2_0>(row, block);
    const float d = half_to_float(x->d);
#pragma unroll
    for (int j = 0; j < QK2_0; ++j) {
        const int q = (x->qs[j/4] >> 2*(j%4)) & 3;
        out[j] = (q - 1)*d;
    }
}

} // namespace quant_detail

template<int TYPE>
static __device__ __forceinline__ float vec_dot_q(
        const void * weights, const block_q8_1 * x, int kbx, int iqs, const Codebooks * cb) {
    static_assert(TYPE >= 3 && TYPE <= 23, "vec_dot_q supports quantized types 3..23");
    if constexpr (TYPE == 3) return quant_detail::vec_dot_q8_0(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 4) return quant_detail::vec_dot_q4_0(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 5) return quant_detail::vec_dot_q4_1(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 6) return quant_detail::vec_dot_q5_0(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 7) return quant_detail::vec_dot_q5_1(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 8) return quant_detail::vec_dot_iq4_nl(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 9) return quant_detail::vec_dot_mxfp4(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 10) return quant_detail::vec_dot_q2_K(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 11) return quant_detail::vec_dot_q3_K(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 12) return quant_detail::vec_dot_q4_K(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 13) return quant_detail::vec_dot_q5_K(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 14) return quant_detail::vec_dot_q6_K(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 15) return quant_detail::vec_dot_iq4_xs(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 16) return quant_detail::vec_dot_iq2_xxs(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 17) return quant_detail::vec_dot_iq2_xs(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 18) return quant_detail::vec_dot_iq2_s(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 19) return quant_detail::vec_dot_iq3_xxs(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 20) return quant_detail::vec_dot_iq3_s(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 21) return quant_detail::vec_dot_iq1_s(weights, x, kbx, iqs, cb);
    else if constexpr (TYPE == 22) return quant_detail::vec_dot_iq1_m(weights, x, kbx, iqs, cb);
    else return quant_detail::vec_dot_q2_0(weights, x, kbx, iqs, cb);
}

template<int TYPE>
static __device__ __forceinline__ void dequant_block(
        const void * row, int block, float * out, const Codebooks * cb) {
    static_assert(TYPE >= 0 && TYPE <= 23, "dequant_block supports types 0..23");
    if constexpr (TYPE == 0) quant_detail::dequant_f32(row, block, out, cb);
    else if constexpr (TYPE == 1) quant_detail::dequant_f16(row, block, out, cb);
    else if constexpr (TYPE == 2) quant_detail::dequant_bf16(row, block, out, cb);
    else if constexpr (TYPE == 3) quant_detail::dequant_q8_0(row, block, out, cb);
    else if constexpr (TYPE == 4) quant_detail::dequant_q4_0(row, block, out, cb);
    else if constexpr (TYPE == 5) quant_detail::dequant_q4_1(row, block, out, cb);
    else if constexpr (TYPE == 6) quant_detail::dequant_q5_0(row, block, out, cb);
    else if constexpr (TYPE == 7) quant_detail::dequant_q5_1(row, block, out, cb);
    else if constexpr (TYPE == 8) quant_detail::dequant_iq4_nl(row, block, out, cb);
    else if constexpr (TYPE == 9) quant_detail::dequant_mxfp4(row, block, out, cb);
    else if constexpr (TYPE == 10) quant_detail::dequant_q2_K(row, block, out, cb);
    else if constexpr (TYPE == 11) quant_detail::dequant_q3_K(row, block, out, cb);
    else if constexpr (TYPE == 12) quant_detail::dequant_q4_K(row, block, out, cb);
    else if constexpr (TYPE == 13) quant_detail::dequant_q5_K(row, block, out, cb);
    else if constexpr (TYPE == 14) quant_detail::dequant_q6_K(row, block, out, cb);
    else if constexpr (TYPE == 15) quant_detail::dequant_iq4_xs(row, block, out, cb);
    else if constexpr (TYPE == 16) quant_detail::dequant_iq2_xxs(row, block, out, cb);
    else if constexpr (TYPE == 17) quant_detail::dequant_iq2_xs(row, block, out, cb);
    else if constexpr (TYPE == 18) quant_detail::dequant_iq2_s(row, block, out, cb);
    else if constexpr (TYPE == 19) quant_detail::dequant_iq3_xxs(row, block, out, cb);
    else if constexpr (TYPE == 20) quant_detail::dequant_iq3_s(row, block, out, cb);
    else if constexpr (TYPE == 21) quant_detail::dequant_iq1_s(row, block, out, cb);
    else if constexpr (TYPE == 22) quant_detail::dequant_iq1_m(row, block, out, cb);
    else quant_detail::dequant_q2_0(row, block, out, cb);
}
