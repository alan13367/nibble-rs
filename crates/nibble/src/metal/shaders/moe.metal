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
    float eps;
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

// Fused `x[t][i] += moe[t][i] + shared[t][i] * sigmoid(gate[t])`, then
// `out[t][i] = rmsnorm(x[t])[i] * norm[i]`: one threadgroup per token.
kernel void moe_residual(
        constant MoeSumArgs & a [[buffer(0)]],
        device const float * moe [[buffer(1)]],
        device const float * shared [[buffer(2)]],
        device const float * gate [[buffer(3)]],
        device float * x [[buffer(4)]],
        device const float * norm [[buffer(5)]],
        device float * out [[buffer(6)]],
        uint t [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float buf[N_SIMDWIDTH];
    const uint64_t row = (uint64_t) t * a.n;
    const float g = sigmoid(gate[t]);
    float sum = 0.0f;
    for (uint i = tid; i < a.n; i += ntg) {
        const float v = x[row + i] + (moe[row + i] + shared[row + i] * g);
        x[row + i] = v;
        sum += v * v;
    }
    sum = threadgroup_sum(sum, buf, tiisg, sgitg, nsg);
    const float scale = 1.0f / sqrt(sum / a.n + a.eps);
    for (uint i = tid; i < a.n; i += ntg) {
        out[row + i] = x[row + i] * scale * norm[i];
    }
}

// --- Expert products of a prompt batch ---------------------------------------
//
// ggml-metal's `kernel_mul_mm_id_map0` and `kernel_mul_mm_id` (llama.cpp
// `ggml/src/ggml-metal/kernels/mul_mm.metal`, b12723): the batch's slots
// are sorted by expert into tiles of up to 32, and each tile is multiplied
// by its expert's matrix with `matmul_impl`'s tiles, so each expert's
// weights are read once per tile instead of once per slot.

// `MoeMapArgs` in `metal/kernels.rs`: `n` tokens of `k` slots, `n_expert`
// experts.
struct MoeMapArgs {
    uint n;
    uint k;
    uint n_expert;
};

// One threadgroup of `n_expert` threads, thread `e` for expert `e`: lists
// the slots `t * k + j` routed to each expert, one after another (each
// expert's list padded by one entry) in `ids`, and describes tiles of up to
// 32 of them in `tiles`: their count, then (expert, first entry, entries)
// for each.
kernel void moe_map(
        constant MoeMapArgs & a [[buffer(0)]],
        device const float * route [[buffer(1)]],
        device uint * tiles [[buffer(2)]],
        device int * ids [[buffer(3)]],
        threadgroup char * shmem [[threadgroup(0)]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const ushort ide = tpitg;
    const uint k = a.k;
    threadgroup uint * cnt = (threadgroup uint *) shmem;
    threadgroup uint * bas = cnt + a.n_expert;
    threadgroup ushort * sids = (threadgroup ushort *) (bas + a.n_expert);

    uint n_all = 0;
    for (uint i21 = 0; i21 < a.n; i21 += ntg) {
        if (i21 + tpitg < a.n) {
            device const float * r = route + (uint64_t) (i21 + tpitg) * 2 * k;
            for (uint j = 0; j < k; ++j) {
                sids[tpitg * k + j] = (ushort) (uint) r[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort t = 0; t < ntg; t++) {
            if (i21 + t >= a.n) {
                break;
            }
            for (uint j = 0; j < k; ++j) {
                n_all += sids[t * k + j] == ide;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    cnt[ide] = n_all;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tpitg == 0) {
        uint base = 0;
        uint tile = 0;
        for (uint e = 0; e < a.n_expert; ++e) {
            const uint n = cnt[e];
            bas[e] = base;
            for (uint i = 0; i < n; i += 32) {
                tiles[1 + 3 * tile] = e;
                tiles[1 + 3 * tile + 1] = base + i;
                tiles[1 + 3 * tile + 2] = min(n - i, 32u);
                tile++;
            }
            base += n + 1;
        }
        tiles[0] = tile;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ient = bas[ide];
    for (uint i21 = 0; i21 < a.n; i21 += ntg) {
        if (i21 + tpitg < a.n) {
            device const float * r = route + (uint64_t) (i21 + tpitg) * 2 * k;
            for (uint j = 0; j < k; ++j) {
                sids[tpitg * k + j] = (ushort) (uint) r[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort t = 0; t < ntg; t++) {
            if (i21 + t >= a.n) {
                break;
            }
            short first = -1;
            short n_ents = 0;
            for (uint j = 0; j < k; ++j) {
                const short hit = sids[t * k + j] == ide;
                first = hit && first < 0 ? (short) j : first;
                n_ents += hit;
            }
            // Every thread stores, matched or not (the entry after the
            // expert's list takes the stores of unmatched tokens).
            ids[ient] = first >= 0 ? (int) ((i21 + t) * k + first) : 0;
            if (n_ents > 1) {
                short m = 1;
                for (uint j = 0; j < k; ++j) {
                    if (sids[t * k + j] == ide && (short) j != first) {
                        ids[ient + m] = (int) ((i21 + t) * k + j);
                        m += 1;
                    }
                }
            }
            ient += n_ents;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// `MatmulIdArgs` in `metal/kernels.rs`: experts of `rows` rows of `cols`
// values, rows `row_bytes` and experts `expert_bytes` apart; slot `s`
// takes input row `s` with `per_slot`, else row `s / k`, and writes output
// row `s`.
struct MatmulIdArgs {
    uint cols;
    uint rows;
    uint row_bytes;
    uint expert_bytes;
    uint k;
    uint per_slot;
    uint mode;
};

template <typename S0, typename S1, typename block_q, short nl,
          void (*dequantize_func)(device const block_q *, short, thread typename value_type_4x4<S0>::type &),
          typename T1>
void matmul_id_impl(
        constant MatmulIdArgs & args,
        device const char * src0,
        device const char * src0_up,
        device const char * src1,
        device const uint * tpe,
        device const int * ids,
        device float * dst,
        threadgroup char * shmem,
        uint3 tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    using S0_4x4 = typename value_type_4x4<S0>::type;
    using S0_8x8 = typename value_type_8x8<S0>::type;
    using S1_2x4 = typename value_type_2x4<S1>::type;
    using S1_8x8 = typename value_type_8x8<S1>::type;
    using T1_2x4 = typename value_type_2x4<T1>::type;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + NR0*NK*sizeof(S0));

    const int tile = tgpig.x;
    if (tile >= (int) tpe[0]) {
        return;
    }
    const bool is_swiglu = args.mode == MATVEC_SWIGLU;
    const int r0 = is_swiglu ? tgpig.y * (NR0 / 2) : tgpig.y * NR0;
    const int im = tpe[1 + 3*tile];
    const int idb = tpe[1 + 3*tile + 1];
    const short nr1 = (short) min(tpe[1 + 3*tile + 2], (uint) NR1);

    const int ne0 = args.rows;
    const int ne00 = args.cols;
    const short row_tile = is_swiglu ? (NR0 / 2) : NR0;
    const short nr0 = (ne0 - r0 < row_tile) ? (ne0 - r0) : row_tile;
    const short row_id = (short) tiitg / NL0;
    const bool is_up = is_swiglu && row_id >= (NR0 / 2);
    const short raw_row = is_up ? (row_id - NR0 / 2) : row_id;
    const short lr0 = raw_row < nr0 ? raw_row : nr0 - 1;
    const short il0 = (tiitg % NL0);
    const short offset1 = il0/nl;
    const short iy = 8*(tiitg % NL1);
    constexpr short NR1H = NR1/2;

    S0_8x8 ma[4];
    S1_8x8 mb[2];
    simdgroup_float8x8 mc[8];

    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;
    short il = il0;
    const int id = ids[idb + lr1];
    const int in_row = args.per_slot != 0 ? id : id / (int) args.k;

    device const char * w_src = is_up ? src0_up : src0;
    device const block_q * x = (device const block_q *)(w_src + (uint64_t) im*args.expert_bytes + (uint64_t) args.row_bytes*(r0 + lr0)) + offset1;
    device const T1 * y = (device const T1 *)(src1 + (uint64_t) 4*ne00*in_row + 4*iy);

    const bool has_hi = nr1 > NR1H;
    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += NK) {
        S0_4x4 temp_a;
        dequantize_func(x, il, temp_a);
        const S1_2x4 temp_b = (S1_2x4)(*((device T1_2x4 *) y));

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = temp_b;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (has_hi) {
            threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
            threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));
            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }
                lsma += 8*64;
                lsmb += 4*64;
            }
        } else {
            // Up to 16 slots: all 4 simdgroups split the 64 rows as 16 rows
            // each instead of leaving simdgroups 2 and 3 idle.
            threadgroup const S0 * lsma = (sa + 2*64*sgitg);
            threadgroup const S1 * lsmb = sb;
            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 4; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/2], ma[i%2], mc[i]);
                }
                lsma += 8*64;
                lsmb += 4*64;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (has_hi) {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 16*sgitg;
        for (short i = 0; i < 4; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%2) + 8*NR0*(i/2), NR0, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_swiglu) {
        constexpr int H = NR0 / 2;
        for (short j = sgitg; j < nr1; j += 4) {
            const int slot = ids[idb + j];
            device float * D = dst + r0 + (uint64_t) slot * ne0;
            threadgroup float * C = ((threadgroup float *) shmem) + j * NR0;
            if (tiisg < (ushort) nr0) {
                const float g = C[tiisg];
                const float u = C[H + tiisg];
                D[tiisg] = silu(g) * u;
            }
        }
    } else {
        for (short j = sgitg; j < nr1; j += 4) {
            const int slot = ids[idb + j];
            device float  * D  = dst + r0 + (uint64_t) slot*ne0;
            device float4 * D4 = (device float4 *) D;
            threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
            threadgroup float4 * C4 = (threadgroup float4 *) C;
            int i = tiisg;
            for (; i < nr0/4; i += 32) {
                *(D4 + i) = *(C4 + i);
            }
            i = (4*(nr0/4)) + tiisg;
            for (; i < nr0; i += 32) {
                *(D + i) = *(C + i);
            }
        }
    }
}

#define MATMUL_ID_KERNEL(name, S, block_q, nl, deq) \
kernel void name( \
        constant MatmulIdArgs & args [[buffer(0)]], \
        device const char * w [[buffer(1)]], \
        device const char * x [[buffer(2)]], \
        device const uint * tiles [[buffer(3)]], \
        device const int * ids [[buffer(4)]], \
        device float * dst [[buffer(5)]], \
        device const char * w2 [[buffer(6)]], \
        threadgroup char * shmem [[threadgroup(0)]], \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        ushort tiitg [[thread_index_in_threadgroup]], \
        ushort tiisg [[thread_index_in_simdgroup]], \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    matmul_id_impl<S, S, block_q, nl, deq<typename value_type_4x4<S>::type>, float>(args, w, w2, x, tiles, ids, dst, shmem, tgpig, tiitg, tiisg, sgitg); \
}

MATMUL_ID_KERNEL(matmul_f32_id,     half,   float4x4,      1,     dequantize_f32)
MATMUL_ID_KERNEL(matmul_f16_id,     half,   half4x4,       1,     dequantize_f16)
MATMUL_ID_KERNEL(matmul_bf16_id,    bfloat, bfloat4x4,     1,     dequantize_bf16)
MATMUL_ID_KERNEL(matmul_q2_0_id,    half,   block_q2_0,    4,     dequantize_q2_0)
MATMUL_ID_KERNEL(matmul_q4_0_id,    half,   block_q4_0,    2,     dequantize_q4_0)
MATMUL_ID_KERNEL(matmul_q4_1_id,    half,   block_q4_1,    2,     dequantize_q4_1)
MATMUL_ID_KERNEL(matmul_q5_0_id,    half,   block_q5_0,    2,     dequantize_q5_0)
MATMUL_ID_KERNEL(matmul_q5_1_id,    half,   block_q5_1,    2,     dequantize_q5_1)
MATMUL_ID_KERNEL(matmul_q8_0_id,    half,   block_q8_0,    2,     dequantize_q8_0)
MATMUL_ID_KERNEL(matmul_mxfp4_id,   half,   block_mxfp4,   2,     dequantize_mxfp4)
MATMUL_ID_KERNEL(matmul_q2_K_id,    half,   block_q2_K,    QK_NL, dequantize_q2_K)
MATMUL_ID_KERNEL(matmul_q3_K_id,    half,   block_q3_K,    QK_NL, dequantize_q3_K)
MATMUL_ID_KERNEL(matmul_q4_K_id,    half,   block_q4_K,    QK_NL, dequantize_q4_K)
MATMUL_ID_KERNEL(matmul_q5_K_id,    half,   block_q5_K,    QK_NL, dequantize_q5_K)
MATMUL_ID_KERNEL(matmul_q6_K_id,    half,   block_q6_K,    QK_NL, dequantize_q6_K)
MATMUL_ID_KERNEL(matmul_iq2_xxs_id, half,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs)
MATMUL_ID_KERNEL(matmul_iq2_xs_id,  half,   block_iq2_xs,  QK_NL, dequantize_iq2_xs)
MATMUL_ID_KERNEL(matmul_iq2_s_id,   half,   block_iq2_s,   QK_NL, dequantize_iq2_s)
MATMUL_ID_KERNEL(matmul_iq3_xxs_id, half,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs)
MATMUL_ID_KERNEL(matmul_iq3_s_id,   half,   block_iq3_s,   QK_NL, dequantize_iq3_s)
MATMUL_ID_KERNEL(matmul_iq1_s_id,   half,   block_iq1_s,   QK_NL, dequantize_iq1_s)
MATMUL_ID_KERNEL(matmul_iq1_m_id,   half,   block_iq1_m,   QK_NL, dequantize_iq1_m)
MATMUL_ID_KERNEL(matmul_iq4_nl_id,  half,   block_iq4_nl,  2,     dequantize_iq4_nl)
MATMUL_ID_KERNEL(matmul_iq4_xs_id,  half,   block_iq4_xs,  QK_NL, dequantize_iq4_xs)
