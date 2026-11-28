//! The CUDA prompt-batch kernels (the GEMMs, flash attention and the
//! DeltaNet tail) against the one-token kernels, on random inputs.
//!
//! Batches of more than eight tokens round differently from one token at a
//! time (as llama.cpp's batched products do), so these compare within the
//! tolerance each test documents. Each test skips (with a note on stderr)
//! where no CUDA device with compute capability 8.0 or newer exists.

use nibble_gguf::GgmlType;

use super::CudaBackend;
use crate::backend::{
    Backend, DeltaNetDims, DeltaNetParams, DeltaNetState, ModelData, Params, QkNorm,
};
use crate::cpu::Weight;
use crate::model::{Architecture, ModelConfig};
use crate::quant::test_data::{k_row, values};
use crate::quant::{self, Q8_0Inverse};

/// Every type the GEMM covers.
const GEMM_TYPES: [GgmlType; 10] = [
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q2K,
    GgmlType::Q3K,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
];

/// A model configuration with only attention shapes set.
fn config(n_head: usize, n_head_kv: usize, head_dim: usize, rope_dims: usize) -> ModelConfig {
    ModelConfig {
        architecture: Architecture::Qwen3,
        name: None,
        n_vocab: 1,
        n_embd: 1,
        n_layer: 1,
        n_head,
        n_head_kv,
        head_dim,
        n_ff: 1,
        n_ctx_train: 1,
        rope_base: 10_000.0,
        rope_dims,
        rms_eps: 1e-6,
        tied_embeddings: true,
        weight_type: GgmlType::F32,
    }
}

/// A backend for `data`, or `None` without a device that runs the prompt
/// kernels.
fn backend(data: ModelData<'_>, n_ctx: usize) -> Option<CudaBackend<'_>> {
    match CudaBackend::new(data, n_ctx, Default::default()) {
        Ok(b) if b.cc >= (8, 0) => Some(b),
        Ok(b) => {
            eprintln!(
                "skipping: compute capability {:?} runs no prompt kernels",
                b.cc
            );
            None
        }
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

/// `rows` random rows of `cols` weights of type `ty`, appended to `bytes`.
fn push_matrix(bytes: &mut Vec<u8>, ty: GgmlType, rows: usize, cols: usize, seed: u32) -> Weight {
    let offset = bytes.len();
    for r in 0..rows {
        let seed = seed + r as u32;
        if ty == GgmlType::Q8_0 {
            let mut row = vec![0; cols / quant::QK8_0 * quant::Q8_0_BLOCK_BYTES];
            quant::quantize_row_q8_0(&values(cols, seed), &mut row, Q8_0Inverse::NATIVE);
            bytes.extend_from_slice(&row);
        } else {
            let block = ty.layout().unwrap().block_size as usize;
            bytes.extend_from_slice(&k_row(ty, cols / block, seed));
        }
    }
    // Weights start on 32-byte boundaries, as in a GGUF file.
    let row_bytes = (bytes.len() - offset) / rows;
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    Weight {
        ty,
        cols,
        rows,
        offset,
        row_bytes,
    }
}

/// The largest difference between `got` and `want` relative to the largest
/// value of `want`.
fn relative_error(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = got
        .iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    diff / scale.max(f32::MIN_POSITIVE)
}

/// Runs `op` on `n` tokens of `cols` inputs from `x` in one batch, and one
/// token at a time, returning both outputs (`out_len` values per token).
fn batch_and_steps(
    b: &mut CudaBackend<'_>,
    x: &[f32],
    cols: usize,
    out_len: usize,
    n: usize,
    op: &mut dyn FnMut(&mut CudaBackend<'_>, &super::CudaVector, &mut super::CudaVector),
) -> (Vec<f32>, Vec<f32>) {
    let mut xv = b.vector(n * cols).unwrap();
    let mut out = b.vector(n * out_len).unwrap();
    b.write(&mut xv, &x[..n * cols]).unwrap();
    b.begin(0, n).unwrap();
    op(b, &xv, &mut out);
    b.end().unwrap();
    let mut batch = vec![0.0; n * out_len];
    b.read(&out, &mut batch).unwrap();

    let mut xv = b.vector(cols).unwrap();
    let mut out = b.vector(out_len).unwrap();
    let mut steps = vec![0.0; n * out_len];
    for t in 0..n {
        b.write(&mut xv, &x[t * cols..(t + 1) * cols]).unwrap();
        b.begin(t, 1).unwrap();
        op(b, &xv, &mut out);
        b.end().unwrap();
        b.read(&out, &mut steps[t * out_len..(t + 1) * out_len])
            .unwrap();
    }
    (batch, steps)
}

/// The GEMM (both its tiling and stream-K, every token tile, partial row
/// tiles and, for blocks of 32, a partial last stage) gives each token what
/// the one-token kernel does, up to rounding: the activations' scales are
/// kept in f32 rather than f16, and the sums are added in another order.
#[test]
fn gemm_matches_one_token_products() {
    let config = config(1, 1, 64, 64);
    for ty in GEMM_TYPES {
        let block = ty.layout().unwrap().block_size as usize;
        let narrow = if block == 32 { 544 } else { 512 };
        // 300 rows take stream-K; 6144 fill the waves of a 48-SM GPU.
        for (rows, cols) in [(300, narrow), (6144, 256)] {
            let mut bytes = Vec::new();
            let w = push_matrix(&mut bytes, ty, rows, cols, 7);
            let data = ModelData {
                bytes: &bytes,
                mapped: false,
                params: &[],
                config: &config,
                largest_tensor: bytes.len(),
            };
            let Some(mut b) = backend(data, 16) else {
                return;
            };
            let x = values(130 * cols, 3);
            for n in [9, 40, 130] {
                let (batch, steps) =
                    batch_and_steps(&mut b, &x, cols, rows, n, &mut |b, x, out| {
                        b.matvecs(x, [(&w, out)]);
                    });
                let err = relative_error(&batch, &steps);
                assert!(err <= 2e-3, "{ty} {rows}x{cols}, {n} tokens: error {err}");
            }
        }
    }
}

/// The fused gate/up GEMM gives each token the one-token SwiGLU, as above.
#[test]
fn fused_swiglu_gemm_matches_one_token_products() {
    let config = config(1, 1, 64, 64);
    for ty in [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
        let (rows, cols) = (200, 512);
        let mut bytes = Vec::new();
        let gate = push_matrix(&mut bytes, ty, rows, cols, 11);
        let up = push_matrix(&mut bytes, ty, rows, cols, 5000);
        let data = ModelData {
            bytes: &bytes,
            mapped: false,
            params: &[],
            config: &config,
            largest_tensor: bytes.len(),
        };
        let Some(mut b) = backend(data, 16) else {
            return;
        };
        let x = values(130 * cols, 4);
        let mut up_out = b.vector(130 * rows).unwrap();
        for n in [9, 40, 130] {
            let (batch, steps) = batch_and_steps(&mut b, &x, cols, rows, n, &mut |b, x, out| {
                b.swiglu(&gate, &up, x, out, &mut up_out);
            });
            let err = relative_error(&batch, &steps);
            assert!(err <= 4e-3, "{ty}, {n} tokens: error {err}");
        }
    }
}

/// Flash attention over an F16 cache, in batches that leave partial query
/// blocks and follow earlier ones in the cache, gives the one-token result
/// up to the rounding of the queries and probabilities to f16.
#[test]
fn flash_attention_matches_one_token_attention() {
    let n_pos = 304;
    // (query heads, key heads, head size, rotated dims, gated)
    for (n_head, n_head_kv, head_dim, rope_dims, gated) in [
        (16, 8, 128, 128, false),
        (8, 2, 256, 64, true),
        (4, 2, 64, 64, false),
    ] {
        let config = config(n_head, n_head_kv, head_dim, rope_dims);
        let mut params = Params::default();
        let weight = |seed| -> Vec<f32> {
            values(head_dim, seed)
                .iter()
                .map(|v| 1.0 + 0.5 * v)
                .collect()
        };
        let q_norm = params.push(&weight(1));
        let k_norm = params.push(&weight(2));
        let params = params.into_vec();
        let bytes = [0u8; 16];
        let data = ModelData {
            bytes: &bytes,
            mapped: false,
            params: &params,
            config: &config,
            largest_tensor: 0,
        };
        let norm = QkNorm {
            weights: Some((q_norm, k_norm)),
            eps: 1e-6,
            gated,
            multi_rope: false,
        };
        let q_len = n_head * head_dim * if gated { 2 } else { 1 };
        let kv_len = n_head_kv * head_dim;
        let out_len = n_head * head_dim;
        let q = values(n_pos * q_len, 10);
        let k = values(n_pos * kv_len, 11);
        let v = values(n_pos * kv_len, 12);
        let run = |sizes: &[usize]| -> Option<Vec<f32>> {
            let mut b = backend(data, n_pos)?;
            let mut cache = b.kv_cache(n_pos).unwrap();
            let mut outputs = Vec::new();
            let mut at = 0;
            for &n in sizes {
                let mut qv = b.vector(n * q_len).unwrap();
                let mut kv = b.vector(n * kv_len).unwrap();
                let mut vv = b.vector(n * kv_len).unwrap();
                let mut out = b.vector(n * out_len).unwrap();
                b.write(&mut qv, &q[at * q_len..(at + n) * q_len]).unwrap();
                b.write(&mut kv, &k[at * kv_len..(at + n) * kv_len])
                    .unwrap();
                b.write(&mut vv, &v[at * kv_len..(at + n) * kv_len])
                    .unwrap();
                b.begin(at, n).unwrap();
                b.attention(&mut qv, &mut kv, &vv, norm, &mut cache, &mut out);
                b.end().unwrap();
                let mut got = vec![0.0; n * out_len];
                b.read(&out, &mut got).unwrap();
                outputs.extend(got);
                at += n;
            }
            Some(outputs)
        };
        let Some(want) = run(&[1; 304]) else {
            return;
        };
        let got = run(&[40, 17, 100, 147]).unwrap();
        for (i, (&g, &e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() <= 1e-2 * (1.0 + e.abs()),
                "{n_head}/{n_head_kv} heads of {head_dim}, position {}, value {}: {g} vs {e}",
                i / out_len,
                i % out_len
            );
        }
    }
}

/// A long batch that leaves its last tokens pending gives every token the
/// output, and leaves the state, that the batch before them and a batch of
/// their own do, bit for bit, also after the next batch applies them.
#[test]
fn delta_net_tail_equals_a_batch_of_its_own() {
    for (n_k_heads, n_v_heads, d_k) in [(2, 4, 128), (2, 2, 64), (1, 2, 256)] {
        let d = DeltaNetDims {
            conv_kernel: 4,
            n_k_heads,
            n_v_heads,
            d_k,
            d_v: d_k,
        };
        let mut params = Params::default();
        let conv_taps: Vec<f32> = values(d.conv_kernel * d.conv_dim(), 5)
            .iter()
            .map(|v| 0.5 * v)
            .collect();
        let p = DeltaNetParams {
            conv: params.push(&conv_taps),
            dt_bias: params.push(&values(n_v_heads, 6)),
            a: params.push(
                &values(n_v_heads, 7)
                    .iter()
                    .map(|v| -0.5 - v.abs())
                    .collect::<Vec<_>>(),
            ),
            norm: params.push(
                &values(d.d_v, 8)
                    .iter()
                    .map(|v| 1.0 + 0.5 * v)
                    .collect::<Vec<_>>(),
            ),
        };
        let params = params.into_vec();
        let config = config(2, 1, 64, 64);
        let bytes = [0u8; 16];
        let data = ModelData {
            bytes: &bytes,
            mapped: false,
            params: &params,
            config: &config,
            largest_tensor: 0,
        };
        let n = 48;
        let qkv = values(n * d.conv_dim(), 20);
        let beta = values(n * n_v_heads, 21);
        let alpha = values(n * n_v_heads, 22);
        let z = values(n * d.value_dim(), 23);

        // Evaluates the tokens in batches of (size, pending tail), returning
        // every token's output, the state after the first batch (pending
        // tokens, committed state) and the final committed state.
        type Run = (Vec<f32>, (usize, Vec<f32>), Vec<f32>);
        let run = |batches: &[(usize, usize)]| -> Option<Run> {
            let mut b = backend(data, 64)?;
            assert!(b.delta_net_tail(d), "heads of {d_k}");
            let mut state = DeltaNetState {
                conv: b.vector(d.conv_len()).unwrap(),
                oldest: 0,
                ssm: b.vector(d.state_len()).unwrap(),
                pending: 0,
                keep: 0,
                pending_ba: b.vector(d.pending_len()).unwrap(),
                pending_row: 0,
            };
            let mut outputs = Vec::new();
            let mut first = None;
            let mut at = 0;
            for &(m, tail) in batches {
                let rows = |v: &[f32], len: usize| v[at * len..(at + m) * len].to_vec();
                let mut x = b.vector(m * d.conv_dim()).unwrap();
                let mut conv = b.vector(m * d.conv_dim()).unwrap();
                let mut zv = b.vector(m * d.value_dim()).unwrap();
                let mut bv = b.vector(m * n_v_heads).unwrap();
                let mut av = b.vector(m * n_v_heads).unwrap();
                let mut out = b.vector(m * d.value_dim()).unwrap();
                b.write(&mut x, &rows(&qkv, d.conv_dim())).unwrap();
                b.write(&mut zv, &rows(&z, d.value_dim())).unwrap();
                b.write(&mut bv, &rows(&beta, n_v_heads)).unwrap();
                b.write(&mut av, &rows(&alpha, n_v_heads)).unwrap();
                b.set_delta_net_tail(tail);
                b.begin(at, m).unwrap();
                b.delta_net(
                    d, p, 1e-6, &mut state, &x, &mut conv, &mut zv, &mut bv, &mut av, &mut out,
                );
                b.end().unwrap();
                b.set_delta_net_tail(0);
                let mut got = vec![0.0; m * d.value_dim()];
                b.read(&out, &mut got).unwrap();
                outputs.extend(got);
                at += m;
                if at == 40 {
                    let mut ssm = vec![0.0; d.state_len()];
                    b.read(&state.ssm, &mut ssm).unwrap();
                    first = Some((state.keep, ssm));
                }
            }
            let mut ssm = vec![0.0; d.state_len()];
            b.read(&state.ssm, &mut ssm).unwrap();
            Some((outputs, first.unwrap(), ssm))
        };
        let Some(want) = run(&[(36, 0), (4, 0), (1, 0), (7, 0)]) else {
            return;
        };
        let got = run(&[(40, 4), (1, 0), (7, 0)]).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let what = format!("{n_k_heads}/{n_v_heads} heads of {d_k}");
        assert_eq!(got.1.0, 4, "pending tokens, {what}");
        assert_eq!(want.1.0, 4, "pending tokens, {what}");
        assert_eq!(bits(&got.0), bits(&want.0), "outputs, {what}");
        assert_eq!(bits(&got.1.1), bits(&want.1.1), "committed state, {what}");
        assert_eq!(bits(&got.2), bits(&want.2), "final state, {what}");
    }
}
