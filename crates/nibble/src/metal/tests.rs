//! The Metal kernels against the CPU reference, on random inputs.
//!
//! Each test skips (with a note on stderr) where no Metal device exists.
//! Whole-model comparisons with the CPU backend are in `tests/metal.rs`.

use nibble_gguf::GgmlType;

use super::device::{Buffer, Device};
use super::kernels::{EmbedArgs, Pipelines};
use super::weights::Weights;
use super::{MetalBackend, Output, Size, encode_matvec, load, store};
use crate::backend::{Backend, ModelData, Params, QkNorm};
use crate::cpu::{CpuBackend, Weight};
use crate::model::{Architecture, ModelConfig};
use crate::quant::test_data::{k_row, values};
use crate::quant::{self, Q8_0Inverse};

/// Every weight type the backend supports.
const TYPES: [GgmlType; 9] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q3K,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Iq4Xs,
];

fn gpu() -> Option<(Device, Pipelines)> {
    let device = match Device::system_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: {e}");
            return None;
        }
    };
    let pipelines = Pipelines::new(&device).expect("the shaders compile");
    Some((device, pipelines))
}

/// A row of `cols` random weights of type `ty`.
fn weight_row(ty: GgmlType, cols: usize, seed: u32) -> Vec<u8> {
    let x = values(cols, seed);
    match ty {
        GgmlType::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlType::F16 => x
            .iter()
            .flat_map(|&v| quant::f32_to_f16(v).to_le_bytes())
            .collect(),
        GgmlType::Bf16 => x
            .iter()
            .flat_map(|&v| quant::f32_to_bf16(v).to_le_bytes())
            .collect(),
        GgmlType::Q8_0 => {
            let mut out = vec![0; cols / quant::QK8_0 * quant::Q8_0_BLOCK_BYTES];
            quant::quantize_row_q8_0(&x, &mut out, Q8_0Inverse::NATIVE);
            out
        }
        _ => k_row(ty, cols / quant::QK_K, seed),
    }
}

fn block_size(ty: GgmlType) -> usize {
    ty.layout().unwrap().block_size as usize
}

/// A `rows × cols` matrix of type `ty` and its rows decoded to f32.
fn matrix(ty: GgmlType, rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Weight, Vec<Vec<f32>>) {
    let mut bytes = Vec::new();
    let mut decoded = Vec::new();
    for r in 0..rows {
        let row = weight_row(ty, cols, seed + r as u32);
        let mut f = vec![0.0; cols];
        quant::dequantize_row(ty, &row, &mut f);
        decoded.push(f);
        bytes.extend_from_slice(&row);
    }
    let weight = Weight {
        ty,
        cols,
        rows,
        offset: 0,
        row_bytes: bytes.len() / rows,
    };
    (bytes, weight, decoded)
}

fn floats(device: &Device, data: &[f32]) -> Buffer {
    let mut b = device.buffer(4 * data.len()).unwrap();
    // SAFETY: the buffer is new.
    unsafe { store(&mut b, data) };
    b
}

fn read(b: &Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    // SAFETY: called after the commands that write `b` completed.
    unsafe { load(b, &mut out) };
    out
}

/// `w · x` in f64, and the sum of the magnitudes of its terms.
fn dot(w: &[f32], x: &[f32]) -> (f64, f64) {
    w.iter().zip(x).fold((0.0, 0.0), |(s, m), (&a, &b)| {
        (
            s + f64::from(a) * f64::from(b),
            m + f64::from((a * b).abs()),
        )
    })
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Every matrix-vector kernel in every mode against the products of the
/// decoded weights: stored, added and SwiGLU outputs.
/// Every weight type the CPU runs has Metal kernels, so `Auto` never picks
/// Metal for a model it cannot run.
#[test]
fn every_supported_type_has_kernels() {
    assert_eq!(TYPES, crate::quant::SUPPORTED);
}

/// A product over several inputs gives each input, bit for bit, the output
/// of a product over that input alone: for every weight type and mode, and
/// numbers of inputs that use each kernel width and group them.
#[test]
fn batched_matvec_equals_one_input_at_a_time() {
    let Some((device, pipelines)) = gpu() else {
        return;
    };
    for ty in TYPES {
        for (rows, cols) in [(65, 1024), (130, 3584), (9, 4160)] {
            if cols % block_size(ty) != 0 {
                continue;
            }
            let (bytes, w, _) = matrix(ty, rows, cols, 300);
            let (up_bytes, mut up, _) = matrix(ty, rows, cols, 700);
            up.offset = bytes.len();
            let largest = bytes.len().max(up_bytes.len());
            let all = [bytes, up_bytes].concat();
            let wbuf = Weights::build(&device, &all, false, largest, usize::MAX).unwrap();
            for output in [Output::Store, Output::Add, Output::SwiGlu(&up)] {
                let run = |x: &[f32], initial: &[f32], ncols: usize| {
                    let (xbuf, out) = (floats(&device, x), floats(&device, initial));
                    let mut commands = device.commands().unwrap();
                    encode_matvec(
                        &mut commands,
                        &pipelines,
                        &wbuf,
                        &w,
                        (&xbuf, 0),
                        (&out, 0),
                        output,
                        ncols,
                    )
                    .unwrap();
                    commands.commit();
                    commands.wait().unwrap();
                    read(&out, initial.len())
                };
                for ncols in [2, 3, 4, 5, 8] {
                    let x = values(ncols * cols, 17 + ncols as u32);
                    let initial = values(ncols * rows, 23);
                    let all = run(&x, &initial, ncols);
                    for c in 0..ncols {
                        let alone = run(
                            &x[c * cols..(c + 1) * cols],
                            &initial[c * rows..(c + 1) * rows],
                            1,
                        );
                        let got = &all[c * rows..(c + 1) * rows];
                        let same = got
                            .iter()
                            .zip(&alone)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                        assert!(same, "{ty} {rows}x{cols} {output:?}: input {c} of {ncols}");
                    }
                }
            }
        }
    }
}

#[test]
fn matvec_matches_cpu_reference() {
    let Some((device, pipelines)) = gpu() else {
        return;
    };
    let shapes = [
        (1, 256),
        (3, 512),
        (65, 1024),
        (130, 3584),
        (5, 100),
        (9, 4160),
    ];
    for ty in TYPES {
        assert!(pipelines.matvec(ty).is_some(), "no matvec kernel for {ty}");
        for (rows, cols) in shapes {
            if cols % block_size(ty) != 0 {
                continue;
            }
            let (bytes, w, decoded) = matrix(ty, rows, cols, 1000);
            let (up_bytes, mut up, up_decoded) = matrix(ty, rows, cols, 5000);
            up.offset = bytes.len();
            let largest = bytes.len().max(up_bytes.len());
            let all = [bytes, up_bytes].concat();
            let page = super::device::page_size();
            let x = values(cols, 7);
            let initial = values(rows, 11);
            let xbuf = floats(&device, &x);
            // Also split into buffers barely longer than a matrix, so the
            // two matrices land in different buffers.
            let split = largest.next_multiple_of(page) + page;
            for (max_len, output) in [usize::MAX, split]
                .into_iter()
                .flat_map(|m| [Output::Store, Output::Add, Output::SwiGlu(&up)].map(|o| (m, o)))
            {
                let wbuf = Weights::build(&device, &all, false, largest, max_len).unwrap();
                if max_len == split && all.len() > split {
                    assert!(wbuf.pieces() > 1);
                }
                let out = floats(&device, &initial);
                let mut commands = device.commands().unwrap();
                encode_matvec(
                    &mut commands,
                    &pipelines,
                    &wbuf,
                    &w,
                    (&xbuf, 0),
                    (&out, 0),
                    output,
                    1,
                )
                .unwrap();
                commands.commit();
                commands.wait().unwrap();
                let got = read(&out, rows);
                for r in 0..rows {
                    let (g, gm) = dot(&decoded[r], &x);
                    let (expected, magnitude) = match output {
                        Output::Store => (g, gm),
                        Output::Add => (f64::from(initial[r]) + g, gm),
                        Output::SwiGlu(_) => {
                            let (u, um) = dot(&up_decoded[r], &x);
                            (silu(g) * u, (gm + um) * (1.0 + u.abs() + g.abs()))
                        }
                    };
                    let err = (f64::from(got[r]) - expected).abs();
                    assert!(
                        err <= 2e-6 * magnitude + 1e-6,
                        "{ty} {rows}x{cols} {output:?} row {r}: got {}, expected {expected}",
                        got[r]
                    );
                }
            }
        }
    }
}

#[test]
fn embed_matches_dequantize() {
    let Some((device, pipelines)) = gpu() else {
        return;
    };
    for ty in TYPES {
        let (pipeline, per_thread) = pipelines
            .embed(ty)
            .unwrap_or_else(|| panic!("no embedding kernel for {ty}"));
        let cols = 1024;
        let (bytes, w, decoded) = matrix(ty, 4, cols, 1000);
        let wbuf = device.buffer_with(&bytes).unwrap();
        // The row from the arguments, and from a token slot.
        for (row, from_slot) in [(0, false), (3, false), (2, true)] {
            let out = device.buffer(4 * cols).unwrap();
            let slot = device.buffer_with(&(row as u32).to_le_bytes()).unwrap();
            let mut commands = device.commands().unwrap();
            commands.set_pipeline(pipeline);
            commands.set_bytes(
                0,
                &EmbedArgs {
                    cols: cols as u32,
                    row: if from_slot { 0 } else { row as u32 },
                    row_bytes: w.row_bytes as u32,
                    from_slot: u32::from(from_slot),
                },
            );
            commands.set_buffer(1, &wbuf, 0);
            commands.set_buffer(2, &out, 0);
            commands.set_buffer(3, &slot, 0);
            let threads = cols.div_ceil(per_thread);
            commands.dispatch(Size::new(threads.div_ceil(64), 1, 1), Size::new(64, 1, 1));
            commands.commit();
            commands.wait().unwrap();
            let got = read(&out, cols);
            for (i, (&g, &e)) in got.iter().zip(&decoded[row]).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-6 * e.abs().max(1e-3),
                    "{ty} row {row} value {i}: got {g}, expected {e}"
                );
            }
        }
    }
}

/// The attention kernels against the CPU backend over a sequence of
/// positions: group sizes 1 to 8, heads of 64 to 512 values, gated and
/// ungated, with the automatic chunking and with chunks forced to 64 and
/// 256 positions (one chunk, and the last chunk partly full).
#[test]
fn attention_matches_the_cpu_backend() {
    if gpu().is_none() {
        return;
    }
    let n_pos = 300;
    // (query heads, key heads, head size, rotated dims, gated, Q/K norms)
    let shapes = [
        (16, 8, 128, 128, false, true),
        (8, 2, 256, 64, true, true),
        (3, 3, 64, 64, false, false),
        (16, 2, 64, 32, false, true),
        (8, 2, 512, 128, false, true),
    ];
    for (n_head, n_head_kv, head_dim, rope_dims, gated, norms) in shapes {
        let config = attention_config(n_head, n_head_kv, head_dim, rope_dims);
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
            weights: norms.then_some((q_norm, k_norm)),
            eps: 1e-6,
            gated,
        };
        let q_len = n_head * head_dim * if gated { 2 } else { 1 };
        let kv_len = n_head_kv * head_dim;
        let inputs: Vec<[Vec<f32>; 3]> = (0..n_pos as u32)
            .map(|p| {
                [
                    values(q_len, 3 * p + 10),
                    values(kv_len, 3 * p + 11),
                    values(kv_len, 3 * p + 12),
                ]
            })
            .collect();

        let mut cpu = CpuBackend::new(data, n_pos, 1);
        let mut expected = Vec::new();
        let mut cache = cpu.kv_cache(n_pos).unwrap();
        let (mut q, mut k, v) = (vec![0.0; q_len], vec![0.0; kv_len], vec![0.0; kv_len]);
        let mut out = vec![0.0; n_head * head_dim];
        let mut v = v;
        for (pos, [qi, ki, vi]) in inputs.iter().enumerate() {
            q.copy_from_slice(qi);
            k.copy_from_slice(ki);
            v.copy_from_slice(vi);
            cpu.begin(pos, 1).unwrap();
            cpu.attention(&mut q, &mut k, &v, norm, &mut cache, &mut out);
            cpu.end().unwrap();
            expected.push(out.clone());
        }

        for chunk in [None, Some(64), Some(256)] {
            let mut metal = MetalBackend::new(data, n_pos).unwrap();
            if let Some(chunk) = chunk
                && !metal.force_attention_chunk(chunk)
            {
                continue;
            }
            let mut cache = metal.kv_cache(n_pos).unwrap();
            let mut q = metal.vector(q_len).unwrap();
            let mut k = metal.vector(kv_len).unwrap();
            let mut v = metal.vector(kv_len).unwrap();
            let mut out = metal.vector(n_head * head_dim).unwrap();
            let mut got = vec![0.0; n_head * head_dim];
            for (pos, [qi, ki, vi]) in inputs.iter().enumerate() {
                metal.write(&mut q, qi).unwrap();
                metal.write(&mut k, ki).unwrap();
                metal.write(&mut v, vi).unwrap();
                metal.begin(pos, 1).unwrap();
                metal.attention(&mut q, &mut k, &v, norm, &mut cache, &mut out);
                metal.end().unwrap();
                metal.read(&out, &mut got).unwrap();
                let what = format!(
                    "{n_head}/{n_head_kv} heads of {head_dim}, chunk {chunk:?}, position {pos}"
                );
                for (i, (&g, &e)) in got.iter().zip(&expected[pos]).enumerate() {
                    // The caches are f16: a key that rounds the other way
                    // after RoPE moves the scores slightly.
                    assert!(
                        (g - e).abs() <= 2e-3 * (1.0 + e.abs()),
                        "{what}, value {i}: {g} vs {e}"
                    );
                }
            }
        }
    }
}

/// A model configuration with only attention shapes set.
fn attention_config(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    rope_dims: usize,
) -> ModelConfig {
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

/// A backend refuses, when it starts, attention its kernels do not cover,
/// which makes `Auto` fall back to the CPU.
#[test]
fn uncovered_attention_shapes_fail_when_the_backend_starts() {
    if gpu().is_none() {
        return;
    }
    let bytes = [0u8; 16];
    // 16 query heads per key head; a head size that is not a multiple of 4;
    // 8 heads per key head of 512 values.
    for config in [
        attention_config(32, 2, 128, 128),
        attention_config(4, 4, 6, 6),
        attention_config(16, 2, 512, 128),
    ] {
        let data = ModelData {
            bytes: &bytes,
            mapped: false,
            params: &[],
            config: &config,
            largest_tensor: 0,
        };
        assert!(MetalBackend::new(data, 16).is_err(), "{config:?}");
    }
    let config = attention_config(16, 8, 128, 128);
    let data = ModelData {
        bytes: &bytes,
        mapped: false,
        params: &[],
        config: &config,
        largest_tensor: 0,
    };
    assert!(MetalBackend::new(data, 16).is_ok());
}

/// The kernels for several inputs that are built all reproduce the one-input
/// kernels on this device, so the check when the backend starts keeps them.
#[test]
fn every_wide_kernel_passes_the_startup_check() {
    let Some((device, mut pipelines)) = gpu() else {
        return;
    };
    let before: Vec<(GgmlType, Vec<usize>)> = TYPES
        .iter()
        .map(|&ty| (ty, pipelines.matvec(ty).unwrap().widths()))
        .collect();
    super::check_wide_matvecs(&device, &mut pipelines).unwrap();
    for (ty, widths) in before {
        assert_eq!(pipelines.matvec(ty).unwrap().widths(), widths, "{ty}");
    }
}
