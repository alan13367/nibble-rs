//! The Metal kernels against the CPU reference, on random inputs.
//!
//! Each test skips (with a note on stderr) where no Metal device exists.
//! Whole-model comparisons with the CPU backend are in `tests/metal.rs`.

use nibble_gguf::GgmlType;

use super::device::{Buffer, Device};
use super::kernels::{EmbedArgs, Pipelines};
use super::weights::Weights;
use super::{MetalBackend, Output, Size, encode_matmul, encode_matvec, load, store};
use crate::backend::{
    Backend, DeltaNetDims, DeltaNetParams, DeltaNetState, HeadOrder, ModelData, Params, QkNorm,
    Rotation,
};
use crate::cpu::{CpuBackend, Weight};
use crate::model::{Architecture, ModelConfig};
use crate::quant::test_data::{k_row, values};
use crate::quant::{self, Q8_0Inverse};

/// Every weight type the backend supports: all the CPU does.
const TYPES: &[GgmlType] = crate::quant::SUPPORTED;

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
        _ => k_row(ty, cols / block_size(ty), seed),
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
    let Some((_, pipelines)) = gpu() else {
        return;
    };
    for &ty in TYPES {
        assert!(
            pipelines.matvec(ty).is_some(),
            "no matrix-vector kernel for {ty}"
        );
        assert!(
            pipelines.embed(ty).is_some(),
            "no embedding kernel for {ty}"
        );
    }
}

/// A product over several inputs gives each input, bit for bit, the output
/// of a product over that input alone: for every weight type and mode, and
/// numbers of inputs that use each kernel width and group them.
#[test]
fn batched_matvec_equals_one_input_at_a_time() {
    let Some((device, pipelines)) = gpu() else {
        return;
    };
    for &ty in TYPES {
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
                let dummy = device.buffer(4).unwrap();
                let run = |x: &[f32], initial: &[f32], ncols: usize| {
                    let (xbuf, out) = (floats(&device, x), floats(&device, initial));
                    let mut commands = device.commands().unwrap();
                    encode_matvec(
                        &mut commands,
                        &pipelines,
                        &wbuf,
                        &dummy,
                        &dummy,
                        None,
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
    for &ty in TYPES {
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
                let dummy = device.buffer(4).unwrap();
                let mut commands = device.commands().unwrap();
                encode_matvec(
                    &mut commands,
                    &pipelines,
                    &wbuf,
                    &dummy,
                    &dummy,
                    None,
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
                        Output::Add | Output::ResidualNorm { .. } => {
                            (f64::from(initial[r]) + g, gm)
                        }
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
    for &ty in TYPES {
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

/// `rotate` and `unrotate` give the CPU backend's results bit for bit, for
/// blocks of 1 and of 64 to 2048 values, with and without signs and head
/// order.
#[test]
fn rotate_matches_the_cpu_backend() {
    if gpu().is_none() {
        return;
    }
    let config = attention_config(4, 4, 64, 64);
    let n = 3;
    // (block, head size, key heads, repeats): rows of `hd * nk * rep` values.
    let cases = [
        (1, 4, 2, 2),
        (64, 64, 2, 3),
        (128, 32, 4, 2),
        (256, 128, 2, 3),
        (512, 128, 4, 3),
        (1024, 128, 16, 3),
        (2048, 64, 8, 4),
    ];
    for (block, hd, nk, rep) in cases {
        let len = hd * nk * rep;
        let mut params = Params::default();
        let signs: Vec<f32> = values(len, 5)
            .iter()
            .map(|&v| if v < 0.0 { -1.0 } else { 1.0 })
            .collect();
        let signs = params.push(&signs);
        let params = params.into_vec();
        let bytes = [0u8; 16];
        let data = ModelData {
            bytes: &bytes,
            mapped: false,
            params: &params,
            config: &config,
            largest_tensor: 0,
        };
        let x = values(n * len, block as u32);
        let mut cpu = CpuBackend::new(data, 16, 1, Default::default());
        let mut metal = MetalBackend::new(data, 16, Default::default()).unwrap();
        let heads = HeadOrder {
            head_dim: hd,
            n_k: nk,
            rep,
        };
        for (signs, heads) in [
            (None, None),
            (Some(signs), None),
            (Some(signs), Some(heads)),
        ] {
            let r = Rotation {
                len,
                block,
                signs,
                heads,
            };
            let inverse = Rotation { heads: None, ..r };
            metal.check_rotation(r).unwrap();
            cpu.begin(0, n).unwrap();
            let mut want = vec![0.0; n * len];
            cpu.rotate(&x, r, &mut want);
            let mut want_inverse = x.clone();
            cpu.unrotate(&mut want_inverse, inverse);
            let mut mx = metal.vector(n * len).unwrap();
            let mut out = metal.vector(n * len).unwrap();
            metal.write(&mut mx, &x).unwrap();
            metal.begin(0, n).unwrap();
            metal.rotate(&mx, r, &mut out);
            metal.unrotate(&mut mx, inverse);
            metal.end().unwrap();
            let what = format!(
                "block {block}, signs {}, heads {}",
                signs.is_some(),
                heads.is_some()
            );
            let mut got = vec![0.0; n * len];
            metal.read(&out, &mut got).unwrap();
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "rotate, {what}"
            );
            metal.read(&mx, &mut got).unwrap();
            assert!(
                got.iter()
                    .zip(&want_inverse)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "unrotate, {what}"
            );
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
        (24, 4, 256, 64, true, true),
        (16, 2, 256, 64, true, true),
        (5, 1, 128, 128, false, true),
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
            multi_rope: false,
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

        let mut cpu = CpuBackend::new(data, n_pos, 1, Default::default());
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

        // Prompt batches (`attention_prompt`, or one token at a time for the
        // shapes it does not cover), of sizes that leave partial tiles.
        let mut metal = MetalBackend::new(data, n_pos, Default::default()).unwrap();
        let mut cache = metal.kv_cache(n_pos).unwrap();
        let mut start = 0;
        for n in [40, 17, 100, 143] {
            let batch = &inputs[start..start + n];
            let rows = |i: usize| -> Vec<f32> { batch.iter().flat_map(|x| x[i].clone()).collect() };
            let mut q = metal.vector(n * q_len).unwrap();
            let mut k = metal.vector(n * kv_len).unwrap();
            let mut v = metal.vector(n * kv_len).unwrap();
            let mut out = metal.vector(n * n_head * head_dim).unwrap();
            metal.write(&mut q, &rows(0)).unwrap();
            metal.write(&mut k, &rows(1)).unwrap();
            metal.write(&mut v, &rows(2)).unwrap();
            metal.begin(start, n).unwrap();
            metal.attention(&mut q, &mut k, &v, norm, &mut cache, &mut out);
            metal.end().unwrap();
            let mut got = vec![0.0; n * n_head * head_dim];
            metal.read(&out, &mut got).unwrap();
            for (t, row) in got.chunks(n_head * head_dim).enumerate() {
                let pos = start + t;
                for (i, (&g, &e)) in row.iter().zip(&expected[pos]).enumerate() {
                    // The queries are rounded to f16 too.
                    assert!(
                        (g - e).abs() <= 1e-2 * (1.0 + e.abs()),
                        "{n_head}/{n_head_kv} heads of {head_dim}, batch of {n}, position {pos}, value {i}: {g} vs {e}"
                    );
                }
            }
            start += n;
        }

        for chunk in [None, Some(64), Some(256)] {
            let mut metal = MetalBackend::new(data, n_pos, Default::default()).unwrap();
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
        assert!(
            MetalBackend::new(data, 16, Default::default()).is_err(),
            "{config:?}"
        );
    }
    let config = attention_config(16, 8, 128, 128);
    let data = ModelData {
        bytes: &bytes,
        mapped: false,
        params: &[],
        config: &config,
        largest_tensor: 0,
    };
    assert!(MetalBackend::new(data, 16, Default::default()).is_ok());
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

/// The value a matrix-matrix kernel multiplies for `v`: `half`, or
/// `bfloat` for BF16 weights.
fn tile_value(ty: GgmlType, v: f32) -> f64 {
    f64::from(if ty == GgmlType::Bf16 {
        quant::bf16_to_f32(quant::f32_to_bf16(v))
    } else {
        quant::f16_to_f32(quant::f32_to_f16(v))
    })
}

/// The matrix-matrix kernel of every type, stored and added, against f64
/// products of the weights and inputs rounded as its tiles hold them, for
/// whole and partial tiles of rows and inputs.
#[test]
fn matmul_matches_rounded_products() {
    let Some((device, pipelines)) = gpu() else {
        return;
    };
    for &ty in TYPES {
        assert!(pipelines.matmul(ty).is_some(), "no matrix kernel for {ty}");
        for (rows, cols, n) in [(64, 256, 32), (130, 512, 45), (9, 1024, 17)] {
            if cols % block_size(ty) != 0 {
                continue;
            }
            let (bytes, w, decoded) = matrix(ty, rows, cols, 7);
            let weights = Weights::build(&device, &bytes, false, bytes.len(), usize::MAX).unwrap();
            let x: Vec<f32> = (0..n).flat_map(|t| values(cols, 1000 + t as u32)).collect();
            let base: Vec<f32> = values(n * rows, 99);
            for add in [false, true] {
                let xb = floats(&device, &x);
                let out = floats(&device, &base);
                let mut commands = device.commands().unwrap();
                let mode = if add { Output::Add } else { Output::Store };
                encode_matmul(&mut commands, &pipelines, &weights, &w, &xb, &out, mode, n).unwrap();
                commands.commit();
                commands.wait().unwrap();
                let got = read(&out, n * rows);
                for t in 0..n {
                    for r in 0..rows {
                        let (mut want, mut mag) = (0.0, 0.0);
                        for c in 0..cols {
                            let term =
                                tile_value(ty, decoded[r][c]) * tile_value(ty, x[t * cols + c]);
                            want += term;
                            mag += term.abs();
                        }
                        if add {
                            want += f64::from(base[t * rows + r]);
                        }
                        let g = f64::from(got[t * rows + r]);
                        assert!(
                            (g - want).abs() <= 1e-5 * mag + 1e-6,
                            "{ty} {rows}x{cols} n={n} add={add} input {t} row {r}: {g} vs {want}"
                        );
                    }
                }
            }
        }
    }
}

/// Times the Q8_0 matrix kernel on the shapes of a 4B model's products
/// (run with `--ignored --nocapture`).
#[test]
#[ignore = "a timing, not a check"]
fn matmul_timing() {
    let Some((device, pipelines)) = gpu() else {
        return;
    };
    let n = 512;
    for ty in [GgmlType::Q8_0, GgmlType::Q4K] {
        for (rows, cols) in [(2560, 2560), (9216, 2560), (2560, 9216), (3584, 1024)] {
            let row = weight_row(ty, cols, 1);
            let bytes: Vec<u8> = (0..rows).flat_map(|_| row.iter().copied()).collect();
            let w = Weight {
                ty,
                cols,
                rows,
                offset: 0,
                row_bytes: row.len(),
            };
            let weights = Weights::build(&device, &bytes, false, bytes.len(), usize::MAX).unwrap();
            let xb = floats(&device, &values(n * cols, 3));
            let out = floats(&device, &vec![0.0; n * rows]);
            let reps = 20;
            let start = std::time::Instant::now();
            let mut commands = device.commands().unwrap();
            for _ in 0..reps {
                encode_matmul(
                    &mut commands,
                    &pipelines,
                    &weights,
                    &w,
                    &xb,
                    &out,
                    Output::Store,
                    n,
                )
                .unwrap();
            }
            commands.commit();
            commands.wait().unwrap();
            let s = start.elapsed().as_secs_f64() / f64::from(reps);
            let flops = 2.0 * (rows * cols * n) as f64;
            eprintln!(
                "{ty} {rows}x{cols} n={n}: {:.3} ms, {:.2} TFLOP/s",
                s * 1e3,
                flops / s / 1e12
            );
        }
    }
}

/// A prompt batch through a DeltaNet layer (`delta_prep` and `delta_scan`,
/// after a short pending batch that it replays first) gives, bit for bit,
/// the outputs and state of the tokens evaluated one at a time; with heads
/// of 128 (`delta_chunk_*`), the same within rounding.
#[test]
fn delta_net_prompt_batch_equals_steps() {
    if gpu().is_none() {
        return;
    }
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
        let config = attention_config(2, 1, 64, 64);
        let bytes = [0u8; 16];
        let data = ModelData {
            bytes: &bytes,
            mapped: false,
            params: &params,
            config: &config,
            largest_tensor: 0,
        };
        let (short, long) = (3, 100);
        let n = short + long;
        let qkv: Vec<f32> = values(n * d.conv_dim(), 20);
        let beta: Vec<f32> = values(n * n_v_heads, 21);
        let alpha: Vec<f32> = values(n * n_v_heads, 22);
        let z: Vec<f32> = values(n * d.value_dim(), 23);

        // Evaluates `tokens` in batches of the sizes given, returning every
        // token's output and the final state.
        let run = |sizes: &[usize]| -> (Vec<f32>, Vec<f32>) {
            let mut metal = MetalBackend::new(data, 64, Default::default()).unwrap();
            let mut state = DeltaNetState {
                conv: metal.vector(d.conv_len()).unwrap(),
                oldest: 0,
                ssm: metal.vector(d.state_len()).unwrap(),
                pending: 0,
                keep: 0,
                pending_ba: metal.vector(d.pending_len()).unwrap(),
                pending_row: 0,
            };
            let mut outputs = Vec::new();
            let mut at = 0;
            for &m in sizes {
                let rows = |v: &[f32], len: usize| v[at * len..(at + m) * len].to_vec();
                let mut x = metal.vector(m.next_multiple_of(32) * d.conv_dim()).unwrap();
                // Whole chunks of rows, as the chunked delta rule reads.
                let mut conv = metal.vector(m.next_multiple_of(32) * d.conv_dim()).unwrap();
                let mut zv = metal.vector(m * d.value_dim()).unwrap();
                let mut b = metal.vector(m * n_v_heads).unwrap();
                let mut al = metal.vector(m * n_v_heads).unwrap();
                let mut out = metal.vector(m * d.value_dim()).unwrap();
                let mut xs = rows(&qkv, d.conv_dim());
                xs.resize(x.len, 0.0);
                metal.write(&mut x, &xs).unwrap();
                metal.write(&mut zv, &rows(&z, d.value_dim())).unwrap();
                metal.write(&mut b, &rows(&beta, n_v_heads)).unwrap();
                metal.write(&mut al, &rows(&alpha, n_v_heads)).unwrap();
                metal.begin(at, m).unwrap();
                metal.delta_net(
                    d, p, 1e-6, &mut state, &x, &mut conv, &mut zv, &mut b, &mut al, &mut out,
                );
                metal.end().unwrap();
                let mut got = vec![0.0; m * d.value_dim()];
                metal.read(&out, &mut got).unwrap();
                outputs.extend(got);
                at += m;
            }
            // A last step applies a pending batch to the state.
            let mut ssm = vec![0.0; d.state_len()];
            metal.read(&state.ssm, &mut ssm).unwrap();
            (outputs, ssm)
        };
        let (want_out, want_state) = run(&vec![1; n]);
        let (got_out, got_state) = run(&[short, long]);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let what = format!("{n_k_heads}/{n_v_heads} heads of {d_k}");
        if d_k == 128 {
            // The chunked rule (heads of 128): the same in exact arithmetic.
            let err = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
                    .fold(0.0f32, f32::max)
            };
            let (e_out, e_state) = (err(&got_out, &want_out), err(&got_state, &want_state));
            assert!(
                e_out <= 2e-5 && e_state <= 2e-5,
                "{what}: errors {e_out}, {e_state}"
            );
        } else {
            assert_eq!(bits(&got_out), bits(&want_out), "outputs, {what}");
            assert_eq!(bits(&got_state), bits(&want_state), "state, {what}");
        }
    }
}

/// Attention over caches of every type (with llama.cpp's rotations of
/// quantized ones) against the CPU backend, one token at a time and in
/// prompt batches.
#[test]
fn attention_with_cache_types_matches_the_cpu_backend() {
    use crate::backend::{KvCacheType as T, KvTypes};
    if gpu().is_none() {
        return;
    }
    let n_pos = 200;
    let types = [
        (T::Q8_0, T::Q8_0),
        (T::Q4_0, T::F16),
        (T::F16, T::Q5_1),
        (T::Bf16, T::Bf16),
        (T::F32, T::Iq4Nl),
        (T::Q4_1, T::Q5_0),
        (T::F32, T::F32),
        (T::Iq4Nl, T::Q5_1),
        (T::Q5_0, T::Q4_1),
        (T::Q5_1, T::Q8_0),
        (T::Q4_1, T::Iq4Nl),
        (T::F16, T::Q4_0),
    ];
    // (query heads, key heads, head size, rotated dims, gated)
    let shapes = [
        (8, 2, 256, 64, true),
        (16, 8, 128, 128, false),
        (6, 1, 64, 64, true),
    ];
    for (n_head, n_head_kv, head_dim, rope_dims, gated) in shapes {
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
            weights: Some((q_norm, k_norm)),
            eps: 1e-6,
            gated,
            multi_rope: false,
        };
        let q_len = n_head * head_dim * if gated { 2 } else { 1 };
        let kv_len = n_head_kv * head_dim;
        let o_len = n_head * head_dim;
        let inputs: Vec<[Vec<f32>; 3]> = (0..n_pos as u32)
            .map(|p| {
                [
                    values(q_len, 3 * p + 10),
                    values(kv_len, 3 * p + 11),
                    values(kv_len, 3 * p + 12),
                ]
            })
            .collect();
        for (k_type, v_type) in types {
            let kv = KvTypes {
                k: k_type,
                v: v_type,
            };
            let mut cpu = CpuBackend::new(data, n_pos, 1, kv);
            let mut cache = cpu.kv_cache(n_pos).unwrap();
            let mut expected = Vec::new();
            let mut out = vec![0.0; o_len];
            for (pos, [qi, ki, vi]) in inputs.iter().enumerate() {
                let (mut q, mut k) = (qi.clone(), ki.clone());
                cpu.begin(pos, 1).unwrap();
                cpu.attention(&mut q, &mut k, vi, norm, &mut cache, &mut out);
                cpu.end().unwrap();
                expected.push(out.clone());
            }
            let tol = if k_type.is_quantized() || v_type.is_quantized() {
                4e-2
            } else {
                1e-2
            };
            for sizes in [&[1usize; 200][..], &[40, 17, 100, 43][..]] {
                let mut metal = MetalBackend::new(data, n_pos, kv).unwrap();
                let mut cache = metal.kv_cache(n_pos).unwrap();
                let mut start = 0;
                let mut worst = 0.0f32;
                for &n in sizes {
                    let batch = &inputs[start..start + n];
                    let rows = |i: usize| -> Vec<f32> {
                        batch.iter().flat_map(|x| x[i].clone()).collect()
                    };
                    let mut q = metal.vector(n * q_len).unwrap();
                    let mut k = metal.vector(n * kv_len).unwrap();
                    let mut v = metal.vector(n * kv_len).unwrap();
                    let mut out = metal.vector(n * o_len).unwrap();
                    metal.write(&mut q, &rows(0)).unwrap();
                    metal.write(&mut k, &rows(1)).unwrap();
                    metal.write(&mut v, &rows(2)).unwrap();
                    metal.begin(start, n).unwrap();
                    metal.attention(&mut q, &mut k, &v, norm, &mut cache, &mut out);
                    metal.end().unwrap();
                    let mut got = vec![0.0; n * o_len];
                    metal.read(&out, &mut got).unwrap();
                    for (t, row) in got.chunks(o_len).enumerate() {
                        for (&g, &e) in row.iter().zip(&expected[start + t]) {
                            worst = worst.max((g - e).abs() / (1.0 + e.abs()));
                        }
                    }
                    start += n;
                }
                assert!(
                    worst <= tol,
                    "{n_head}/{n_head_kv} heads of {head_dim}, K {k_type} V {v_type}, batches of {}: error {worst}",
                    sizes[0]
                );
            }
        }
    }
}

/// Times a DeltaNet layer over a 512-token batch with Qwen3.5-4B's sizes
/// (run with `--ignored --nocapture`).
#[test]
#[ignore = "a timing, not a check"]
fn delta_net_timing() {
    if gpu().is_none() {
        return;
    }
    let d = DeltaNetDims {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 32,
        d_k: 128,
        d_v: 128,
    };
    let mut params = Params::default();
    let p = DeltaNetParams {
        conv: params.push(&values(d.conv_kernel * d.conv_dim(), 5)),
        dt_bias: params.push(&values(d.n_v_heads, 6)),
        a: params.push(
            &values(d.n_v_heads, 7)
                .iter()
                .map(|v| -0.5 - v.abs())
                .collect::<Vec<_>>(),
        ),
        norm: params.push(&values(d.d_v, 8)),
    };
    let params = params.into_vec();
    let config = attention_config(2, 1, 64, 64);
    let bytes = [0u8; 16];
    let data = ModelData {
        bytes: &bytes,
        mapped: false,
        params: &params,
        config: &config,
        largest_tensor: 0,
    };
    let n = 512;
    let mut metal = MetalBackend::new(data, 64, Default::default()).unwrap();
    let mut state = DeltaNetState {
        conv: metal.vector(d.conv_len()).unwrap(),
        oldest: 0,
        ssm: metal.vector(d.state_len()).unwrap(),
        pending: 0,
        keep: 0,
        pending_ba: metal.vector(d.pending_len()).unwrap(),
        pending_row: 0,
    };
    let mut x = metal.vector(n * d.conv_dim()).unwrap();
    metal.write(&mut x, &values(n * d.conv_dim(), 1)).unwrap();
    let mut conv = metal.vector(n * d.conv_dim()).unwrap();
    let mut z = metal.vector(n * d.value_dim()).unwrap();
    let mut b = metal.vector(n * d.n_v_heads).unwrap();
    let mut al = metal.vector(n * d.n_v_heads).unwrap();
    let mut out = metal.vector(n * d.value_dim()).unwrap();
    let reps = 20;
    let mut run = |metal: &mut MetalBackend<'_>| {
        metal.begin(0, n).unwrap();
        metal.delta_net(
            d, p, 1e-6, &mut state, &x, &mut conv, &mut z, &mut b, &mut al, &mut out,
        );
        metal.end().unwrap();
    };
    run(&mut metal);
    let start = std::time::Instant::now();
    for _ in 0..reps {
        run(&mut metal);
    }
    drop(run);
    let mut sink = vec![0.0; n * d.value_dim()];
    metal.read(&out, &mut sink).unwrap();
    let s = start.elapsed().as_secs_f64() / f64::from(reps);
    eprintln!("delta_net, {n} tokens: {:.3} ms", s * 1e3);
}

#[test]
#[ignore = "a timing, not a check"]
fn attention_timing() {
    use crate::backend::{KvCacheType as T, KvTypes};
    let Some((device, pipelines)) = gpu() else {
        return;
    };
    for hd in [64, 128, 256] {
        let p_f16 = pipelines.attention_prompt(hd).unwrap().max_threads();
        let p_stg = pipelines.attention_prompt_staged(hd).unwrap().max_threads();
        eprintln!("hd={hd}: prompt max_threads={p_f16}, staged max_threads={p_stg}");
    }
    for (label, n_head, n_head_kv, hd, gated, n_past, kv) in [
        (
            "qwen3-0.6b d0 f16",
            16,
            8,
            128,
            false,
            0,
            KvTypes::default(),
        ),
        (
            "qwen35-4b d4096 f16",
            16,
            4,
            256,
            true,
            4096,
            KvTypes::default(),
        ),
        (
            "qwen35-4b d4096 q8_0",
            16,
            4,
            256,
            true,
            4096,
            KvTypes {
                k: T::Q8_0,
                v: T::Q8_0,
            },
        ),
    ] {
        let config = attention_config(n_head, n_head_kv, hd, hd.min(128));
        let mut params = Params::default();
        let qn = params.push(&vec![1.0; hd]);
        let kn = params.push(&vec![1.0; hd]);
        let params = params.into_vec();
        let bytes = [0u8; 16];
        let data = ModelData {
            bytes: &bytes,
            mapped: false,
            params: &params,
            config: &config,
            largest_tensor: 0,
        };
        let n = 512;
        let n_ctx = n_past + n;
        let mut metal = MetalBackend::new(data, n_ctx, kv).unwrap();
        let mut cache = metal.kv_cache(n_ctx).unwrap();
        let q_len = n * n_head * hd * if gated { 2 } else { 1 };
        let kv_len = n * n_head_kv * hd;
        let q_init = values(q_len, 1);
        let k_init = values(kv_len, 2);
        let v_init = values(kv_len, 3);
        let mut q = metal.vector(q_len).unwrap();
        let mut k = metal.vector(kv_len).unwrap();
        let mut v = metal.vector(kv_len).unwrap();
        let mut out = metal.vector(n * n_head * hd).unwrap();
        metal.write(&mut k, &k_init).unwrap();
        metal.write(&mut v, &v_init).unwrap();
        let norm = QkNorm {
            weights: Some((qn, kn)),
            eps: 1e-6,
            gated,
            multi_rope: false,
        };
        let shape = super::kernels::PromptAttention::new(
            n_head / n_head_kv,
            hd,
            kv != KvTypes::default(),
            device.max_threadgroup_memory(),
        );
        eprintln!("{label}: shape={shape:?}");
        // Populate `n_past` positions with distinct values per chunk so `cache`
        // is non-repeating and `need_scale` triggers like a real prompt.
        let mut filled = 0;
        let mut seed = 10u32;
        while filled < n_past {
            let chunk = n.min(n_past - filled);
            metal.write(&mut q, &values(q_len, seed)).unwrap();
            metal.write(&mut k, &values(kv_len, seed + 1)).unwrap();
            metal.write(&mut v, &values(kv_len, seed + 2)).unwrap();
            metal.begin(filled, chunk).unwrap();
            metal.attention(&mut q, &mut k, &v, norm, &mut cache, &mut out);
            metal.end().unwrap();
            filled += chunk;
            seed += 3;
        }
        metal.sync().unwrap();
        // Warm up and time `reps` evaluations at `n_past` (use fresh q in a backup buffer).
        let mut q_src = metal.vector(q_len).unwrap();
        metal.write(&mut q_src, &q_init).unwrap();
        let reps = 10;
        let mut run = |m: &mut MetalBackend<'_>, batch: usize| {
            cache.len = n_past;
            m.begin(n_past, batch).unwrap();
            m.copy_rows(&q_src, 0, &mut q, 0, batch, q_len / n);
            m.attention(&mut q, &mut k, &v, norm, &mut cache, &mut out);
            m.end().unwrap();
        };
        run(&mut metal, n);
        metal.sync().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..reps {
            run(&mut metal, n);
        }
        metal.sync().unwrap();
        let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        run(&mut metal, 508);
        metal.sync().unwrap();
        let start508 = std::time::Instant::now();
        for _ in 0..reps {
            run(&mut metal, 508);
        }
        metal.sync().unwrap();
        let ms508 = start508.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        eprintln!("{label}: n=512: {ms:.3} ms/layer, n=508: {ms508:.3} ms/layer");
    }
}

/// The deferred RMSNorm of `residual` (batches of up to `MATMUL_MIN_BATCH`
/// tokens): the fixed-point sum of squares is within 2e-6 of the exact sum
/// (a few float roundings of the scale), for activations as small as an
/// embedding's (RMS near 0.007, where rounding each square to the fixed
/// point costs the most) and large ones, and the scale it gives reaches
/// both a product that reads the vector and a `read` of it.
#[test]
fn deferred_rmsnorm_matches_the_exact_norm() {
    if gpu().is_none() {
        return;
    }
    let (rows, cols, rows2) = (1024, 256, 64);
    let (bytes1, w, _) = matrix(GgmlType::F16, rows, cols, 1);
    let (bytes2, mut w2, decoded2) = matrix(GgmlType::F16, rows2, rows, 5000);
    w2.offset = bytes1.len();
    let bytes = [bytes1, bytes2].concat();
    let mut params = Params::default();
    let norm_w: Vec<f32> = values(rows, 7).iter().map(|v| 1.0 + 0.5 * v).collect();
    let norm = params.push(&norm_w);
    let params = params.into_vec();
    let config = attention_config(1, 1, 64, 64);
    let data = ModelData {
        bytes: &bytes,
        mapped: false,
        params: &params,
        config: &config,
        largest_tensor: bytes.len(),
    };
    let eps = 1e-6;
    for scale in [0.01f32, 4.0] {
        for n in [1, 3] {
            let mut metal =
                MetalBackend::new(data, 64, crate::backend::KvTypes::default()).unwrap();
            let a0: Vec<f32> = values(n * cols, 11)
                .iter()
                .map(|v| v * scale / 16.0)
                .collect();
            let x0: Vec<f32> = values(n * rows, 13).iter().map(|v| v * scale).collect();
            let mut a = metal.vector(n * cols).unwrap();
            let mut x = metal.vector(n * rows).unwrap();
            let mut out = metal.vector(n * rows).unwrap();
            let mut y = metal.vector(n * rows2).unwrap();
            metal.write(&mut a, &a0).unwrap();
            metal.write(&mut x, &x0).unwrap();
            metal.begin(0, n).unwrap();
            metal.residual(&w, &a, &mut x, norm, eps, &mut out);
            metal.matvecs(&out, [(&w2, &mut y)]);
            metal.end().unwrap();
            let mut xs = vec![0.0; n * rows];
            metal.read(&x, &mut xs).unwrap();
            let mut ys = vec![0.0; n * rows2];
            metal.read(&y, &mut ys).unwrap();
            // `read` synced and moved the pending sums to the first slots.
            // SAFETY: every committed command buffer has completed.
            let sums = unsafe { metal.rms_sums.bytes() }[..8 * n].to_vec();
            let mut outs = vec![0.0; n * rows];
            metal.read(&out, &mut outs).unwrap();
            for t in 0..n {
                let x = &xs[t * rows..(t + 1) * rows];
                let exact: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
                let raw = u64::from_ne_bytes(sums[8 * t..8 * t + 8].try_into().unwrap());
                let fixed = raw as f64 / f64::from(super::RMS_FP_SCALE);
                let rms = (exact / rows as f64).sqrt();
                let rel = (fixed - exact).abs() / exact;
                assert!(
                    rel <= 2e-6,
                    "RMS {rms:.4}, token {t} of {n}: sum of squares {fixed} vs {exact} (relative error {rel:e})"
                );
                let s = 1.0 / (exact / rows as f64 + f64::from(eps)).sqrt();
                let xn: Vec<f64> = x
                    .iter()
                    .zip(&norm_w)
                    .map(|(&v, &w)| f64::from(v) * f64::from(w) * s)
                    .collect();
                for (i, (&got, &want)) in outs[t * rows..(t + 1) * rows].iter().zip(&xn).enumerate()
                {
                    assert!(
                        (f64::from(got) - want).abs() <= 2e-6 * want.abs(),
                        "RMS {rms:.4}, token {t} of {n}, read value {i}: {got} vs {want}"
                    );
                }
                for (r, row) in decoded2.iter().enumerate() {
                    let (want, magnitude) =
                        row.iter().zip(&xn).fold((0.0, 0.0), |(s, m), (&w, &v)| {
                            (s + f64::from(w) * v, m + (f64::from(w) * v).abs())
                        });
                    let got = f64::from(ys[t * rows2 + r]);
                    assert!(
                        (got - want).abs() <= 4e-6 * magnitude,
                        "RMS {rms:.4}, token {t} of {n}, product row {r}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
