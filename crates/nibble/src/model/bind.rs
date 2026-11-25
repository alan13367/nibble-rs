//! Binding GGUF tensors to weights, checking every shape and type.

use nibble_gguf::Gguf;

use crate::backend::{Param, Params};
use crate::cpu::{Experts, Weight};
use crate::error::{Error, Result};
use crate::quant;

/// Binds a weight matrix with `rows` rows of `cols` elements (ggml shape
/// `[cols, rows]`; a single row may also be stored as the vector `[cols]`).
pub(crate) fn matrix(g: &Gguf<'_>, name: &str, cols: usize, rows: usize) -> Result<Weight> {
    bind(g, name, cols, rows, &[cols as u64, rows as u64])
}

/// Binds `count` expert matrices of `rows` rows of `cols` elements, stored
/// as one tensor of ggml shape `[cols, rows, count]`.
pub(crate) fn experts(
    g: &Gguf<'_>,
    name: &str,
    cols: usize,
    rows: usize,
    count: usize,
) -> Result<Experts> {
    let all_rows = rows
        .checked_mul(count)
        .ok_or_else(|| Error::InvalidMetadata {
            key: name.to_owned(),
            reason: "too many expert rows".into(),
        })?;
    let all = bind(
        g,
        name,
        cols,
        all_rows,
        &[cols as u64, rows as u64, count as u64],
    )?;
    Ok(Experts {
        first: Weight { rows, ..all },
        count,
    })
}

/// Binds tensor `name` of ggml shape `expected` as `rows` rows of `cols`.
fn bind(g: &Gguf<'_>, name: &str, cols: usize, rows: usize, expected: &[u64]) -> Result<Weight> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
    let dims = t.shape().dims();
    if dims != expected && !(rows == 1 && dims == [cols as u64]) {
        return Err(Error::TensorShape {
            tensor: name.to_owned(),
            expected: expected.to_vec(),
            found: dims.to_vec(),
        });
    }
    let ty = t.ggml_type();
    let unsupported = || Error::UnsupportedTensorType {
        tensor: name.to_owned(),
        ggml_type: ty,
    };
    if !quant::is_supported(ty) {
        return Err(unsupported());
    }
    let row_bytes = ty
        .layout()
        .and_then(|l| l.row_bytes(cols as u64))
        .ok_or_else(unsupported)?;
    Ok(Weight {
        ty,
        cols,
        rows,
        offset: g.data_offset() + t.offset() as usize,
        row_bytes: row_bytes as usize,
    })
}

/// Reads a tensor of `shape` (ggml order, e.g. `[4, 6144]`) into an `f32` vector,
/// for small parameters such as norm weights, biases and convolution kernels.
pub(crate) fn tensor_f32(g: &Gguf<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
    let expected: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    if t.shape().dims() != expected {
        return Err(Error::TensorShape {
            tensor: name.to_owned(),
            expected,
            found: t.shape().dims().to_vec(),
        });
    }
    let ty = t.ggml_type();
    let data = t.data().filter(|_| quant::is_supported(ty));
    let data = data.ok_or_else(|| Error::UnsupportedTensorType {
        tensor: name.to_owned(),
        ggml_type: ty,
    })?;
    // Rows are whole blocks laid out back to back, so the tensor decodes as one run.
    let mut out = vec![0.0; shape.iter().product()];
    quant::dequantize_row(ty, data, &mut out);
    Ok(out)
}

/// Reads a 1-D tensor of `len` elements as `f32` into the parameter arena.
pub(crate) fn vector(g: &Gguf<'_>, p: &mut Params, name: &str, len: usize) -> Result<Param> {
    Ok(p.push(&tensor_f32(g, name, &[len])?))
}
