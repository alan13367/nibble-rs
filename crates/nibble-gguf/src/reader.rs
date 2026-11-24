use std::collections::HashMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::types::GgmlType;
use crate::value::{Array, GgufStr, Value, ValueType, split_value};

/// Tensor data alignment used when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: usize = 32;

/// Maximum number of tensor dimensions (`GGML_MAX_DIMS`).
pub const MAX_DIMS: usize = 4;

const MAGIC: [u8; 4] = *b"GGUF";

/// Smallest possible metadata entry: empty key (8) + type (4) + one-byte value.
const MIN_KV_BYTES: usize = 13;
/// Smallest possible tensor info: empty name (8) + n_dims (4) + type (4) + offset (8).
const MIN_TENSOR_INFO_BYTES: usize = 24;

/// A parsed GGUF file. Everything is borrowed from the input buffer.
pub struct Gguf<'a> {
    version: u32,
    alignment: usize,
    data_offset: usize,
    metadata: Vec<(&'a str, Value<'a>)>,
    metadata_index: HashMap<&'a str, usize>,
    tensors: Vec<TensorInfo<'a>>,
    tensor_index: HashMap<&'a str, usize>,
}

impl<'a> Gguf<'a> {
    /// Parses a GGUF file (version 2 or 3) from its bytes.
    ///
    /// Every length, count and offset is bounds-checked, so a corrupt or
    /// hostile file produces an error rather than a panic or a huge allocation.
    /// Tensors of unknown types are listed with no data.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] describing the first problem found.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut r = Cursor { buf: bytes, pos: 0 };

        let magic = r.array::<4>()?;
        if magic != MAGIC {
            return Err(Error::BadMagic(magic));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;

        let cap = capacity_hint(n_kv, r.remaining(), MIN_KV_BYTES);
        let mut metadata = Vec::with_capacity(cap);
        let mut metadata_index = HashMap::with_capacity(cap);
        for _ in 0..n_kv {
            let key = r.utf8()?;
            let value_type = r.value_type()?;
            let value = r.value(value_type)?;
            if metadata_index.insert(key, metadata.len()).is_some() {
                return Err(Error::DuplicateKey(key.to_owned()));
            }
            metadata.push((key, value));
        }

        let alignment = match metadata_index
            .get("general.alignment")
            .map(|&i| metadata[i].1)
        {
            None => DEFAULT_ALIGNMENT,
            Some(Value::U32(a)) if a.is_power_of_two() => a as usize,
            Some(other) => return Err(Error::InvalidAlignment(format!("{other:?}"))),
        };

        let cap = capacity_hint(n_tensors, r.remaining(), MIN_TENSOR_INFO_BYTES);
        let mut raw_tensors = Vec::with_capacity(cap);
        for _ in 0..n_tensors {
            raw_tensors.push(r.tensor_info()?);
        }

        // The data section starts at the next multiple of the alignment. A file
        // with no tensor data (e.g. vocab-only) may end before it.
        let data_offset = r
            .pos
            .checked_next_multiple_of(alignment)
            .ok_or(Error::TooLarge { offset: r.pos })?;
        let data = bytes.get(data_offset..).unwrap_or(&[]);

        let mut tensors = Vec::with_capacity(raw_tensors.len());
        let mut tensor_index = HashMap::with_capacity(raw_tensors.len());
        for raw in raw_tensors {
            let tensor = raw.validate(data, alignment)?;
            if tensor_index.insert(tensor.name, tensors.len()).is_some() {
                return Err(Error::DuplicateTensor(tensor.name.to_owned()));
            }
            tensors.push(tensor);
        }

        Ok(Self {
            version,
            alignment,
            data_offset,
            metadata,
            metadata_index,
            tensors,
            tensor_index,
        })
    }

    /// The GGUF format version (2 or 3).
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Alignment of tensor data, in bytes.
    #[must_use]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    /// Byte offset of the data section from the start of the file.
    /// A tensor's absolute file offset is `data_offset() + tensor.offset()`.
    #[must_use]
    pub const fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// All metadata entries, in file order.
    #[must_use]
    pub fn metadata(&self) -> &[(&'a str, Value<'a>)] {
        &self.metadata
    }

    /// The metadata value stored under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Value<'a>> {
        self.metadata_index.get(key).map(|&i| self.metadata[i].1)
    }

    /// The model architecture (`general.architecture`), such as `qwen35`.
    #[must_use]
    pub fn architecture(&self) -> Option<&'a str> {
        self.get("general.architecture")?.as_str()
    }

    /// All tensors, in file order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo<'a>] {
        &self.tensors
    }

    /// The tensor named `name`.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo<'a>> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }
}

impl fmt::Debug for Gguf<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gguf")
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("data_offset", &self.data_offset)
            .field("metadata", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

/// A tensor's dimensions, innermost (contiguous) first, as in ggml.
///
/// GGUF stores dimensions in the reverse of PyTorch order: a PyTorch weight
/// of shape `[out, in]` has `dims() == [in, out]`, and `dims()[0]` is the
/// length of one contiguous row.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u64; MAX_DIMS],
    n_dims: u8,
}

impl Shape {
    /// The dimensions that are present in the file.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims[..usize::from(self.n_dims)]
    }

    /// Dimension `i`, or 1 for dimensions past [`Shape::dims`] (as ggml's `ne[i]`).
    #[must_use]
    pub fn ne(&self, i: usize) -> u64 {
        self.dims.get(i).copied().unwrap_or(1)
    }

    /// Total number of elements. Guaranteed by the parser to fit in `i64`.
    #[must_use]
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.dims()).finish()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Pad as a single unit so the shape lines up in tables.
        let mut s = String::from("[");
        for (i, d) in self.dims().iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&d.to_string());
        }
        s.push(']');
        f.pad(&s)
    }
}

/// A tensor's description and data.
#[derive(Clone)]
pub struct TensorInfo<'a> {
    name: &'a str,
    ggml_type: GgmlType,
    shape: Shape,
    offset: u64,
    data: Option<&'a [u8]>,
}

impl<'a> TensorInfo<'a> {
    /// The tensor's name, such as `blk.0.attn_norm.weight`.
    #[must_use]
    pub const fn name(&self) -> &'a str {
        self.name
    }

    /// The element type.
    #[must_use]
    pub const fn ggml_type(&self) -> GgmlType {
        self.ggml_type
    }

    /// The tensor's dimensions.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Offset of the data, relative to the start of the data section.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// The tensor's encoded bytes, or `None` when the type is unknown.
    #[must_use]
    pub const fn data(&self) -> Option<&'a [u8]> {
        self.data
    }
}

impl fmt::Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorInfo")
            .field("name", &self.name)
            .field("ggml_type", &self.ggml_type)
            .field("shape", &self.shape)
            .field("offset", &self.offset)
            .field("size", &self.data.map(<[u8]>::len))
            .finish()
    }
}

/// A tensor info as read, before its data range is validated.
struct RawTensor<'a> {
    name: &'a str,
    ggml_type: GgmlType,
    shape: Shape,
    offset: u64,
}

impl<'a> RawTensor<'a> {
    fn validate(self, data: &'a [u8], alignment: usize) -> Result<TensorInfo<'a>> {
        let invalid_shape = |reason: String| Error::InvalidShape {
            tensor: self.name.to_owned(),
            reason,
        };

        // ggml stores dimensions as int64, so every product must fit in i64.
        let mut n_elements: u64 = 1;
        for &d in self.shape.dims() {
            n_elements = n_elements
                .checked_mul(d)
                .filter(|&n| i64::try_from(n).is_ok())
                .ok_or_else(|| invalid_shape(format!("{:?} has too many elements", self.shape)))?;
        }

        if !self.offset.is_multiple_of(alignment as u64) {
            return Err(Error::MisalignedTensor {
                tensor: self.name.to_owned(),
                offset: self.offset,
                alignment,
            });
        }

        let tensor_data = match self.ggml_type.layout() {
            None => None,
            Some(layout) => {
                let ne0 = self.shape.ne(0);
                let row_bytes = layout.row_bytes(ne0).ok_or_else(|| {
                    invalid_shape(format!(
                        "first dimension {ne0} is not a multiple of the {} block size {}",
                        self.ggml_type, layout.block_size
                    ))
                })?;
                // Checked separately: with ne0 == 0 the element count is 0 even
                // when the other dimensions multiply past u64::MAX.
                let size = self
                    .shape
                    .ne(1)
                    .checked_mul(self.shape.ne(2))
                    .and_then(|n| n.checked_mul(self.shape.ne(3)))
                    .and_then(|rows| rows.checked_mul(row_bytes))
                    .ok_or_else(|| invalid_shape(format!("{:?} is too large", self.shape)))?;
                let out_of_bounds = || Error::TensorOutOfBounds {
                    tensor: self.name.to_owned(),
                    offset: self.offset,
                    size,
                    available: data.len() as u64,
                };
                let start = usize::try_from(self.offset).map_err(|_| out_of_bounds())?;
                let len = usize::try_from(size).map_err(|_| out_of_bounds())?;
                let end = start.checked_add(len).ok_or_else(out_of_bounds)?;
                Some(data.get(start..end).ok_or_else(out_of_bounds)?)
            }
        };

        Ok(TensorInfo {
            name: self.name,
            ggml_type: self.ggml_type,
            shape: self.shape,
            offset: self.offset,
            data: tensor_data,
        })
    }
}

/// Upper bound for pre-allocation: never reserve more entries than the
/// remaining bytes could hold, whatever count the file claims.
fn capacity_hint(count: u64, remaining: usize, min_entry_bytes: usize) -> usize {
    usize::try_from(count)
        .unwrap_or(usize::MAX)
        .min(remaining / min_entry_bytes)
}

/// Little-endian reader over the file bytes. Invariant: `pos <= buf.len()`.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    const fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        let eof = Error::UnexpectedEof {
            offset: self.pos,
            needed: n,
        };
        let n = usize::try_from(n).map_err(|_| eof.clone())?;
        let (head, _) = self.buf[self.pos..].split_at_checked(n).ok_or(eof)?;
        self.pos += n;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let offset = self.pos;
        let (head, _) =
            self.buf[self.pos..]
                .split_first_chunk::<N>()
                .ok_or(Error::UnexpectedEof {
                    offset,
                    needed: N as u64,
                })?;
        self.pos += N;
        Ok(*head)
    }

    fn u32(&mut self) -> Result<u32> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64> {
        self.array().map(u64::from_le_bytes)
    }

    /// A length-prefixed byte string.
    fn string(&mut self) -> Result<&'a [u8]> {
        let len = self.u64()?;
        self.take(len)
    }

    /// A length-prefixed string that must be UTF-8 (keys and tensor names).
    fn utf8(&mut self) -> Result<&'a str> {
        let offset = self.pos + 8;
        std::str::from_utf8(self.string()?).map_err(|_| Error::InvalidUtf8 { offset })
    }

    fn value_type(&mut self) -> Result<ValueType> {
        let offset = self.pos;
        let raw = self.u32()?;
        ValueType::from_raw(raw).ok_or(Error::InvalidValueType { offset, raw })
    }

    fn value(&mut self, ty: ValueType) -> Result<Value<'a>> {
        match ty {
            ValueType::String => Ok(Value::String(GgufStr::new(self.string()?))),
            ValueType::Array => Ok(Value::Array(self.array_value()?)),
            ValueType::Bool => {
                let offset = self.pos;
                match self.array::<1>()?[0] {
                    0 => Ok(Value::Bool(false)),
                    1 => Ok(Value::Bool(true)),
                    raw => Err(Error::InvalidBool { offset, raw }),
                }
            }
            fixed => {
                let offset = self.pos;
                let size = fixed.fixed_size().unwrap_or(0);
                let bytes = self.take(size as u64)?;
                split_value(fixed, bytes)
                    .map(|(value, _)| value)
                    .ok_or(Error::UnexpectedEof {
                        offset,
                        needed: size as u64,
                    })
            }
        }
    }

    fn array_value(&mut self) -> Result<Array<'a>> {
        let type_offset = self.pos;
        let elem = self.value_type()?;
        let count_offset = self.pos;
        let count = self.u64()?;
        let start = self.pos;

        match elem {
            ValueType::Array => {
                return Err(Error::NestedArray {
                    offset: type_offset,
                });
            }
            // Each string needs at least 8 bytes, so a bogus count fails at EOF
            // after at most len/8 iterations.
            ValueType::String => {
                for _ in 0..count {
                    self.string()?;
                }
            }
            fixed => {
                let size = fixed.fixed_size().unwrap_or(0) as u64;
                let total = count.checked_mul(size).ok_or(Error::TooLarge {
                    offset: count_offset,
                })?;
                let bytes = self.take(total)?;
                if fixed == ValueType::Bool
                    && let Some(i) = bytes.iter().position(|&b| b > 1)
                {
                    return Err(Error::InvalidBool {
                        offset: start + i,
                        raw: bytes[i],
                    });
                }
            }
        }

        let len = usize::try_from(count).map_err(|_| Error::TooLarge {
            offset: count_offset,
        })?;
        Ok(Array::new(elem, len, &self.buf[start..self.pos]))
    }

    fn tensor_info(&mut self) -> Result<RawTensor<'a>> {
        let name = self.utf8()?;
        let n_dims = self.u32()?;
        if n_dims as usize > MAX_DIMS {
            return Err(Error::InvalidShape {
                tensor: name.to_owned(),
                reason: format!("{n_dims} dimensions (at most {MAX_DIMS} are supported)"),
            });
        }
        let mut dims = [1u64; MAX_DIMS];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = self.u64()?;
        }
        let ggml_type = GgmlType::from_raw(self.u32()?);
        let offset = self.u64()?;
        // n_dims <= MAX_DIMS (4), checked above.
        let shape = Shape {
            dims,
            n_dims: n_dims as u8,
        };
        Ok(RawTensor {
            name,
            ggml_type,
            shape,
            offset,
        })
    }
}
