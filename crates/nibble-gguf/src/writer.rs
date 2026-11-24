use std::io::{self, Write};

use crate::reader::{DEFAULT_ALIGNMENT, MAX_DIMS};
use crate::types::GgmlType;
use crate::value::ValueType;

/// An owned metadata value, for [`GgufWriter`].
#[derive(Clone, Debug, PartialEq)]
pub enum OwnedValue {
    /// `u8`
    U8(u8),
    /// `i8`
    I8(i8),
    /// `u16`
    U16(u16),
    /// `i16`
    I16(i16),
    /// `u32`
    U32(u32),
    /// `i32`
    I32(i32),
    /// `f32`
    F32(f32),
    /// `bool`
    Bool(bool),
    /// A string
    String(String),
    /// An array: element type and elements. Elements must all have that type
    /// and must not be arrays.
    Array(ValueType, Vec<OwnedValue>),
    /// `u64`
    U64(u64),
    /// `i64`
    I64(i64),
    /// `f64`
    F64(f64),
}

impl OwnedValue {
    /// Builds an array of `elem`-typed values.
    pub fn array<T: Into<Self>>(elem: ValueType, items: impl IntoIterator<Item = T>) -> Self {
        Self::Array(elem, items.into_iter().map(Into::into).collect())
    }

    /// The value's type.
    #[must_use]
    pub const fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(..) => ValueType::Array,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }

    fn write_payload(&self, out: &mut Vec<u8>) {
        match self {
            Self::U8(v) => out.push(*v),
            Self::I8(v) => out.extend(v.to_le_bytes()),
            Self::U16(v) => out.extend(v.to_le_bytes()),
            Self::I16(v) => out.extend(v.to_le_bytes()),
            Self::U32(v) => out.extend(v.to_le_bytes()),
            Self::I32(v) => out.extend(v.to_le_bytes()),
            Self::F32(v) => out.extend(v.to_le_bytes()),
            Self::Bool(v) => out.push(u8::from(*v)),
            Self::String(s) => put_str(out, s),
            Self::Array(elem, items) => {
                out.extend(elem.to_raw().to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                for item in items {
                    item.write_payload(out);
                }
            }
            Self::U64(v) => out.extend(v.to_le_bytes()),
            Self::I64(v) => out.extend(v.to_le_bytes()),
            Self::F64(v) => out.extend(v.to_le_bytes()),
        }
    }
}

macro_rules! owned_from {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(impl From<$t> for OwnedValue {
            fn from(v: $t) -> Self {
                Self::$variant(v)
            }
        })*
    };
}

owned_from!(u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32, f32 => F32,
    bool => Bool, String => String, u64 => U64, i64 => I64, f64 => F64);

impl From<&str> for OwnedValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

/// Builds GGUF v3 files: test fixtures, tiny synthetic models, conversions.
///
/// Tensors are laid out in the order they are added, each padded to the alignment.
#[derive(Clone, Debug)]
pub struct GgufWriter {
    alignment: u32,
    metadata: Vec<(String, OwnedValue)>,
    tensors: Vec<PendingTensor>,
}

#[derive(Clone, Debug)]
struct PendingTensor {
    name: String,
    ggml_type: GgmlType,
    dims: Vec<u64>,
    data: Vec<u8>,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    /// An empty file with the default 32-byte alignment.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT as u32,
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Sets the tensor data alignment, written as `general.alignment` when not 32.
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is not a non-zero power of two.
    pub fn set_alignment(&mut self, alignment: u32) -> &mut Self {
        assert!(
            alignment.is_power_of_two(),
            "alignment must be a non-zero power of two"
        );
        self.alignment = alignment;
        self
    }

    /// Adds a metadata entry.
    ///
    /// # Panics
    ///
    /// Panics if `key` was already added or is `general.alignment` (use
    /// [`GgufWriter::set_alignment`]), or if an array holds arrays or
    /// elements of the wrong type.
    pub fn add(&mut self, key: impl Into<String>, value: impl Into<OwnedValue>) -> &mut Self {
        let key = key.into();
        let value = value.into();
        assert!(
            key != "general.alignment",
            "use set_alignment for general.alignment"
        );
        assert!(
            self.metadata.iter().all(|(k, _)| *k != key),
            "duplicate metadata key `{key}`"
        );
        if let OwnedValue::Array(elem, items) = &value {
            assert!(
                *elem != ValueType::Array,
                "nested arrays are not supported (`{key}`)"
            );
            assert!(
                items.iter().all(|item| item.value_type() == *elem),
                "array `{key}` has elements that are not {elem}"
            );
        }
        self.metadata.push((key, value));
        self
    }

    /// Adds a tensor with `dims` in ggml order (innermost first) and encoded `data`.
    ///
    /// # Panics
    ///
    /// Panics if the name was already added, there are more than four
    /// dimensions, or (for known types) `data` has the wrong length for the shape.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        ggml_type: GgmlType,
        dims: &[u64],
        data: impl Into<Vec<u8>>,
    ) -> &mut Self {
        let name = name.into();
        let data = data.into();
        assert!(
            dims.len() <= MAX_DIMS,
            "tensor `{name}` has more than {MAX_DIMS} dimensions"
        );
        assert!(
            self.tensors.iter().all(|t| t.name != name),
            "duplicate tensor `{name}`"
        );
        if let Some(layout) = ggml_type.layout() {
            let ne0 = dims.first().copied().unwrap_or(1);
            let rows: u64 = dims.iter().skip(1).product();
            let expected = layout.row_bytes(ne0).map(|row| row * rows);
            assert_eq!(
                expected,
                Some(data.len() as u64),
                "tensor `{name}`: data length does not match {ggml_type} {dims:?}"
            );
        }
        self.tensors.push(PendingTensor {
            name,
            ggml_type,
            dims: dims.to_vec(),
            data,
        });
        self
    }

    /// Encodes the file.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let align = self.alignment as usize;
        let mut out = Vec::new();
        out.extend(b"GGUF");
        out.extend(3u32.to_le_bytes());
        out.extend((self.tensors.len() as u64).to_le_bytes());

        let write_alignment = align != DEFAULT_ALIGNMENT;
        let n_kv = self.metadata.len() + usize::from(write_alignment);
        out.extend((n_kv as u64).to_le_bytes());
        if write_alignment {
            put_str(&mut out, "general.alignment");
            out.extend(ValueType::U32.to_raw().to_le_bytes());
            out.extend(self.alignment.to_le_bytes());
        }
        for (key, value) in &self.metadata {
            put_str(&mut out, key);
            out.extend(value.value_type().to_raw().to_le_bytes());
            value.write_payload(&mut out);
        }

        let mut offset = 0usize;
        for t in &self.tensors {
            put_str(&mut out, &t.name);
            out.extend((t.dims.len() as u32).to_le_bytes());
            for &d in &t.dims {
                out.extend(d.to_le_bytes());
            }
            out.extend(t.ggml_type.to_raw().to_le_bytes());
            out.extend((offset as u64).to_le_bytes());
            offset = (offset + t.data.len()).next_multiple_of(align);
        }

        out.resize(out.len().next_multiple_of(align), 0);
        for t in &self.tensors {
            out.extend(&t.data);
            out.resize(out.len().next_multiple_of(align), 0);
        }
        out
    }

    /// Encodes the file into `writer`.
    ///
    /// # Errors
    ///
    /// Returns any error from `writer`.
    pub fn write_to(&self, mut writer: impl Write) -> io::Result<()> {
        writer.write_all(&self.to_bytes())
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}
