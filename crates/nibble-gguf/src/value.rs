use std::fmt;
use std::str::Utf8Error;

/// The type of a metadata value, as stored in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// `u8`
    U8,
    /// `i8`
    I8,
    /// `u16`
    U16,
    /// `i16`
    I16,
    /// `u32`
    U32,
    /// `i32`
    I32,
    /// `f32`
    F32,
    /// `bool`, stored as one byte
    Bool,
    /// Length-prefixed string
    String,
    /// Length-prefixed array of one element type
    Array,
    /// `u64`
    U64,
    /// `i64`
    I64,
    /// `f64`
    F64,
}

impl ValueType {
    /// Maps a raw type id to a value type.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    /// The raw type id.
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// Size in bytes of one value, or `None` for strings and arrays.
    #[must_use]
    pub const fn fixed_size(self) -> Option<usize> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::String | Self::Array => None,
        }
    }

    /// Short lowercase name, such as `u32` or `str`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "str",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.name())
    }
}

/// A string stored in a GGUF file, borrowed from the file's bytes.
///
/// GGUF strings should be UTF-8, but tokenizer vocabularies occasionally
/// contain invalid sequences, so validation happens on access.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GgufStr<'a>(&'a [u8]);

impl<'a> GgufStr<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }

    /// The raw bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.0
    }

    /// The string, if it is valid UTF-8.
    ///
    /// # Errors
    ///
    /// Returns an error when the bytes are not valid UTF-8.
    pub const fn to_str(&self) -> Result<&'a str, Utf8Error> {
        std::str::from_utf8(self.0)
    }
}

impl fmt::Display for GgufStr<'_> {
    /// Writes the string, replacing invalid UTF-8 with U+FFFD.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for chunk in self.0.utf8_chunks() {
            f.write_str(chunk.valid())?;
            if !chunk.invalid().is_empty() {
                f.write_str("\u{FFFD}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for GgufStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_str() {
            Ok(s) => fmt::Debug::fmt(s, f),
            Err(_) => write!(f, "GgufStr({:x?})", self.0),
        }
    }
}

/// A metadata value, borrowed from the file's bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<'a> {
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
    String(GgufStr<'a>),
    /// An array
    Array(Array<'a>),
    /// `u64`
    U64(u64),
    /// `i64`
    I64(i64),
    /// `f64`
    F64(f64),
}

impl<'a> Value<'a> {
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
            Self::Array(_) => ValueType::Array,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }

    /// The value as `u64`, for any integer type holding a non-negative value.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value as `u32`, for any integer type holding a value in range.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|v| u32::try_from(v).ok())
    }

    /// The value as `i64`, for any integer type holding a value in range.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => i64::try_from(v).ok(),
            Self::I8(v) => Some(v.into()),
            Self::I16(v) => Some(v.into()),
            Self::I32(v) => Some(v.into()),
            Self::I64(v) => Some(v),
            _ => None,
        }
    }

    /// The value as `f32`, for `f32` values and (rounded) `f64` values.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            // Rounding to the nearest f32 is the intended conversion.
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    /// The value as `bool`.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// The value as a UTF-8 string.
    #[must_use]
    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            Self::String(s) => s.to_str().ok(),
            _ => None,
        }
    }

    /// The value as an array.
    #[must_use]
    pub const fn as_array(&self) -> Option<Array<'a>> {
        match *self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// An array of metadata values, borrowed from the file's bytes.
///
/// Elements are decoded lazily: iterating a 250k-entry vocabulary allocates nothing.
#[derive(Clone, Copy, PartialEq)]
pub struct Array<'a> {
    elem: ValueType,
    len: usize,
    /// Encoded elements, already validated by the parser.
    bytes: &'a [u8],
}

impl<'a> Array<'a> {
    pub(crate) const fn new(elem: ValueType, len: usize, bytes: &'a [u8]) -> Self {
        Self { elem, len, bytes }
    }

    /// The type of every element.
    #[must_use]
    pub const fn elem_type(&self) -> ValueType {
        self.elem
    }

    /// The number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the array has no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The element at `index`. Constant time for fixed-size elements, linear for strings.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value<'a>> {
        self.iter().nth(index)
    }

    /// Iterates over the elements.
    #[must_use]
    pub const fn iter(&self) -> ArrayIter<'a> {
        ArrayIter {
            elem: self.elem,
            remaining: self.len,
            bytes: self.bytes,
        }
    }
}

impl fmt::Debug for Array<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("elem", &self.elem)
            .field("len", &self.len)
            .finish()
    }
}

impl<'a> IntoIterator for Array<'a> {
    type Item = Value<'a>;
    type IntoIter = ArrayIter<'a>;

    fn into_iter(self) -> ArrayIter<'a> {
        self.iter()
    }
}

/// Iterator over the elements of an [`Array`].
#[derive(Clone, Debug)]
pub struct ArrayIter<'a> {
    elem: ValueType,
    remaining: usize,
    bytes: &'a [u8],
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = Value<'a>;

    fn next(&mut self) -> Option<Value<'a>> {
        if self.remaining == 0 {
            return None;
        }
        let (value, rest) = split_value(self.elem, self.bytes)?;
        self.bytes = rest;
        self.remaining -= 1;
        Some(value)
    }

    fn nth(&mut self, n: usize) -> Option<Value<'a>> {
        if let Some(size) = self.elem.fixed_size() {
            if n >= self.remaining {
                self.remaining = 0;
                self.bytes = &[];
                return None;
            }
            self.bytes = self.bytes.get(n * size..)?;
            self.remaining -= n;
            return self.next();
        }
        for _ in 0..n {
            self.next()?;
        }
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for ArrayIter<'_> {}

/// Decodes one encoded value of type `ty` from the front of `bytes`.
///
/// Returns `None` for arrays and when `bytes` is too short. Booleans decode as
/// `byte != 0`; the parser has already rejected bytes other than 0 and 1.
pub(crate) fn split_value(ty: ValueType, bytes: &[u8]) -> Option<(Value<'_>, &[u8])> {
    macro_rules! fixed {
        ($n:literal, $variant:ident, $t:ty) => {{
            let (head, rest) = bytes.split_first_chunk::<$n>()?;
            Some((Value::$variant(<$t>::from_le_bytes(*head)), rest))
        }};
    }
    match ty {
        ValueType::U8 => fixed!(1, U8, u8),
        ValueType::I8 => fixed!(1, I8, i8),
        ValueType::U16 => fixed!(2, U16, u16),
        ValueType::I16 => fixed!(2, I16, i16),
        ValueType::U32 => fixed!(4, U32, u32),
        ValueType::I32 => fixed!(4, I32, i32),
        ValueType::F32 => fixed!(4, F32, f32),
        ValueType::U64 => fixed!(8, U64, u64),
        ValueType::I64 => fixed!(8, I64, i64),
        ValueType::F64 => fixed!(8, F64, f64),
        ValueType::Bool => {
            let (&byte, rest) = bytes.split_first()?;
            Some((Value::Bool(byte != 0), rest))
        }
        ValueType::String => {
            let (len, rest) = bytes.split_first_chunk::<8>()?;
            let len = usize::try_from(u64::from_le_bytes(*len)).ok()?;
            let (s, rest) = rest.split_at_checked(len)?;
            Some((Value::String(GgufStr::new(s)), rest))
        }
        ValueType::Array => None,
    }
}
