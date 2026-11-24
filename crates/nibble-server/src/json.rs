//! JSON parsing and serialization: strict RFC 8259, `std` only.
//!
//! * [`parse`] turns text into a [`Json`] tree and reports errors with
//!   1-based line and column positions.
//! * [`Json`] implements `Display`, so `value.to_string()` writes compact
//!   JSON that round-trips through [`parse`].
//!
//! The parser makes a single pass over the input bytes, copies
//! escape-free strings in one slice, and tracks container nesting on an
//! explicit stack instead of the call stack, so no input can overflow the
//! stack. It never panics and never allocates from an untrusted size.

use std::fmt;
use std::fmt::Write;
use std::mem;

/// A JSON value. Objects keep their keys in document order.
///
/// Build values with the `From` impls, [`Json::object`] and [`Json::array`];
/// read them with the `as_*` accessors and [`Json::get`]; serialize with
/// `Display` (`value.to_string()` writes compact JSON).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    /// The `null` value.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number; see [`Number`].
    Number(Number),
    /// A string.
    String(String),
    /// An array, in document order.
    Array(Vec<Json>),
    /// An object: key/value pairs in document order. Duplicate keys are all
    /// kept in this list; [`Json::get`] returns the last one.
    Object(Vec<(String, Json)>),
}

/// A JSON number. Integers that fit are kept exact.
///
/// Floating-point values are always finite; non-finite floats can only be
/// introduced by constructing them in Rust, and serialize as `null`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Number {
    /// Integer literal that fits i64 (no fraction, no exponent).
    I64(i64),
    /// Integer literal above i64::MAX that fits u64.
    U64(u64),
    /// Everything else (finite only).
    F64(f64),
}

/// Why a document failed to parse.
///
/// This enum is `#[non_exhaustive]`: new variants may be added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum ErrorKind {
    /// The document ended while a value was still required.
    UnexpectedEof,
    /// A byte that cannot start (or continue) the expected token.
    UnexpectedCharacter,
    /// A backslash escape that is not one of JSON's escapes.
    InvalidEscape,
    /// A malformed `\uXXXX` escape or a lone surrogate.
    InvalidUnicodeEscape,
    /// A raw control character (U+0000..U+001F) inside a string.
    ControlCharacter,
    /// A number that does not match the JSON grammar.
    InvalidNumber,
    /// A number whose value overflows f64 to infinity.
    NumberOutOfRange,
    /// Extra non-whitespace input after the top-level value.
    TrailingCharacters,
    /// Arrays or objects nested deeper than the configured limit.
    DepthLimitExceeded,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            ErrorKind::UnexpectedEof => "unexpected end of input",
            ErrorKind::UnexpectedCharacter => "unexpected character",
            ErrorKind::InvalidEscape => "invalid escape",
            ErrorKind::InvalidUnicodeEscape => "invalid \\u escape",
            ErrorKind::ControlCharacter => "control character in string",
            ErrorKind::InvalidNumber => "invalid number",
            ErrorKind::NumberOutOfRange => "number out of range",
            ErrorKind::TrailingCharacters => "trailing characters",
            ErrorKind::DepthLimitExceeded => "nesting too deep",
        };
        f.write_str(text)
    }
}

/// A parse failure at a specific input position.
///
/// `line` and `column` are 1-based; the column counts characters since the
/// last line break (lines break at `\n`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParseError {
    /// 1-based line of the offending byte.
    pub(crate) line: usize,
    /// 1-based column (in characters) of the offending byte.
    pub(crate) column: usize,
    /// 0-based byte offset of the offending byte, or `input.len()` when the
    /// input ended too early.
    pub(crate) offset: usize,
    /// What went wrong.
    pub(crate) kind: ErrorKind,
}

impl fmt::Display for ParseError {
    /// Formats as `<reason> at line <l>, column <c>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ParseError {
            line, column, kind, ..
        } = *self;
        write!(f, "{kind} at line {line}, column {column}")
    }
}

impl std::error::Error for ParseError {}

/// The maximum array/object nesting depth accepted by [`parse`]: 128
/// levels. Deeper documents fail with [`ErrorKind::DepthLimitExceeded`].
pub(crate) const DEFAULT_MAX_DEPTH: usize = 128;

/// Parses a complete JSON document from `input`.
///
/// Grammar and limits follow RFC 8259 strictly: no comments, no trailing
/// commas, no leading zeros, no `NaN`/`Infinity`, at least one digit after
/// `.` and in exponents, and control characters must be escaped in
/// strings. Any value may appear at the top level; trailing whitespace is
/// allowed, and anything else after the value is
/// [`ErrorKind::TrailingCharacters`].
///
/// Container nesting is limited to [`DEFAULT_MAX_DEPTH`] levels; use
/// [`parse_with_depth`] for a different limit. The parser never panics and
/// never overflows the call stack: nesting is tracked on an explicit stack,
/// not by recursion.
///
/// # Errors
///
/// Returns a [`ParseError`] with 1-based line and column positions
/// describing the first invalid character.
///
/// # Examples
///
/// ```ignore
/// use nibble_json::json::{self, Json};
///
/// let value = json::parse(r#"{"a": [1, 2.5], "b": null}"#).unwrap();
/// assert_eq!(value.get("a").unwrap().as_array().unwrap().len(), 2);
/// assert_eq!(value.get("b"), Some(&Json::Null));
/// assert_eq!(value.to_string(), r#"{"a":[1,2.5],"b":null}"#);
/// ```
pub(crate) fn parse(input: &str) -> Result<Json, ParseError> {
    parse_with_depth(input, DEFAULT_MAX_DEPTH)
}

/// Parses a complete JSON document with a caller-chosen maximum nesting
/// depth for arrays and objects.
///
/// `max_depth` is the number of container levels allowed, so
/// `parse_with_depth(input, 0)` rejects every array and object while still
/// accepting scalars. The parser is iterative: any limit, however large,
/// costs heap space proportional to the real nesting, never call-stack
/// space.
///
/// # Errors
///
/// Returns a [`ParseError`] on invalid input; nesting past `max_depth`
/// gives [`ErrorKind::DepthLimitExceeded`].
///
/// # Examples
///
/// ```ignore
/// use nibble_json::json::{self, ErrorKind};
///
/// // 999 nested arrays fit within a limit of 1000.
/// let deep = format!("{}{}", "[".repeat(999), "]".repeat(999));
/// assert!(json::parse_with_depth(&deep, 1000).is_ok());
///
/// // One level past the limit is rejected.
/// let too_deep = format!("{}{}", "[".repeat(1001), "]".repeat(1001));
/// let err = json::parse_with_depth(&too_deep, 1000).unwrap_err();
/// assert_eq!(err.kind, ErrorKind::DepthLimitExceeded);
/// ```
pub(crate) fn parse_with_depth(input: &str, max_depth: usize) -> Result<Json, ParseError> {
    let mut p = Parser::new(input);
    let mut stack: Vec<Frame> = Vec::new();
    let mut state = State::Value;

    // Each iteration consumes at least one byte, errors out, or completes a
    // value, so the loop always terminates.
    let value = loop {
        let flow = match state {
            State::Value => p.value(&mut stack, max_depth)?,
            State::Sep => p.sep(&mut stack)?,
            State::Key => p.key(&mut stack)?,
            State::Colon => p.colon()?,
        };
        match flow {
            Flow::Continue(next) => state = next,
            Flow::Done(v) => match stack.last_mut() {
                Some(Frame::Array { items }) => {
                    items.push(v);
                    state = State::Sep;
                }
                Some(Frame::Object { pairs, key }) => {
                    pairs.push((mem::take(key), v));
                    state = State::Sep;
                }
                // The stack is empty: `v` is the complete top-level value.
                None => break v,
            },
        }
    };

    // Whitespace may follow the value; anything else is an error.
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(p.error(p.pos, ErrorKind::TrailingCharacters));
    }
    Ok(value)
}

/// A container currently being parsed.
enum Frame {
    /// An array whose elements are still being collected.
    Array {
        /// Elements parsed so far.
        items: Vec<Json>,
    },
    /// An object whose members are still being collected.
    Object {
        /// Members parsed so far, in document order.
        pairs: Vec<(String, Json)>,
        /// Key of the member whose value is currently being parsed (empty
        /// between members).
        key: String,
    },
}

/// What the parser expects at the cursor.
enum State {
    /// A value: any scalar token or the start of a container.
    Value,
    /// After a container element: `,` or the matching closer.
    Sep,
    /// After `{` or `,` inside an object: a quoted key.
    Key,
    /// After an object key: `:`.
    Colon,
}

/// The result of one parser step.
enum Flow {
    /// The step succeeded; the parser next expects the given state.
    Continue(State),
    /// A value completed and must be delivered to its parent container (or,
    /// if the stack is empty, finishes the document).
    Done(Json),
}

/// A single-pass cursor over the input.
struct Parser<'a> {
    /// The whole input, for slicing out lexemes and strings.
    input: &'a str,
    /// `input` as bytes, for cheap single-byte lookahead.
    bytes: &'a [u8],
    /// Current byte offset; always a char boundary, at most `bytes.len()`.
    pos: usize,
}

impl Parser<'_> {
    fn new(input: &str) -> Parser<'_> {
        Parser {
            input,
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    /// The byte at the cursor, if any.
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Skips JSON whitespace: space, tab, LF, CR.
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Builds a `ParseError` at byte `offset`, computing the 1-based line
    /// and column. Only called once per failed parse, so the scan is cheap.
    fn error(&self, offset: usize, kind: ErrorKind) -> ParseError {
        let (line, column) = line_column(self.input, offset);
        ParseError {
            line,
            column,
            offset,
            kind,
        }
    }

    /// Handles the state where a value is required.
    fn value(&mut self, stack: &mut Vec<Frame>, max_depth: usize) -> Result<Flow, ParseError> {
        self.skip_ws();
        match self.peek() {
            None => Err(self.error(self.pos, ErrorKind::UnexpectedEof)),
            Some(b'[') => self.open_array(stack, max_depth),
            Some(b'{') => self.open_object(stack, max_depth),
            Some(b'"') => Ok(Flow::Done(Json::String(self.parse_string()?))),
            Some(b't') => {
                self.parse_literal(b"true")?;
                Ok(Flow::Done(Json::Bool(true)))
            }
            Some(b'f') => {
                self.parse_literal(b"false")?;
                Ok(Flow::Done(Json::Bool(false)))
            }
            Some(b'n') => {
                self.parse_literal(b"null")?;
                Ok(Flow::Done(Json::Null))
            }
            Some(b'-') | Some(b'0'..=b'9') => Ok(Flow::Done(self.parse_number()?)),
            Some(_) => Err(self.error(self.pos, ErrorKind::UnexpectedCharacter)),
        }
    }

    /// Handles `[`: an empty array completes immediately, otherwise a new
    /// frame opens for its elements.
    fn open_array(&mut self, stack: &mut Vec<Frame>, max_depth: usize) -> Result<Flow, ParseError> {
        if stack.len() >= max_depth {
            return Err(self.error(self.pos, ErrorKind::DepthLimitExceeded));
        }
        self.pos += 1;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Flow::Done(Json::Array(Vec::new())));
        }
        stack.push(Frame::Array { items: Vec::new() });
        Ok(Flow::Continue(State::Value))
    }

    /// Handles `{`: an empty object completes immediately, otherwise a new
    /// frame opens and a key is required.
    fn open_object(
        &mut self,
        stack: &mut Vec<Frame>,
        max_depth: usize,
    ) -> Result<Flow, ParseError> {
        if stack.len() >= max_depth {
            return Err(self.error(self.pos, ErrorKind::DepthLimitExceeded));
        }
        self.pos += 1;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Flow::Done(Json::Object(Vec::new())));
        }
        stack.push(Frame::Object {
            pairs: Vec::new(),
            key: String::new(),
        });
        Ok(Flow::Continue(State::Key))
    }

    /// Handles the state after a container element: `,` continues, the
    /// matching closer completes the container.
    fn sep(&mut self, stack: &mut Vec<Frame>) -> Result<Flow, ParseError> {
        self.skip_ws();
        match (self.peek(), stack.last()) {
            (None, _) => Err(self.error(self.pos, ErrorKind::UnexpectedEof)),
            (Some(b','), Some(Frame::Array { .. })) => {
                self.pos += 1;
                Ok(Flow::Continue(State::Value))
            }
            (Some(b','), Some(Frame::Object { .. })) => {
                self.pos += 1;
                Ok(Flow::Continue(State::Key))
            }
            (Some(b']'), Some(Frame::Array { .. })) => {
                self.pos += 1;
                match stack.pop() {
                    Some(Frame::Array { items }) => Ok(Flow::Done(Json::Array(items))),
                    _ => Err(self.error(self.pos, ErrorKind::UnexpectedCharacter)),
                }
            }
            (Some(b'}'), Some(Frame::Object { .. })) => {
                self.pos += 1;
                match stack.pop() {
                    Some(Frame::Object { pairs, .. }) => Ok(Flow::Done(Json::Object(pairs))),
                    _ => Err(self.error(self.pos, ErrorKind::UnexpectedCharacter)),
                }
            }
            // A mismatched closer (or anything else) is not a separator.
            (Some(_), _) => Err(self.error(self.pos, ErrorKind::UnexpectedCharacter)),
        }
    }

    /// Handles the object-key state: a quoted string is required, so
    /// trailing commas (`{"a":1,}`) are rejected here.
    fn key(&mut self, stack: &mut [Frame]) -> Result<Flow, ParseError> {
        self.skip_ws();
        match self.peek() {
            None => Err(self.error(self.pos, ErrorKind::UnexpectedEof)),
            Some(b'"') => {
                let name = self.parse_string()?;
                match stack.last_mut() {
                    Some(Frame::Object { key, .. }) => {
                        *key = name;
                        Ok(Flow::Continue(State::Colon))
                    }
                    // Keys are only parsed while an object frame is open.
                    _ => Err(self.error(self.pos, ErrorKind::UnexpectedCharacter)),
                }
            }
            Some(_) => Err(self.error(self.pos, ErrorKind::UnexpectedCharacter)),
        }
    }

    /// Handles the state after an object key: `:` is required.
    fn colon(&mut self) -> Result<Flow, ParseError> {
        self.skip_ws();
        match self.peek() {
            None => Err(self.error(self.pos, ErrorKind::UnexpectedEof)),
            Some(b':') => {
                self.pos += 1;
                Ok(Flow::Continue(State::Value))
            }
            Some(_) => Err(self.error(self.pos, ErrorKind::UnexpectedCharacter)),
        }
    }

    /// Consumes the exact bytes of a keyword literal such as `true`.
    ///
    /// A truncated keyword reports an unexpected end of input; any other
    /// byte reports an unexpected character at the first mismatch.
    fn parse_literal(&mut self, lit: &[u8]) -> Result<(), ParseError> {
        let rem = self.bytes.len() - self.pos;
        if rem < lit.len() {
            self.pos = self.bytes.len();
            return Err(self.error(self.pos, ErrorKind::UnexpectedEof));
        }
        let got = &self.bytes[self.pos..self.pos + lit.len()];
        if got != lit {
            let mismatch = got.iter().zip(lit).position(|(a, b)| a != b);
            let at = self.pos + mismatch.unwrap_or(0);
            return Err(self.error(at, ErrorKind::UnexpectedCharacter));
        }
        self.pos += lit.len();
        Ok(())
    }

    /// Parses a string literal; the cursor must be at the opening quote.
    ///
    /// Returns the decoded contents. Raw control characters (U+0000 through
    /// U+001F) are rejected, every JSON escape is decoded, and lone
    /// surrogates are an error. Runs without escapes are copied in one
    /// slice; the scan only ever stops at ASCII bytes, so every slice taken
    /// is bounded by char boundaries and cannot panic.
    fn parse_string(&mut self) -> Result<String, ParseError> {
        // Skip the opening quote.
        self.pos += 1;
        let mut out = String::new();
        let mut plain = self.pos;
        loop {
            match self.peek() {
                None => return Err(self.error(self.pos, ErrorKind::UnexpectedEof)),
                Some(b'"') => {
                    // Fast path: one copy for the whole plain run.
                    out.push_str(&self.input[plain..self.pos]);
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    out.push_str(&self.input[plain..self.pos]);
                    self.pos += 1;
                    match self.peek() {
                        None => return Err(self.error(self.pos, ErrorKind::UnexpectedEof)),
                        Some(b'u') => {
                            let esc = self.pos - 1;
                            self.pos += 1;
                            self.parse_unicode_escape(&mut out, esc)?;
                        }
                        Some(b'"') => {
                            out.push('"');
                            self.pos += 1;
                        }
                        Some(b'\\') => {
                            out.push('\\');
                            self.pos += 1;
                        }
                        Some(b'/') => {
                            out.push('/');
                            self.pos += 1;
                        }
                        Some(b'b') => {
                            out.push('\u{8}');
                            self.pos += 1;
                        }
                        Some(b'f') => {
                            out.push('\u{c}');
                            self.pos += 1;
                        }
                        Some(b'n') => {
                            out.push('\n');
                            self.pos += 1;
                        }
                        Some(b'r') => {
                            out.push('\r');
                            self.pos += 1;
                        }
                        Some(b't') => {
                            out.push('\t');
                            self.pos += 1;
                        }
                        Some(_) => return Err(self.error(self.pos, ErrorKind::InvalidEscape)),
                    }
                    plain = self.pos;
                }
                Some(b) if b < 0x20 => {
                    return Err(self.error(self.pos, ErrorKind::ControlCharacter));
                }
                // Ordinary bytes, including UTF-8 continuation bytes.
                Some(_) => self.pos += 1,
            }
        }
    }

    /// Decodes a `\uXXXX` escape whose backslash is at byte `esc`; the
    /// cursor is at the first hex digit. High surrogates must be followed by
    /// a matching low surrogate; any other surrogate is an error.
    fn parse_unicode_escape(&mut self, out: &mut String, esc: usize) -> Result<(), ParseError> {
        let first = self.hex4()?;
        // `char::from_u32` accepts every BMP code point except surrogates.
        if let Some(c) = char::from_u32(first) {
            out.push(c);
            return Ok(());
        }
        if (0xDC00..=0xDFFF).contains(&first) {
            // A lone low surrogate.
            return Err(self.error(esc, ErrorKind::InvalidUnicodeEscape));
        }
        // High surrogate: the escape must continue with "\u" + low surrogate.
        let second_esc = self.pos;
        if self.peek() == Some(b'\\') && self.bytes.get(self.pos + 1).copied() == Some(b'u') {
            self.pos += 2;
            let second = self.hex4()?;
            if (0xDC00..=0xDFFF).contains(&second) {
                let c = 0x1_0000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                out.push(char::from_u32(c).unwrap_or(char::REPLACEMENT_CHARACTER));
                Ok(())
            } else {
                // e.g. "\ud800\ud800": the second escape is not a low surrogate.
                Err(self.error(second_esc, ErrorKind::InvalidUnicodeEscape))
            }
        } else {
            Err(self.error(esc, ErrorKind::InvalidUnicodeEscape))
        }
    }

    /// Reads exactly four hex digits; the cursor fails (at its position) on
    /// the first byte that is not one.
    fn hex4(&mut self) -> Result<u32, ParseError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let b = match self.peek() {
                Some(b) => b,
                None => return Err(self.error(self.pos, ErrorKind::InvalidUnicodeEscape)),
            };
            let d = match b {
                b'0'..=b'9' => u32::from(b - b'0'),
                b'a'..=b'f' => u32::from(b - b'a') + 10,
                b'A'..=b'F' => u32::from(b - b'A') + 10,
                _ => return Err(self.error(self.pos, ErrorKind::InvalidUnicodeEscape)),
            };
            v = (v << 4) | d;
            self.pos += 1;
        }
        Ok(v)
    }

    /// Parses a number literal; the cursor is at the leading `-` or digit.
    ///
    /// The lexeme is validated against the JSON grammar as it is scanned,
    /// then interpreted: integers without a fraction or exponent become
    /// `I64` (or `U64` for values above i64::MAX) when they fit, and
    /// everything else goes through `str::parse::<f64>`, which is correctly
    /// rounded. Overflow to infinity is an error.
    fn parse_number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        let mut float = false;

        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    // JSON forbids leading zeros: "01" is invalid.
                    return Err(self.error(self.pos, ErrorKind::InvalidNumber));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            // A `-` with no digits behind it (or any other malformed start).
            _ => return Err(self.error(self.pos, ErrorKind::InvalidNumber)),
        }
        if self.peek() == Some(b'.') {
            float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error(self.pos, ErrorKind::InvalidNumber));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error(self.pos, ErrorKind::InvalidNumber));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        let lexeme = &self.input[start..self.pos];
        if !float {
            // Integer lexeme: keep it exact when it fits.
            if let Ok(v) = lexeme.parse::<i64>() {
                return Ok(Json::Number(Number::I64(v)));
            }
            // (Only non-negative values can reach the u64 branch.)
            if let Ok(v) = lexeme.parse::<u64>() {
                return Ok(Json::Number(Number::U64(v)));
            }
        }
        // Floats, and integers too large for u64, go through f64.
        let v: f64 = lexeme
            .parse()
            .map_err(|_| self.error(start, ErrorKind::InvalidNumber))?;
        if !v.is_finite() {
            return Err(self.error(start, ErrorKind::NumberOutOfRange));
        }
        Ok(Json::Number(Number::F64(v)))
    }
}

/// Returns the 1-based line and 1-based character column of byte `offset`.
///
/// Lines break at `\n`; every other character, including `\r`, counts as
/// one column.
fn line_column(input: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut column = 0;
    for (i, c) in input.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            column = 0;
        } else {
            column += 1;
        }
    }
    (line, column + 1)
}

impl Json {
    /// Builds an object from `(key, value)` pairs, in iteration order.
    /// Later pairs win over earlier ones for [`Json::get`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use nibble_json::json::Json;
    ///
    /// let value = Json::object([
    ///     ("id", Json::from(7)),
    ///     ("ids", Json::array([Json::from(1), Json::from(2)])),
    /// ]);
    /// assert_eq!(value.to_string(), r#"{"id":7,"ids":[1,2]}"#);
    /// ```
    pub(crate) fn object<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Json)>) -> Json {
        Json::Object(
            pairs
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    /// Builds an array from `items`, in iteration order.
    pub(crate) fn array(items: impl IntoIterator<Item = Json>) -> Json {
        Json::Array(items.into_iter().collect())
    }

    /// Looks up `key` in an object and returns the value. With duplicate
    /// keys, the last one wins; any other kind of value, or a missing key,
    /// gives `None`.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string contents, if this is a string.
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// The boolean, if this is a bool.
    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Any number as `f64` (integers outside f64's exact range lose
    /// precision); `None` for values that are not numbers.
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => match *n {
                Number::I64(v) => Some(v as f64),
                Number::U64(v) => Some(v as f64),
                Number::F64(v) => Some(v),
            },
            _ => None,
        }
    }

    /// The number as `i64`, if it is an `I64`, a `U64` that fits, or an
    /// integral `F64` that fits.
    pub(crate) fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Number(n) => match *n {
                Number::I64(v) => Some(v),
                Number::U64(v) => i64::try_from(v).ok(),
                Number::F64(v) => f64_to_i64(v),
            },
            _ => None,
        }
    }

    /// The number as `u64`, if it is a `U64`, an `I64` that fits, or an
    /// integral `F64` that fits.
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(n) => match *n {
                Number::I64(v) => u64::try_from(v).ok(),
                Number::U64(v) => Some(v),
                Number::F64(v) => f64_to_u64(v),
            },
            _ => None,
        }
    }

    /// The array elements, if this is an array.
    pub(crate) fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The object's key/value pairs in document order, if this is an object.
    pub(crate) fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(pairs) => Some(pairs),
            _ => None,
        }
    }

    /// Whether this is `null`.
    pub(crate) fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

/// Converts `v` to `i64` if it is an integer that fits exactly.
///
/// The upper bound is checked against 2^63 (not `i64::MAX as f64`, which
/// rounds up to 2^63) so the cast can never saturate.
fn f64_to_i64(v: f64) -> Option<i64> {
    if v.is_finite()
        && v.fract() == 0.0
        && (i64::MIN as f64) <= v
        && v < 9_223_372_036_854_775_808.0
    {
        Some(v as i64)
    } else {
        None
    }
}

/// Converts `v` to `u64` if it is an integer that fits exactly (checked
/// against 2^64, the first value that does not fit).
fn f64_to_u64(v: f64) -> Option<u64> {
    if v.is_finite() && v.fract() == 0.0 && (0.0..18_446_744_073_709_551_616.0).contains(&v) {
        Some(v as u64)
    } else {
        None
    }
}

impl fmt::Display for Number {
    /// Writes JSON: exact integers, shortest round-trip floats, `null` for
    /// non-finite values.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Number::I64(v) => write!(f, "{v}"),
            Number::U64(v) => write!(f, "{v}"),
            Number::F64(v) => {
                if v.is_finite() {
                    // `Debug` is Rust's shortest round-trip form ("1.0",
                    // "1e-7", "1e300"), always valid JSON for finite values.
                    write!(f, "{v:?}")
                } else {
                    // NaN and infinities are not representable in JSON.
                    f.write_str("null")
                }
            }
        }
    }
}

impl fmt::Display for Json {
    /// Writes compact JSON: no spaces between tokens.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Json::Null => f.write_str("null"),
            Json::Bool(b) => f.write_str(if *b { "true" } else { "false" }),
            Json::Number(n) => write!(f, "{n}"),
            Json::String(s) => {
                f.write_char('"')?;
                escape_into(f, s)?;
                f.write_char('"')
            }
            Json::Array(items) => {
                f.write_char('[')?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_char(']')
            }
            Json::Object(pairs) => {
                f.write_char('{')?;
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    f.write_char('"')?;
                    escape_into(f, key)?;
                    f.write_str("\":")?;
                    write!(f, "{value}")?;
                }
                f.write_char('}')
            }
        }
    }
}

/// The short JSON escape for `c`, if it has one.
fn short_escape(c: char) -> Option<&'static str> {
    match c {
        '"' => Some("\\\""),
        '\\' => Some("\\\\"),
        '\u{8}' => Some("\\b"),
        '\u{c}' => Some("\\f"),
        '\n' => Some("\\n"),
        '\r' => Some("\\r"),
        '\t' => Some("\\t"),
        _ => None,
    }
}

/// Writes `s` (without surrounding quotes) into `w`, escaped per the JSON
/// string rules: runs of safe characters are copied in one piece.
fn escape_into<W: fmt::Write>(w: &mut W, s: &str) -> fmt::Result {
    let mut plain = 0;
    for (i, c) in s.char_indices() {
        if let Some(short) = short_escape(c) {
            w.write_str(&s[plain..i])?;
            w.write_str(short)?;
            plain = i + c.len_utf8();
        } else if (c as u32) < 0x20 {
            w.write_str(&s[plain..i])?;
            let hex = c as u32;
            write!(w, "\\u{hex:04x}")?;
            plain = i + c.len_utf8();
        }
    }
    w.write_str(&s[plain..])
}

impl From<bool> for Json {
    fn from(v: bool) -> Json {
        Json::Bool(v)
    }
}

impl From<i32> for Json {
    fn from(v: i32) -> Json {
        Json::Number(Number::I64(i64::from(v)))
    }
}

impl From<i64> for Json {
    fn from(v: i64) -> Json {
        Json::Number(Number::I64(v))
    }
}

impl From<u32> for Json {
    fn from(v: u32) -> Json {
        Json::Number(Number::I64(i64::from(v)))
    }
}

impl From<u64> for Json {
    fn from(v: u64) -> Json {
        // Small values become I64, mirroring what parsing produces.
        match i64::try_from(v) {
            Ok(small) => Json::Number(Number::I64(small)),
            Err(_) => Json::Number(Number::U64(v)),
        }
    }
}

impl From<usize> for Json {
    fn from(v: usize) -> Json {
        Json::from(v as u64)
    }
}

impl From<f32> for Json {
    fn from(v: f32) -> Json {
        Json::Number(Number::F64(f64::from(v)))
    }
}

impl From<f64> for Json {
    fn from(v: f64) -> Json {
        Json::Number(Number::F64(v))
    }
}

impl From<&str> for Json {
    fn from(v: &str) -> Json {
        Json::String(v.to_owned())
    }
}

impl From<String> for Json {
    fn from(v: String) -> Json {
        Json::String(v)
    }
}

impl From<Vec<Json>> for Json {
    fn from(v: Vec<Json>) -> Json {
        Json::Array(v)
    }
}

impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Json {
        match v {
            None => Json::Null,
            Some(t) => t.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    /// Parses `input`, panicking with the error message on failure.
    fn ok(input: &str) -> Json {
        parse(input).unwrap_or_else(|e| panic!("parse({input:?}) failed: {e}"))
    }

    /// Parses `input`, requiring an `I64`.
    fn i64_of(input: &str) -> i64 {
        match ok(input) {
            Json::Number(Number::I64(v)) => v,
            other => panic!("expected I64 from {input:?}, got {other:?}"),
        }
    }

    /// Parses `input`, requiring a `U64`.
    fn u64_of(input: &str) -> u64 {
        match ok(input) {
            Json::Number(Number::U64(v)) => v,
            other => panic!("expected U64 from {input:?}, got {other:?}"),
        }
    }

    /// Parses `input`, requiring an `F64`.
    fn f64_of(input: &str) -> f64 {
        match ok(input) {
            Json::Number(Number::F64(v)) => v,
            other => panic!("expected F64 from {input:?}, got {other:?}"),
        }
    }

    #[test]
    fn parse_scalars_top_level() {
        assert_eq!(ok("null"), Json::Null);
        assert_eq!(ok("true"), Json::Bool(true));
        assert_eq!(ok("false"), Json::Bool(false));
        assert_eq!(ok("  null  "), Json::Null);
        assert_eq!(ok("\t[\n1\r]\n"), Json::array([Json::from(1)]));
        assert_eq!(ok("\"x\""), Json::from("x"));
        assert_eq!(ok("1"), Json::Number(Number::I64(1)));
    }

    #[test]
    fn parse_integers() {
        assert_eq!(i64_of("0"), 0);
        // "-0" has no fraction or exponent, so it stays an exact integer.
        assert_eq!(i64_of("-0"), 0);
        assert_eq!(i64_of("42"), 42);
        assert_eq!(i64_of("-42"), -42);
        assert_eq!(i64_of("9223372036854775807"), i64::MAX);
        assert_eq!(i64_of("-9223372036854775808"), i64::MIN);
        // Whitespace is allowed everywhere a value may appear.
        assert_eq!(ok(" -7 ").as_i64(), Some(-7));
        assert_eq!(ok("[ -7 ]").as_array().unwrap()[0].as_i64(), Some(-7));
    }

    #[test]
    fn parse_large_integers() {
        // Above i64::MAX but within u64.
        assert_eq!(
            ok("9223372036854775808"),
            Json::Number(Number::U64(9223372036854775808))
        );
        assert_eq!(u64_of("18446744073709551615"), u64::MAX);
        // u64::MAX + 1 (and anything below i64::MIN) becomes an F64.
        assert_eq!(f64_of("18446744073709551616"), 18446744073709551616.0);
        assert_eq!(f64_of("-9223372036854775809"), -9223372036854775809.0);
    }

    #[test]
    fn parse_floats() {
        assert_eq!(f64_of("0.1"), 0.1);
        let neg_zero = f64_of("-0.0");
        assert_eq!(neg_zero, 0.0);
        assert!(neg_zero.is_sign_negative(), "-0.0 must stay negative zero");
        assert_eq!(f64_of("1e308"), 1e308);
        assert_eq!(f64_of("1E5"), 1e5); // uppercase exponent marker
        assert_eq!(f64_of("1e+5"), 1e5); // explicit plus sign
        assert_eq!(f64_of("5e-324"), 5e-324); // smallest subnormal
        assert_eq!(f64_of("1.5e-3"), 0.0015);
        // Exponent or fraction makes the value a float, even when integral.
        assert!(matches!(ok("2.0"), Json::Number(Number::F64(_))));
        assert!(matches!(ok("2e0"), Json::Number(Number::F64(_))));
    }

    #[test]
    fn parse_arrays() {
        assert_eq!(ok("[]"), Json::Array(Vec::new()));
        assert_eq!(ok("[1]"), Json::array([Json::from(1)]));
        assert_eq!(
            ok("[1, 2, 3]"),
            Json::array([Json::from(1), Json::from(2), Json::from(3)])
        );
        assert_eq!(
            ok(" [ 1 , 2 ] "),
            Json::array([Json::from(1), Json::from(2)])
        );
        let expected = Json::array([
            Json::array([Json::from(1)]),
            Json::array([Json::from(2), Json::array([Json::from(3)])]),
        ]);
        assert_eq!(ok("[[1],[2,[3]]]"), expected);
        assert_eq!(
            ok("[1,\"two\",null,true,[]]"),
            Json::array([
                Json::from(1),
                Json::from("two"),
                Json::Null,
                Json::Bool(true),
                Json::Array(Vec::new()),
            ])
        );
    }

    #[test]
    fn parse_objects() {
        assert_eq!(ok("{}"), Json::Object(Vec::new()));
        assert_eq!(ok(r#"{"a":1}"#), Json::object([("a", Json::from(1))]));
        assert_eq!(
            ok(r#"{ "a" : 1 , "b" : false }"#),
            Json::object([("a", Json::from(1)), ("b", Json::Bool(false))])
        );
        // Keys keep document order.
        let v = ok(r#"{"b":1,"a":2}"#);
        let pairs = v.as_object().unwrap();
        assert_eq!((pairs[0].0.as_str(), pairs[1].0.as_str()), ("b", "a"));
        // Empty key.
        assert_eq!(ok(r#"{"":0}"#).get(""), Some(&Json::Number(Number::I64(0))));
        // Nested containers.
        let doc = r#"{"a":{"b":[1,{"c":null}]}}"#;
        assert_eq!(ok(doc).to_string(), doc);
    }

    #[test]
    fn parse_kitchen_sink() {
        let doc = r#"{
            "name": "nibble",
            "tags": ["fast", "strict"],
            "pi": 3.7,
            "ok": true,
            "nothing": null,
            "counts": {"a": 1, "a": 2},
            "emoji": "😀"
        }"#;
        let v = ok(doc);
        assert_eq!(v.get("name").and_then(Json::as_str), Some("nibble"));
        assert_eq!(
            v.get("tags").unwrap().as_array().unwrap()[0].as_str(),
            Some("fast")
        );
        assert_eq!(v.get("tags").unwrap().as_array().unwrap().len(), 2);
        assert_eq!(v.get("pi").and_then(Json::as_f64), Some(3.7));
        assert_eq!(v.get("ok").and_then(Json::as_bool), Some(true));
        assert!(v.get("nothing").unwrap().is_null());
        // Duplicate keys: the last value wins for `get`, all pairs are kept.
        assert_eq!(v.get("counts").unwrap().as_object().unwrap().len(), 2);
        assert_eq!(
            v.get("counts").unwrap().get("a"),
            Some(&Json::Number(Number::I64(2)))
        );
        assert_eq!(v.get("emoji").and_then(Json::as_str), Some("😀"));
    }

    #[test]
    fn parse_all_escapes() {
        assert_eq!(ok(r#""""#), Json::String(String::new()));
        assert_eq!(ok(r#""A""#), Json::from("A"));
        assert_eq!(ok(r#""\"""#), Json::from("\""));
        assert_eq!(ok(r#""\\""#), Json::from("\\"));
        assert_eq!(ok(r#""\/""#), Json::from("/"));
        assert_eq!(ok(r#""\b""#), Json::from("\u{8}"));
        assert_eq!(ok(r#""\f""#), Json::from("\u{c}"));
        assert_eq!(ok(r#""\n""#), Json::from("\n"));
        assert_eq!(ok(r#""\r""#), Json::from("\r"));
        assert_eq!(ok(r#""\t""#), Json::from("\t"));
        assert_eq!(ok(r#""\u0041""#), Json::from("A"));
        assert_eq!(ok(r#""\u00e9""#), Json::from("é"));
        assert_eq!(ok(r#""\uFFFF""#), Json::from("\u{ffff}"));
        // Mixed-case hex digits.
        assert_eq!(ok(r#""\uAbCd""#), Json::from("\u{abcd}"));
        // Several escapes inside one string.
        assert_eq!(ok(r#""a\tb\nc""#), Json::from("a\tb\nc"));
        // Raw non-ASCII passes through untouched.
        assert_eq!(ok(r#""héllo""#), Json::from("héllo"));
    }

    #[test]
    fn parse_surrogate_pairs() {
        assert_eq!(ok(r#""😀""#), Json::from("😀"));
        assert_eq!(ok(r#""\ud83d\ude00""#), Json::from("😀"));
        assert_eq!(ok(r#""\uD83D\uDE00""#), Json::from("😀"));
        // U+10000, the first supplementary-plane character.
        assert_eq!(ok(r#""\ud800\udc00""#), Json::from("\u{10000}"));
        // A pair in the middle of other text.
        assert_eq!(ok(r#""x\ud83d\ude03y""#), Json::from("x😃y"));
    }

    /// Structural errors: unterminated values, separators and garbage.
    const STRUCTURAL: &[(&str, ErrorKind)] = &[
        // No value at all.
        ("", ErrorKind::UnexpectedEof),
        ("   ", ErrorKind::UnexpectedEof),
        // Unterminated containers.
        ("[", ErrorKind::UnexpectedEof),
        ("[1", ErrorKind::UnexpectedEof),
        ("[1,", ErrorKind::UnexpectedEof),
        ("{", ErrorKind::UnexpectedEof),
        ("{\"a\"", ErrorKind::UnexpectedEof),
        ("{\"a\":", ErrorKind::UnexpectedEof),
        ("{\"a\":1", ErrorKind::UnexpectedEof),
        // Unterminated strings.
        ("\"abc", ErrorKind::UnexpectedEof),
        ("\"abc\\\"", ErrorKind::UnexpectedEof),
        // Truncated keywords.
        ("tru", ErrorKind::UnexpectedEof),
        ("fals", ErrorKind::UnexpectedEof),
        ("nul", ErrorKind::UnexpectedEof),
        // Trailing commas.
        ("[1,]", ErrorKind::UnexpectedCharacter),
        ("{\"a\":1,}", ErrorKind::UnexpectedCharacter),
        // Missing or wrong separators.
        ("[1 2]", ErrorKind::UnexpectedCharacter),
        ("{\"a\" 1}", ErrorKind::UnexpectedCharacter),
        ("{\"a\":}", ErrorKind::UnexpectedCharacter),
        ("{\"a\"}", ErrorKind::UnexpectedCharacter),
        ("{1:2}", ErrorKind::UnexpectedCharacter),
        ("{:1}", ErrorKind::UnexpectedCharacter),
        ("{\"a\":1 \"b\":2}", ErrorKind::UnexpectedCharacter),
        // Wrong or unmatched closers.
        ("[,]", ErrorKind::UnexpectedCharacter),
        ("[}", ErrorKind::UnexpectedCharacter),
        ("{]", ErrorKind::UnexpectedCharacter),
        ("[1}", ErrorKind::UnexpectedCharacter),
        ("{\"a\":1]", ErrorKind::UnexpectedCharacter),
        // Bad value starts.
        (".5", ErrorKind::UnexpectedCharacter),
        ("+1", ErrorKind::UnexpectedCharacter),
        ("NaN", ErrorKind::UnexpectedCharacter),
        ("Infinity", ErrorKind::UnexpectedCharacter),
        ("'a'", ErrorKind::UnexpectedCharacter),
        ("truX", ErrorKind::UnexpectedCharacter),
        // Trailing garbage after a complete value.
        ("1 2", ErrorKind::TrailingCharacters),
        ("[1] x", ErrorKind::TrailingCharacters),
        ("1.2.3", ErrorKind::TrailingCharacters),
        ("[1,2]]", ErrorKind::TrailingCharacters),
        ("nulll", ErrorKind::TrailingCharacters),
        ("1,2", ErrorKind::TrailingCharacters),
    ];

    #[test]
    fn invalid_structure() {
        for (input, kind) in STRUCTURAL {
            match parse(input) {
                Ok(v) => panic!("expected an error for {input:?}, got {v:?}"),
                Err(e) => assert_eq!(e.kind, *kind, "wrong kind for {input:?}"),
            }
        }
    }

    /// String errors: bad escapes, surrogates and raw control characters.
    const STRING_ERRORS: &[(&str, ErrorKind)] = &[
        ("\"\\x\"", ErrorKind::InvalidEscape),
        ("\"\\q\"", ErrorKind::InvalidEscape),
        ("\"\\u12\"", ErrorKind::InvalidUnicodeEscape),
        ("\"\\uZZZZ\"", ErrorKind::InvalidUnicodeEscape),
        // Lone high surrogate.
        ("\"\\ud800\"", ErrorKind::InvalidUnicodeEscape),
        // Lone low surrogate.
        ("\"\\udc00\"", ErrorKind::InvalidUnicodeEscape),
        // High surrogate followed by another high surrogate.
        ("\"\\ud800\\ud800\"", ErrorKind::InvalidUnicodeEscape),
        // High surrogate with no second escape.
        ("\"\\ud800x\"", ErrorKind::InvalidUnicodeEscape),
        // Raw control characters in strings.
        ("\"a\nb\"", ErrorKind::ControlCharacter),
        ("\"a\tb\"", ErrorKind::ControlCharacter),
        ("\"a\u{0}b\"", ErrorKind::ControlCharacter),
    ];

    #[test]
    fn invalid_strings() {
        for (input, kind) in STRING_ERRORS {
            match parse(input) {
                Ok(v) => panic!("expected an error for {input:?}, got {v:?}"),
                Err(e) => assert_eq!(e.kind, *kind, "wrong kind for {input:?}"),
            }
        }
    }

    /// Number errors: grammar violations and range violations.
    const NUMBER_ERRORS: &[(&str, ErrorKind)] = &[
        // Leading zeros.
        ("01", ErrorKind::InvalidNumber),
        ("00", ErrorKind::InvalidNumber),
        ("-01", ErrorKind::InvalidNumber),
        // Missing fraction digits.
        ("1.", ErrorKind::InvalidNumber),
        ("1.e5", ErrorKind::InvalidNumber),
        // Missing exponent digits.
        ("1e", ErrorKind::InvalidNumber),
        ("1e+", ErrorKind::InvalidNumber),
        ("1ex", ErrorKind::InvalidNumber),
        // Missing integer digits.
        ("-", ErrorKind::InvalidNumber),
        ("- 1", ErrorKind::InvalidNumber),
        ("-x", ErrorKind::InvalidNumber),
        // Overflow to infinity.
        ("1e400", ErrorKind::NumberOutOfRange),
        ("-1e400", ErrorKind::NumberOutOfRange),
    ];

    #[test]
    fn invalid_numbers() {
        for (input, kind) in NUMBER_ERRORS {
            match parse(input) {
                Ok(v) => panic!("expected an error for {input:?}, got {v:?}"),
                Err(e) => assert_eq!(e.kind, *kind, "wrong kind for {input:?}"),
            }
        }
    }

    #[test]
    fn number_out_of_range() {
        // A 401-digit integer also overflows f64 to infinity.
        let huge = format!("1{}", "0".repeat(400));
        assert_eq!(parse(&huge).unwrap_err().kind, ErrorKind::NumberOutOfRange);
        // 1e308 is well within range.
        assert!(matches!(ok("1e308"), Json::Number(Number::F64(v)) if v == 1e308));
    }

    #[test]
    fn error_positions() {
        // Missing ':' after a key, on line 2.
        let err = parse("{\n  \"a\" 1\n}").unwrap_err();
        assert_eq!(
            (err.line, err.column, err.offset, err.kind),
            (2, 7, 8, ErrorKind::UnexpectedCharacter)
        );
        assert_eq!(err.to_string(), "unexpected character at line 2, column 7");

        // Line 3, columns counted in characters.
        let err = parse("{\n\"a\":1,\n\"b\" 2\n}").unwrap_err();
        assert_eq!((err.line, err.column, err.offset), (3, 5, 13));

        // Columns count characters, not bytes: "é" is two bytes, one column.
        let err = parse("{\n\"é\" 1}").unwrap_err();
        assert_eq!((err.line, err.column, err.offset), (2, 5, 7));

        // End-of-input positions.
        let err = parse("\"abc").unwrap_err();
        assert_eq!((err.line, err.column, err.offset), (1, 5, 4));
        let err = parse("[1,\n").unwrap_err();
        assert_eq!((err.line, err.column, err.offset), (2, 1, 4));

        // CRLF: lines break at '\n'; the '\r' counts as a column character.
        let err = parse("{\r\n\"a\" 1}").unwrap_err();
        assert_eq!((err.line, err.column, err.offset), (2, 5, 7));

        // Trailing characters.
        let err = parse("1 x").unwrap_err();
        assert_eq!(
            (err.line, err.column, err.kind),
            (1, 3, ErrorKind::TrailingCharacters)
        );

        // Raw control character position.
        let err = parse("\"a\u{3}b\"").unwrap_err();
        assert_eq!(
            (err.line, err.column, err.kind),
            (1, 3, ErrorKind::ControlCharacter)
        );

        // Empty input.
        let err = parse("").unwrap_err();
        assert_eq!(
            (err.line, err.column, err.offset, err.kind),
            (1, 1, 0, ErrorKind::UnexpectedEof)
        );
    }

    #[test]
    fn error_display_and_trait() {
        assert_eq!(
            parse("").unwrap_err().to_string(),
            "unexpected end of input at line 1, column 1"
        );
        assert_eq!(
            parse("-").unwrap_err().to_string(),
            "invalid number at line 1, column 2"
        );
        assert_eq!(
            parse("01").unwrap_err().to_string(),
            "invalid number at line 1, column 2"
        );
        assert_eq!(
            parse("1e400").unwrap_err().to_string(),
            "number out of range at line 1, column 1"
        );
        assert_eq!(
            parse("\"\\ud800\"").unwrap_err().to_string(),
            "invalid \\u escape at line 1, column 2"
        );
        assert_eq!(
            parse("[").unwrap_err().to_string(),
            "unexpected end of input at line 1, column 2"
        );

        // ParseError implements std::error::Error.
        let err = parse("bad").unwrap_err();
        let _: &dyn std::error::Error = &err;
        assert!(err.source().is_none());
    }

    #[test]
    fn accessors() {
        let null = Json::Null;
        let b = Json::Bool(true);
        let s = Json::from("hi");
        let a = Json::array([Json::from(1), Json::from(2)]);
        let o = Json::object([("k", Json::from("v"))]);

        assert!(null.is_null());
        assert!(!b.is_null());

        assert_eq!(s.as_str(), Some("hi"));
        assert_eq!(null.as_str(), None);
        assert_eq!(b.as_str(), None);

        assert_eq!(b.as_bool(), Some(true));
        assert_eq!(null.as_bool(), None);
        assert_eq!(s.as_bool(), None);

        assert_eq!(a.as_array().map(|items| items.len()), Some(2));
        assert_eq!(o.as_array(), None);

        let pairs = o.as_object().unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "k");
        assert_eq!(o.as_object().map(|p| p[0].1.as_str()), Some(Some("v")));
        assert_eq!(a.as_object(), None);

        assert_eq!(o.get("k").and_then(Json::as_str), Some("v"));
        assert_eq!(o.get("missing"), None);
        // get on non-objects.
        assert_eq!(a.get("k"), None);
        assert_eq!(s.get("k"), None);
        assert_eq!(null.get("k"), None);

        // Duplicate keys: get returns the last value.
        let dup = ok(r#"{"a": 1, "a": 2}"#);
        assert_eq!(dup.as_object().unwrap().len(), 2);
        assert_eq!(dup.get("a"), Some(&Json::Number(Number::I64(2))));
    }

    #[test]
    fn number_accessors() {
        assert_eq!(ok("-3").as_i64(), Some(-3));
        assert_eq!(ok("-3").as_u64(), None);
        assert_eq!(ok("18446744073709551615").as_i64(), None);
        assert_eq!(ok("18446744073709551615").as_u64(), Some(u64::MAX));
        assert_eq!(ok("2.0").as_i64(), Some(2));
        assert_eq!(ok("2.0").as_u64(), Some(2));
        assert_eq!(ok("-2.0").as_u64(), None);
        assert_eq!(ok("2.5").as_i64(), None);
        assert_eq!(ok("2.5").as_u64(), None);
        assert_eq!(ok("1e300").as_i64(), None);
        assert_eq!(ok("1e300").as_u64(), None);
        assert_eq!(Json::from("s").as_i64(), None);

        // Boundaries.
        assert_eq!(ok("9223372036854775807").as_i64(), Some(i64::MAX));
        assert_eq!(ok("-9223372036854775808").as_i64(), Some(i64::MIN));
        assert_eq!(ok("9223372036854775808").as_i64(), None); // 2^63
        assert_eq!(
            ok("9223372036854775808").as_u64(),
            Some(9223372036854775808)
        );
        assert_eq!(ok("18446744073709551616").as_u64(), None); // 2^64 as f64
        // Integral floats that fit exactly.
        assert_eq!(ok("-9223372036854775808.0").as_i64(), Some(i64::MIN));
        assert_eq!(ok("9223372036854775808.0").as_i64(), None); // exactly 2^63
        assert_eq!(ok("18446744073709551616.0").as_u64(), None); // exactly 2^64

        // as_f64 accepts every number kind.
        assert_eq!(ok("-3").as_f64(), Some(-3.0));
        assert_eq!(
            ok("18446744073709551615").as_f64(),
            Some(18446744073709551615.0)
        );
        assert_eq!(ok("0.5").as_f64(), Some(0.5));
        assert_eq!(Json::from("s").as_f64(), None);
    }

    #[test]
    fn construction_from_impls() {
        assert_eq!(Json::from(true), Json::Bool(true));
        assert_eq!(Json::from(false), Json::Bool(false));
        assert_eq!(Json::from(7i32), Json::Number(Number::I64(7)));
        assert_eq!(Json::from(-7i64), Json::Number(Number::I64(-7)));
        assert_eq!(Json::from(7u32), Json::Number(Number::I64(7)));
        // Small u64/usize become I64, large ones U64, like the parser.
        assert_eq!(Json::from(7u64), Json::Number(Number::I64(7)));
        assert_eq!(Json::from(u64::MAX), Json::Number(Number::U64(u64::MAX)));
        assert_eq!(Json::from(7usize), Json::Number(Number::I64(7)));
        assert_eq!(
            Json::from(usize::MAX),
            Json::Number(Number::U64(usize::MAX as u64))
        );
        assert_eq!(Json::from(0.5f32), Json::Number(Number::F64(0.5)));
        assert_eq!(Json::from(0.25f64), Json::Number(Number::F64(0.25)));
        assert_eq!(Json::from("lo"), Json::String("lo".to_owned()));
        assert_eq!(
            Json::from(String::from("lo")),
            Json::String("lo".to_owned())
        );
        assert_eq!(
            Json::from(vec![Json::from(1)]),
            Json::Array(vec![Json::from(1)])
        );
        // Option<T>: None becomes Null.
        assert_eq!(Json::from(None::<i64>), Json::Null);
        assert_eq!(Json::from(Some(9i64)), Json::Number(Number::I64(9)));
        assert_eq!(
            Json::from(Some(Json::from("x"))),
            Json::String("x".to_owned())
        );
    }

    #[test]
    fn construction_object_and_array() {
        // Keys may be any Into<String>.
        let v = Json::object([
            (String::from("a"), Json::from(1)),
            (String::from("b"), Json::from("two")),
        ]);
        let pairs = v.as_object().unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!((pairs[0].0.as_str(), pairs[1].0.as_str()), ("a", "b"));
        assert_eq!(pairs[0].1, Json::Number(Number::I64(1)));
        assert_eq!(pairs[1].1.as_str(), Some("two"));

        let empty = Json::object(Vec::<(&str, Json)>::new());
        assert!(empty.as_object().unwrap().is_empty());

        let arr = Json::array([Json::from(1), Json::Bool(false)]);
        assert_eq!(arr.as_array().map(|items| items.len()), Some(2));
        let empty_arr = Json::array(Vec::<Json>::new());
        assert!(empty_arr.as_array().unwrap().is_empty());
    }

    #[test]
    fn display_is_compact() {
        assert_eq!(Json::Null.to_string(), "null");
        assert_eq!(Json::Bool(true).to_string(), "true");
        assert_eq!(Json::Bool(false).to_string(), "false");
        assert_eq!(Json::from(1).to_string(), "1");
        assert_eq!(Json::from("s").to_string(), "\"s\"");
        assert_eq!(Json::from(-1.5f64).to_string(), "-1.5");
        let v = Json::array([Json::from(1), Json::from("two"), Json::Null]);
        assert_eq!(v.to_string(), "[1,\"two\",null]");
        let v = Json::object([
            ("k", Json::from(vec![Json::Bool(true)])),
            ("z", Json::from(0)),
        ]);
        assert_eq!(v.to_string(), "{\"k\":[true],\"z\":0}");
    }

    #[test]
    fn display_escapes_strings() {
        assert_eq!(Json::from("").to_string(), "\"\"");
        assert_eq!(Json::from("plain").to_string(), "\"plain\"");
        // '/' is never escaped.
        assert_eq!(Json::from("q\"b\\s/n").to_string(), "\"q\\\"b\\\\s/n\"");
        assert_eq!(
            Json::from("\n\r\t\u{8}\u{c}").to_string(),
            "\"\\n\\r\\t\\b\\f\""
        );
        // Other control characters use \u00xx.
        assert_eq!(
            Json::from("\u{1}\u{1f}\u{0}").to_string(),
            "\"\\u0001\\u001f\\u0000\""
        );
        // Non-ASCII is written as raw UTF-8.
        assert_eq!(Json::from("héllo 😀").to_string(), "\"héllo 😀\"");
        // DEL (U+007F) is not a control character for JSON.
        assert_eq!(Json::from("del\u{7f}").to_string(), "\"del\u{7f}\"");
    }

    #[test]
    fn display_numbers() {
        assert_eq!(Json::from(i64::MIN).to_string(), "-9223372036854775808");
        assert_eq!(Json::from(i64::MAX).to_string(), "9223372036854775807");
        assert_eq!(Json::from(u64::MAX).to_string(), "18446744073709551615");
        // Shortest round-trip forms.
        assert_eq!(Json::from(1.0f64).to_string(), "1.0");
        assert_eq!(Json::from(0.1f64).to_string(), "0.1");
        assert_eq!(Json::from(1e-7f64).to_string(), "1e-7");
        assert_eq!(Json::from(1e300f64).to_string(), "1e300");
        // Non-finite floats serialize as null.
        assert_eq!(Json::from(f64::NAN).to_string(), "null");
        assert_eq!(Json::from(f64::INFINITY).to_string(), "null");
        assert_eq!(Json::from(f64::NEG_INFINITY).to_string(), "null");
    }

    #[test]
    fn round_trips() {
        let values: Vec<Json> = vec![
            Json::Null,
            Json::Bool(true),
            Json::Bool(false),
            Json::from(0i64),
            Json::from(-1i64),
            Json::from(i64::MIN),
            Json::from(i64::MAX),
            Json::from(u64::MAX),
            Json::from(0.1f64),
            Json::from(1.0f64),
            Json::from(-0.0f64),
            Json::from(1e-7f64),
            Json::from(1e300f64),
            Json::from(5e-324f64),
            Json::from(-2.5f64),
            Json::from(f64::MAX),
            Json::from(""),
            Json::from("hello"),
            // Every escape in one string.
            Json::from("quote\"back\\slash/b\u{8}f\u{c}n\nr\rt\u{1}"),
            Json::from("😀 héllo 日本語"),
            Json::from("del\u{7f}"),
            Json::Array(Vec::new()),
            Json::Object(Vec::new()),
            Json::array([
                Json::from(1),
                Json::from("two"),
                Json::Null,
                Json::Bool(false),
            ]),
            Json::object([
                ("k\"ey\\nl", Json::from("val\u{1}😀")),
                ("num", Json::from(-0.5)),
                ("nested", Json::object([("inner", Json::Array(Vec::new()))])),
            ]),
            {
                let mut v = Json::from("leaf");
                for _ in 0..20 {
                    v = Json::Array(vec![v]);
                }
                v
            },
        ];
        for value in values {
            let text = value.to_string();
            assert_eq!(ok(&text), value, "round trip failed for {text:?}");
        }
    }

    #[test]
    fn depth_limit_default() {
        assert_eq!(DEFAULT_MAX_DEPTH, 128);
        // Exactly at the default limit: fine.
        let at_limit = format!("{}{}", "[".repeat(128), "]".repeat(128));
        assert!(parse(&at_limit).is_ok());
        // One level deeper fails cleanly, without stack overflow.
        let err = parse(&"[".repeat(129)).unwrap_err();
        assert_eq!(err.kind, ErrorKind::DepthLimitExceeded);
        assert_eq!(err.offset, 128);
        // 200 open brackets, as the spec calls out.
        let err = parse(&"[".repeat(200)).unwrap_err();
        assert_eq!(err.kind, ErrorKind::DepthLimitExceeded);
    }

    #[test]
    fn depth_limit_custom() {
        // 999 nested arrays fit within a limit of 1000.
        let deep = format!("{}{}", "[".repeat(999), "]".repeat(999));
        let v = parse_with_depth(&deep, 1000).unwrap();
        assert!(v.as_array().is_some());

        // One level past the limit is rejected.
        let over = format!("{}{}", "[".repeat(1001), "]".repeat(1001));
        assert_eq!(
            parse_with_depth(&over, 1000).unwrap_err().kind,
            ErrorKind::DepthLimitExceeded
        );

        // Scalars are never depth-limited; containers need at least one level.
        assert!(parse_with_depth("1", 0).is_ok());
        assert_eq!(
            parse_with_depth("[]", 0).unwrap_err().kind,
            ErrorKind::DepthLimitExceeded
        );
        assert_eq!(
            parse_with_depth("{}", 0).unwrap_err().kind,
            ErrorKind::DepthLimitExceeded
        );

        // parse() is parse_with_depth() at the default limit.
        let doc = r#"[1, {"a": [true, null]}]"#;
        assert_eq!(parse(doc), parse_with_depth(doc, DEFAULT_MAX_DEPTH));
    }

    #[test]
    fn no_panics_on_prefixes() {
        let docs = [
            r#"{"name":"nibble","tags":["fast","strict"],"pi":3.14159,"ok":true,"emoji":"😀","nested":{"deep":[1,[2,[3]]]}}"#,
            r#"[1, 2.5e-3, "héllo", null, {"k":"A\n😀"}, -0.0]"#,
            r#""\"quoted\" with \\ backslash""#,
        ];
        for doc in &docs {
            assert!(parse(doc).is_ok());
            for end in 0..=doc.len() {
                // Byte-prefix cuts can split multi-byte characters;
                // from_utf8_lossy turns those into U+FFFD.
                let prefix = String::from_utf8_lossy(&doc.as_bytes()[..end]);
                // Prefixes may parse or fail; they must never panic.
                let _ = parse(&prefix);
            }
        }
    }

    /// A tiny xorshift64 PRNG, so the fuzz-ish tests are deterministic.
    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn no_panics_on_random_bytes() {
        let mut state = 0x2545F4914F6CDD1Du64;
        for _ in 0..10_000 {
            let len = (xorshift(&mut state) % 65) as usize;
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                bytes.push(xorshift(&mut state) as u8);
            }
            let text = String::from_utf8_lossy(&bytes);
            // Any outcome is fine; the point is that parsing never panics.
            let _ = parse(&text);
        }
    }

    #[test]
    fn no_panics_on_random_json_tokens() {
        // Random strings over the JSON alphabet exercise the deeper parser
        // states: strings, escapes, surrogates, numbers, keywords, nesting.
        const ALPHABET: &[u8] = b"[]{}\":,\\/-+.eE0123456789abcdfnrtu \n";
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..10_000 {
            let len = (xorshift(&mut state) % 64 + 1) as usize;
            let mut text = String::with_capacity(len);
            for _ in 0..len {
                let b = ALPHABET[(xorshift(&mut state) % ALPHABET.len() as u64) as usize];
                text.push(char::from(b));
            }
            let _ = parse(&text);
        }
    }
}
