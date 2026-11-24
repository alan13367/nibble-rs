//! HTTP/1.1 request parsing and response writing, using only `std`.
//!
//! This module only parses requests and writes responses; routing, JSON
//! encoding and threading belong to the caller. Reading is incremental and
//! strictly bounded: no buffer ever grows past the configured [`Limits`], and
//! no input can cause a panic. Writing covers complete responses
//! ([`write_response`]), interim `100 Continue` responses
//! ([`write_continue`]), streamed chunked responses ([`ChunkedWriter`]) and
//! server-sent events ([`SseWriter`]). A usage sketch:
//!
//! ```text
//! let head = http::read_head(&mut reader, &limits)?;
//! let body = http::read_body(&mut reader, &head, &limits)?;
//! http::write_response(&mut stream, &http::Response::json(200, "{\"ok\":true}"), head.keep_alive(), false)?;
//! ```
//!
//! Lines end in CRLF; a bare LF is also accepted as a line ending, while a
//! bare CR is rejected as malformed.

use std::fmt;
use std::io::{self, BufRead, Write};

/// Maximum length in bytes of a chunk-size line (hex size plus extensions)
/// or of a single trailer line.
const CHUNK_LINE_LIMIT: usize = 1024;

/// Bounded sizes used while parsing a request.
///
/// These are the only knobs that decide how much memory parsing may use:
/// buffers grow incrementally and never past these limits, so a hostile
/// client cannot make the reader allocate unboundedly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Limits {
    /// Maximum number of bytes of the request line plus header lines.
    /// Leading empty lines and the blank line that ends the headers count
    /// towards this limit. Default: 16 KiB.
    pub(crate) max_head_bytes: usize,
    /// Maximum number of header lines. Default: 100.
    pub(crate) max_headers: usize,
    /// Maximum decoded size in bytes of a request body. Default: 8 MiB.
    pub(crate) max_body_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_head_bytes: 16 * 1024,
            max_headers: 100,
            max_body_bytes: 8 * 1024 * 1024,
        }
    }
}

/// A parsed request head: the request line plus header lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RequestHead {
    /// Request method exactly as sent, e.g. `"POST"`. Only the token syntax
    /// is checked; unknown methods are passed through as-is.
    pub(crate) method: String,
    /// Request target without the query string, e.g. `"/v1/models"`. For
    /// absolute-form targets (`"http://host:8080/v1/models?x"`) only the
    /// path is kept; `"*"` is kept for asterisk-form targets.
    pub(crate) path: String,
    /// Raw query string after `?` (no percent-decoding), if a `?` was
    /// present in the target.
    pub(crate) query: Option<String>,
    /// HTTP minor version: `0` for HTTP/1.0, `1` for HTTP/1.1. Only these
    /// two versions are accepted.
    pub(crate) version_minor: u8,
    /// Header pairs in the order sent, names exactly as sent, values trimmed
    /// of leading and trailing spaces and tabs.
    pub(crate) headers: Vec<(String, String)>,
}

impl RequestHead {
    /// Returns the value of the first header whose name matches `name`
    /// case-insensitively.
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// Returns whether the connection should be kept alive after this
    /// request. For HTTP/1.1 the default is yes unless a `close` token is
    /// present in a `Connection` header; for HTTP/1.0 the default is no
    /// unless a `keep-alive` token is present. Token lists such as
    /// `"keep-alive, Upgrade"` are honoured.
    pub(crate) fn keep_alive(&self) -> bool {
        let mut saw_keep_alive = false;
        let mut saw_close = false;
        for (name, value) in &self.headers {
            if name.eq_ignore_ascii_case("connection") {
                for token in value.split(',') {
                    let token = token.trim();
                    if token.eq_ignore_ascii_case("keep-alive") {
                        saw_keep_alive = true;
                    } else if token.eq_ignore_ascii_case("close") {
                        saw_close = true;
                    }
                }
            }
        }
        if self.version_minor == 0 {
            saw_keep_alive
        } else {
            !saw_close
        }
    }

    /// Returns whether the request announces `Expect: 100-continue`
    /// (case-insensitive, honouring comma-separated token lists).
    pub(crate) fn expects_continue(&self) -> bool {
        for (name, value) in &self.headers {
            if name.eq_ignore_ascii_case("expect") {
                for token in value.split(',') {
                    if token.trim().eq_ignore_ascii_case("100-continue") {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Errors that can occur while reading a request.
#[non_exhaustive]
#[derive(Debug)]
pub(crate) enum HttpError {
    /// The request was syntactically invalid; answer with 400.
    Malformed(&'static str),
    /// The connection ended before the request was complete; answer with 400.
    Truncated,
    /// The request used an HTTP version this parser does not support;
    /// answer with 505.
    UnsupportedVersion,
    /// The request used a transfer encoding other than exactly `chunked`;
    /// answer with 501.
    UnsupportedTransferEncoding,
    /// The request line plus headers exceeded the configured limits;
    /// answer with 431.
    HeadTooLarge,
    /// The decoded body exceeded `Limits::max_body_bytes`; answer with 413.
    BodyTooLarge,
    /// An underlying I/O error occurred.
    Io(io::Error),
}

impl HttpError {
    /// Returns the HTTP status code a server should answer with, or `None`
    /// for I/O errors, which have no status of their own.
    pub(crate) fn status(&self) -> Option<u16> {
        match self {
            HttpError::Malformed(_) => Some(400),
            HttpError::Truncated => Some(400),
            HttpError::UnsupportedVersion => Some(505),
            HttpError::UnsupportedTransferEncoding => Some(501),
            HttpError::HeadTooLarge => Some(431),
            HttpError::BodyTooLarge => Some(413),
            HttpError::Io(_) => None,
        }
    }
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HttpError::Malformed(what) => write!(f, "malformed request: {what}"),
            HttpError::Truncated => {
                write!(f, "connection closed before the request was complete")
            }
            HttpError::UnsupportedVersion => write!(f, "unsupported HTTP version"),
            HttpError::UnsupportedTransferEncoding => write!(f, "unsupported transfer encoding"),
            HttpError::HeadTooLarge => write!(f, "request head exceeds the configured limits"),
            HttpError::BodyTooLarge => write!(f, "request body exceeds the configured limits"),
            HttpError::Io(err) => write!(f, "i/o error: {err}"),
        }
    }
}

impl std::error::Error for HttpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HttpError::Io(err) => Some(err),
            _ => None,
        }
    }
}

/// The body framing a request head announces.
enum Framing {
    /// Neither `Content-Length` nor `Transfer-Encoding` is present.
    None,
    /// A `Content-Length` body of that many bytes.
    Length(u64),
    /// A `Transfer-Encoding: chunked` body.
    Chunked,
}

/// Validates the body framing the head announces: `Content-Length` must be
/// all digits (multiple values must be identical), `Transfer-Encoding` must
/// be exactly `chunked`, announcing both is rejected (request smuggling),
/// and a body over `Limits::max_body_bytes` is rejected before any body byte
/// is read.
fn body_framing(head: &RequestHead, limits: &Limits) -> Result<Framing, HttpError> {
    let mut content_length: Option<u64> = None;
    let mut transfer_encoding_count = 0usize;
    let mut transfer_encoding_is_chunked = false;
    for (name, value) in &head.headers {
        if name.eq_ignore_ascii_case("content-length") {
            let len = parse_content_length(value)?;
            match content_length {
                None => content_length = Some(len),
                Some(prev) if prev == len => {}
                Some(_) => {
                    return Err(HttpError::Malformed("conflicting Content-Length values"));
                }
            }
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            transfer_encoding_count += 1;
            transfer_encoding_is_chunked = value.trim().eq_ignore_ascii_case("chunked");
        }
    }
    if transfer_encoding_count > 0 {
        // Transfer-Encoding would win over Content-Length, but announcing
        // both at once is a request-smuggling vector: reject it outright.
        if content_length.is_some() {
            return Err(HttpError::Malformed(
                "both Content-Length and Transfer-Encoding are present",
            ));
        }
        if transfer_encoding_count != 1 || !transfer_encoding_is_chunked {
            return Err(HttpError::UnsupportedTransferEncoding);
        }
        return Ok(Framing::Chunked);
    }
    match content_length {
        Some(len) if len > limits.max_body_bytes as u64 => Err(HttpError::BodyTooLarge),
        Some(len) => Ok(Framing::Length(len)),
        None => Ok(Framing::None),
    }
}

/// Parses a `Content-Length` value: all ASCII digits, or the request is
/// malformed. An all-digit value too large to represent is certainly over any
/// body limit, so it is reported as too large.
fn parse_content_length(value: &str) -> Result<u64, HttpError> {
    let value = value.trim();
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(HttpError::Malformed("invalid Content-Length"));
    }
    match value.parse::<u64>() {
        Ok(len) => Ok(len),
        Err(_) => Err(HttpError::BodyTooLarge),
    }
}

/// Parses a chunk-size line: hexadecimal digits, optionally followed by `;`
/// chunk extensions which are skipped. Hex sizes that do not fit in `u64`
/// are certainly over any body limit, so they are reported as too large.
fn parse_chunk_size(content: &[u8]) -> Result<u64, HttpError> {
    let hex_part = content.split(|&b| b == b';').next().unwrap_or_default();
    if hex_part.is_empty() {
        return Err(HttpError::Malformed("missing chunk size"));
    }
    if !hex_part.iter().all(|b| b.is_ascii_hexdigit()) {
        return Err(HttpError::Malformed("invalid chunk size"));
    }
    let text = match std::str::from_utf8(hex_part) {
        Ok(text) => text,
        Err(_) => return Err(HttpError::Malformed("invalid chunk size")),
    };
    match u64::from_str_radix(text, 16) {
        Ok(size) => Ok(size),
        Err(_) => Err(HttpError::BodyTooLarge),
    }
}

/// Parses a header line (without its line ending): `name ":" OWS value OWS`.
/// The name must be a non-empty token with no whitespace before the colon,
/// and the value must not contain NUL.
fn parse_header_line(content: &[u8]) -> Result<(String, String), HttpError> {
    let colon = match content.iter().position(|&b| b == b':') {
        Some(colon) => colon,
        None => return Err(HttpError::Malformed("header line without a colon")),
    };
    let name_bytes = &content[..colon];
    let value_bytes = trim_ows(&content[colon + 1..]);
    if name_bytes.is_empty() || !name_bytes.iter().all(|&b| is_tchar(b)) {
        return Err(HttpError::Malformed("invalid header field name"));
    }
    if value_bytes.contains(&b'\0') {
        return Err(HttpError::Malformed("NUL in header value"));
    }
    let name = match std::str::from_utf8(name_bytes) {
        Ok(name) => name,
        Err(_) => return Err(HttpError::Malformed("header name is not valid UTF-8")),
    };
    let value = match std::str::from_utf8(value_bytes) {
        Ok(value) => value,
        Err(_) => return Err(HttpError::Malformed("header value is not valid UTF-8")),
    };
    Ok((name.to_string(), value.to_string()))
}

/// Trims spaces and tabs from both ends of a byte slice.
fn trim_ows(mut s: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = s.split_first() {
        if first != b' ' && first != b'\t' {
            break;
        }
        s = rest;
    }
    while let Some((&last, rest)) = s.split_last() {
        if last != b' ' && last != b'\t' {
            break;
        }
        s = rest;
    }
    s
}

/// Returns whether a byte is an RFC 9110 token character (`tchar`).
fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

/// The outcome of a bounded line read.
enum LineOutcome {
    /// A complete line, including its terminator, was read into the buffer.
    Line,
    /// The stream ended before the first byte of this line was read.
    EofAtStart,
}

/// Errors of the bounded line reader.
enum LineError {
    /// The stream ended in the middle of a line.
    Truncated,
    /// The line would exceed the byte limit.
    Overflow,
    /// An underlying I/O failure.
    Io(io::Error),
}

/// Reads one line, terminated by LF (optionally preceded by CR), into `out`.
/// At most `limit` bytes are consumed for this line; anything longer is an
/// error, so the buffer never grows past the limit.
fn read_line_bounded<R: BufRead>(
    reader: &mut R,
    out: &mut Vec<u8>,
    limit: usize,
) -> Result<LineOutcome, LineError> {
    let start_len = out.len();
    let mut remaining = limit;
    loop {
        let available = match reader.fill_buf() {
            Ok(available) => available,
            Err(err) => return Err(LineError::Io(err)),
        };
        if available.is_empty() {
            return if out.len() == start_len {
                Ok(LineOutcome::EofAtStart)
            } else {
                Err(LineError::Truncated)
            };
        }
        if let Some(i) = available.iter().position(|&b| b == b'\n') {
            let n = i + 1;
            if n > remaining {
                return Err(LineError::Overflow);
            }
            out.extend_from_slice(&available[..n]);
            reader.consume(n);
            return Ok(LineOutcome::Line);
        }
        if available.len() >= remaining {
            // Even the whole chunk does not bring the line ending within the
            // limit, so the line is too long.
            return Err(LineError::Overflow);
        }
        out.extend_from_slice(available);
        let n = available.len();
        reader.consume(n);
        remaining -= n;
    }
}

/// Strips a line's trailing LF and an optional preceding CR and returns the
/// content. A CR left inside the content is a bare CR, which is malformed.
fn line_content(line: &[u8]) -> Result<&[u8], HttpError> {
    let mut content = line;
    if content.last() == Some(&b'\n') {
        content = &content[..content.len() - 1];
    }
    if content.last() == Some(&b'\r') {
        content = &content[..content.len() - 1];
    }
    if content.contains(&b'\r') {
        return Err(HttpError::Malformed("bare CR within a line"));
    }
    Ok(content)
}

/// Reads exactly one byte; an end of stream is reported as truncated.
fn read_one<R: BufRead>(reader: &mut R) -> Result<u8, HttpError> {
    let available = reader.fill_buf().map_err(HttpError::Io)?;
    if available.is_empty() {
        return Err(HttpError::Truncated);
    }
    let byte = available[0];
    reader.consume(1);
    Ok(byte)
}

/// Reads exactly `count` bytes and appends them to `out`. `count` must have
/// been validated against the caller's limits first, which keeps the
/// allocation bounded by configuration rather than by untrusted input.
fn read_exact_bytes<R: BufRead>(
    reader: &mut R,
    out: &mut Vec<u8>,
    count: usize,
) -> Result<(), HttpError> {
    out.reserve(count);
    let mut remaining = count;
    while remaining > 0 {
        let available = reader.fill_buf().map_err(HttpError::Io)?;
        if available.is_empty() {
            return Err(HttpError::Truncated);
        }
        let take = available.len().min(remaining);
        out.extend_from_slice(&available[..take]);
        reader.consume(take);
        remaining -= take;
    }
    Ok(())
}

/// The parsed pieces of a request line.
struct RequestLine {
    /// Request method exactly as sent.
    method: String,
    /// Target path without the query string.
    path: String,
    /// Raw query string, if the target had one.
    query: Option<String>,
    /// HTTP minor version.
    version_minor: u8,
}

/// Parses a request line: `METHOD SP target SP HTTP-version`.
fn parse_request_line(content: &[u8]) -> Result<RequestLine, HttpError> {
    let text = match std::str::from_utf8(content) {
        Ok(text) => text,
        Err(_) => return Err(HttpError::Malformed("request line is not valid UTF-8")),
    };
    let mut parts = text.split(' ');
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    let version = parts.next().unwrap_or("");
    if parts.next().is_some() || method.is_empty() || target.is_empty() || version.is_empty() {
        return Err(HttpError::Malformed("malformed request line"));
    }
    if !method.bytes().all(is_tchar) {
        return Err(HttpError::Malformed("invalid method token"));
    }
    let version_minor = parse_version(version)?;
    if target.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return Err(HttpError::Malformed("control byte in request target"));
    }
    let (path, query) = parse_target(target)?;
    Ok(RequestLine {
        method: method.to_string(),
        path,
        query,
        version_minor,
    })
}

/// Parses an HTTP version: exactly `HTTP/1.0` or `HTTP/1.1` are accepted.
/// Any other well-formed `HTTP/x.y` is an unsupported version; anything else
/// is malformed (the `HTTP` name is case-sensitive).
fn parse_version(version: &str) -> Result<u8, HttpError> {
    let bytes = version.as_bytes();
    if bytes.len() == 8
        && bytes.starts_with(b"HTTP/")
        && bytes[5].is_ascii_digit()
        && bytes[6] == b'.'
        && bytes[7].is_ascii_digit()
    {
        if bytes[5] == b'1' && (bytes[7] == b'0' || bytes[7] == b'1') {
            return Ok(bytes[7] - b'0');
        }
        return Err(HttpError::UnsupportedVersion);
    }
    Err(HttpError::Malformed("malformed HTTP version"))
}

/// Parses a request target: origin-form (`/path?query`), absolute-form
/// (`http://host/path`, of which only the path and query are kept), or `*`.
/// Anything else is malformed.
fn parse_target(target: &str) -> Result<(String, Option<String>), HttpError> {
    if target == "*" {
        return Ok((String::from("*"), None));
    }
    let origin = match strip_absolute_scheme(target) {
        // Absolute-form: keep only the path and query that follow the
        // authority component.
        Some(rest) => match rest.find(['/', '?']) {
            Some(i) if rest.as_bytes()[i] == b'/' => &rest[i..],
            Some(i) => return Ok((String::from("/"), Some(rest[i + 1..].to_string()))),
            None => return Ok((String::from("/"), None)),
        },
        // Origin-form.
        None => {
            if !target.starts_with('/') {
                return Err(HttpError::Malformed("unsupported request target"));
            }
            target
        }
    };
    match origin.split_once('?') {
        Some((path, query)) => Ok((path.to_string(), Some(query.to_string()))),
        None => Ok((origin.to_string(), None)),
    }
}

/// Strips a case-insensitive `http://` or `https://` scheme prefix, if any.
fn strip_absolute_scheme(target: &str) -> Option<&str> {
    for scheme in ["http://", "https://"] {
        let n = scheme.len();
        if target.len() >= n && target[..n].eq_ignore_ascii_case(scheme) {
            return Some(&target[n..]);
        }
    }
    None
}

/// Reads the request line, skipping up to four tolerated empty lines first
/// (RFC 9112 §2.2). `consumed` tracks the head bytes read so far and is
/// advanced so the header loop can share one head-byte budget.
/// `Ok(None)` means the connection ended before a request began.
fn read_request_line<R: BufRead>(
    reader: &mut R,
    limits: &Limits,
    consumed: &mut usize,
) -> Result<Option<RequestLine>, HttpError> {
    let mut line: Vec<u8> = Vec::new();
    let mut empty_lines = 0u32;
    loop {
        line.clear();
        let remaining = limits.max_head_bytes.saturating_sub(*consumed);
        let content = match read_line_bounded(reader, &mut line, remaining) {
            Ok(LineOutcome::Line) => {
                *consumed += line.len();
                line_content(&line)?
            }
            // A clean end of the connection before any request bytes arrived
            // (tolerated empty lines aside): the normal end of a keep-alive
            // connection, with no request to answer.
            Ok(LineOutcome::EofAtStart) => return Ok(None),
            Err(LineError::Io(err)) => return Err(HttpError::Io(err)),
            Err(LineError::Truncated) => return Err(HttpError::Truncated),
            Err(LineError::Overflow) => return Err(HttpError::HeadTooLarge),
        };
        if content.is_empty() {
            empty_lines += 1;
            if empty_lines > 4 {
                return Err(HttpError::Malformed("more than four leading empty lines"));
            }
            continue;
        }
        return Ok(Some(parse_request_line(content)?));
    }
}

/// Reads one request head from `reader`.
///
/// Returns `Ok(None)` if the connection closed cleanly before the first
/// byte of a request, which is the normal end of a keep-alive connection.
/// Up to four empty lines before the request line are tolerated
/// (RFC 9112 §2.2). Over-size heads and too many headers are rejected with
/// [`HttpError::HeadTooLarge`]. The announced body framing
/// (`Content-Length` / `Transfer-Encoding`) is validated here as well, so a
/// caller can reject a bad or oversized request before reading its body.
///
/// Lines end in CRLF; a bare LF is also accepted as a line ending, while a
/// bare CR is malformed. Never panics on any input.
pub(crate) fn read_head<R: BufRead>(
    reader: &mut R,
    limits: &Limits,
) -> Result<Option<RequestHead>, HttpError> {
    let mut consumed = 0usize;
    let request_line = match read_request_line(reader, limits, &mut consumed)? {
        Some(parsed) => parsed,
        None => return Ok(None),
    };

    let mut line: Vec<u8> = Vec::new();
    let mut headers: Vec<(String, String)> = Vec::new();
    loop {
        line.clear();
        let remaining = limits.max_head_bytes.saturating_sub(consumed);
        match read_line_bounded(reader, &mut line, remaining) {
            Ok(LineOutcome::Line) => {
                consumed += line.len();
            }
            Ok(LineOutcome::EofAtStart) | Err(LineError::Truncated) => {
                return Err(HttpError::Truncated);
            }
            Err(LineError::Overflow) => return Err(HttpError::HeadTooLarge),
            Err(LineError::Io(err)) => return Err(HttpError::Io(err)),
        }
        let content = line_content(&line)?;
        if content.is_empty() {
            // The blank line that ends the header section.
            break;
        }
        if matches!(content.first(), Some(&b' ') | Some(&b'\t')) {
            return Err(HttpError::Malformed(
                "obsolete line folding is not accepted",
            ));
        }
        if headers.len() >= limits.max_headers {
            return Err(HttpError::HeadTooLarge);
        }
        headers.push(parse_header_line(content)?);
    }

    let head = RequestHead {
        method: request_line.method,
        path: request_line.path,
        query: request_line.query,
        version_minor: request_line.version_minor,
        headers,
    };
    body_framing(&head, limits)?;
    Ok(Some(head))
}

/// Reads the body that `head` announces: a `Content-Length` body, a
/// `Transfer-Encoding: chunked` body (chunk extensions are skipped and
/// trailers are read and discarded), or an empty body when neither header is
/// present. Bodies over `Limits::max_body_bytes` are rejected before any
/// unbounded allocation. Never panics on any input.
pub(crate) fn read_body<R: BufRead>(
    reader: &mut R,
    head: &RequestHead,
    limits: &Limits,
) -> Result<Vec<u8>, HttpError> {
    match body_framing(head, limits)? {
        Framing::None => Ok(Vec::new()),
        Framing::Length(len) => {
            let mut body = Vec::new();
            read_exact_bytes(reader, &mut body, len as usize)?;
            Ok(body)
        }
        Framing::Chunked => read_chunked(reader, limits),
    }
}

/// Reads a `Transfer-Encoding: chunked` body and returns the decoded bytes.
fn read_chunked<R: BufRead>(reader: &mut R, limits: &Limits) -> Result<Vec<u8>, HttpError> {
    let mut body: Vec<u8> = Vec::new();
    let mut line: Vec<u8> = Vec::new();
    loop {
        line.clear();
        match read_line_bounded(reader, &mut line, CHUNK_LINE_LIMIT) {
            Ok(LineOutcome::Line) => {}
            Ok(LineOutcome::EofAtStart) | Err(LineError::Truncated) => {
                return Err(HttpError::Truncated);
            }
            Err(LineError::Overflow) => {
                return Err(HttpError::Malformed("chunk size line too long"));
            }
            Err(LineError::Io(err)) => return Err(HttpError::Io(err)),
        }
        let content = line_content(&line)?;
        let size = parse_chunk_size(content)?;
        if size == 0 {
            discard_trailers(reader, limits, &mut line)?;
            return Ok(body);
        }
        let remaining = limits.max_body_bytes.saturating_sub(body.len()) as u64;
        if size > remaining {
            return Err(HttpError::BodyTooLarge);
        }
        read_exact_bytes(reader, &mut body, size as usize)?;
        expect_chunk_terminator(reader)?;
    }
}

/// Reads and discards the trailer section after the last chunk. Trailer
/// lines must still be shaped like header lines, and the section is bounded
/// by the chunk line limit and by `Limits::max_head_bytes` in total.
fn discard_trailers<R: BufRead>(
    reader: &mut R,
    limits: &Limits,
    line: &mut Vec<u8>,
) -> Result<(), HttpError> {
    let mut budget = limits.max_head_bytes;
    loop {
        line.clear();
        let limit = CHUNK_LINE_LIMIT.min(budget);
        match read_line_bounded(reader, line, limit) {
            Ok(LineOutcome::Line) => {}
            Ok(LineOutcome::EofAtStart) | Err(LineError::Truncated) => {
                return Err(HttpError::Truncated);
            }
            Err(LineError::Overflow) => {
                return Err(HttpError::Malformed("trailer section too large"));
            }
            Err(LineError::Io(err)) => return Err(HttpError::Io(err)),
        }
        budget = budget.saturating_sub(line.len());
        let content = line_content(line)?;
        if content.is_empty() {
            return Ok(());
        }
        parse_header_line(content)?;
    }
}

/// Reads the CRLF (or bare LF) that must follow each chunk's data.
fn expect_chunk_terminator<R: BufRead>(reader: &mut R) -> Result<(), HttpError> {
    match read_one(reader)? {
        b'\n' => Ok(()),
        b'\r' => match read_one(reader)? {
            b'\n' => Ok(()),
            _ => Err(HttpError::Malformed("bad terminator after chunk data")),
        },
        _ => Err(HttpError::Malformed("missing CRLF after chunk data")),
    }
}

/// A complete HTTP response ready to be written with [`write_response`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Response {
    /// Status code, e.g. 200.
    pub(crate) status: u16,
    /// Extra headers, written before the automatically added `Content-Length`
    /// and `Connection` headers.
    pub(crate) headers: Vec<(String, String)>,
    /// Response body bytes. For `head_only` writes the body is not written
    /// but still measured for the `Content-Length` header.
    pub(crate) body: Vec<u8>,
}

impl Response {
    /// Creates a response with the given status, no headers and an empty
    /// body.
    pub(crate) fn new(status: u16) -> Self {
        Response {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Creates a response with `Content-Type: application/json`.
    pub(crate) fn json(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Response {
            status,
            headers: vec![(
                String::from("Content-Type"),
                String::from("application/json"),
            )],
            body: body.into(),
        }
    }

    /// Builder: appends a header and returns the response.
    #[must_use]
    pub(crate) fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// Returns the standard reason phrase for a status code, or `"Unknown"` for
/// codes without a standard phrase.
#[must_use]
pub(crate) fn reason_phrase(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        101 => "Switching Protocols",
        102 => "Processing",
        103 => "Early Hints",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        203 => "Non-Authoritative Information",
        204 => "No Content",
        205 => "Reset Content",
        206 => "Partial Content",
        300 => "Multiple Choices",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        407 => "Proxy Authentication Required",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        411 => "Length Required",
        412 => "Precondition Failed",
        413 => "Content Too Large",
        414 => "URI Too Long",
        415 => "Unsupported Media Type",
        416 => "Range Not Satisfiable",
        417 => "Expectation Failed",
        418 => "I'm a teapot",
        421 => "Misdirected Request",
        422 => "Unprocessable Content",
        425 => "Too Early",
        426 => "Upgrade Required",
        428 => "Precondition Required",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        451 => "Unavailable For Legal Reasons",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        506 => "Variant Also Negotiates",
        507 => "Insufficient Storage",
        508 => "Loop Detected",
        510 => "Not Extended",
        511 => "Network Authentication Required",
        _ => "Unknown",
    }
}

/// Rejects header names and values that contain CR or LF, to prevent header
/// injection. Nothing is written when validation fails.
fn validate_headers(headers: &[(String, String)]) -> io::Result<()> {
    for (name, value) in headers {
        if name.contains('\r') || name.contains('\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "header names must not contain CR or LF",
            ));
        }
        if value.contains('\r') || value.contains('\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "header values must not contain CR or LF",
            ));
        }
    }
    Ok(())
}

/// Serialises a status line, the given headers, then the raw `tail` header
/// lines, then the `Connection` header and the blank line, after validating
/// the given headers against CR/LF injection.
fn serialize_head(
    status: u16,
    headers: &[(String, String)],
    tail: &[&str],
    keep_alive: bool,
) -> io::Result<Vec<u8>> {
    validate_headers(headers)?;
    let reason = reason_phrase(status);
    let connection = if keep_alive { "keep-alive" } else { "close" };
    let mut buf = Vec::with_capacity(128);
    write!(buf, "HTTP/1.1 {status} {reason}\r\n")?;
    for (name, value) in headers {
        write!(buf, "{name}: {value}\r\n")?;
    }
    for line in tail {
        buf.extend_from_slice(line.as_bytes());
        buf.extend_from_slice(b"\r\n");
    }
    write!(buf, "Connection: {connection}\r\n\r\n")?;
    Ok(buf)
}

/// Writes a complete response: the status line, the response headers, an
/// added `Content-Length` and a `Connection: keep-alive` or
/// `Connection: close` header per `keep_alive`, then the body. No `Date`
/// header is added. For a HEAD request pass `head_only = true`: the headers,
/// including `Content-Length`, are written but the body is not.
///
/// Header names or values containing CR or LF are rejected with an
/// `io::Error` of kind `InvalidInput` before anything is written, to prevent
/// header injection.
pub(crate) fn write_response<W: Write>(
    w: &mut W,
    response: &Response,
    keep_alive: bool,
    head_only: bool,
) -> io::Result<()> {
    let len = response.body.len();
    let mut buf = serialize_head(
        response.status,
        &response.headers,
        &[&format!("Content-Length: {len}")],
        keep_alive,
    )?;
    if !head_only {
        buf.extend_from_slice(&response.body);
    }
    w.write_all(&buf)?;
    w.flush()
}

/// Writes `HTTP/1.1 100 Continue` with an empty head and flushes, to
/// acknowledge a request that announced `Expect: 100-continue` before the
/// client sends its body.
pub(crate) fn write_continue<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    w.flush()
}

/// A streamed response writer using `Transfer-Encoding: chunked`.
///
/// [`start`](Self::start) writes the status line and headers;
/// [`write_chunk`](Self::write_chunk) writes one chunk at a time (useful for
/// streaming, e.g. token-by-token LLM output); [`finish`](Self::finish)
/// terminates the chunked body and returns the underlying writer.
pub(crate) struct ChunkedWriter<W: Write> {
    inner: W,
}

impl<W: Write> ChunkedWriter<W> {
    /// Writes the status line and headers, adding a
    /// `Transfer-Encoding: chunked` header and the `Connection` header, and
    /// returns a chunk writer. The given headers get the same CR/LF
    /// validation as [`write_response`].
    pub(crate) fn start(
        mut w: W,
        status: u16,
        headers: &[(String, String)],
        keep_alive: bool,
    ) -> io::Result<Self> {
        let head = serialize_head(status, headers, &["Transfer-Encoding: chunked"], keep_alive)?;
        w.write_all(&head)?;
        w.flush()?;
        Ok(ChunkedWriter { inner: w })
    }

    /// Writes `data` as one chunk. Empty `data` writes nothing (it must not
    /// terminate the body). Flushes so the chunk reaches the peer.
    pub(crate) fn write_chunk(&mut self, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            return self.inner.flush();
        }
        let len = data.len();
        let mut buf = Vec::with_capacity(len + 20);
        write!(buf, "{len:x}\r\n")?;
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\r\n");
        self.inner.write_all(&buf)?;
        self.inner.flush()
    }

    /// Writes the terminating zero-size chunk, flushes, and returns the
    /// underlying writer.
    pub(crate) fn finish(mut self) -> io::Result<W> {
        self.inner.write_all(b"0\r\n\r\n")?;
        self.inner.flush()?;
        Ok(self.inner)
    }
}

/// Server-sent events on top of a [`ChunkedWriter`].
///
/// [`send`](Self::send) emits `data:` events, and [`comment`](Self::comment)
/// emits a comment line that clients ignore, which makes it a convenient
/// keep-alive ping for long LLM generations.
pub(crate) struct SseWriter<W: Write> {
    inner: ChunkedWriter<W>,
}

impl<W: Write> SseWriter<W> {
    /// Starts a 200 response with `Content-Type: text/event-stream`,
    /// `Cache-Control: no-cache`, the given `extra_headers`, and the chunked
    /// and connection headers.
    pub(crate) fn start(
        w: W,
        extra_headers: &[(String, String)],
        keep_alive: bool,
    ) -> io::Result<Self> {
        let mut headers: Vec<(String, String)> = vec![
            (
                String::from("Content-Type"),
                String::from("text/event-stream"),
            ),
            (String::from("Cache-Control"), String::from("no-cache")),
        ];
        headers.extend(extra_headers.iter().cloned());
        let inner = ChunkedWriter::start(w, 200, &headers, keep_alive)?;
        Ok(SseWriter { inner })
    }

    /// Sends `data` as one event. Each line of `data` (split on `\n`, `\r\n`
    /// and `\r`) becomes its own `data: ` line, and the event ends with a
    /// blank line. Sent as a single chunk and flushed.
    pub(crate) fn send(&mut self, data: &str) -> io::Result<()> {
        let mut buf = Vec::with_capacity(data.len() + 32);
        for line in sse_lines(data) {
            buf.extend_from_slice(b"data: ");
            buf.extend_from_slice(line.as_bytes());
            buf.extend_from_slice(b"\n");
        }
        buf.push(b'\n');
        self.inner.write_chunk(&buf)
    }

    /// Sends an SSE comment line (`": text\n\n"`), which clients ignore;
    /// useful as a keep-alive ping.
    pub(crate) fn comment(&mut self, text: &str) -> io::Result<()> {
        let mut buf = Vec::with_capacity(text.len() + 8);
        buf.extend_from_slice(b": ");
        buf.extend_from_slice(text.as_bytes());
        buf.extend_from_slice(b"\n\n");
        self.inner.write_chunk(&buf)
    }

    /// Terminates the chunked response, flushes, and returns the underlying
    /// writer.
    pub(crate) fn finish(self) -> io::Result<W> {
        self.inner.finish()
    }
}

/// Splits `data` into lines on `\n`, `\r\n` and `\r`. A trailing line ending
/// does not produce a trailing empty line.
fn sse_lines(data: &str) -> Vec<&str> {
    let bytes = data.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\r' => {
                lines.push(&data[start..i]);
                i = if bytes.get(i + 1) == Some(&b'\n') {
                    i + 2
                } else {
                    i + 1
                };
                start = i;
            }
            b'\n' => {
                lines.push(&data[start..i]);
                i += 1;
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&data[start..]);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor, Read};

    /// Runs a byte string through `read_head` with the default limits.
    fn parse(input: &[u8]) -> Result<Option<RequestHead>, HttpError> {
        read_head(&mut Cursor::new(input), &Limits::default())
    }

    /// Parses a request that must be valid, panicking otherwise.
    fn request(input: &str) -> RequestHead {
        match parse(input.as_bytes()) {
            Ok(Some(head)) => head,
            other => panic!("failed to parse request: {other:?}"),
        }
    }

    /// Asserts that a request is rejected with the given HTTP status.
    fn assert_status(input: &[u8], expected: u16) {
        let err = parse(input).expect_err("input should have been rejected");
        assert_eq!(err.status(), Some(expected), "wrong error: {err}");
    }

    /// Builds a request head by hand, for driving `read_body` directly.
    fn synthetic_head(headers: &[(&str, &str)]) -> RequestHead {
        RequestHead {
            method: String::from("POST"),
            path: String::from("/"),
            query: None,
            version_minor: 1,
            headers: headers
                .iter()
                .map(|(name, value)| (String::from(*name), String::from(*value)))
                .collect(),
        }
    }

    #[test]
    fn get_with_query() {
        let head = request(
            "GET /v1/models?limit=5&x=%20 HTTP/1.1\r\nHost: localhost\r\nAccept: */*\r\n\r\n",
        );
        assert_eq!(head.method, "GET");
        assert_eq!(head.path, "/v1/models");
        assert_eq!(head.query.as_deref(), Some("limit=5&x=%20"));
        assert_eq!(head.version_minor, 1);
        assert_eq!(head.header("HOST"), Some("localhost"));
        assert_eq!(head.header("accept"), Some("*/*"));
        assert_eq!(head.header("X-Missing"), None);
        // "get" is not a known method but it is a valid token: kept as sent.
        assert_eq!(request("get /lower HTTP/1.1\r\n\r\n").method, "get");
    }

    #[test]
    fn post_with_content_length() {
        let input = b"POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhelloEXTRA";
        let mut reader = Cursor::new(input);
        let limits = Limits::default();
        let head = read_head(&mut reader, &limits).unwrap().unwrap();
        assert_eq!(head.method, "POST");
        assert_eq!(head.header("content-LENGTH"), Some("5"));
        assert_eq!(read_body(&mut reader, &head, &limits).unwrap(), b"hello");
        // The reader stops exactly at the end of the announced body.
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"EXTRA");
    }

    #[test]
    fn chunked_body_with_extensions_and_trailers() {
        let input = b"POST /upload HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: CHUNKED\r\n\r\n\
                      4;name=value\r\nWiki\r\n5\r\npedia\r\n0\r\nX-Trailer: ignored\r\n\r\n";
        let mut reader = Cursor::new(input);
        let limits = Limits::default();
        let head = read_head(&mut reader, &limits).unwrap().unwrap();
        assert_eq!(head.header("transfer-encoding"), Some("CHUNKED"));
        assert_eq!(
            read_body(&mut reader, &head, &limits).unwrap(),
            b"Wikipedia"
        );
    }

    #[test]
    fn absolute_form_target() {
        let head = request("GET http://example.com:8080/v1/models?x HTTP/1.1\r\nHost: y\r\n\r\n");
        assert_eq!(head.path, "/v1/models");
        assert_eq!(head.query.as_deref(), Some("x"));
        // Case-insensitive scheme, authority without a path.
        let head = request("GET HTTPS://example.com HTTP/1.1\r\n\r\n");
        assert_eq!(head.path, "/");
        assert_eq!(head.query, None);
        // Query without a path.
        let head = request("GET http://example.com?x HTTP/1.1\r\n\r\n");
        assert_eq!(head.path, "/");
        assert_eq!(head.query.as_deref(), Some("x"));
    }

    #[test]
    fn asterisk_form() {
        let head = request("OPTIONS * HTTP/1.1\r\nHost: x\r\n\r\n");
        assert_eq!(head.method, "OPTIONS");
        assert_eq!(head.path, "*");
        assert_eq!(head.query, None);
    }

    #[test]
    fn pipelined_requests_then_eof() {
        let input = b"GET /a HTTP/1.1\r\nHost: x\r\n\r\nGET /b?k=v HTTP/1.0\r\nHost: x\r\n\r\n";
        let mut reader = Cursor::new(input);
        let limits = Limits::default();
        let first = read_head(&mut reader, &limits).unwrap().unwrap();
        assert_eq!(first.path, "/a");
        assert!(read_body(&mut reader, &first, &limits).unwrap().is_empty());
        let second = read_head(&mut reader, &limits).unwrap().unwrap();
        assert_eq!(second.path, "/b");
        assert_eq!(second.query.as_deref(), Some("k=v"));
        assert_eq!(second.version_minor, 0);
        assert!(read_body(&mut reader, &second, &limits).unwrap().is_empty());
        // A clean end of the connection.
        assert!(read_head(&mut reader, &limits).unwrap().is_none());
        // An empty input is also a clean end.
        assert!(parse(b"").unwrap().is_none());
        // So is an end after tolerated leading empty lines.
        assert!(parse(b"\r\n\r\n").unwrap().is_none());
    }

    #[test]
    fn leading_empty_lines() {
        let head = request("\r\n\r\n\r\n\r\nGET / HTTP/1.1\r\nHost: x\r\n\r\n");
        assert_eq!(head.path, "/");
        // Five leading empty lines are one too many.
        assert_status(b"\r\n\r\n\r\n\r\n\r\nGET / HTTP/1.1\r\n\r\n", 400);
    }

    #[test]
    fn lf_only_line_endings() {
        let head = request("GET /a?b=c HTTP/1.1\nHost: x\nX-Y: 1\n\n");
        assert_eq!(head.path, "/a");
        assert_eq!(head.query.as_deref(), Some("b=c"));
        assert_eq!(head.header("host"), Some("x"));
        assert_eq!(head.header("x-y"), Some("1"));
    }

    #[test]
    fn keep_alive_and_expect_rules() {
        // HTTP/1.1 defaults to keep-alive.
        assert!(request("GET / HTTP/1.1\r\n\r\n").keep_alive());
        assert!(!request("GET / HTTP/1.1\r\nConnection: close\r\n\r\n").keep_alive());
        assert!(request("GET / HTTP/1.1\r\nConnection: keep-alive\r\n\r\n").keep_alive());
        // Token lists must work, in both versions.
        assert!(request("GET / HTTP/1.1\r\nConnection: keep-alive, Upgrade\r\n\r\n").keep_alive());
        // "close" wins over a later "keep-alive" token.
        assert!(!request("GET / HTTP/1.1\r\nConnection: close, keep-alive\r\n\r\n").keep_alive());
        // HTTP/1.0 defaults to close.
        assert!(!request("GET / HTTP/1.0\r\n\r\n").keep_alive());
        assert!(request("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n").keep_alive());
        assert!(request("GET / HTTP/1.0\r\nConnection: Keep-Alive, Upgrade\r\n\r\n").keep_alive());
        assert!(!request("GET / HTTP/1.0\r\nConnection: close\r\n\r\n").keep_alive());
        // Expect: 100-continue, case-insensitive, in token lists.
        assert!(request("POST / HTTP/1.1\r\nExpect: 100-Continue\r\n\r\n").expects_continue());
        assert!(
            request("POST / HTTP/1.1\r\nExpect: token, 100-CONTINUE\r\n\r\n").expects_continue()
        );
        assert!(!request("POST / HTTP/1.1\r\nExpect: 200-ok\r\n\r\n").expects_continue());
        assert!(!request("POST / HTTP/1.1\r\n\r\n").expects_continue());
    }

    #[test]
    fn malformed_request_line() {
        assert_status(b"G\x01T / HTTP/1.1\r\n\r\n", 400); // control byte in method
        assert_status(b"G@T / HTTP/1.1\r\n\r\n", 400); // '@' is not a token char
        assert_status(b"GET /\x00 HTTP/1.1\r\n\r\n", 400); // NUL in target
        assert_status(b"GET / HTTP/1.1 extra\r\n\r\n", 400); // four parts
        assert_status(b"GET  / HTTP/1.1\r\n\r\n", 400); // empty target
        assert_status(b"GET /HTTP/1.1\r\n\r\n", 400); // missing separator
        assert_status(b"GET / HTTP/1.1\r\nGET /\r\n\r\n", 400); // not a header line
    }

    #[test]
    fn version_errors() {
        assert_status(b"GET / HTTP/2.0\r\n\r\n", 505);
        assert_status(b"GET / HTTP/1.9\r\n\r\n", 505);
        assert_status(b"GET / HTTP/0.9\r\n\r\n", 505);
        assert_status(b"GET / http/1.1\r\n\r\n", 400); // the name is case-sensitive
        assert_status(b"GET / HTTP/11.1\r\n\r\n", 400);
        assert_status(b"GET / HTTP/1.x\r\n\r\n", 400);
    }

    #[test]
    fn header_errors() {
        assert_status(b"GET / HTTP/1.1\r\nHost : x\r\n\r\n", 400); // space before colon
        assert_status(b"GET / HTTP/1.1\r\nX-A: 1\r\n  fold\r\n\r\n", 400); // obs-fold
        assert_status(b"GET / HTTP/1.1\r\nX-A: 1\r\n\tfold\r\n\r\n", 400); // obs-fold (tab)
        assert_status(b"GET / HTTP/1.1\r\nNoColon\r\n\r\n", 400);
        assert_status(b"GET / HTTP/1.1\r\n: empty\r\n\r\n", 400); // empty name
        assert_status(b"GET / HTTP/1.1\r\nX-Nul: a\x00b\r\n\r\n", 400);
        assert_status(b"GET / HTTP/1.1\r\nX-Cr: a\rb\r\n\r\n", 400); // bare CR
    }

    #[test]
    fn head_limits() {
        // An over-size head is rejected with 431 without ever buffering it.
        let mut huge = b"GET / HTTP/1.1\r\nX-Big: ".to_vec();
        huge.resize(huge.len() + 20 * 1024, b'a');
        huge.extend_from_slice(b"\r\n\r\n");
        assert_status(&huge, 431);
        // One header too many is rejected with 431 ...
        let mut many = b"GET / HTTP/1.1\r\n".to_vec();
        for i in 0..101 {
            many.extend_from_slice(format!("X-{i:03}: v\r\n").as_bytes());
        }
        many.extend_from_slice(b"\r\n");
        assert_status(&many, 431);
        // ... while exactly `max_headers` headers are fine.
        let mut hundred = b"GET / HTTP/1.1\r\n".to_vec();
        for i in 0..100 {
            hundred.extend_from_slice(format!("X-{i:03}: v\r\n").as_bytes());
        }
        hundred.extend_from_slice(b"\r\n");
        assert!(parse(&hundred).unwrap().is_some());
    }

    #[test]
    fn content_length_errors() {
        assert_status(b"POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n", 400);
        assert_status(b"POST / HTTP/1.1\r\nContent-Length: \r\n\r\n", 400);
        assert_status(b"POST / HTTP/1.1\r\nContent-Length: +5\r\n\r\n", 400);
        assert_status(b"POST / HTTP/1.1\r\nContent-Length: 5, 5\r\n\r\n", 400);
        assert_status(
            b"POST / HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 6\r\n\r\n",
            400,
        );
        // Both framing headers at once: request smuggling, rejected.
        assert_status(
            b"POST / HTTP/1.1\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n",
            400,
        );
        // Transfer encodings other than exactly "chunked".
        assert_status(b"POST / HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n", 501);
        assert_status(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\n\r\n",
            501,
        );
        assert_status(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n",
            501,
        );
        // A Content-Length over the limit is rejected before reading, and one
        // that does not even fit in u64 likewise counts as too large.
        assert_status(b"POST / HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n", 413);
        assert_status(
            b"POST / HTTP/1.1\r\nContent-Length: 99999999999999999999999\r\n\r\n",
            413,
        );
    }

    #[test]
    fn oversized_body_is_413() {
        let limits = Limits {
            max_body_bytes: 8,
            ..Limits::default()
        };
        // Content-Length over the limit: rejected at head time.
        let mut reader =
            Cursor::new(b"POST / HTTP/1.1\r\nContent-Length: 16\r\n\r\n0123456789abcdef");
        let err = read_head(&mut reader, &limits).unwrap_err();
        assert_eq!(err.status(), Some(413));
        // A chunked body whose chunks sum over the limit: rejected mid-read.
        let mut reader = Cursor::new(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n01234\r\n5\r\n56789\r\n0\r\n\r\n",
        );
        let head = read_head(&mut reader, &limits).unwrap().unwrap();
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(413));
        // A single chunk over the limit.
        let mut reader = Cursor::new(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n9\r\n012345678\r\n0\r\n\r\n",
        );
        let head = read_head(&mut reader, &limits).unwrap().unwrap();
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(413));
    }

    #[test]
    fn chunk_framing_errors() {
        let limits = Limits::default();
        let head = request("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        // Not a hex size.
        let mut reader = Cursor::new(b"zz\r\n");
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(400));
        // Missing CRLF after chunk data.
        let mut reader = Cursor::new(b"3\r\nabcX");
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(400));
        // CR not followed by LF.
        let mut reader = Cursor::new(b"3\r\nabc\rX");
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(400));
        // A chunk size line longer than the internal limit.
        let mut long = b"3".to_vec();
        long.resize(long.len() + 2000, b';');
        long.extend_from_slice(b"\r\nabc\r\n0\r\n\r\n");
        let mut reader = Cursor::new(long);
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(400));
        // A trailer section that is not header-shaped.
        let mut reader = Cursor::new(b"0\r\nnot a header\r\n\r\n");
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(400));
        // Bare LF line endings are accepted throughout the chunked body.
        let mut reader = Cursor::new(b"3\nabc\n0\n\n");
        assert_eq!(read_body(&mut reader, &head, &limits).unwrap(), b"abc");
    }

    #[test]
    fn truncated_requests() {
        let limits = Limits::default();
        // Truncated in the middle of the request line.
        let err = parse(b"GET / HT").unwrap_err();
        assert_eq!(err.status(), Some(400));
        assert!(matches!(err, HttpError::Truncated));
        // Truncated in the middle of the headers.
        assert_eq!(
            parse(b"GET / HTTP/1.1\r\nHost: x").unwrap_err().status(),
            Some(400)
        );
        // Headers not terminated by a blank line.
        assert_eq!(
            parse(b"GET / HTTP/1.1\r\nHost: x\r\n")
                .unwrap_err()
                .status(),
            Some(400)
        );
        // Truncated in the middle of a Content-Length body.
        let mut reader = Cursor::new(b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nshort");
        let head = read_head(&mut reader, &limits).unwrap().unwrap();
        let err = read_body(&mut reader, &head, &limits).unwrap_err();
        assert_eq!(err.status(), Some(400));
        assert!(matches!(err, HttpError::Truncated));
        // Truncated in the middle of a chunked body.
        let mut reader =
            Cursor::new(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nab");
        let head = read_head(&mut reader, &limits).unwrap().unwrap();
        assert_eq!(
            read_body(&mut reader, &head, &limits).unwrap_err().status(),
            Some(400)
        );
        // Truncated before the trailer section ends.
        let mut reader = Cursor::new(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n");
        let head = read_head(&mut reader, &limits).unwrap().unwrap();
        assert_eq!(
            read_body(&mut reader, &head, &limits).unwrap_err().status(),
            Some(400)
        );
    }

    #[test]
    fn write_response_bytes() {
        let response = Response::json(200, "{\"ok\":true}");
        let mut out = Vec::new();
        write_response(&mut out, &response, true, false).unwrap();
        assert_eq!(
            out,
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\
              Connection: keep-alive\r\n\r\n{\"ok\":true}"
        );
        // Keep-alive off.
        let mut out = Vec::new();
        write_response(&mut out, &response, false, false).unwrap();
        assert_eq!(
            out,
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\
              Connection: close\r\n\r\n{\"ok\":true}"
        );
        // HEAD: the headers, including Content-Length, but no body.
        let mut out = Vec::new();
        write_response(&mut out, &response, true, true).unwrap();
        assert_eq!(
            out,
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\
              Connection: keep-alive\r\n\r\n"
        );
        // Plain text plus a builder header.
        let mut response = Response::new(404)
            .header("Content-Type", "text/plain; charset=utf-8")
            .header("X-Reason", "missing");
        response.body = b"nope".to_vec();
        let mut out = Vec::new();
        write_response(&mut out, &response, false, false).unwrap();
        assert_eq!(
            out,
            b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain; charset=utf-8\r\nX-Reason: missing\r\n\
              Content-Length: 4\r\nConnection: close\r\n\r\nnope"
        );
        // Empty body.
        let mut out = Vec::new();
        write_response(&mut out, &Response::new(204), true, false).unwrap();
        assert_eq!(
            out,
            b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n"
        );
        // The interim 100 response.
        let mut out = Vec::new();
        write_continue(&mut out).unwrap();
        assert_eq!(out, b"HTTP/1.1 100 Continue\r\n\r\n");
    }

    #[test]
    fn rejects_header_injection() {
        for bad in ["x\r\nEvil: 1", "x\nEvil: 1", "x\rEvil: 1"] {
            let response = Response::new(200).header("X-Bad", bad);
            let mut out = Vec::new();
            let err = write_response(&mut out, &response, true, false).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
            // Nothing was written.
            assert!(out.is_empty());
        }
        // A CR/LF in a header name is rejected too.
        let response = Response::new(200).header("X-\rBad", "v");
        let mut out = Vec::new();
        assert!(write_response(&mut out, &response, true, false).is_err());
        assert!(out.is_empty());
        // And through the chunked and SSE writers.
        let bad = vec![(String::from("X-Bad"), String::from("v\r\n"))];
        let mut out = Vec::new();
        assert!(ChunkedWriter::start(&mut out, 200, &bad, true).is_err());
        assert!(out.is_empty());
        let mut out = Vec::new();
        assert!(SseWriter::start(&mut out, &bad, true).is_err());
        assert!(out.is_empty());
    }

    #[test]
    fn chunked_writer_bytes() {
        let headers = vec![(
            String::from("Content-Type"),
            String::from("application/x-ndjson"),
        )];
        let mut out = Vec::new();
        let mut writer = ChunkedWriter::start(&mut out, 200, &headers, true).unwrap();
        writer.write_chunk(b"hello").unwrap();
        writer.write_chunk(b"").unwrap(); // writes nothing, flushes
        writer.write_chunk(b"world").unwrap();
        writer.finish().unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(b"HTTP/1.1 200 OK\r\n");
        expected.extend_from_slice(b"Content-Type: application/x-ndjson\r\n");
        expected.extend_from_slice(b"Transfer-Encoding: chunked\r\n");
        expected.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
        expected.extend_from_slice(b"5\r\nhello\r\n");
        expected.extend_from_slice(b"5\r\nworld\r\n");
        expected.extend_from_slice(b"0\r\n\r\n");
        assert_eq!(out, expected);
    }

    #[test]
    fn sse_writer_bytes() {
        let mut out = Vec::new();
        let mut writer = SseWriter::start(&mut out, &[], true).unwrap();
        // Multi-line data: each line becomes its own "data: " line.
        writer.send("first\nsecond\r\nthird\rfourth").unwrap();
        writer.comment("keep-alive").unwrap();
        writer.send("done").unwrap();
        writer.finish().unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(b"HTTP/1.1 200 OK\r\n");
        expected.extend_from_slice(b"Content-Type: text/event-stream\r\n");
        expected.extend_from_slice(b"Cache-Control: no-cache\r\n");
        expected.extend_from_slice(b"Transfer-Encoding: chunked\r\n");
        expected.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
        // 0x33 = 51 bytes: four data lines plus the blank line.
        expected.extend_from_slice(
            b"33\r\ndata: first\ndata: second\ndata: third\ndata: fourth\n\n\r\n",
        );
        // 0x0e = 14 bytes: the comment plus its blank line.
        expected.extend_from_slice(b"e\r\n: keep-alive\n\n\r\n");
        // 0x0c = 12 bytes.
        expected.extend_from_slice(b"c\r\ndata: done\n\n\r\n");
        expected.extend_from_slice(b"0\r\n\r\n");
        assert_eq!(out, expected);
    }

    #[test]
    fn sse_extra_headers() {
        let extra = vec![(String::from("X-Accel-Buffering"), String::from("no"))];
        let mut out = Vec::new();
        let writer = SseWriter::start(&mut out, &extra, false).unwrap();
        writer.finish().unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(b"HTTP/1.1 200 OK\r\n");
        expected.extend_from_slice(b"Content-Type: text/event-stream\r\n");
        expected.extend_from_slice(b"Cache-Control: no-cache\r\n");
        expected.extend_from_slice(b"X-Accel-Buffering: no\r\n");
        expected.extend_from_slice(b"Transfer-Encoding: chunked\r\n");
        expected.extend_from_slice(b"Connection: close\r\n\r\n0\r\n\r\n");
        assert_eq!(out, expected);
    }

    #[test]
    fn sse_line_splitting() {
        assert_eq!(sse_lines("a\nb"), vec!["a", "b"]);
        assert_eq!(sse_lines("a\r\nb"), vec!["a", "b"]);
        assert_eq!(sse_lines("a\rb"), vec!["a", "b"]);
        assert_eq!(sse_lines("a\n"), vec!["a"]);
        assert_eq!(sse_lines("a"), vec!["a"]);
        assert_eq!(sse_lines(""), Vec::<&str>::new());
        assert_eq!(sse_lines("a\n\nb"), vec!["a", "", "b"]);
    }

    #[test]
    fn reason_phrases() {
        assert_eq!(reason_phrase(100), "Continue");
        assert_eq!(reason_phrase(200), "OK");
        assert_eq!(reason_phrase(400), "Bad Request");
        assert_eq!(reason_phrase(404), "Not Found");
        assert_eq!(reason_phrase(413), "Content Too Large");
        assert_eq!(reason_phrase(431), "Request Header Fields Too Large");
        assert_eq!(reason_phrase(501), "Not Implemented");
        assert_eq!(reason_phrase(505), "HTTP Version Not Supported");
        assert_eq!(reason_phrase(599), "Unknown");
    }

    #[test]
    fn http_error_trait_impls() {
        let err = HttpError::Io(io::Error::new(io::ErrorKind::InvalidInput, "boom"));
        assert!(std::error::Error::source(&err).is_some());
        assert_eq!(err.status(), None);
        assert_eq!(err.to_string(), "i/o error: boom");
        let err = HttpError::Malformed("bad");
        assert!(std::error::Error::source(&err).is_none());
        assert_eq!(err.to_string(), "malformed request: bad");
        assert_eq!(err.status(), Some(400));
    }

    /// A tiny xorshift64 generator for the fuzz test.
    struct XorShift(u64);

    impl XorShift {
        fn new(seed: u64) -> Self {
            // The generator sticks at zero, so force a set bit.
            XorShift(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn no_panics_on_prefixes() {
        let samples: [&[u8]; 5] = [
            b"GET /v1/models?x=1 HTTP/1.1\r\nHost: a\r\nAccept: */*\r\n\r\n",
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\nhello",
            b"POST /up HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n\
              4;ext=1\r\nWiki\r\n5\r\npedia\r\n0\r\nT: 1\r\n\r\n",
            b"GET http://h/x?y HTTP/1.0\r\nConnection: keep-alive\r\n\r\n",
            b"OPTIONS * HTTP/1.1\r\n\r\n",
        ];
        let limits = Limits::default();
        for sample in samples {
            for end in 0..=sample.len() {
                let mut reader = Cursor::new(&sample[..end]);
                if let Ok(Some(head)) = read_head(&mut reader, &limits) {
                    let pos = reader.position() as usize;
                    let mut body_reader = Cursor::new(&sample[pos..end]);
                    let _ = read_body(&mut body_reader, &head, &limits);
                }
            }
        }
    }

    #[test]
    fn no_panics_on_random_bytes() {
        let mut rng = XorShift::new(0x9E37_79B9_7F4A_7C15);
        let limits = Limits::default();
        let tiny = Limits {
            max_head_bytes: 40,
            max_headers: 3,
            max_body_bytes: 8,
        };
        let length_head = synthetic_head(&[("Content-Length", "8")]);
        let chunked_head = synthetic_head(&[("Transfer-Encoding", "chunked")]);
        for _ in 0..5000 {
            let len = (rng.next_u64() % 97) as usize;
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                bytes.push((rng.next_u64() & 0xff) as u8);
            }
            // Through read_head, and read_body if a head comes out.
            let mut reader = Cursor::new(bytes.clone());
            if let Ok(Some(head)) = read_head(&mut reader, &limits) {
                let pos = reader.position() as usize;
                let mut body_reader = Cursor::new(&bytes[pos..]);
                let _ = read_body(&mut body_reader, &head, &limits);
            }
            // Through read_head again with tiny limits, to reach the limit
            // error paths.
            let mut reader = Cursor::new(bytes.clone());
            let _ = read_head(&mut reader, &tiny);
            // And through read_body directly, with both framing kinds.
            let mut body_reader = Cursor::new(bytes.clone());
            let _ = read_body(&mut body_reader, &length_head, &tiny);
            let mut body_reader = Cursor::new(bytes);
            let _ = read_body(&mut body_reader, &chunked_head, &tiny);
        }
    }

    #[test]
    fn end_to_end_tcp_with_continue_and_sse() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let stream = std::net::TcpStream::connect(addr).unwrap();
            let mut writer = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream);
            let body = b"{\"messages\":[]}";
            let len = body.len();
            let request = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
                 Expect: 100-continue\r\nContent-Length: {len}\r\n\r\n"
            );
            writer.write_all(request.as_bytes()).unwrap();
            // The client waits for the interim 100 response before sending
            // the body.
            let mut interim = [0u8; 25];
            reader.read_exact(&mut interim).unwrap();
            assert_eq!(&interim, b"HTTP/1.1 100 Continue\r\n\r\n");
            writer.write_all(body).unwrap();
            // Read the whole streamed response until the server closes.
            let mut response = Vec::new();
            reader.read_to_end(&mut response).unwrap();
            let mut expected = Vec::new();
            expected.extend_from_slice(b"HTTP/1.1 200 OK\r\n");
            expected.extend_from_slice(b"Content-Type: text/event-stream\r\n");
            expected.extend_from_slice(b"Cache-Control: no-cache\r\n");
            expected.extend_from_slice(b"Transfer-Encoding: chunked\r\n");
            expected.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
            expected.extend_from_slice(b"f\r\ndata: token 1\n\n\r\n");
            expected.extend_from_slice(b"f\r\ndata: token 2\n\n\r\n");
            expected.extend_from_slice(b"f\r\ndata: token 3\n\n\r\n");
            expected.extend_from_slice(b"0\r\n\r\n");
            assert_eq!(response, expected);
        });

        let (stream, _) = listener.accept().unwrap();
        let mut write_half = stream.try_clone().unwrap();
        let limits = Limits::default();
        // Scope the read half so the server's end of the socket closes
        // before we wait for the client thread.
        let head = {
            let mut read_half = BufReader::new(stream);
            let head = read_head(&mut read_half, &limits).unwrap().unwrap();
            assert_eq!(head.method, "POST");
            assert_eq!(head.path, "/v1/chat/completions");
            assert_eq!(head.query, None);
            assert!(head.expects_continue());
            assert!(head.keep_alive());
            write_continue(&mut write_half).unwrap();
            let body = read_body(&mut read_half, &head, &limits).unwrap();
            assert_eq!(body, b"{\"messages\":[]}");
            head
        };
        let mut sse = SseWriter::start(&mut write_half, &[], head.keep_alive()).unwrap();
        sse.send("token 1").unwrap();
        sse.send("token 2").unwrap();
        sse.send("token 3").unwrap();
        sse.finish().unwrap();
        drop(write_half);
        client.join().unwrap();
    }
}
