//! `nibble inspect`: print a GGUF file's header, metadata and tensor table.

use std::error::Error;
use std::fmt::Write as _;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use nibble::gguf::{Array, GgmlType, Gguf, Storage, Value};

/// Array elements shown before truncating, without `--all`.
const ARRAY_PREVIEW: usize = 6;
/// Characters of a metadata string shown before truncating, without `--all`.
const STRING_PREVIEW: usize = 72;
/// Characters of each string inside an array shown before truncating, without `--all`.
const ELEMENT_PREVIEW: usize = 24;

pub(crate) struct Options {
    pub(crate) path: PathBuf,
    pub(crate) all: bool,
    pub(crate) metadata: bool,
    pub(crate) tensors: bool,
    pub(crate) mmap: bool,
}

pub(crate) fn run(opts: &Options) -> Result<(), Box<dyn Error>> {
    let display = opts.path.display();
    let storage = if opts.mmap {
        // SAFETY: model files are read-only inputs. As in llama.cpp, we assume
        // nothing modifies or truncates the file while it is mapped.
        unsafe { Storage::open(&opts.path) }
    } else {
        Storage::read(&opts.path)
    }
    .map_err(|e| format!("{display}: {e}"))?;
    let gguf = Gguf::parse(&storage).map_err(|e| format!("{display}: {e}"))?;

    let mut out = BufWriter::new(io::stdout().lock());
    summary(&mut out, opts, &storage, &gguf)?;
    if opts.metadata {
        metadata(&mut out, &gguf, opts.all)?;
    }
    if opts.tensors {
        tensors(&mut out, &gguf)?;
    }
    out.flush()?;
    Ok(())
}

fn summary(
    out: &mut impl Write,
    opts: &Options,
    storage: &Storage,
    g: &Gguf<'_>,
) -> io::Result<()> {
    let how = if storage.is_mapped() {
        "memory-mapped"
    } else {
        "read into memory"
    };
    writeln!(
        out,
        "file     {} ({}, {how})",
        opts.path.display(),
        fmt_bytes(storage.len() as u64)
    )?;
    writeln!(
        out,
        "format   GGUF v{}, alignment {}, data at byte {}",
        g.version(),
        g.alignment(),
        g.data_offset()
    )?;

    let arch = g.architecture().unwrap_or("(no general.architecture)");
    match g.get("general.name").and_then(|v| v.as_str()) {
        Some(name) => writeln!(out, "model    {name} [{arch}]")?,
        None => writeln!(out, "model    [{arch}]")?,
    }

    let params: u64 = g.tensors().iter().map(|t| t.shape().n_elements()).sum();
    let bytes: u64 = g
        .tensors()
        .iter()
        .filter_map(|t| t.data())
        .map(|d| d.len() as u64)
        .sum();
    writeln!(
        out,
        "tensors  {} tensors, {} parameters, {}",
        g.tensors().len(),
        fmt_count(params),
        fmt_bytes(bytes)
    )?;

    // Per-type totals, largest first.
    let mut by_type: Vec<(GgmlType, usize, u64)> = Vec::new();
    for t in g.tensors() {
        let size = t.data().map_or(0, |d| d.len() as u64);
        match by_type.iter_mut().find(|(ty, ..)| *ty == t.ggml_type()) {
            Some((_, count, total)) => {
                *count += 1;
                *total += size;
            }
            None => by_type.push((t.ggml_type(), 1, size)),
        }
    }
    by_type.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));
    let parts: Vec<String> = by_type
        .iter()
        .map(|(ty, count, total)| match ty.layout() {
            Some(_) => format!("{ty} x{count} ({})", fmt_bytes(*total)),
            None => format!("{ty} x{count} (unknown type, size unknown)"),
        })
        .collect();
    if !parts.is_empty() {
        writeln!(out, "types    {}", parts.join(", "))?;
    }
    Ok(())
}

fn metadata(out: &mut impl Write, g: &Gguf<'_>, all: bool) -> io::Result<()> {
    writeln!(out, "\nmetadata ({} keys)", g.metadata().len())?;
    let key_width = g
        .metadata()
        .iter()
        .map(|(k, _)| k.len())
        .max()
        .unwrap_or(0)
        .min(48);
    let described: Vec<(String, String)> =
        g.metadata().iter().map(|(_, v)| describe(v, all)).collect();
    let type_width = described.iter().map(|(ty, _)| ty.len()).max().unwrap_or(0);
    for ((key, _), (ty, value)) in g.metadata().iter().zip(&described) {
        writeln!(out, "  {key:<key_width$}  {ty:<type_width$}  {value}")?;
    }
    Ok(())
}

fn tensors(out: &mut impl Write, g: &Gguf<'_>) -> io::Result<()> {
    writeln!(out, "\ntensors ({})", g.tensors().len())?;
    let rows: Vec<[String; 6]> = g
        .tensors()
        .iter()
        .map(|t| {
            [
                t.name().to_owned(),
                t.ggml_type().to_string(),
                t.shape().to_string(),
                fmt_count(t.shape().n_elements()),
                t.data()
                    .map_or_else(|| "?".to_owned(), |d| fmt_bytes(d.len() as u64)),
                format!("{:#x}", g.data_offset() as u64 + t.offset()),
            ]
        })
        .collect();
    let header = ["name", "type", "shape", "params", "size", "offset"].map(str::to_owned);
    let mut widths = header.clone().map(|h| h.len());
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.len());
        }
    }
    let index_width = rows.len().saturating_sub(1).to_string().len().max(1);
    let line = |out: &mut dyn Write, index: &str, r: &[String; 6]| {
        let [w0, w1, w2, w3, w4, w5] = widths;
        writeln!(
            out,
            "  {index:>index_width$}  {:<w0$}  {:<w1$}  {:<w2$}  {:>w3$}  {:>w4$}  {:>w5$}",
            r[0], r[1], r[2], r[3], r[4], r[5]
        )
    };
    line(out, "#", &header)?;
    for (i, row) in rows.iter().enumerate() {
        line(out, &i.to_string(), row)?;
    }
    Ok(())
}

/// The type column and the value column for one metadata entry.
fn describe(value: &Value<'_>, all: bool) -> (String, String) {
    match value {
        Value::Array(a) => (
            format!("[{}; {}]", a.elem_type(), a.len()),
            fmt_array(a, all),
        ),
        v => (
            v.value_type().to_string(),
            fmt_scalar(v, if all { usize::MAX } else { STRING_PREVIEW }),
        ),
    }
}

fn fmt_array(a: &Array<'_>, all: bool) -> String {
    let (limit, string_limit) = if all {
        (usize::MAX, usize::MAX)
    } else {
        (ARRAY_PREVIEW, ELEMENT_PREVIEW)
    };
    let mut out = String::from("[");
    for (i, v) in a.iter().enumerate() {
        if i == limit {
            let _ = write!(out, ", ... {} more", a.len() - i);
            break;
        }
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&fmt_scalar(&v, string_limit));
    }
    out.push(']');
    out
}

fn fmt_scalar(value: &Value<'_>, max_chars: usize) -> String {
    match value {
        Value::U8(v) => v.to_string(),
        Value::I8(v) => v.to_string(),
        Value::U16(v) => v.to_string(),
        Value::I16(v) => v.to_string(),
        Value::U32(v) => v.to_string(),
        Value::I32(v) => v.to_string(),
        Value::U64(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
        // Debug formatting keeps small values readable (`1e-6`, not `0.000001`).
        Value::F32(v) => format!("{v:?}"),
        Value::F64(v) => format!("{v:?}"),
        Value::Bool(v) => v.to_string(),
        Value::String(s) => quote(&s.to_string(), max_chars),
        Value::Array(a) => format!("[{}; {}]", a.elem_type(), a.len()),
    }
}

/// Quotes and escapes `s`, keeping at most `max_chars` characters.
fn quote(s: &str, max_chars: usize) -> String {
    let mut out = String::from("\"");
    for (i, c) in s.chars().enumerate() {
        if i == max_chars {
            let rest = s.chars().count() - i;
            let _ = write!(out, "\"... (+{rest} chars)");
            return out;
        }
        out.extend(c.escape_debug());
    }
    out.push('"');
    out
}

fn fmt_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn fmt_count(n: u64) -> String {
    let v = n as f64;
    if v >= 1e9 {
        format!("{:.2}B", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.2}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.1}K", v / 1e3)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_sizes_and_counts() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(1536), "1.5 KiB");
        assert_eq!(fmt_bytes(833_592_096), "795.0 MiB");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(254_279_680), "254.28M");
        assert_eq!(fmt_count(1_500_000_000), "1.50B");
    }

    #[test]
    fn quotes_and_truncates_strings() {
        assert_eq!(quote("a\"b\n", 10), r#""a\"b\n""#);
        assert_eq!(quote("abcdef", 3), "\"abc\"... (+3 chars)");
    }
}
