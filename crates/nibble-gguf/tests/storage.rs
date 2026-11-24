//! Storage tests: memory mapping and reading agree with the file on disk.

use std::path::PathBuf;

use nibble_gguf::{Gguf, GgufWriter, OwnedValue, Storage, ValueType};

/// A file in the temp directory, removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str, contents: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!("nibble-gguf-{}-{name}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn sample() -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add("general.architecture", "test")
        .add("tokens", OwnedValue::array(ValueType::String, ["x", "y"]));
    w.add_tensor(
        "t",
        nibble_gguf::GgmlType::F32,
        &[2],
        [0u8, 0, 128, 63, 0, 0, 0, 64],
    );
    w.to_bytes()
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot call mmap")]
fn mapped_and_read_bytes_match_the_file() {
    let bytes = sample();
    let file = TempFile::new("match.gguf", &bytes);

    // SAFETY: the temp file is private to this test and not modified while mapped.
    let mapped = unsafe { Storage::open(&file.0) }.unwrap();
    assert_eq!(mapped.is_mapped(), nibble_gguf::Mmap::SUPPORTED);
    assert_eq!(&*mapped, &bytes[..]);

    let read = Storage::read(&file.0).unwrap();
    assert!(!read.is_mapped());
    assert_eq!(&*read, &bytes[..]);
    assert_eq!(read.as_ptr().addr() % 64, 0);

    let g = Gguf::parse(&mapped).unwrap();
    assert_eq!(g.architecture(), Some("test"));
    assert_eq!(
        g.tensor("t").unwrap().data().unwrap(),
        &bytes[g.data_offset()..g.data_offset() + 8]
    );
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot call mmap")]
fn empty_files_map_to_empty_slices() {
    let file = TempFile::new("empty.gguf", &[]);
    // SAFETY: the temp file is private to this test and not modified while mapped.
    let mapped = unsafe { Storage::open(&file.0) }.unwrap();
    assert!(mapped.is_empty());
    assert!(Gguf::parse(&mapped).is_err());
}

#[test]
fn missing_files_report_not_found() {
    let path = std::env::temp_dir().join("nibble-gguf-does-not-exist.gguf");
    let err = Storage::read(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
