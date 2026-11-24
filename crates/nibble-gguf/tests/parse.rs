//! Parser tests: round trips through `GgufWriter`, and hand-built corrupt files.

use nibble_gguf::{Error, GgmlType, Gguf, GgufWriter, OwnedValue, Value, ValueType};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn sample_writer() -> GgufWriter {
    let mut w = GgufWriter::new();
    w.add("general.architecture", "test")
        .add("test.u8", 7u8)
        .add("test.i8", -7i8)
        .add("test.u16", 700u16)
        .add("test.i16", -700i16)
        .add("test.u32", 70_000u32)
        .add("test.i32", -70_000i32)
        .add("test.f32", 1.5f32)
        .add("test.bool", true)
        .add("test.u64", u64::MAX)
        .add("test.i64", i64::MIN)
        .add("test.f64", -2.25f64)
        .add(
            "test.tokens",
            OwnedValue::array(ValueType::String, ["a", "bc", "", "é"]),
        )
        .add(
            "test.scores",
            OwnedValue::array(ValueType::F32, [0.5f32, -1.0, 2.0]),
        )
        .add(
            "test.flags",
            OwnedValue::array(ValueType::Bool, [true, false, true]),
        )
        .add(
            "test.empty",
            OwnedValue::array(ValueType::U32, Vec::<u32>::new()),
        );
    w.add_tensor(
        "weight",
        GgmlType::F32,
        &[4, 2],
        f32_bytes(&[1., 2., 3., 4., 5., 6., 7., 8.]),
    )
    .add_tensor("quant", GgmlType::Q8_0, &[64], vec![0xAB; 68])
    .add_tensor("norm", GgmlType::F32, &[3], f32_bytes(&[0.25, 0.5, 0.75]));
    w
}

#[test]
fn metadata_round_trips() {
    let bytes = sample_writer().to_bytes();
    let g = Gguf::parse(&bytes).unwrap();

    assert_eq!(g.version(), 3);
    assert_eq!(g.architecture(), Some("test"));
    assert_eq!(g.metadata().len(), 16);
    assert_eq!(g.get("test.u8"), Some(Value::U8(7)));
    assert_eq!(g.get("test.i8"), Some(Value::I8(-7)));
    assert_eq!(g.get("test.u16"), Some(Value::U16(700)));
    assert_eq!(g.get("test.i16"), Some(Value::I16(-700)));
    assert_eq!(g.get("test.u32").and_then(|v| v.as_u32()), Some(70_000));
    assert_eq!(g.get("test.i32").and_then(|v| v.as_i64()), Some(-70_000));
    assert_eq!(g.get("test.f32").and_then(|v| v.as_f32()), Some(1.5));
    assert_eq!(g.get("test.bool").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(g.get("test.u64").and_then(|v| v.as_u64()), Some(u64::MAX));
    assert_eq!(g.get("test.i64").and_then(|v| v.as_i64()), Some(i64::MIN));
    assert_eq!(g.get("test.f64"), Some(Value::F64(-2.25)));
    assert_eq!(g.get("missing"), None);

    let tokens = g.get("test.tokens").and_then(|v| v.as_array()).unwrap();
    assert_eq!(tokens.elem_type(), ValueType::String);
    let tokens: Vec<_> = tokens
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(tokens, ["a", "bc", "", "é"]);

    let scores = g.get("test.scores").and_then(|v| v.as_array()).unwrap();
    assert_eq!(scores.len(), 3);
    assert_eq!(scores.get(2), Some(Value::F32(2.0)));
    assert_eq!(scores.get(3), None);

    let flags: Vec<_> = g
        .get("test.flags")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(
        flags,
        [Value::Bool(true), Value::Bool(false), Value::Bool(true)]
    );

    let empty = g.get("test.empty").unwrap().as_array().unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.iter().next(), None);
}

#[test]
fn tensors_round_trip() {
    let bytes = sample_writer().to_bytes();
    let g = Gguf::parse(&bytes).unwrap();

    assert_eq!(g.alignment(), 32);
    assert_eq!(g.data_offset() % 32, 0);
    let names: Vec<_> = g.tensors().iter().map(|t| t.name()).collect();
    assert_eq!(names, ["weight", "quant", "norm"]);

    let weight = g.tensor("weight").unwrap();
    assert_eq!(weight.ggml_type(), GgmlType::F32);
    assert_eq!(weight.shape().dims(), [4, 2]);
    assert_eq!(weight.shape().ne(3), 1);
    assert_eq!(weight.shape().n_elements(), 8);
    assert_eq!(weight.offset(), 0);
    assert_eq!(
        weight.data().unwrap(),
        f32_bytes(&[1., 2., 3., 4., 5., 6., 7., 8.])
    );

    let quant = g.tensor("quant").unwrap();
    assert_eq!(quant.offset(), 32);
    assert_eq!(quant.data().unwrap(), [0xAB; 68]);

    let norm = g.tensor("norm").unwrap();
    assert_eq!(norm.offset(), 128);
    assert_eq!(norm.data().unwrap(), f32_bytes(&[0.25, 0.5, 0.75]));
    assert!(g.tensor("missing").is_none());
}

#[test]
fn custom_alignment_is_honoured() {
    let mut w = sample_writer();
    w.set_alignment(64);
    let bytes = w.to_bytes();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.alignment(), 64);
    assert_eq!(g.data_offset() % 64, 0);
    assert_eq!(g.tensor("quant").unwrap().offset(), 64);
    assert_eq!(g.get("general.alignment"), Some(Value::U32(64)));
}

#[test]
fn unknown_tensor_types_are_listed_without_data() {
    let mut w = GgufWriter::new();
    w.add_tensor("mystery", GgmlType::from_raw(999), &[10, 10], vec![1, 2, 3]);
    let bytes = w.to_bytes();
    let g = Gguf::parse(&bytes).unwrap();
    let t = g.tensor("mystery").unwrap();
    assert_eq!(t.ggml_type(), GgmlType::Unknown(999));
    assert_eq!(t.ggml_type().to_string(), "type#999");
    assert!(t.data().is_none());
}

#[test]
fn every_truncation_fails_cleanly() {
    let bytes = sample_writer().to_bytes();
    let full = Gguf::parse(&bytes).unwrap();
    let last = full.tensors().last().unwrap();
    let needed = full.data_offset() + last.offset() as usize + last.data().unwrap().len();

    // Under Miri, sample the prefixes to keep the run short.
    let step = if cfg!(miri) { 7 } else { 1 };
    for len in (0..bytes.len()).step_by(step) {
        let result = Gguf::parse(&bytes[..len]);
        if len < needed {
            assert!(result.is_err(), "prefix of {len} bytes parsed");
        }
    }
}

/// Hand-assembles GGUF bytes, for files the writer refuses to produce.
#[derive(Default)]
struct Raw {
    kv: Vec<u8>,
    n_kv: u64,
    tensors: Vec<u8>,
    n_tensors: u64,
    data_len: usize,
    alignment: usize,
}

impl Raw {
    fn new() -> Self {
        Self {
            alignment: 32,
            ..Self::default()
        }
    }

    fn kv(mut self, key: &str, ty: u32, payload: &[u8]) -> Self {
        put_str(&mut self.kv, key);
        self.kv.extend(ty.to_le_bytes());
        self.kv.extend(payload);
        self.n_kv += 1;
        self
    }

    fn tensor(mut self, name: &str, dims: &[u64], ty: GgmlType, offset: u64) -> Self {
        put_str(&mut self.tensors, name);
        self.tensors.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            self.tensors.extend(d.to_le_bytes());
        }
        self.tensors.extend(ty.to_raw().to_le_bytes());
        self.tensors.extend(offset.to_le_bytes());
        self.n_tensors += 1;
        self
    }

    fn data(mut self, len: usize) -> Self {
        self.data_len = len;
        self
    }

    fn build(self) -> Vec<u8> {
        self.build_with_counts(self.n_kv, self.n_tensors)
    }

    fn build_with_counts(&self, n_kv: u64, n_tensors: u64) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(n_tensors.to_le_bytes());
        out.extend(n_kv.to_le_bytes());
        out.extend(&self.kv);
        out.extend(&self.tensors);
        out.resize(out.len().next_multiple_of(self.alignment), 0);
        out.resize(out.len() + self.data_len, 0);
        out
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn u32_payload(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn parse_err(bytes: &[u8]) -> Error {
    Gguf::parse(bytes).expect_err("parse should fail")
}

#[test]
fn rejects_bad_magic_and_versions() {
    assert!(matches!(parse_err(b"GGML\x03\0\0\0"), Error::BadMagic(m) if &m == b"GGML"));
    for version in [0u32, 1, 4] {
        let mut bytes = Raw::new().build();
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        assert_eq!(parse_err(&bytes), Error::UnsupportedVersion(version));
    }
    assert!(
        Gguf::parse(&Raw::new().build()).is_ok(),
        "an empty v3 file is valid"
    );
}

#[test]
fn rejects_invalid_alignment() {
    for alignment in [0u32, 3, 48] {
        let bytes = Raw::new()
            .kv("general.alignment", 4, &u32_payload(alignment))
            .build();
        assert!(
            matches!(parse_err(&bytes), Error::InvalidAlignment(_)),
            "{alignment}"
        );
    }
    // Stored as u64 instead of u32.
    let bytes = Raw::new()
        .kv("general.alignment", 10, &32u64.to_le_bytes())
        .build();
    assert!(matches!(parse_err(&bytes), Error::InvalidAlignment(_)));
}

#[test]
fn rejects_bad_metadata() {
    let bytes = Raw::new().kv("a", 13, &[]).build();
    assert!(matches!(
        parse_err(&bytes),
        Error::InvalidValueType { raw: 13, .. }
    ));

    let bytes = Raw::new().kv("a", 7, &[2]).build();
    assert!(matches!(
        parse_err(&bytes),
        Error::InvalidBool { raw: 2, .. }
    ));

    let mut bool_array = u32_payload(7);
    bool_array.extend(2u64.to_le_bytes());
    bool_array.extend([1, 5]);
    let bytes = Raw::new().kv("a", 9, &bool_array).build();
    assert!(matches!(
        parse_err(&bytes),
        Error::InvalidBool { raw: 5, .. }
    ));

    let mut nested = u32_payload(9);
    nested.extend(0u64.to_le_bytes());
    let bytes = Raw::new().kv("a", 9, &nested).build();
    assert!(matches!(parse_err(&bytes), Error::NestedArray { .. }));

    let bytes = Raw::new()
        .kv("a", 4, &u32_payload(1))
        .kv("a", 4, &u32_payload(2))
        .build();
    assert_eq!(parse_err(&bytes), Error::DuplicateKey("a".into()));

    let mut bad_key = Raw::new();
    bad_key.kv.extend(2u64.to_le_bytes());
    bad_key.kv.extend([0xFF, 0xFE]);
    bad_key.kv.extend(u32_payload(4));
    bad_key.kv.extend(u32_payload(1));
    bad_key.n_kv = 1;
    assert!(matches!(
        parse_err(&bad_key.build()),
        Error::InvalidUtf8 { .. }
    ));
}

#[test]
fn huge_counts_fail_without_allocating() {
    let raw = Raw::new();
    assert!(matches!(
        parse_err(&raw.build_with_counts(u64::MAX, 0)),
        Error::UnexpectedEof { .. }
    ));
    assert!(matches!(
        parse_err(&raw.build_with_counts(0, u64::MAX)),
        Error::UnexpectedEof { .. }
    ));

    // A string array claiming u64::MAX elements.
    let mut strings = u32_payload(8);
    strings.extend(u64::MAX.to_le_bytes());
    let bytes = Raw::new().kv("a", 9, &strings).build();
    assert!(matches!(parse_err(&bytes), Error::UnexpectedEof { .. }));

    // A u32 array whose byte size overflows u64.
    let mut ints = u32_payload(4);
    ints.extend(u64::MAX.to_le_bytes());
    let bytes = Raw::new().kv("a", 9, &ints).build();
    assert!(matches!(parse_err(&bytes), Error::TooLarge { .. }));

    // A string whose length prefix is u64::MAX.
    let bytes = Raw::new().kv("a", 8, &u64::MAX.to_le_bytes()).build();
    assert!(matches!(parse_err(&bytes), Error::UnexpectedEof { .. }));
}

#[test]
fn rejects_bad_tensors() {
    let bytes = Raw::new()
        .tensor("t", &[4], GgmlType::F32, 8)
        .data(64)
        .build();
    assert!(matches!(
        parse_err(&bytes),
        Error::MisalignedTensor { offset: 8, .. }
    ));

    let bytes = Raw::new()
        .tensor("t", &[16], GgmlType::F32, 32)
        .data(64)
        .build();
    assert!(matches!(
        parse_err(&bytes),
        Error::TensorOutOfBounds {
            size: 64,
            available: 64,
            ..
        }
    ));

    let bytes = Raw::new()
        .tensor("t", &[31], GgmlType::Q8_0, 0)
        .data(64)
        .build();
    assert!(matches!(parse_err(&bytes), Error::InvalidShape { .. }));

    let bytes = Raw::new()
        .tensor("t", &[1, 1, 1, 1, 1], GgmlType::F32, 0)
        .data(64)
        .build();
    assert!(matches!(parse_err(&bytes), Error::InvalidShape { .. }));

    let bytes = Raw::new()
        .tensor("t", &[u64::MAX, 2], GgmlType::F32, 0)
        .data(64)
        .build();
    assert!(matches!(parse_err(&bytes), Error::InvalidShape { .. }));

    let bytes = Raw::new()
        .tensor("t", &[0, u64::MAX, u64::MAX], GgmlType::F32, 0)
        .data(64)
        .build();
    assert!(matches!(parse_err(&bytes), Error::InvalidShape { .. }));

    let bytes = Raw::new()
        .tensor("t", &[4], GgmlType::F32, 0)
        .tensor("t", &[4], GgmlType::F32, 32)
        .data(64)
        .build();
    assert_eq!(parse_err(&bytes), Error::DuplicateTensor("t".into()));
}

#[test]
fn empty_tensors_are_valid() {
    let bytes = Raw::new().tensor("t", &[0, 4], GgmlType::F32, 0).build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.tensor("t").unwrap().data(), Some(&[][..]));
}
