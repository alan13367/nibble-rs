//! Weights with a Hadamard rotation folded in (PrismML's `prism.hadamard.*`
//! metadata, as llama.cpp's `load_hparams_hadamard` reads it).
//!
//! A folded weight is `W · D · H`: the weight `W` with ±1 signs `D` and the
//! orthonormal block Walsh-Hadamard transform `H` folded into its columns.
//! Since `H · H = I` and `D · D = I`, `W · x` is the folded weight times
//! `H · (D · x)`, so every product with a folded weight first rotates its
//! input ([`Rotation`], [`crate::backend::Backend::rotate`]). A table read
//! by row (the token embedding) may instead be stored rotated, its rows
//! turned back with `D · (H · z)` after the lookup.
//!
//! A model binds each folded weight's rotation with [`Folding::input`] (or
//! [`Folding::delta_out`] for a Gated DeltaNet's output projection) and
//! calls [`Folding::finish`] at the end, which fails if a weight the file
//! lists was not bound through a rotation-aware product, rather than
//! computing with it unrotated.

use std::cell::RefCell;
use std::collections::BTreeSet;

use nibble_gguf::Gguf;

use crate::backend::{DeltaNetDims, HeadOrder, Param, Params, Rotation};
use crate::error::{Error, Result};

const PREFIX: &str = "prism.hadamard.";

/// The weight kinds llama.cpp folds (`is_foldable_weight`): those it
/// multiplies through `build_lora_mm`, which applies the rotation.
const FOLDABLE: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_gate",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "ssm_out",
];

/// The folded weights of a model file.
#[derive(Debug, Default)]
pub(crate) struct Folding {
    /// The transform's block size (0 when nothing is folded).
    block: usize,
    /// The sign vector for each input width, if the signs are explicit.
    signs: Vec<(usize, Param)>,
    explicit_signs: bool,
    /// Whether a DeltaNet output projection was folded with its value heads
    /// grouped by key head.
    gdn_v_grouped: bool,
    /// Folded weights not yet bound.
    unbound: RefCell<BTreeSet<String>>,
    /// Tables stored rotated, read by row, not yet bound.
    inverse: RefCell<BTreeSet<String>>,
    /// Every distinct rotation handed out.
    used: RefCell<Vec<Rotation>>,
}

fn invalid(key: &str, reason: impl Into<String>) -> Error {
    Error::InvalidMetadata {
        key: format!("{PREFIX}{key}"),
        reason: reason.into(),
    }
}

fn get<'a>(g: &Gguf<'a>, key: &str) -> Option<nibble_gguf::Value<'a>> {
    g.get(&format!("{PREFIX}{key}"))
}

fn required<'a>(g: &Gguf<'a>, key: &str) -> Result<nibble_gguf::Value<'a>> {
    get(g, key).ok_or_else(|| Error::MissingMetadata(format!("{PREFIX}{key}")))
}

fn string<'a>(g: &Gguf<'a>, key: &str) -> Result<&'a str> {
    required(g, key)?
        .as_str()
        .ok_or_else(|| invalid(key, "expected a string"))
}

fn strings(g: &Gguf<'_>, key: &str) -> Result<Vec<String>> {
    let Some(v) = get(g, key) else {
        return Ok(Vec::new());
    };
    let array = v
        .as_array()
        .ok_or_else(|| invalid(key, "expected an array of strings"))?;
    array
        .iter()
        .map(|e| {
            e.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid(key, "expected an array of strings"))
        })
        .collect()
}

fn integers(g: &Gguf<'_>, key: &str) -> Result<Vec<i64>> {
    let array = required(g, key)?
        .as_array()
        .ok_or_else(|| invalid(key, "expected an array of integers"))?;
    array
        .iter()
        .map(|e| {
            e.as_i64()
                .ok_or_else(|| invalid(key, "expected an array of integers"))
        })
        .collect()
}

/// Whether llama.cpp applies the rotation to weight `name`: the output
/// head, or `blk.{N}.{kind}.weight` of a [`FOLDABLE`] kind.
fn is_foldable(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    let Some(kind) = rest[digits..]
        .strip_prefix('.')
        .and_then(|r| r.strip_suffix(".weight"))
    else {
        return false;
    };
    digits > 0 && FOLDABLE.contains(&kind)
}

impl Folding {
    /// Reads and checks the metadata (nothing is folded without it), with
    /// the sign vectors pushed to the parameter arena.
    pub(crate) fn read(g: &Gguf<'_>, p: &mut Params) -> Result<Self> {
        let tied = match get(g, "tied_output") {
            None => false,
            Some(v) => v
                .as_bool()
                .ok_or_else(|| invalid("tied_output", "expected a boolean"))?,
        };
        let Some(version) = get(g, "version") else {
            return Ok(Self::default());
        };
        let version = version
            .as_u64()
            .ok_or_else(|| invalid("version", "expected an unsigned integer"))?;
        if version != 1 && version != 2 {
            return Err(invalid("version", format!("unsupported version {version}")));
        }
        if (version == 2) != tied {
            return Err(invalid(
                "tied_output",
                "version 2 requires tied_output = true; version 1 forbids it",
            ));
        }
        let has_output = g.tensor("output.weight").is_some();
        if tied && has_output {
            return Err(invalid(
                "tied_output",
                "requires output.weight to be absent",
            ));
        }
        let block = required(g, "block_size")?
            .as_u64()
            .and_then(|b| usize::try_from(b).ok())
            .filter(|b| b.is_power_of_two())
            .ok_or_else(|| invalid("block_size", "must be a power of two"))?;
        let checks = [
            ("transform", "normalized-sylvester-walsh-hadamard"),
            ("axis", "input-last-dimension"),
        ];
        for (key, want) in checks {
            let found = string(g, key)?;
            if found != want {
                return Err(invalid(key, format!("unsupported value `{found}`")));
            }
        }
        let explicit_signs = match string(g, "sign_mode")? {
            "identity" => false,
            "explicit" => true,
            other => return Err(invalid("sign_mode", format!("unsupported value `{other}`"))),
        };
        let mut signs = Vec::new();
        if explicit_signs {
            let widths = integers(g, "sign_widths")?;
            let values = integers(g, "sign_values")?;
            // No widths would make every sign the identity, a different model.
            if widths.is_empty() {
                return Err(invalid("sign_widths", "is empty with explicit signs"));
            }
            let mut at = 0usize;
            for w in widths {
                let width = usize::try_from(w)
                    .ok()
                    .filter(|&w| w > 0 && w % block == 0)
                    .ok_or_else(|| invalid("sign_widths", format!("invalid width {w}")))?;
                if signs.iter().any(|&(seen, _)| seen == width) {
                    return Err(invalid("sign_widths", format!("duplicate width {width}")));
                }
                let end = at
                    .checked_add(width)
                    .filter(|&end| end <= values.len())
                    .ok_or_else(|| invalid("sign_values", "too few values for sign_widths"))?;
                let vector = values[at..end]
                    .iter()
                    .map(|&v| match v {
                        1 => Ok(1.0),
                        -1 => Ok(-1.0),
                        _ => Err(invalid("sign_values", "values must be +1 or -1")),
                    })
                    .collect::<Result<Vec<f32>>>()?;
                signs.push((width, p.push(&vector)));
                at = end;
            }
            if at != values.len() {
                return Err(invalid(
                    "sign_values",
                    "more values than sign_widths add up to",
                ));
            }
        }
        let gdn_v_grouped = match get(g, "gdn_v_grouped") {
            None => false,
            Some(v) => v
                .as_bool()
                .ok_or_else(|| invalid("gdn_v_grouped", "expected a boolean"))?,
        };
        let names = strings(g, "weight_names")?;
        if names.is_empty() {
            return Err(invalid("weight_names", "is empty"));
        }
        let mut unbound = BTreeSet::new();
        for name in names {
            if !is_foldable(&name) {
                return Err(invalid(
                    "weight_names",
                    format!("`{name}` is not on a product that applies the rotation"),
                ));
            }
            if g.tensor(&name).is_none() {
                return Err(Error::MissingTensor(name));
            }
            if !unbound.insert(name.clone()) {
                return Err(invalid(
                    "weight_names",
                    format!("duplicate weight `{name}`"),
                ));
            }
        }
        let mut inverse = BTreeSet::new();
        for name in strings(g, "inverse_weight_names")? {
            // llama.cpp turns back the token embedding's lookups only.
            if name != "token_embd.weight" || !inverse.insert(name) {
                return Err(invalid(
                    "inverse_weight_names",
                    "only token_embd.weight may be stored rotated, once",
                ));
            }
        }
        let embd_latent = inverse.contains("token_embd.weight");
        if tied {
            if !embd_latent {
                return Err(invalid("tied_output", "requires a rotated token embedding"));
            }
            // The output head is the embedding, folded.
            unbound.insert("token_embd.weight".to_owned());
        } else if embd_latent && !has_output {
            return Err(invalid(
                "tied_output",
                "a rotated embedding tied to the output requires version 2",
            ));
        }
        Ok(Self {
            block,
            signs,
            explicit_signs,
            gdn_v_grouped,
            unbound: RefCell::new(unbound),
            inverse: RefCell::new(inverse),
            used: RefCell::new(Vec::new()),
        })
    }

    /// The rotation of rows of `len` values, without a head order.
    fn rotation(&self, name: &str, len: usize) -> Result<Rotation> {
        if !len.is_multiple_of(self.block) {
            return Err(invalid(
                "block_size",
                format!(
                    "{} does not divide the {len} columns of `{name}`",
                    self.block
                ),
            ));
        }
        let signs = if self.explicit_signs {
            let param = self.signs.iter().find(|&&(w, _)| w == len).map(|&(_, p)| p);
            Some(param.ok_or_else(|| {
                invalid(
                    "sign_widths",
                    format!("no signs for the {len} columns of `{name}`"),
                )
            })?)
        } else {
            None
        };
        Ok(Rotation {
            len,
            block: self.block,
            signs,
            heads: None,
        })
    }

    /// Records `r` among the rotations handed out, and returns it.
    fn note(&self, r: Rotation) -> Rotation {
        let mut used = self.used.borrow_mut();
        if !used.contains(&r) {
            used.push(r);
        }
        r
    }

    /// The rotation the input of weights `names` (of `cols` columns, all
    /// reading the same input) goes through, if they are folded. They must
    /// all be folded or all not: nibble multiplies them in one pass.
    pub(crate) fn input(&self, names: &[&str], cols: usize) -> Result<Option<Rotation>> {
        Ok(self.bind(names, cols)?.map(|r| self.note(r)))
    }

    /// [`Folding::input`] without recording the rotation.
    fn bind(&self, names: &[&str], cols: usize) -> Result<Option<Rotation>> {
        let mut unbound = self.unbound.borrow_mut();
        let folded: Vec<bool> = names.iter().map(|n| unbound.remove(*n)).collect();
        match folded.iter().filter(|&&f| f).count() {
            0 => Ok(None),
            n if n == names.len() => Ok(Some(self.rotation(names[0], cols)?)),
            _ => Err(invalid(
                "weight_names",
                format!(
                    "weights {} share an input but are not all folded",
                    names.join(", ")
                ),
            )),
        }
    }

    /// [`Folding::input`] for a Gated DeltaNet's output projection `name`,
    /// whose input has the layer's value heads in ggml's tiled order.
    pub(crate) fn delta_out(&self, name: &str, d: DeltaNetDims) -> Result<Option<Rotation>> {
        let Some(mut r) = self.bind(&[name], d.value_dim())? else {
            return Ok(None);
        };
        if self.gdn_v_grouped {
            if !d.n_v_heads.is_multiple_of(d.n_k_heads) {
                return Err(invalid(
                    "gdn_v_grouped",
                    format!("bad head geometry for `{name}`"),
                ));
            }
            r.heads = Some(HeadOrder {
                head_dim: d.d_v,
                n_k: d.n_k_heads,
                rep: d.n_v_heads / d.n_k_heads,
            });
        }
        Ok(Some(self.note(r)))
    }

    /// The rotation that turns rows looked up in table `name` (of `cols`
    /// columns) back, if the table is stored rotated.
    pub(crate) fn lookup(&self, name: &str, cols: usize) -> Result<Option<Rotation>> {
        if self.inverse.borrow_mut().remove(name) {
            Ok(Some(self.note(self.rotation(name, cols)?)))
        } else {
            Ok(None)
        }
    }

    /// Every distinct rotation handed out, for the backend to check and the
    /// scratch vector to fit; fails if a weight the file folds was not bound.
    pub(crate) fn finish(self) -> Result<Vec<Rotation>> {
        let unbound = self.unbound.borrow();
        let inverse = self.inverse.borrow();
        match unbound.iter().chain(inverse.iter()).next() {
            None => Ok(self.used.take()),
            Some(name) => Err(invalid(
                "weight_names",
                format!("`{name}` is folded but nibble does not read it through a rotation"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use nibble_gguf::{GgmlType, GgufWriter, OwnedValue, Storage, ValueType};

    use super::*;

    /// `prism.hadamard.*` metadata, keys without the prefix.
    type Meta = Vec<(&'static str, OwnedValue)>;

    /// The metadata of a folded file with weights `blk.0.ffn_{gate,up}`
    /// (16 columns) and `ffn_down` (32), the token embedding (16) and an
    /// output head: version 1, blocks of 8, explicit signs.
    fn metadata() -> Meta {
        let names = [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output.weight",
        ];
        let signs: Vec<i32> = (0..48).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        vec![
            ("version", 1u32.into()),
            ("block_size", 8u32.into()),
            ("transform", "normalized-sylvester-walsh-hadamard".into()),
            ("axis", "input-last-dimension".into()),
            ("sign_mode", "explicit".into()),
            ("sign_widths", OwnedValue::array(ValueType::I32, [16, 32])),
            ("sign_values", OwnedValue::array(ValueType::I32, signs)),
            ("weight_names", OwnedValue::array(ValueType::String, names)),
            (
                "inverse_weight_names",
                OwnedValue::array(ValueType::String, ["token_embd.weight"]),
            ),
        ]
    }

    /// A file with `meta` (keys after `prism.hadamard.`) and the tensors,
    /// `output.weight` only if `output`.
    fn file(meta: &[(&str, OwnedValue)], output: bool) -> Storage {
        let mut w = GgufWriter::new();
        for (k, v) in meta {
            w.add(format!("{PREFIX}{k}"), v.clone());
        }
        let tensor = |w: &mut GgufWriter, name: &str, cols: u64, rows: u64| {
            let bytes = vec![0u8; (4 * cols * rows) as usize];
            w.add_tensor(name, GgmlType::F32, &[cols, rows], bytes);
        };
        tensor(&mut w, "token_embd.weight", 16, 4);
        tensor(&mut w, "blk.0.ffn_gate.weight", 16, 32);
        tensor(&mut w, "blk.0.ffn_up.weight", 16, 32);
        tensor(&mut w, "blk.0.ffn_down.weight", 32, 16);
        if output {
            tensor(&mut w, "output.weight", 16, 4);
        }
        Storage::from_bytes(&w.to_bytes())
    }

    fn read(meta: &[(&str, OwnedValue)], output: bool) -> Result<(Folding, Vec<f32>)> {
        let storage = file(meta, output);
        let g = Gguf::parse(&storage)?;
        let mut p = Params::default();
        let f = Folding::read(&g, &mut p)?;
        Ok((f, p.into_vec()))
    }

    /// `meta` with `key` set to `value`, or removed.
    fn with(key: &str, value: Option<OwnedValue>) -> Meta {
        let mut m: Vec<_> = metadata().into_iter().filter(|(k, _)| *k != key).collect();
        if let Some(v) = value {
            let key = metadata().iter().map(|(k, _)| *k).find(|k| *k == key);
            m.push((key.unwrap_or("tied_output"), v));
        }
        m
    }

    #[test]
    fn reads_explicit_signs_and_binds_every_weight() {
        let (f, params) = read(&metadata(), true).unwrap();
        let r = f
            .input(&["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"], 16)
            .unwrap()
            .unwrap();
        assert_eq!((r.len, r.block, r.heads), (16, 8, None));
        let signs = &params[r.signs.unwrap().range()];
        assert_eq!(signs[..4], [-1.0, 1.0, 1.0, -1.0]);
        let down = f.input(&["blk.0.ffn_down.weight"], 32).unwrap().unwrap();
        // The 32-wide signs follow the 16-wide ones.
        assert_eq!(params[down.signs.unwrap().range()][0], 1.0);
        assert!(f.lookup("token_embd.weight", 16).unwrap().is_some());
        assert!(f.input(&["output.weight"], 16).unwrap().is_some());
        // The gate/up input and the head's are the same rotation.
        assert_eq!(f.finish().unwrap().len(), 2);
    }

    #[test]
    fn identity_signs_and_one_value_blocks() {
        let mut m = with("sign_mode", Some("identity".into()));
        m.retain(|(k, _)| !k.starts_with("sign_w") && !k.starts_with("sign_v"));
        m.retain(|(k, _)| *k != "block_size");
        m.push(("block_size", 1u32.into()));
        let (f, _) = read(&m, true).unwrap();
        let r = f.input(&["blk.0.ffn_down.weight"], 32).unwrap().unwrap();
        assert_eq!((r.block, r.signs), (1, None));
    }

    #[test]
    fn an_unbound_weight_fails_the_load() {
        let (f, _) = read(&metadata(), true).unwrap();
        f.input(&["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"], 16)
            .unwrap();
        assert!(
            f.finish().is_err(),
            "ffn_down, output and the embedding are unbound"
        );
        // Weights that share an input must be folded together.
        let (f, _) = read(&metadata(), true).unwrap();
        assert!(
            f.input(&["blk.0.ffn_up.weight", "blk.0.attn_gate.weight"], 16)
                .is_err()
        );
    }

    #[test]
    fn invalid_metadata_is_refused() {
        let s = |v: &str| Some(OwnedValue::from(v));
        let ints = |v: &[i32]| Some(OwnedValue::array(ValueType::I32, v.to_vec()));
        let strs = |v: &[&str]| Some(OwnedValue::array(ValueType::String, v.to_vec()));
        // (case, the key the error names, metadata, whether there is an
        // `output.weight`; without one it is not among the folded weights)
        let cases: Vec<(&str, &str, Meta, bool)> = vec![
            (
                "version 3",
                "version",
                with("version", Some(3u32.into())),
                true,
            ),
            (
                "version 2 untied",
                "tied_output",
                with("version", Some(2u32.into())),
                true,
            ),
            (
                "tied version 1",
                "tied_output",
                with("tied_output", Some(true.into())),
                false,
            ),
            (
                "block 0",
                "block_size",
                with("block_size", Some(0u32.into())),
                true,
            ),
            (
                "block 12",
                "block_size",
                with("block_size", Some(12u32.into())),
                true,
            ),
            ("no block", "block_size", with("block_size", None), true),
            (
                "transform",
                "transform",
                with("transform", s("hadamard")),
                true,
            ),
            ("axis", "axis", with("axis", s("output")), true),
            (
                "sign mode",
                "sign_mode",
                with("sign_mode", s("random")),
                true,
            ),
            (
                "no widths",
                "sign_widths",
                with("sign_widths", ints(&[])),
                true,
            ),
            (
                "width 12",
                "sign_widths",
                with("sign_widths", ints(&[12, 36])),
                true,
            ),
            (
                "same width",
                "sign_widths",
                with("sign_widths", ints(&[16, 16])),
                true,
            ),
            (
                "few values",
                "sign_values",
                with("sign_widths", ints(&[16, 64])),
                true,
            ),
            (
                "more values",
                "sign_values",
                with("sign_widths", ints(&[16])),
                true,
            ),
            (
                "sign 2",
                "sign_values",
                with("sign_values", ints(&[2; 48])),
                true,
            ),
            (
                "no names",
                "weight_names",
                with("weight_names", strs(&[])),
                true,
            ),
            (
                "beta",
                "weight_names",
                with("weight_names", strs(&["blk.0.ssm_beta.weight"])),
                true,
            ),
            (
                "missing",
                "blk.3.ffn_up",
                with("weight_names", strs(&["blk.3.ffn_up.weight"])),
                true,
            ),
            (
                "twice",
                "weight_names",
                with("weight_names", strs(&["output.weight"; 2])),
                true,
            ),
            (
                "inverse",
                "inverse_weight_names",
                with("inverse_weight_names", strs(&["output.weight"])),
                true,
            ),
            // A rotated embedding as the only head needs version 2.
            ("latent head", "tied_output", metadata(), false),
        ];
        let trunk = [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        for (what, key, meta, output) in cases {
            let meta: Vec<_> = meta
                .into_iter()
                .map(|(k, v)| match k {
                    "weight_names" if !output => (k, OwnedValue::array(ValueType::String, trunk)),
                    _ => (k, v),
                })
                .collect();
            let err = read(&meta, output).err().map(|e| e.to_string());
            assert!(
                err.as_ref().is_some_and(|e| e.contains(key)),
                "{what}: {err:?}"
            );
        }
        // Version 2 with a tied, rotated embedding loads.
        let mut tied = with("version", Some(2u32.into()));
        tied.push(("tied_output", true.into()));
        tied.retain(|(k, _)| *k != "weight_names");
        let names = [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        tied.push(("weight_names", OwnedValue::array(ValueType::String, names)));
        let (f, _) = read(&tied, false).unwrap();
        assert!(
            f.input(&["token_embd.weight"], 16).unwrap().is_some(),
            "the tied head"
        );
    }

    #[test]
    fn foldable_names_follow_llama_cpp() {
        for name in [
            "output.weight",
            "blk.0.attn_qkv.weight",
            "blk.63.ssm_out.weight",
            "blk.12.ffn_down_exps.weight",
        ] {
            assert!(is_foldable(name), "{name}");
        }
        for name in [
            "token_embd.weight",
            "blk.0.ssm_beta.weight",
            "blk..attn_q.weight",
            "blk.1.attn_q.bias",
            "blk.64.nextn.eh_proj.weight",
            "blk.x.ffn_up.weight",
        ] {
            assert!(!is_foldable(name), "{name}");
        }
    }

    #[test]
    fn head_order_groups_value_heads_by_key_head() {
        // Two key heads, three value heads each, heads of 2: value head
        // `k + 2r` moves to place `r + 3k`.
        let h = HeadOrder {
            head_dim: 2,
            n_k: 2,
            rep: 3,
        };
        let heads: Vec<usize> = (0..6).map(|head| h.target(2 * head) / 2).collect();
        assert_eq!(heads, [0, 3, 1, 4, 2, 5]);
        assert_eq!(h.target(3), 2 * 3 + 1);
    }
}
