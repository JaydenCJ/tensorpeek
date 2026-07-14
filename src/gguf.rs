//! GGUF header parser (versions 2 and 3, little-endian). Reads the metadata
//! key/value section and the tensor-info table sequentially from the start
//! of the file — tensor data itself is never touched; its size and the
//! expected end of file are computed from the ggml type table.

use std::io::Read;

use crate::json::Json;
use crate::report::{PeekError, Report, Tensor};

pub const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DIMS: u32 = 8;
const MAX_ARRAY_DEPTH: usize = 4;

/// ggml tensor types: (id, name, elements per block, bytes per block).
/// Ids 4, 5 and 31..=33, 36..=38 were removed from ggml and are absent here;
/// files using them get an "unknown type" problem, which is accurate enough
/// for an inspector.
const GGML_TYPES: &[(u32, &str, u64, u64)] = &[
    (0, "f32", 1, 4),
    (1, "f16", 1, 2),
    (2, "q4_0", 32, 18),
    (3, "q4_1", 32, 20),
    (6, "q5_0", 32, 22),
    (7, "q5_1", 32, 24),
    (8, "q8_0", 32, 34),
    (9, "q8_1", 32, 36),
    (10, "q2_k", 256, 84),
    (11, "q3_k", 256, 110),
    (12, "q4_k", 256, 144),
    (13, "q5_k", 256, 176),
    (14, "q6_k", 256, 210),
    (15, "q8_k", 256, 292),
    (16, "iq2_xxs", 256, 66),
    (17, "iq2_xs", 256, 74),
    (18, "iq3_xxs", 256, 98),
    (19, "iq1_s", 256, 50),
    (20, "iq4_nl", 32, 18),
    (21, "iq3_s", 256, 110),
    (22, "iq2_s", 256, 82),
    (23, "iq4_xs", 256, 136),
    (24, "i8", 1, 1),
    (25, "i16", 1, 2),
    (26, "i32", 1, 4),
    (27, "i64", 1, 8),
    (28, "f64", 1, 8),
    (29, "iq1_m", 256, 56),
    (30, "bf16", 1, 2),
    (34, "tq1_0", 256, 54),
    (35, "tq2_0", 256, 66),
    (39, "mxfp4", 32, 17),
];

fn ggml_type(id: u32) -> Option<(&'static str, u64, u64)> {
    GGML_TYPES
        .iter()
        .find(|t| t.0 == id)
        .map(|t| (t.1, t.2, t.3))
}

/// Block geometry (elements, bytes) for a type id — shared with the fixture
/// builder so written and parsed sizes always agree.
pub(crate) fn type_geometry(id: u32) -> Option<(u64, u64)> {
    ggml_type(id).map(|(_, block_len, block_bytes)| (block_len, block_bytes))
}

/// Metadata value-type ids as stored on disk.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

fn type_name(ty: u32) -> &'static str {
    match ty {
        T_U8 => "u8",
        T_I8 => "i8",
        T_U16 => "u16",
        T_I16 => "i16",
        T_U32 => "u32",
        T_I32 => "i32",
        T_F32 => "f32",
        T_BOOL => "bool",
        T_STRING => "string",
        T_ARRAY => "array",
        T_U64 => "u64",
        T_I64 => "i64",
        T_F64 => "f64",
        _ => "unknown",
    }
}

/// Smallest possible on-disk footprint of one value of `ty`, used to reject
/// implausible counts before allocating anything.
fn min_value_size(ty: u32) -> u64 {
    match ty {
        T_U8 | T_I8 | T_BOOL => 1,
        T_U16 | T_I16 => 2,
        T_U32 | T_I32 | T_F32 => 4,
        T_STRING | T_U64 | T_I64 | T_F64 => 8,
        T_ARRAY => 12,
        _ => 1,
    }
}

// ---------------------------------------------------------------------------
// Tracking reader
// ---------------------------------------------------------------------------

struct Rd<'a, R: Read> {
    r: &'a mut R,
    consumed: u64,
    file_len: u64,
}

impl<'a, R: Read> Rd<'a, R> {
    fn take(&mut self, n: u64) -> Result<Vec<u8>, PeekError> {
        if n > self.remaining() {
            return Err(PeekError(format!(
                "header needs {n} more bytes at offset {} but the file ends at {} — truncated header",
                self.consumed, self.file_len
            )));
        }
        let mut buf = vec![0u8; n as usize];
        self.r
            .read_exact(&mut buf)
            .map_err(|e| PeekError(format!("read failed: {e}")))?;
        self.consumed += n;
        Ok(buf)
    }

    fn remaining(&self) -> u64 {
        self.file_len.saturating_sub(self.consumed)
    }

    fn u32(&mut self) -> Result<u32, PeekError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, PeekError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<(String, bool), PeekError> {
        let len = self.u64()?;
        if len > self.remaining() {
            return Err(PeekError(format!(
                "string length {len} at offset {} exceeds the rest of the file",
                self.consumed - 8
            )));
        }
        let bytes = self.take(len)?;
        match String::from_utf8(bytes) {
            Ok(s) => Ok((s, true)),
            Err(e) => Ok((String::from_utf8_lossy(e.as_bytes()).into_owned(), false)),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a GGUF header. `array_limit` bounds how many elements of each
/// metadata array land in the report (`usize::MAX` = keep everything).
pub fn parse<R: Read>(
    r: &mut R,
    file_len: u64,
    path: &str,
    array_limit: usize,
) -> Result<Report, PeekError> {
    let mut rd = Rd {
        r,
        consumed: 0,
        file_len,
    };

    let magic = rd.take(4)?;
    if magic != b"GGUF" {
        return Err(PeekError("not a GGUF file (bad magic)".into()));
    }
    let version = rd.u32()?;
    if version == 1 {
        return Err(PeekError("GGUF v1 is obsolete and unsupported".into()));
    }
    if !(2..=3).contains(&version) {
        if (1..=16).contains(&version.swap_bytes()) {
            return Err(PeekError(
                "this looks like a big-endian GGUF file, which tensorpeek does not support".into(),
            ));
        }
        return Err(PeekError(format!("unsupported GGUF version {version}")));
    }
    let tensor_count = rd.u64()?;
    let kv_count = rd.u64()?;
    // Minimal on-disk sizes: a KV is at least key-length + type + 1 byte,
    // a tensor info at least name-length + n_dims + type + offset.
    let need = tensor_count
        .checked_mul(24)
        .and_then(|t| kv_count.checked_mul(13).map(|k| (t, k)));
    match need {
        Some((t, k)) if t.saturating_add(k) <= rd.remaining() => {}
        _ => {
            return Err(PeekError(format!(
                "header declares {kv_count} metadata keys and {tensor_count} tensors, \
                 which cannot fit in a {file_len}-byte file"
            )))
        }
    }

    let mut report = Report::new(path, "gguf", file_len);
    let mut problems: Vec<String> = Vec::new();

    // --- metadata -----------------------------------------------------------
    let mut meta: Vec<(String, Json)> = Vec::with_capacity(kv_count.min(1024) as usize);
    for _ in 0..kv_count {
        let (key, clean) = rd.string()?;
        if !clean {
            problems.push(format!("metadata key '{key}' is not valid UTF-8"));
        }
        if meta.iter().any(|(k, _)| *k == key) {
            problems.push(format!("duplicate metadata key '{key}'"));
        }
        let ty = rd.u32()?;
        let value = read_value(&mut rd, ty, 0, array_limit, &mut problems)?;
        meta.push((key, value));
    }

    // --- alignment -----------------------------------------------------------
    let mut alignment = DEFAULT_ALIGNMENT;
    if let Some((_, v)) = meta.iter().find(|(k, _)| k == "general.alignment") {
        match v.as_int() {
            Some(a) if a > 0 && (a as u64).is_power_of_two() => alignment = a as u64,
            _ => problems.push(format!(
                "general.alignment is {} — not a positive power of two; using the default 32",
                v.compact()
            )),
        }
    }

    // --- tensor infos ---------------------------------------------------------
    let mut data_end: u64 = 0;
    for i in 0..tensor_count {
        let (name, clean) = rd.string()?;
        if !clean {
            problems.push(format!("tensor #{i} name is not valid UTF-8"));
        }
        let n_dims = rd.u32()?;
        if n_dims > MAX_DIMS {
            return Err(PeekError(format!(
                "tensor '{name}' declares {n_dims} dimensions (max {MAX_DIMS}) — corrupt header"
            )));
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(rd.u64()?);
        }
        let type_id = rd.u32()?;
        let offset = rd.u64()?;

        let numel = shape
            .iter()
            .fold(1u128, |acc, &d| acc.saturating_mul(d as u128));
        let (dtype, bytes) = match ggml_type(type_id) {
            Some((tname, block_len, block_bytes)) => {
                let ne0 = shape.first().copied().unwrap_or(1);
                if block_len > 1 && ne0 % block_len != 0 {
                    problems.push(format!(
                        "tensor '{name}': first dimension {ne0} is not divisible by the {tname} block length {block_len}"
                    ));
                }
                let blocks = numel.div_ceil(block_len as u128);
                let bytes = blocks.saturating_mul(block_bytes as u128);
                (tname.to_string(), u64::try_from(bytes).unwrap_or(u64::MAX))
            }
            None => {
                problems.push(format!("tensor '{name}': unknown ggml type id {type_id}"));
                (format!("unknown({type_id})"), 0)
            }
        };
        data_end = data_end.max(offset.saturating_add(bytes));
        report
            .tensors
            .push(Tensor::new(&name, &dtype, shape, offset, bytes));
    }

    // --- accounting ------------------------------------------------------------
    let header_end = rd.consumed;
    let data_start = align_up(header_end, alignment);
    report.header_bytes = data_start.min(file_len);
    report.data_bytes = data_end;
    let expected = data_start.saturating_add(data_end);
    if file_len < expected {
        problems.push(format!(
            "tensor data needs {} bytes but only {} are present ({} missing) — truncated file",
            data_end,
            file_len.saturating_sub(data_start),
            expected - file_len
        ));
    } else if file_len - expected >= alignment {
        let extra = file_len - expected;
        problems.push(format!(
            "{extra} trailing byte{} after the last tensor",
            if extra == 1 { "" } else { "s" }
        ));
    }

    let arch = meta
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .map(|(_, v)| v.clone());
    let mut details: Vec<(String, Json)> = vec![
        ("version".into(), Json::Int(version as i128)),
        ("alignment".into(), Json::from(alignment)),
        ("data_start".into(), Json::from(data_start)),
    ];
    if let Some(a) = arch {
        details.push(("architecture".into(), a));
    }
    report.details = Json::Obj(details);
    report.metadata = Json::Obj(meta);
    report.problems = problems;
    Ok(report)
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a).saturating_mul(a)
}

fn read_value<R: Read>(
    rd: &mut Rd<R>,
    ty: u32,
    depth: usize,
    limit: usize,
    problems: &mut Vec<String>,
) -> Result<Json, PeekError> {
    Ok(match ty {
        T_U8 => Json::Int(rd.take(1)?[0] as i128),
        T_I8 => Json::Int(rd.take(1)?[0] as i8 as i128),
        T_U16 => {
            let b = rd.take(2)?;
            Json::Int(u16::from_le_bytes([b[0], b[1]]) as i128)
        }
        T_I16 => {
            let b = rd.take(2)?;
            Json::Int(i16::from_le_bytes([b[0], b[1]]) as i128)
        }
        T_U32 => Json::Int(rd.u32()? as i128),
        T_I32 => Json::Int(rd.u32()? as i32 as i128),
        T_F32 => {
            let b = rd.take(4)?;
            Json::Float(f32::from_le_bytes(b.try_into().unwrap()) as f64)
        }
        T_BOOL => {
            let b = rd.take(1)?[0];
            if b > 1 {
                problems.push(format!("bool metadata value has byte {b} (must be 0 or 1)"));
            }
            Json::Bool(b != 0)
        }
        T_STRING => {
            let (s, clean) = rd.string()?;
            if !clean {
                problems.push("string metadata value is not valid UTF-8".into());
            }
            Json::Str(s)
        }
        T_U64 => Json::Int(rd.u64()? as i128),
        T_I64 => Json::Int(rd.u64()? as i64 as i128),
        T_F64 => {
            let b = rd.take(8)?;
            Json::Float(f64::from_le_bytes(b.try_into().unwrap()))
        }
        T_ARRAY => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(PeekError("metadata arrays nested too deeply".into()));
            }
            let elem = rd.u32()?;
            let count = rd.u64()?;
            if count.saturating_mul(min_value_size(elem)) > rd.remaining() {
                return Err(PeekError(format!(
                    "array declares {count} elements, which cannot fit in the rest of the file"
                )));
            }
            let keep = (count as usize).min(limit);
            let mut items = Vec::with_capacity(keep.min(4096));
            for i in 0..count {
                let v = read_value(rd, elem, depth + 1, limit, problems)?;
                if (i as usize) < keep {
                    items.push(v);
                }
            }
            if (count as usize) > limit {
                // Large arrays (tokenizer vocabularies…) are summarized so the
                // report stays readable; --full-arrays lifts the limit.
                Json::Obj(vec![(
                    "$array".into(),
                    Json::Obj(vec![
                        ("elem".into(), Json::Str(type_name(elem).into())),
                        ("len".into(), Json::from(count)),
                        ("first".into(), Json::Arr(items)),
                    ]),
                )])
            } else {
                Json::Arr(items)
            }
        }
        _ => return Err(PeekError(format!("unknown metadata value type {ty}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{GgufBuilder, Gv};
    use std::io::Cursor;

    fn parse_bytes(b: &[u8], limit: usize) -> Result<Report, PeekError> {
        parse(&mut Cursor::new(b), b.len() as u64, "test.gguf", limit)
    }

    #[test]
    fn parses_the_demo_model() {
        let bytes = GgufBuilder::demo().build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert_eq!(r.format, "gguf");
        assert_eq!(r.details.get("version").and_then(Json::as_int), Some(3));
        assert_eq!(
            r.details.get("architecture").and_then(Json::as_str),
            Some("llama")
        );
        assert_eq!(
            r.metadata.get("general.name").and_then(Json::as_str),
            Some("tinyllama-demo")
        );
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn quantized_sizes_use_block_geometry_and_shapes_keep_disk_order() {
        let bytes = GgufBuilder::demo().build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        let t = r
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .unwrap();
        assert_eq!(t.dtype, "q8_0");
        // 64*256 elements / 32 per block * 34 bytes per block.
        assert_eq!(t.bytes, 64 * 256 / 32 * 34);
        assert_eq!(
            t.shape,
            [64, 256],
            "ne0 (embedding width) must come first, as stored"
        );
    }

    #[test]
    fn header_and_data_accounting_add_up() {
        let bytes = GgufBuilder::demo().build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert_eq!(r.header_bytes % 32, 0, "data start must be aligned");
        assert_eq!(r.header_bytes + r.data_bytes, bytes.len() as u64);
    }

    #[test]
    fn custom_alignment_is_honored() {
        let bytes = GgufBuilder::demo()
            .kv("general.alignment", Gv::U32(64))
            .build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert_eq!(r.details.get("alignment").and_then(Json::as_int), Some(64));
        assert_eq!(r.header_bytes % 64, 0);
    }

    #[test]
    fn bad_alignment_falls_back_to_default_with_a_problem() {
        let bytes = GgufBuilder::demo()
            .kv("general.alignment", Gv::U32(48))
            .build_with_alignment(32);
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert_eq!(r.details.get("alignment").and_then(Json::as_int), Some(32));
        assert!(
            r.problems.iter().any(|p| p.contains("power of two")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn truncation_is_reported_with_the_missing_byte_count() {
        let mut bytes = GgufBuilder::demo().build();
        let n = bytes.len();
        bytes.truncate(n - 500);
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert!(
            r.problems.iter().any(|p| p.contains("500 missing")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn arrays_are_summarized_at_the_limit_and_kept_in_full_without_one() {
        let bytes = GgufBuilder::demo().build();
        let r = parse_bytes(&bytes, 4).unwrap();
        let tokens = r.metadata.get("tokenizer.ggml.tokens").unwrap();
        let marker = tokens.get("$array").expect("summarized array marker");
        assert_eq!(marker.get("len").and_then(Json::as_int), Some(12));
        assert_eq!(marker.get("first").and_then(Json::as_arr).unwrap().len(), 4);

        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        let tokens = r.metadata.get("tokenizer.ggml.tokens").unwrap();
        assert_eq!(tokens.as_arr().unwrap().len(), 12);
    }

    #[test]
    fn scalar_metadata_types_round_trip() {
        let bytes = GgufBuilder::demo()
            .kv("test.f32", Gv::F32(0.5))
            .kv("test.bool", Gv::Bool(true))
            .kv("test.i64", Gv::I64(-9_000_000_000))
            .kv("test.u16", Gv::U16(65535))
            .build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert_eq!(r.metadata.get("test.f32"), Some(&Json::Float(0.5)));
        assert_eq!(r.metadata.get("test.bool"), Some(&Json::Bool(true)));
        assert_eq!(
            r.metadata.get("test.i64").and_then(Json::as_int),
            Some(-9_000_000_000)
        );
        assert_eq!(
            r.metadata.get("test.u16").and_then(Json::as_int),
            Some(65535)
        );
    }

    #[test]
    fn bad_magic_and_unsupported_versions_are_fatal() {
        let err = parse_bytes(b"GGML1234rest-of-file", usize::MAX).unwrap_err();
        assert!(err.0.contains("bad magic"));

        let mut v1 = b"GGUF".to_vec();
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&[0u8; 16]);
        assert!(parse_bytes(&v1, usize::MAX).unwrap_err().0.contains("v1"));

        let mut v9 = b"GGUF".to_vec();
        v9.extend_from_slice(&9999u32.to_le_bytes());
        v9.extend_from_slice(&[0u8; 16]);
        assert!(parse_bytes(&v9, usize::MAX)
            .unwrap_err()
            .0
            .contains("unsupported"));

        // Version 3 with swapped bytes reads as 0x03000000 — call it out.
        let mut be = b"GGUF".to_vec();
        be.extend_from_slice(&3u32.to_be_bytes());
        be.extend_from_slice(&[0u8; 16]);
        let err = parse_bytes(&be, usize::MAX).unwrap_err();
        assert!(err.0.contains("big-endian"), "got: {err}");
    }

    #[test]
    fn hostile_counts_and_lengths_are_rejected_before_allocation() {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor_count
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // kv_count
        let err = parse_bytes(&b, usize::MAX).unwrap_err();
        assert!(err.0.contains("cannot fit"), "got: {err}");

        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // one kv
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // key length: absurd
        b.extend_from_slice(&[0u8; 64]);
        let err = parse_bytes(&b, usize::MAX).unwrap_err();
        assert!(err.0.contains("exceeds"), "got: {err}");
    }

    #[test]
    fn unknown_tensor_type_is_a_problem_not_a_crash() {
        let bytes = GgufBuilder::demo()
            .tensor("weird.weight", &[32], 99)
            .build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        let t = r.tensors.iter().find(|t| t.name == "weird.weight").unwrap();
        assert_eq!(t.dtype, "unknown(99)");
        assert!(r.problems.iter().any(|p| p.contains("unknown ggml type")));
    }

    #[test]
    fn non_divisible_block_row_is_flagged() {
        // 40 is not divisible by the q8_0 block length of 32.
        let bytes = GgufBuilder::demo().tensor("odd.weight", &[40], 8).build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert!(
            r.problems.iter().any(|p| p.contains("not divisible")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn duplicate_metadata_keys_are_noted() {
        let bytes = GgufBuilder::demo()
            .kv("general.name", Gv::Str("again".into()))
            .build();
        let r = parse_bytes(&bytes, usize::MAX).unwrap();
        assert!(
            r.problems
                .iter()
                .any(|p| p.contains("duplicate metadata key")),
            "problems: {:?}",
            r.problems
        );
    }
}
