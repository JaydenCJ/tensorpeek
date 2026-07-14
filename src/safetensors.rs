//! safetensors header parser. The format is an 8-byte little-endian header
//! length, a JSON header mapping tensor names to `{dtype, shape,
//! data_offsets}` (plus an optional `__metadata__` string map), then the raw
//! data region. Only the first `8 + header_len` bytes are ever read.

use std::io::Read;

use crate::json::Json;
use crate::report::{PeekError, Report, Tensor};

/// The reference implementation caps headers at 100 MB; a bigger claim is a
/// corrupt or hostile file, not a real checkpoint.
const MAX_HEADER: u64 = 100 * 1024 * 1024;

/// Known dtypes and their size in bits.
fn dtype_bits(dtype: &str) -> Option<u32> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 64,
        "F32" | "I32" | "U32" => 32,
        "F16" | "BF16" | "I16" | "U16" => 16,
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" | "BOOL" => 8,
        "F4" => 4,
        _ => return None,
    })
}

/// Canonical (lowercase) dtype name for the report.
fn canonical(dtype: &str) -> String {
    dtype.to_ascii_lowercase()
}

pub fn parse<R: Read>(r: &mut R, file_len: u64, path: &str) -> Result<Report, PeekError> {
    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)
        .map_err(|_| PeekError("file too small for a safetensors header".into()))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > MAX_HEADER {
        return Err(PeekError(format!(
            "declared header length {header_len} exceeds the 100 MB format cap"
        )));
    }
    if header_len.saturating_add(8) > file_len {
        return Err(PeekError(format!(
            "declared header length {header_len} exceeds the file itself ({file_len} bytes)"
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    r.read_exact(&mut header)
        .map_err(|_| PeekError("file ends inside the JSON header".into()))?;
    let doc =
        Json::parse(&header).map_err(|e| PeekError(format!("header is not valid JSON: {e}")))?;
    let Json::Obj(members) = doc else {
        return Err(PeekError("header JSON is not an object".into()));
    };

    let mut report = Report::new(path, "safetensors", file_len);
    report.header_bytes = 8 + header_len;

    let mut max_end: u64 = 0;
    for (name, value) in &members {
        if name == "__metadata__" {
            match value {
                Json::Obj(meta) if meta.iter().all(|(_, v)| matches!(v, Json::Str(_))) => {
                    report.metadata = value.clone();
                }
                _ => report
                    .problems
                    .push("__metadata__ is not a string-to-string map".into()),
            }
            continue;
        }
        match parse_tensor(name, value) {
            Ok((tensor, end, note)) => {
                max_end = max_end.max(end);
                if let Some(n) = note {
                    report.problems.push(n);
                }
                report.tensors.push(tensor);
            }
            Err(msg) => report.problems.push(msg),
        }
    }

    report.data_bytes = max_end;
    let expected = report.header_bytes + max_end;
    if file_len < expected {
        report.problems.push(format!(
            "data section needs {} bytes but only {} are present ({} missing) — truncated file",
            max_end,
            file_len - report.header_bytes,
            expected - file_len
        ));
    } else if file_len > expected {
        let extra = file_len - expected;
        report.problems.push(format!(
            "{extra} trailing byte{} after the last tensor",
            if extra == 1 { "" } else { "s" }
        ));
    }
    report.details = Json::Obj(vec![("header_json_bytes".into(), Json::from(header_len))]);
    Ok(report)
}

/// Returns (tensor, end offset, optional problem note) or a problem message
/// for an entry that cannot be interpreted as a tensor at all.
fn parse_tensor(name: &str, v: &Json) -> Result<(Tensor, u64, Option<String>), String> {
    let dtype = v
        .get("dtype")
        .and_then(Json::as_str)
        .ok_or_else(|| format!("tensor '{name}': missing or non-string dtype"))?;
    let shape_json = v
        .get("shape")
        .and_then(Json::as_arr)
        .ok_or_else(|| format!("tensor '{name}': missing or non-array shape"))?;
    let mut shape = Vec::with_capacity(shape_json.len());
    for d in shape_json {
        match d.as_int() {
            Some(n) if (0..=u64::MAX as i128).contains(&n) => shape.push(n as u64),
            _ => {
                return Err(format!(
                    "tensor '{name}': shape contains a non-integer or negative dimension"
                ))
            }
        }
    }
    let offsets = v
        .get("data_offsets")
        .and_then(Json::as_arr)
        .ok_or_else(|| format!("tensor '{name}': missing data_offsets"))?;
    let (Some(begin), Some(end)) = (
        offsets.first().and_then(Json::as_int),
        offsets.get(1).and_then(Json::as_int),
    ) else {
        return Err(format!(
            "tensor '{name}': data_offsets is not a pair of integers"
        ));
    };
    if offsets.len() != 2 || begin < 0 || end < begin || end > u64::MAX as i128 {
        return Err(format!(
            "tensor '{name}': invalid data_offsets [{begin}, {end}]"
        ));
    }
    let (begin, end) = (begin as u64, end as u64);
    let stored = end - begin;

    let numel = shape
        .iter()
        .try_fold(1u128, |acc, &d| acc.checked_mul(d as u128));
    let mut note = None;
    match (numel, dtype_bits(dtype)) {
        (Some(n), Some(bits)) => {
            let expect = (n * bits as u128).div_ceil(8);
            if expect != stored as u128 {
                note = Some(format!(
                    "tensor '{name}': {dtype} {shape:?} needs {expect} bytes but data_offsets span {stored}"
                ));
            }
        }
        (None, _) => {
            note = Some(format!(
                "tensor '{name}': shape overflows element arithmetic"
            ))
        }
        (_, None) => note = Some(format!("tensor '{name}': unknown dtype '{dtype}'")),
    }

    let canonical_dtype = if dtype_bits(dtype).is_some() {
        canonical(dtype)
    } else {
        dtype.to_string()
    };
    let tensor = Tensor::new(name, &canonical_dtype, shape, begin, stored);
    Ok((tensor, end, note))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder;
    use std::io::Cursor;

    fn parse_bytes(b: &[u8]) -> Result<Report, PeekError> {
        parse(&mut Cursor::new(b), b.len() as u64, "test.safetensors")
    }

    fn demo() -> Vec<u8> {
        builder::safetensors(
            &[
                ("embed.weight", "F32", &[32, 8]),
                ("fc1.weight", "F16", &[8, 16]),
                ("fc1.bias", "F16", &[16]),
            ],
            &[("format", "pt")],
        )
    }

    #[test]
    fn parses_a_well_formed_file_in_header_order() {
        let r = parse_bytes(&demo()).unwrap();
        assert_eq!(r.format, "safetensors");
        assert_eq!(r.tensors.len(), 3);
        assert_eq!(r.parameters(), 32 * 8 + 8 * 16 + 16);
        assert!(
            r.problems.is_empty(),
            "unexpected problems: {:?}",
            r.problems
        );
        let names: Vec<&str> = r.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["embed.weight", "fc1.weight", "fc1.bias"]);
    }

    #[test]
    fn dtype_is_canonicalized_and_bytes_computed() {
        let r = parse_bytes(&demo()).unwrap();
        let t = &r.tensors[1];
        assert_eq!(t.dtype, "f16");
        assert_eq!(t.shape, [8, 16]);
        assert_eq!(t.bytes, 8 * 16 * 2);
        assert_eq!(t.offset, 32 * 8 * 4); // right after embed.weight
    }

    #[test]
    fn metadata_map_is_reported() {
        let r = parse_bytes(&demo()).unwrap();
        assert_eq!(r.metadata.get("format").and_then(Json::as_str), Some("pt"));
    }

    #[test]
    fn header_and_data_accounting_add_up() {
        let bytes = demo();
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.header_bytes + r.data_bytes, bytes.len() as u64);
        assert_eq!(r.data_bytes, 32 * 8 * 4 + 8 * 16 * 2 + 16 * 2);
    }

    #[test]
    fn truncation_and_trailing_bytes_are_problems_not_errors() {
        let mut bytes = demo();
        bytes.truncate(bytes.len() - 100);
        let r = parse_bytes(&bytes).unwrap();
        assert!(
            r.problems.iter().any(|p| p.contains("100 missing")),
            "problems: {:?}",
            r.problems
        );
        let mut bytes = demo();
        bytes.extend_from_slice(&[0u8; 7]);
        let r = parse_bytes(&bytes).unwrap();
        assert!(
            r.problems.iter().any(|p| p.contains("7 trailing")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn undecidable_headers_are_fatal() {
        // Claims an 800-byte header in a 20-byte file.
        let mut bytes = 800u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{\"a\":1}");
        let err = parse_bytes(&bytes).unwrap_err();
        assert!(err.0.contains("exceeds the file"), "got: {err}");
        // Claims a header beyond the format's 100 MB cap.
        let mut bytes = (MAX_HEADER + 1).to_le_bytes().to_vec();
        bytes.extend_from_slice(&[b' '; 64]);
        let err = parse(&mut Cursor::new(&bytes[..]), u64::MAX, "x").unwrap_err();
        assert!(err.0.contains("100 MB"), "got: {err}");
        // A header that is not JSON at all.
        let mut bytes = 5u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{oops");
        assert!(parse_bytes(&bytes)
            .unwrap_err()
            .0
            .contains("not valid JSON"));
    }

    #[test]
    fn unknown_dtype_is_reported_but_not_fatal() {
        let header = br#"{"t":{"dtype":"F6_E2M3","shape":[4],"data_offsets":[0,3]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 3]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].dtype, "F6_E2M3");
        assert!(r.problems.iter().any(|p| p.contains("unknown dtype")));
    }

    #[test]
    fn size_mismatch_between_dtype_and_offsets_is_flagged() {
        // F32 [4] needs 16 bytes but the offsets span only 8.
        let header = br#"{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 8]);
        let r = parse_bytes(&bytes).unwrap();
        assert!(
            r.problems.iter().any(|p| p.contains("needs 16 bytes")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn scalar_tensor_with_empty_shape() {
        let header = br#"{"s":{"dtype":"F64","shape":[],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 8]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].numel, 1);
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn padded_header_with_trailing_spaces_parses() {
        // The reference writer pads headers with spaces to align the data.
        let header = br#"{"t":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}    "#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 2]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors.len(), 1);
        assert!(r.problems.is_empty());
    }
}
