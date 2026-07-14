//! NumPy `.npy` header parser. An npy file is `\x93NUMPY`, a two-byte
//! version, a little-endian header length (u16 for format 1.0, u32 for
//! 2.0/3.0) and a Python dict literal like
//! `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }`.
//! The dict is parsed with a small purpose-built literal parser — no Python
//! required — including structured (record) dtypes.

use std::io::Read;

use crate::json::Json;
use crate::report::{PeekError, Report, Tensor};

pub const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Everything found in one npy header. Shared with the npz parser, which
/// extracts the same header from each archive member.
#[derive(Debug)]
pub struct NpyHeader {
    /// Format version, e.g. (1, 0).
    pub version: (u8, u8),
    /// Total bytes from the magic through the end of the padded dict —
    /// i.e. the offset where array data begins.
    pub header_total: u64,
    /// The raw `descr` value rendered as JSON (a string, or nested arrays
    /// for structured dtypes).
    pub descr: Json,
    /// Canonical dtype name (`f32`, `i64`, `struct`, …).
    pub dtype: String,
    /// Bytes per element, when the descr is understood.
    pub itemsize: Option<u64>,
    /// `little`, `big`, `native` or `none` (for single-byte types).
    pub byte_order: &'static str,
    pub fortran_order: bool,
    pub shape: Vec<u64>,
    pub problems: Vec<String>,
}

impl NpyHeader {
    pub fn numel(&self) -> u128 {
        self.shape
            .iter()
            .fold(1u128, |acc, &d| acc.saturating_mul(d as u128))
    }

    /// Expected size of the data section, when the itemsize is known.
    pub fn data_bytes(&self) -> Option<u64> {
        let n = self.numel().checked_mul(self.itemsize? as u128)?;
        u64::try_from(n).ok()
    }
}

/// How many bytes of prefix can ever be needed: magic + version + u32 length
/// + a maximal dict. Real headers are < 200 bytes; structured ones a few KB.
pub const MAX_PREFIX: usize = 6 + 2 + 4 + 65536;

pub fn parse<R: Read>(r: &mut R, file_len: u64, path: &str) -> Result<Report, PeekError> {
    let mut prefix = Vec::with_capacity(MAX_PREFIX.min(file_len as usize));
    r.take(MAX_PREFIX as u64).read_to_end(&mut prefix)?;
    let h = parse_header(&prefix)?;

    let mut report = Report::new(path, "npy", file_len);
    report.header_bytes = h.header_total;
    report.problems = h.problems.clone();

    let data_present = file_len.saturating_sub(h.header_total);
    match h.data_bytes() {
        Some(expect) => {
            report.data_bytes = expect;
            if data_present < expect {
                report.problems.push(format!(
                    "array data needs {expect} bytes but only {data_present} are present ({} missing) — truncated file",
                    expect - data_present
                ));
            } else if data_present > expect {
                let extra = data_present - expect;
                report.problems.push(format!(
                    "{extra} trailing byte{} after the array data",
                    if extra == 1 { "" } else { "s" }
                ));
            }
        }
        None => report.data_bytes = data_present,
    }

    let bytes = h.data_bytes().unwrap_or(data_present);
    // The single array in an npy file has no name; the report uses "".
    report
        .tensors
        .push(Tensor::new("", &h.dtype, h.shape.clone(), 0, bytes));

    report.details = Json::Obj(vec![
        (
            "version".into(),
            Json::Str(format!("{}.{}", h.version.0, h.version.1)),
        ),
        ("descr".into(), h.descr.clone()),
        ("byte_order".into(), Json::Str(h.byte_order.into())),
        ("fortran_order".into(), Json::Bool(h.fortran_order)),
        (
            "itemsize".into(),
            h.itemsize.map(Json::from).unwrap_or(Json::Null),
        ),
    ]);
    Ok(report)
}

/// Parse an npy header from a file prefix (must contain the whole header).
pub fn parse_header(b: &[u8]) -> Result<NpyHeader, PeekError> {
    if b.len() < 10 || &b[..6] != MAGIC {
        return Err(PeekError("not an npy file (bad magic)".into()));
    }
    let version = (b[6], b[7]);
    let (len_size, header_len) = match version.0 {
        1 => (2usize, u16::from_le_bytes([b[8], b[9]]) as usize),
        2 | 3 => {
            if b.len() < 12 {
                return Err(PeekError("file ends inside the npy header length".into()));
            }
            (
                4usize,
                u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize,
            )
        }
        v => {
            return Err(PeekError(format!(
                "unsupported npy format version {v}.{}",
                version.1
            )))
        }
    };
    let dict_start = 6 + 2 + len_size;
    let header_total = (dict_start + header_len) as u64;
    let Some(dict_bytes) = b.get(dict_start..dict_start + header_len) else {
        return Err(PeekError(format!(
            "declared header length {header_len} runs past the available bytes — truncated header"
        )));
    };
    let dict_text = std::str::from_utf8(dict_bytes)
        .map_err(|_| PeekError("npy header dict is not valid ASCII/UTF-8".into()))?;

    let py = PyParser::parse_dict(dict_text)?;
    let mut problems = Vec::new();

    let mut descr_py = None;
    let mut fortran = None;
    let mut shape_py = None;
    for (k, v) in &py {
        match k.as_str() {
            "descr" => descr_py = Some(v),
            "fortran_order" => match v {
                Py::Bool(x) => fortran = Some(*x),
                _ => return Err(PeekError("fortran_order is not a boolean".into())),
            },
            "shape" => shape_py = Some(v),
            other => problems.push(format!("unexpected npy header key '{other}'")),
        }
    }
    let descr_py = descr_py.ok_or(PeekError("npy header is missing 'descr'".into()))?;
    let fortran_order = fortran.ok_or(PeekError("npy header is missing 'fortran_order'".into()))?;
    let shape_py = shape_py.ok_or(PeekError("npy header is missing 'shape'".into()))?;

    let Py::Seq(dims, _) = shape_py else {
        return Err(PeekError("npy shape is not a tuple".into()));
    };
    let mut shape = Vec::with_capacity(dims.len());
    for d in dims {
        match d {
            Py::Int(n) if *n >= 0 => shape.push(*n as u64),
            _ => {
                return Err(PeekError(
                    "npy shape contains a non-integer dimension".into(),
                ))
            }
        }
    }

    let (dtype, itemsize, byte_order) = interpret_descr(descr_py, &mut problems);

    Ok(NpyHeader {
        version,
        header_total,
        descr: py_to_json(descr_py),
        dtype,
        itemsize,
        byte_order,
        fortran_order,
        shape,
        problems,
    })
}

// ---------------------------------------------------------------------------
// descr interpretation
// ---------------------------------------------------------------------------

fn interpret_descr(descr: &Py, problems: &mut Vec<String>) -> (String, Option<u64>, &'static str) {
    match descr {
        Py::Str(s) => match parse_descr_str(s) {
            Some((dtype, itemsize, order)) => (dtype, Some(itemsize), order),
            None => {
                problems.push(format!("unknown dtype descr '{s}'"));
                (s.clone(), None, "none")
            }
        },
        Py::Seq(fields, _) => match struct_itemsize(fields) {
            Some(size) => ("struct".into(), Some(size), "none"),
            None => {
                problems.push("structured descr could not be fully interpreted".into());
                ("struct".into(), None, "none")
            }
        },
        _ => {
            problems.push("descr is neither a string nor a field list".into());
            ("unknown".into(), None, "none")
        }
    }
}

/// Decode a simple descr string like `<f4`, `>i8`, `|b1`, `<U16`, `<M8[ns]`.
/// Shared with the fixture builder, which needs itemsizes to lay out data.
pub(crate) fn parse_descr_str(s: &str) -> Option<(String, u64, &'static str)> {
    let bytes = s.as_bytes();
    let (order, rest) = match bytes.first()? {
        b'<' => ("little", &s[1..]),
        b'>' => ("big", &s[1..]),
        b'=' => ("native", &s[1..]),
        b'|' => ("none", &s[1..]),
        _ => ("native", s),
    };
    let kind = *rest.as_bytes().first()? as char;
    let size_part = &rest[1..];
    // datetime64/timedelta64 carry a unit suffix: 'M8[ns]'.
    let digits = size_part.split('[').next()?;
    let size: u64 = digits.parse().ok()?;
    let dtype = match (kind, size) {
        ('f', 2) => "f16".to_string(),
        ('f', 4) => "f32".to_string(),
        ('f', 8) => "f64".to_string(),
        ('f', 16) => "f128".to_string(),
        ('i', 1) => "i8".to_string(),
        ('i', 2) => "i16".to_string(),
        ('i', 4) => "i32".to_string(),
        ('i', 8) => "i64".to_string(),
        ('u', 1) => "u8".to_string(),
        ('u', 2) => "u16".to_string(),
        ('u', 4) => "u32".to_string(),
        ('u', 8) => "u64".to_string(),
        ('b', 1) => "bool".to_string(),
        ('c', 8) => "complex64".to_string(),
        ('c', 16) => "complex128".to_string(),
        ('c', 32) => "complex256".to_string(),
        ('m', 8) => "timedelta64".to_string(),
        ('M', 8) => "datetime64".to_string(),
        ('S', n) | ('a', n) => format!("bytes{n}"),
        ('U', n) => format!("str{n}"),
        ('V', n) => format!("void{n}"),
        _ => return None,
    };
    // 'U' stores UCS-4: four bytes per character.
    let itemsize = if kind == 'U' {
        size.checked_mul(4)?
    } else {
        size
    };
    let order = if itemsize <= 1 { "none" } else { order };
    Some((dtype, itemsize, order))
}

/// Itemsize of a structured descr: sum over fields of itemsize × subshape.
fn struct_itemsize(fields: &[Py]) -> Option<u64> {
    let mut total: u64 = 0;
    for field in fields {
        let Py::Seq(parts, _) = field else {
            return None;
        };
        if parts.len() < 2 || parts.len() > 3 {
            return None;
        }
        // parts[0] is the name (or a (title, name) tuple) — size-irrelevant.
        let elem = match &parts[1] {
            Py::Str(s) => parse_descr_str(s)?.1,
            Py::Seq(nested, _) => struct_itemsize(nested)?, // nested struct
            _ => return None,
        };
        let repeat = match parts.get(2) {
            None => 1u64,
            Some(Py::Int(n)) if *n >= 0 => *n as u64,
            Some(Py::Seq(dims, _)) => {
                let mut r: u64 = 1;
                for d in dims {
                    match d {
                        Py::Int(n) if *n >= 0 => r = r.checked_mul(*n as u64)?,
                        _ => return None,
                    }
                }
                r
            }
            _ => return None,
        };
        total = total.checked_add(elem.checked_mul(repeat)?)?;
    }
    Some(total)
}

// ---------------------------------------------------------------------------
// Python literal mini-parser (dicts, strings, bools, ints, tuples, lists)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Py {
    Str(String),
    Bool(bool),
    Int(i64),
    /// A tuple or list; the bool is true for tuples.
    Seq(Vec<Py>, bool),
}

fn py_to_json(v: &Py) -> Json {
    match v {
        Py::Str(s) => Json::Str(s.clone()),
        Py::Bool(b) => Json::Bool(*b),
        Py::Int(n) => Json::Int(*n as i128),
        Py::Seq(items, _) => Json::Arr(items.iter().map(py_to_json).collect()),
    }
}

struct PyParser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> PyParser<'a> {
    /// Parse a top-level dict with string keys; trailing padding is ignored.
    fn parse_dict(text: &str) -> Result<Vec<(String, Py)>, PeekError> {
        let mut p = PyParser {
            b: text.as_bytes(),
            pos: 0,
        };
        p.ws();
        if !p.eat(b'{') {
            return Err(p.err("npy header does not start with '{'"));
        }
        let mut pairs = Vec::new();
        loop {
            p.ws();
            if p.eat(b'}') {
                break;
            }
            let Py::Str(key) = p.value()? else {
                return Err(p.err("dict key is not a string"));
            };
            p.ws();
            if !p.eat(b':') {
                return Err(p.err("expected ':' in header dict"));
            }
            p.ws();
            let v = p.value()?;
            pairs.push((key, v));
            p.ws();
            if p.eat(b',') {
                continue;
            }
            p.ws();
            if p.eat(b'}') {
                break;
            }
            return Err(p.err("expected ',' or '}' in header dict"));
        }
        Ok(pairs)
    }

    fn err(&self, msg: &str) -> PeekError {
        PeekError(format!("{msg} (at header byte {})", self.pos))
    }

    fn ws(&mut self) {
        while matches!(self.b.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.b.get(self.pos) == Some(&c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Py, PeekError> {
        self.ws();
        match self.b.get(self.pos) {
            Some(b'\'') | Some(b'"') => self.string(),
            Some(b'(') => self.seq(b'(', b')', true),
            Some(b'[') => self.seq(b'[', b']', false),
            Some(b'T') if self.b[self.pos..].starts_with(b"True") => {
                self.pos += 4;
                Ok(Py::Bool(true))
            }
            Some(b'F') if self.b[self.pos..].starts_with(b"False") => {
                self.pos += 5;
                Ok(Py::Bool(false))
            }
            Some(c) if c.is_ascii_digit() || *c == b'-' => self.int(),
            _ => Err(self.err("unsupported literal in header dict")),
        }
    }

    fn string(&mut self) -> Result<Py, PeekError> {
        let quote = self.b[self.pos];
        self.pos += 1;
        let mut s = String::new();
        loop {
            match self.b.get(self.pos) {
                None => return Err(self.err("unterminated string")),
                Some(&c) if c == quote => {
                    self.pos += 1;
                    return Ok(Py::Str(s));
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.b.get(self.pos) {
                        Some(b'\\') => s.push('\\'),
                        Some(b'\'') => s.push('\''),
                        Some(b'"') => s.push('"'),
                        Some(&c) => {
                            // Other escapes are passed through verbatim;
                            // dtype descrs never need them.
                            s.push('\\');
                            s.push(c as char);
                        }
                        None => return Err(self.err("unterminated escape")),
                    }
                    self.pos += 1;
                }
                Some(&c) => {
                    s.push(c as char);
                    self.pos += 1;
                }
            }
        }
    }

    fn seq(&mut self, open: u8, close: u8, is_tuple: bool) -> Result<Py, PeekError> {
        debug_assert_eq!(self.b[self.pos], open);
        self.pos += 1;
        let mut items = Vec::new();
        loop {
            self.ws();
            if self.eat(close) {
                return Ok(Py::Seq(items, is_tuple));
            }
            items.push(self.value()?);
            self.ws();
            if self.eat(b',') {
                continue;
            }
            self.ws();
            if self.eat(close) {
                return Ok(Py::Seq(items, is_tuple));
            }
            return Err(self.err("expected ',' or a closing bracket"));
        }
    }

    fn int(&mut self) -> Result<Py, PeekError> {
        let start = self.pos;
        self.eat(b'-');
        while self.b.get(self.pos).is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.b[start..self.pos]).unwrap();
        text.parse::<i64>()
            .map(Py::Int)
            .map_err(|_| self.err("invalid integer"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder;
    use std::io::Cursor;

    fn parse_bytes(b: &[u8]) -> Result<Report, PeekError> {
        parse(&mut Cursor::new(b), b.len() as u64, "test.npy")
    }

    #[test]
    fn parses_a_v1_float32_array_with_exact_accounting() {
        let bytes = builder::npy("<f4", false, &[2, 3]);
        let r = parse_bytes(&bytes).unwrap();
        let t = &r.tensors[0];
        assert_eq!(t.dtype, "f32");
        assert_eq!(t.shape, [2, 3]);
        assert_eq!(t.bytes, 24);
        assert_eq!(t.name, "", "the npy array is unnamed");
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
        // numpy pads headers so data starts on a 64-byte boundary.
        assert_eq!(r.header_bytes % 64, 0);
        assert_eq!(r.header_bytes + r.data_bytes, bytes.len() as u64);
    }

    #[test]
    fn fortran_order_and_byte_order_are_reported() {
        let bytes = builder::npy(">i8", true, &[4]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.details.get("fortran_order"), Some(&Json::Bool(true)));
        assert_eq!(
            r.details.get("byte_order").and_then(Json::as_str),
            Some("big")
        );
        assert_eq!(r.tensors[0].dtype, "i64");
    }

    #[test]
    fn scalar_and_one_element_tuple_shapes() {
        // A zero-dimensional array: shape (), one element.
        let bytes = builder::npy("<f8", false, &[]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].numel, 1);
        assert_eq!(r.tensors[0].bytes, 8);
        // numpy writes `(5,)` with a trailing comma for 1-D shapes.
        let bytes = builder::npy("<u2", false, &[5]);
        assert!(String::from_utf8_lossy(&bytes[..64]).contains("(5,)"));
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].shape, [5]);
        assert_eq!(r.tensors[0].bytes, 10);
    }

    #[test]
    fn version_2_header_uses_a_u32_length() {
        let bytes = builder::npy_v2("<f4", false, &[8]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.details.get("version").and_then(Json::as_str), Some("2.0"));
        assert_eq!(r.tensors[0].bytes, 32);
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn truncated_data_is_a_problem() {
        let mut bytes = builder::npy("<f4", false, &[100]);
        bytes.truncate(bytes.len() - 40);
        let r = parse_bytes(&bytes).unwrap();
        assert!(
            r.problems.iter().any(|p| p.contains("40 missing")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn unicode_and_bytes_descrs_get_correct_itemsizes() {
        let (dtype, itemsize, _) = parse_descr_str("<U16").unwrap();
        assert_eq!((dtype.as_str(), itemsize), ("str16", 64)); // UCS-4
        let (dtype, itemsize, _) = parse_descr_str("|S10").unwrap();
        assert_eq!((dtype.as_str(), itemsize), ("bytes10", 10));
        let (dtype, itemsize, _) = parse_descr_str("<M8[ns]").unwrap();
        assert_eq!((dtype.as_str(), itemsize), ("datetime64", 8));
    }

    #[test]
    fn structured_descr_computes_the_record_size() {
        let header = "{'descr': [('time', '<u8'), ('pos', '<f4', (3,)), ('flag', '|b1')], \
                      'fortran_order': False, 'shape': (10,), }";
        let bytes = builder::npy_raw(header, 10 * (8 + 12 + 1));
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].dtype, "struct");
        assert_eq!(r.tensors[0].bytes, 210);
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn unknown_descr_is_a_problem_not_a_crash() {
        let header = "{'descr': '<x9', 'fortran_order': False, 'shape': (2,), }";
        let bytes = builder::npy_raw(header, 0);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].dtype, "<x9");
        assert!(r.problems.iter().any(|p| p.contains("unknown dtype descr")));
    }

    #[test]
    fn missing_required_key_is_fatal() {
        let header = "{'descr': '<f4', 'shape': (2,), }";
        let bytes = builder::npy_raw(header, 8);
        let err = parse_bytes(&bytes).unwrap_err();
        assert!(err.0.contains("fortran_order"), "got: {err}");
    }

    #[test]
    fn bad_magic_version_and_oversized_header_length_are_fatal() {
        assert!(parse_bytes(b"\x93NUMPZ\x01\x00xxxx")
            .unwrap_err()
            .0
            .contains("bad magic"));

        let mut b = MAGIC.to_vec();
        b.extend_from_slice(&[9, 0, 10, 0]);
        b.extend_from_slice(&[b' '; 10]);
        assert!(parse_bytes(&b).unwrap_err().0.contains("version 9"));

        let mut b = MAGIC.to_vec();
        b.extend_from_slice(&[1, 0]);
        b.extend_from_slice(&9999u16.to_le_bytes());
        b.extend_from_slice(b"{'descr': '<f4'");
        let err = parse_bytes(&b).unwrap_err();
        assert!(err.0.contains("truncated header"), "got: {err}");
    }

    #[test]
    fn malformed_dict_is_fatal_with_position() {
        let header = "{'descr' '<f4'}";
        let bytes = builder::npy_raw(header, 0);
        let err = parse_bytes(&bytes).unwrap_err();
        assert!(err.0.contains("expected ':'"), "got: {err}");
    }
}
