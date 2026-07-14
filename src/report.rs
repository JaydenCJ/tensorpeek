//! The unified report model every format parser produces, plus its JSON
//! rendering. One schema for four formats is the whole point of tensorpeek:
//! `.tensors[].shape` means the same thing whether the file was written by a
//! Python training loop or a llama.cpp quantizer.

use crate::json::Json;

/// A parse failure that makes the file undecidable (bad magic, malformed
/// header, implausible counts). Non-fatal irregularities go into
/// [`Report::problems`] instead.
#[derive(Debug)]
pub struct PeekError(pub String);

impl std::fmt::Display for PeekError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<std::io::Error> for PeekError {
    fn from(e: std::io::Error) -> Self {
        PeekError(format!("I/O error: {e}"))
    }
}

/// One tensor as described by the file header.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name; empty for the single unnamed array in an `.npy` file.
    pub name: String,
    /// Canonical dtype name (`f32`, `bf16`, `q4_k`, …) or the raw name when
    /// the dtype is unknown to tensorpeek.
    pub dtype: String,
    /// Shape exactly as stored in the file (see docs/output-schema.md for
    /// the GGUF element-order caveat).
    pub shape: Vec<u64>,
    /// Element count (product of the shape; 1 for a scalar).
    pub numel: u128,
    /// Byte offset of this tensor within the data region it lives in.
    pub offset: u64,
    /// Size in bytes, computed from dtype and shape (or taken from the
    /// header when the format stores it directly).
    pub bytes: u64,
    /// Format-specific extras (e.g. `member` and `compression` for npz).
    pub extra: Vec<(String, Json)>,
}

impl Tensor {
    pub fn new(name: &str, dtype: &str, shape: Vec<u64>, offset: u64, bytes: u64) -> Tensor {
        // Saturating: a hostile header can declare dimensions whose product
        // exceeds even u128, and that must not become an overflow panic.
        let numel = shape
            .iter()
            .fold(1u128, |acc, &d| acc.saturating_mul(d as u128));
        Tensor {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            numel,
            offset,
            bytes,
            extra: Vec::new(),
        }
    }
}

/// Everything tensorpeek learned about one file.
#[derive(Debug)]
pub struct Report {
    pub file: String,
    /// Format name: `safetensors`, `gguf`, `npy` or `npz`.
    pub format: &'static str,
    /// Total size of the file on disk.
    pub file_bytes: u64,
    /// Bytes occupied by header/index structures (everything that is not
    /// tensor data).
    pub header_bytes: u64,
    /// Bytes of tensor data the header promises (for npz: the compressed
    /// member payloads actually present in the archive).
    pub data_bytes: u64,
    /// File-level metadata as JSON (safetensors `__metadata__`, GGUF KVs).
    pub metadata: Json,
    /// One format-specific detail object, keyed by format name in the output.
    pub details: Json,
    pub tensors: Vec<Tensor>,
    /// Non-fatal irregularities: truncated data section, size mismatches,
    /// unknown dtypes. `--strict` turns these into a failing exit code.
    pub problems: Vec<String>,
}

impl Report {
    pub fn new(file: &str, format: &'static str, file_bytes: u64) -> Report {
        Report {
            file: file.to_string(),
            format,
            file_bytes,
            header_bytes: 0,
            data_bytes: 0,
            metadata: Json::Obj(Vec::new()),
            details: Json::Obj(Vec::new()),
            tensors: Vec::new(),
            problems: Vec::new(),
        }
    }

    /// Total parameter count across all tensors.
    pub fn parameters(&self) -> u128 {
        self.tensors.iter().map(|t| t.numel).sum()
    }

    /// Render as a JSON object. `filter` keeps only matching tensor names
    /// (glob, comma-separated alternatives); `with_tensors=false` drops the
    /// tensor list entirely but keeps the counts.
    pub fn to_json(&self, with_tensors: bool, filter: Option<&str>) -> Json {
        let mut o: Vec<(String, Json)> = vec![
            ("file".into(), Json::Str(self.file.clone())),
            ("format".into(), Json::Str(self.format.into())),
            ("file_bytes".into(), Json::from(self.file_bytes)),
            ("header_bytes".into(), Json::from(self.header_bytes)),
            ("data_bytes".into(), Json::from(self.data_bytes)),
            ("tensor_count".into(), Json::Int(self.tensors.len() as i128)),
            ("parameters".into(), Json::Int(self.parameters() as i128)),
        ];
        if !matches!(&self.details, Json::Obj(m) if m.is_empty()) {
            o.push((self.format.to_string(), self.details.clone()));
        }
        if !matches!(&self.metadata, Json::Obj(m) if m.is_empty()) {
            o.push(("metadata".into(), self.metadata.clone()));
        }
        if with_tensors {
            let tensors: Vec<Json> = self
                .tensors
                .iter()
                .filter(|t| filter.map_or(true, |f| filter_match(f, &t.name)))
                .map(tensor_json)
                .collect();
            o.push(("tensors".into(), Json::Arr(tensors)));
        }
        if !self.problems.is_empty() {
            let p = self.problems.iter().map(|s| Json::Str(s.clone())).collect();
            o.push(("problems".into(), Json::Arr(p)));
        }
        Json::Obj(o)
    }
}

fn tensor_json(t: &Tensor) -> Json {
    let mut o: Vec<(String, Json)> = vec![
        ("name".into(), Json::Str(t.name.clone())),
        ("dtype".into(), Json::Str(t.dtype.clone())),
        (
            "shape".into(),
            Json::Arr(t.shape.iter().map(|&d| Json::from(d)).collect()),
        ),
        ("numel".into(), Json::Int(t.numel as i128)),
        ("offset".into(), Json::from(t.offset)),
        ("bytes".into(), Json::from(t.bytes)),
    ];
    o.extend(t.extra.iter().cloned());
    Json::Obj(o)
}

/// Match `name` against a comma-separated list of glob alternatives
/// (`*` = any run, `?` = any one character). Used by `--filter`.
pub fn filter_match(patterns: &str, name: &str) -> bool {
    patterns.split(',').any(|p| glob_match(p.trim(), name))
}

fn glob_match(pattern: &str, text: &str) -> bool {
    // Iterative backtracking matcher — no recursion, no pathological blowup.
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_p, mut star_t) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_p = pi;
            star_t = ti;
            pi += 1;
        } else if star_p != usize::MAX {
            pi = star_p + 1;
            star_t += 1;
            ti = star_t;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_handles_scalars_and_zero_dims() {
        assert_eq!(Tensor::new("s", "f32", vec![], 0, 4).numel, 1);
        assert_eq!(Tensor::new("e", "f32", vec![0, 4], 0, 0).numel, 0);
        assert_eq!(Tensor::new("m", "f32", vec![2, 3, 4], 0, 96).numel, 24);
        // A hostile header whose dimension product exceeds u128 must
        // saturate, not panic with an arithmetic overflow.
        let hostile = Tensor::new("h", "f32", vec![u64::MAX; 4], 0, 0);
        assert_eq!(hostile.numel, u128::MAX);
    }

    #[test]
    fn parameters_sum_uses_wide_arithmetic() {
        let mut r = Report::new("x", "safetensors", 0);
        // Two tensors that together overflow u64 element counts must not wrap.
        r.tensors
            .push(Tensor::new("a", "f32", vec![u64::MAX / 2, 3], 0, 0));
        r.tensors
            .push(Tensor::new("b", "f32", vec![u64::MAX / 2, 3], 0, 0));
        assert_eq!(r.parameters(), (u64::MAX / 2) as u128 * 3 * 2);
    }

    #[test]
    fn json_shape_has_stable_top_level_keys() {
        let mut r = Report::new("m.npy", "npy", 100);
        r.tensors.push(Tensor::new("", "f32", vec![2, 3], 0, 24));
        let j = r.to_json(true, None);
        for key in [
            "file",
            "format",
            "file_bytes",
            "header_bytes",
            "data_bytes",
            "tensor_count",
            "parameters",
            "tensors",
        ] {
            assert!(j.get(key).is_some(), "missing key {key}");
        }
        assert!(
            j.get("problems").is_none(),
            "empty problems must be omitted"
        );
    }

    #[test]
    fn filter_narrows_the_tensor_list_but_not_the_counts() {
        let mut r = Report::new("m.safetensors", "safetensors", 0);
        r.tensors
            .push(Tensor::new("fc1.weight", "f32", vec![8], 0, 32));
        r.tensors
            .push(Tensor::new("fc1.bias", "f32", vec![8], 32, 32));
        r.tensors
            .push(Tensor::new("fc2.weight", "f32", vec![8], 64, 32));
        let j = r.to_json(true, Some("*.weight"));
        assert_eq!(j.get("tensors").and_then(Json::as_arr).unwrap().len(), 2);
        assert_eq!(j.get("tensor_count").and_then(Json::as_int), Some(3));
    }

    #[test]
    fn glob_star_question_alternatives_and_backtracking() {
        assert!(filter_match("blk.*.weight", "blk.31.attn_q.weight"));
        assert!(filter_match("fc?.bias", "fc1.bias"));
        assert!(!filter_match("fc?.bias", "fc10.bias"));
        assert!(filter_match("*.bias,*.weight", "fc1.weight"));
        assert!(filter_match("*", ""));
        assert!(!filter_match("", "x"));
        // Multiple stars require backtracking, not greedy first-match.
        assert!(filter_match("*attn*weight", "blk.0.attn_output.weight"));
        assert!(!filter_match("*attn*weight", "blk.0.ffn_down.weight"));
    }
}
