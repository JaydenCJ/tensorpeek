//! Format detection. Magic bytes first — file extensions lie — with a
//! plausibility heuristic for safetensors (which has no magic) and an
//! extension fallback for empty or ambiguous prefixes.

/// The four formats tensorpeek understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Gguf,
    Npy,
    Npz,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Safetensors => "safetensors",
            Format::Gguf => "gguf",
            Format::Npy => "npy",
            Format::Npz => "npz",
        }
    }

    /// For `--as <format>` on the command line.
    pub fn from_name(s: &str) -> Option<Format> {
        match s {
            "safetensors" => Some(Format::Safetensors),
            "gguf" => Some(Format::Gguf),
            "npy" => Some(Format::Npy),
            "npz" => Some(Format::Npz),
            _ => None,
        }
    }
}

/// Detect the format from the first bytes of the file (at least 9 are
/// needed for the safetensors heuristic), the file length and — as a last
/// resort — the file name.
pub fn detect(prefix: &[u8], file_len: u64, path: &str) -> Option<Format> {
    if prefix.starts_with(b"GGUF") {
        return Some(Format::Gguf);
    }
    if prefix.starts_with(crate::npy::MAGIC) {
        return Some(Format::Npy);
    }
    // Any ZIP local-header or (for an empty archive) EOCD signature.
    if prefix.starts_with(b"PK\x03\x04") || prefix.starts_with(b"PK\x05\x06") {
        return Some(Format::Npz);
    }
    // safetensors has no magic: an 8-byte LE header length that fits the
    // file, followed by the '{' of the JSON header, is a strong signal.
    if prefix.len() >= 9 {
        let header_len = u64::from_le_bytes(prefix[..8].try_into().unwrap());
        if header_len > 0 && header_len.saturating_add(8) <= file_len && prefix[8] == b'{' {
            return Some(Format::Safetensors);
        }
    }
    // Extension fallback for files whose prefix proves nothing.
    let lower = path.to_ascii_lowercase();
    for (ext, fmt) in [
        (".safetensors", Format::Safetensors),
        (".gguf", Format::Gguf),
        (".npy", Format::Npy),
        (".npz", Format::Npz),
    ] {
        if lower.ends_with(ext) {
            return Some(fmt);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder;

    #[test]
    fn magic_bytes_identify_each_format() {
        let st = builder::safetensors(&[("t", "F32", &[2])], &[]);
        assert_eq!(
            detect(&st[..16], st.len() as u64, "x.bin"),
            Some(Format::Safetensors)
        );
        let gg = builder::GgufBuilder::demo().build();
        assert_eq!(
            detect(&gg[..16], gg.len() as u64, "x.bin"),
            Some(Format::Gguf)
        );
        let np = builder::npy("<f4", false, &[2]);
        assert_eq!(
            detect(&np[..16], np.len() as u64, "x.bin"),
            Some(Format::Npy)
        );
        let nz = builder::npz(&[("a.npy", builder::npy("<f4", false, &[2]), false)]);
        assert_eq!(
            detect(&nz[..16], nz.len() as u64, "x.bin"),
            Some(Format::Npz)
        );
    }

    #[test]
    fn magic_beats_a_lying_extension() {
        let gg = builder::GgufBuilder::demo().build();
        assert_eq!(
            detect(&gg[..16], gg.len() as u64, "model.safetensors"),
            Some(Format::Gguf)
        );
    }

    #[test]
    fn safetensors_heuristic_requires_a_plausible_length_and_brace() {
        // Header length larger than the file: not safetensors.
        let mut b = u64::MAX.to_le_bytes().to_vec();
        b.push(b'{');
        assert_eq!(detect(&b, 9, "x.bin"), None);
        // Plausible length but no '{': not safetensors.
        let mut b = 8u64.to_le_bytes().to_vec();
        b.push(b'x');
        assert_eq!(detect(&b, 100, "x.bin"), None);
    }

    #[test]
    fn extension_fallback_unknowns_and_name_round_trip() {
        assert_eq!(detect(&[], 0, "weights.npz"), Some(Format::Npz));
        assert_eq!(detect(&[0u8; 16], 16, "model.GGUF"), Some(Format::Gguf));
        assert_eq!(detect(&[0u8; 16], 16, "mystery.bin"), None);
        for f in [Format::Safetensors, Format::Gguf, Format::Npy, Format::Npz] {
            assert_eq!(Format::from_name(f.name()), Some(f));
        }
        assert_eq!(Format::from_name("onnx"), None);
    }
}
