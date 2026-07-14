//! Spec-exact fixture writers for all four formats. Used by the unit tests,
//! the CLI integration tests, `examples/gen_fixtures.rs` and the smoke
//! script, so every test runs against real bytes rather than mocks. Writers
//! panic on inputs a fixture would never use (e.g. an unknown safetensors
//! dtype) — they are test infrastructure, not a public authoring API.

use crate::json::Json;

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

fn st_elem_bytes(dtype: &str) -> u64 {
    match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        other => panic!("fixture writer does not know dtype {other}"),
    }
}

/// Write a safetensors file with zeroed data. Offsets are packed
/// back-to-back; the header is space-padded to an 8-byte boundary exactly
/// like the reference writer.
pub fn safetensors(tensors: &[(&str, &str, &[u64])], metadata: &[(&str, &str)]) -> Vec<u8> {
    let mut members: Vec<(String, Json)> = Vec::new();
    if !metadata.is_empty() {
        let meta = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), Json::Str(v.to_string())))
            .collect();
        members.push(("__metadata__".into(), Json::Obj(meta)));
    }
    let mut offset = 0u64;
    for (name, dtype, shape) in tensors {
        let numel: u64 = shape.iter().product();
        let bytes = numel * st_elem_bytes(dtype);
        members.push((
            name.to_string(),
            Json::Obj(vec![
                ("dtype".into(), Json::Str(dtype.to_string())),
                (
                    "shape".into(),
                    Json::Arr(shape.iter().map(|&d| Json::from(d)).collect()),
                ),
                (
                    "data_offsets".into(),
                    Json::Arr(vec![Json::from(offset), Json::from(offset + bytes)]),
                ),
            ]),
        ));
        offset += bytes;
    }
    let mut header = Json::Obj(members).compact();
    while (header.len() + 8) % 8 != 0 {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat(0u8).take(offset as usize));
    out
}

// ---------------------------------------------------------------------------
// npy
// ---------------------------------------------------------------------------

fn npy_dict(descr: &str, fortran: bool, shape: &[u64]) -> String {
    let shape_text = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    format!(
        "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape_text}, }}",
        if fortran { "True" } else { "False" }
    )
}

fn npy_data_len(descr: &str, shape: &[u64]) -> usize {
    let itemsize = crate::npy::parse_descr_str(descr)
        .unwrap_or_else(|| panic!("fixture writer does not know descr {descr}"))
        .1;
    (shape.iter().product::<u64>() * itemsize) as usize
}

/// Write a format-1.0 npy file (u16 header length, 64-byte alignment, zeroed
/// data), byte-identical to what `np.save` produces for the same dtype/shape.
pub fn npy(descr: &str, fortran: bool, shape: &[u64]) -> Vec<u8> {
    let mut dict = npy_dict(descr, fortran, shape);
    let base = 6 + 2 + 2;
    let pad = (64 - (base + dict.len() + 1) % 64) % 64;
    dict.push_str(&" ".repeat(pad));
    dict.push('\n');
    let mut out = crate::npy::MAGIC.to_vec();
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat(0u8).take(npy_data_len(descr, shape)));
    out
}

/// Write a format-2.0 npy file (u32 header length).
pub fn npy_v2(descr: &str, fortran: bool, shape: &[u64]) -> Vec<u8> {
    let mut dict = npy_dict(descr, fortran, shape);
    let base = 6 + 2 + 4;
    let pad = (64 - (base + dict.len() + 1) % 64) % 64;
    dict.push_str(&" ".repeat(pad));
    dict.push('\n');
    let mut out = crate::npy::MAGIC.to_vec();
    out.extend_from_slice(&[2, 0]);
    out.extend_from_slice(&(dict.len() as u32).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat(0u8).take(npy_data_len(descr, shape)));
    out
}

/// Write an npy file around a verbatim header dict (no padding fix-ups) —
/// for malformed-header tests.
pub fn npy_raw(dict: &str, data_len: usize) -> Vec<u8> {
    let mut out = crate::npy::MAGIC.to_vec();
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat(0u8).take(data_len));
    out
}

// ---------------------------------------------------------------------------
// ZIP / npz
// ---------------------------------------------------------------------------

/// CRC-32 (IEEE 802.3), bitwise — plenty fast for fixture-sized data.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Wrap `data` in DEFLATE *stored* blocks (BTYPE=0) — a fully valid DEFLATE
/// stream any inflater accepts, producible without implementing compression.
pub fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 6);
    let mut chunks = data.chunks(65535).peekable();
    if data.is_empty() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
        return out;
    }
    while let Some(chunk) = chunks.next() {
        let last = chunks.peek().is_none();
        out.push(u8::from(last)); // BFINAL, BTYPE=00
        out.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(chunk.len() as u16)).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out
}

/// One entry as it will appear in the central directory. `flags` and
/// `method` are public so tests can fake encrypted or exotic members.
pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub flags: u16,
    pub crc: u32,
    pub comp_size: u64,
    pub uncomp_size: u64,
    pub local_offset: u64,
}

/// A minimal ZIP writer: stored members, "deflated" members (via stored
/// DEFLATE blocks or pre-compressed bytes) and optional forced ZIP64 records.
pub struct ZipWriter {
    buf: Vec<u8>,
    pub entries: Vec<ZipEntry>,
}

impl Default for ZipWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ZipWriter {
    pub fn new() -> ZipWriter {
        ZipWriter {
            buf: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn add_stored(&mut self, name: &str, data: &[u8]) {
        self.add_member(name, 0, crc32(data), data.len() as u64, data);
    }

    pub fn add_deflated(&mut self, name: &str, data: &[u8]) {
        let comp = deflate_stored(data);
        self.add_member(name, 8, crc32(data), data.len() as u64, &comp);
    }

    /// Add a member whose compressed payload was produced elsewhere.
    pub fn add_precompressed(&mut self, name: &str, comp: &[u8], uncomp_size: u64, crc: u32) {
        self.add_member(name, 8, crc, uncomp_size, comp);
    }

    fn add_member(&mut self, name: &str, method: u16, crc: u32, uncomp_size: u64, payload: &[u8]) {
        let local_offset = self.buf.len() as u64;
        self.buf.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        self.buf.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.buf.extend_from_slice(&method.to_le_bytes());
        self.buf.extend_from_slice(&[0u8; 4]); // dos time/date
        self.buf.extend_from_slice(&crc.to_le_bytes());
        self.buf
            .extend_from_slice(&(payload.len() as u32).to_le_bytes());
        self.buf
            .extend_from_slice(&(uncomp_size as u32).to_le_bytes());
        self.buf
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.buf.extend_from_slice(&0u16.to_le_bytes()); // extra len
        self.buf.extend_from_slice(name.as_bytes());
        self.buf.extend_from_slice(payload);
        self.entries.push(ZipEntry {
            name: name.to_string(),
            method,
            flags: 0,
            crc,
            comp_size: payload.len() as u64,
            uncomp_size,
            local_offset,
        });
    }

    /// Append the central directory and EOCD. With `zip64`, sizes and
    /// offsets in the central directory use 0xFFFFFFFF sentinels resolved by
    /// ZIP64 extra fields, and ZIP64 EOCD + locator records are written —
    /// the layout numpy produces for >4 GiB archives.
    pub fn finish(mut self, zip64: bool) -> Vec<u8> {
        let cd_offset = self.buf.len() as u64;
        for e in &self.entries {
            self.buf.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            self.buf.extend_from_slice(&(20u16 | 0x0300).to_le_bytes()); // made by: unix
            self.buf.extend_from_slice(&20u16.to_le_bytes()); // version needed
            self.buf.extend_from_slice(&e.flags.to_le_bytes());
            self.buf.extend_from_slice(&e.method.to_le_bytes());
            self.buf.extend_from_slice(&[0u8; 4]); // dos time/date
            self.buf.extend_from_slice(&e.crc.to_le_bytes());
            let (comp32, uncomp32, off32, extra_len) = if zip64 {
                (0xFFFF_FFFFu32, 0xFFFF_FFFFu32, 0xFFFF_FFFFu32, 4 + 24u16)
            } else {
                (
                    e.comp_size as u32,
                    e.uncomp_size as u32,
                    e.local_offset as u32,
                    0,
                )
            };
            self.buf.extend_from_slice(&comp32.to_le_bytes());
            self.buf.extend_from_slice(&uncomp32.to_le_bytes());
            self.buf
                .extend_from_slice(&(e.name.len() as u16).to_le_bytes());
            self.buf.extend_from_slice(&extra_len.to_le_bytes());
            self.buf.extend_from_slice(&[0u8; 2]); // comment len
            self.buf.extend_from_slice(&[0u8; 8]); // disk start, int attrs, ext attrs
            self.buf.extend_from_slice(&off32.to_le_bytes());
            self.buf.extend_from_slice(e.name.as_bytes());
            if zip64 {
                self.buf.extend_from_slice(&0x0001u16.to_le_bytes());
                self.buf.extend_from_slice(&24u16.to_le_bytes());
                self.buf.extend_from_slice(&e.uncomp_size.to_le_bytes());
                self.buf.extend_from_slice(&e.comp_size.to_le_bytes());
                self.buf.extend_from_slice(&e.local_offset.to_le_bytes());
            }
        }
        let cd_size = self.buf.len() as u64 - cd_offset;
        let n = self.entries.len();
        if zip64 {
            let eocd64_offset = self.buf.len() as u64;
            self.buf.extend_from_slice(&0x0606_4b50u32.to_le_bytes());
            self.buf.extend_from_slice(&44u64.to_le_bytes()); // size of remainder
            self.buf.extend_from_slice(&(45u16 | 0x0300).to_le_bytes());
            self.buf.extend_from_slice(&45u16.to_le_bytes());
            self.buf.extend_from_slice(&[0u8; 8]); // disk numbers
            self.buf.extend_from_slice(&(n as u64).to_le_bytes());
            self.buf.extend_from_slice(&(n as u64).to_le_bytes());
            self.buf.extend_from_slice(&cd_size.to_le_bytes());
            self.buf.extend_from_slice(&cd_offset.to_le_bytes());
            self.buf.extend_from_slice(&0x0706_4b50u32.to_le_bytes());
            self.buf.extend_from_slice(&[0u8; 4]); // disk with EOCD64
            self.buf.extend_from_slice(&eocd64_offset.to_le_bytes());
            self.buf.extend_from_slice(&1u32.to_le_bytes()); // total disks
        }
        self.buf.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        self.buf.extend_from_slice(&[0u8; 4]); // disk numbers
        let n16 = if zip64 { 0xFFFFu16 } else { n as u16 };
        self.buf.extend_from_slice(&n16.to_le_bytes());
        self.buf.extend_from_slice(&n16.to_le_bytes());
        let (cds32, cdo32) = if zip64 {
            (0xFFFF_FFFFu32, 0xFFFF_FFFFu32)
        } else {
            (cd_size as u32, cd_offset as u32)
        };
        self.buf.extend_from_slice(&cds32.to_le_bytes());
        self.buf.extend_from_slice(&cdo32.to_le_bytes());
        self.buf.extend_from_slice(&[0u8; 2]); // comment len
        self.buf
    }
}

/// Write an npz archive: `(member name, member bytes, deflate?)`.
pub fn npz(members: &[(&str, Vec<u8>, bool)]) -> Vec<u8> {
    let mut z = ZipWriter::new();
    for (name, data, deflate) in members {
        if *deflate {
            z.add_deflated(name, data);
        } else {
            z.add_stored(name, data);
        }
    }
    z.finish(false)
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

/// A GGUF metadata value the builder can serialize.
#[derive(Debug, Clone)]
pub enum Gv {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    AStr(Vec<String>),
    AI32(Vec<i32>),
    AF32(Vec<f32>),
}

/// Builds spec-exact GGUF v3 (little-endian) files with zeroed, aligned
/// tensor data.
pub struct GgufBuilder {
    alignment: u64,
    kvs: Vec<(String, Gv)>,
    tensors: Vec<(String, Vec<u64>, u32)>,
}

impl GgufBuilder {
    pub fn new() -> GgufBuilder {
        GgufBuilder {
            alignment: 32,
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// A small llama-flavored model: realistic keys, a 12-token vocabulary
    /// and four tensors (q8_0, q4_0 and f32).
    pub fn demo() -> GgufBuilder {
        let tokens: Vec<String> = [
            "<unk>", "<s>", "</s>", "the", "ten", "sor", "peek", "of", "and", "to", "in", "a",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let n = tokens.len();
        GgufBuilder::new()
            .kv("general.architecture", Gv::Str("llama".into()))
            .kv("general.name", Gv::Str("tinyllama-demo".into()))
            .kv("general.file_type", Gv::U32(7))
            .kv("general.quantization_version", Gv::U32(2))
            .kv("llama.block_count", Gv::U32(1))
            .kv("llama.context_length", Gv::U32(512))
            .kv("llama.embedding_length", Gv::U32(64))
            .kv("llama.rope.freq_base", Gv::F32(10000.0))
            .kv("tokenizer.ggml.model", Gv::Str("llama".into()))
            .kv("tokenizer.ggml.tokens", Gv::AStr(tokens))
            .kv("tokenizer.ggml.scores", Gv::AF32(vec![0.0; n]))
            .kv("tokenizer.ggml.token_type", Gv::AI32(vec![1; n]))
            .tensor("token_embd.weight", &[64, 256], 8) // q8_0
            .tensor("blk.0.attn_norm.weight", &[64], 0) // f32
            .tensor("blk.0.ffn_down.weight", &[128, 64], 2) // q4_0
            .tensor("output_norm.weight", &[64], 0) // f32
    }

    pub fn kv(mut self, key: &str, value: Gv) -> GgufBuilder {
        if key == "general.alignment" {
            if let Gv::U32(a) = value {
                if a > 0 && (a as u64).is_power_of_two() {
                    self.alignment = a as u64;
                }
            }
        }
        self.kvs.push((key.to_string(), value));
        self
    }

    pub fn tensor(mut self, name: &str, shape: &[u64], type_id: u32) -> GgufBuilder {
        self.tensors
            .push((name.to_string(), shape.to_vec(), type_id));
        self
    }

    pub fn build(&self) -> Vec<u8> {
        self.build_with_alignment(self.alignment)
    }

    /// Build with an explicit physical alignment — used to model files whose
    /// declared `general.alignment` is invalid, where writers fall back to 32.
    pub fn build_with_alignment(&self, alignment: u64) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (key, value) in &self.kvs {
            write_gguf_string(&mut out, key);
            write_gguf_value(&mut out, value);
        }
        // Tensor infos, with offsets packed in aligned order.
        let mut offset = 0u64;
        let mut data_len = 0u64;
        for (name, shape, type_id) in &self.tensors {
            write_gguf_string(&mut out, name);
            out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for d in shape {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&type_id.to_le_bytes());
            offset = offset.div_ceil(alignment) * alignment;
            out.extend_from_slice(&offset.to_le_bytes());
            let numel: u64 = shape.iter().product();
            let bytes = match crate::gguf::type_geometry(*type_id) {
                Some((block_len, block_bytes)) => numel.div_ceil(block_len) * block_bytes,
                None => 0,
            };
            data_len = offset + bytes;
            offset += bytes;
        }
        // Pad to the data start, then write zeroed tensor data.
        while (out.len() as u64) % alignment != 0 {
            out.push(0);
        }
        out.extend(std::iter::repeat(0u8).take(data_len as usize));
        out
    }
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn write_gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_gguf_value(out: &mut Vec<u8>, v: &Gv) {
    match v {
        Gv::U8(x) => {
            out.extend_from_slice(&0u32.to_le_bytes());
            out.push(*x);
        }
        Gv::I8(x) => {
            out.extend_from_slice(&1u32.to_le_bytes());
            out.push(*x as u8);
        }
        Gv::U16(x) => {
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::I16(x) => {
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::I32(x) => {
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::Bool(x) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*x));
        }
        Gv::Str(x) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_gguf_string(out, x);
        }
        Gv::U64(x) => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::I64(x) => {
            out.extend_from_slice(&11u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::F64(x) => {
            out.extend_from_slice(&12u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Gv::AStr(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                write_gguf_string(out, s);
            }
        }
        Gv::AI32(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for x in items {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Gv::AF32(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for x in items {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_reference_check_value() {
        // The classic CRC-32 check: crc32("123456789") == 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn deflate_stored_handles_multi_chunk_and_empty_input() {
        // 70000 bytes forces two stored blocks (max 65535 per block).
        let data: Vec<u8> = (0..70_000u32).map(|i| (i % 256) as u8).collect();
        let stream = deflate_stored(&data);
        let round = crate::inflate::inflate_prefix(&stream, data.len()).unwrap();
        assert_eq!(round.data, data);
        let empty = crate::inflate::inflate_prefix(&deflate_stored(&[]), 10).unwrap();
        assert!(empty.data.is_empty() && empty.complete);
    }

    #[test]
    fn safetensors_header_is_8_byte_aligned() {
        let bytes = safetensors(&[("t", "F32", &[3])], &[]);
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        assert_eq!((8 + header_len) % 8, 0);
        assert_eq!(bytes.len() as u64, 8 + header_len + 12);
    }
}
