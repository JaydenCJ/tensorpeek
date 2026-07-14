//! A bounded raw-DEFLATE (RFC 1951) decoder, used to read npy headers out of
//! compressed `.npz` members without any dependency. It is a *prefix*
//! decoder: decompression stops as soon as `max_out` bytes have been
//! produced, so extracting a 128-byte header from a member that would
//! decompress to gigabytes costs almost nothing. All three block types
//! (stored, fixed Huffman, dynamic Huffman) are supported.

/// Result of a bounded inflate.
#[derive(Debug)]
pub struct Inflated {
    /// The decompressed bytes (at most `max_out` of them).
    pub data: Vec<u8>,
    /// True when the final block finished inside the budget; false when the
    /// decoder stopped early because `max_out` was reached.
    pub complete: bool,
}

/// Decompress up to `max_out` bytes from a raw DEFLATE stream.
pub fn inflate_prefix(input: &[u8], max_out: usize) -> Result<Inflated, String> {
    let mut r = BitReader {
        b: input,
        pos: 0,
        bit: 0,
    };
    let mut out: Vec<u8> = Vec::new();
    loop {
        let bfinal = r.bits(1)?;
        let btype = r.bits(2)?;
        match btype {
            0 => stored_block(&mut r, &mut out, max_out)?,
            1 => {
                let (lit, dist) = fixed_tables();
                huffman_block(&mut r, &lit, &dist, &mut out, max_out)?;
            }
            2 => {
                let (lit, dist) = dynamic_tables(&mut r)?;
                huffman_block(&mut r, &lit, &dist, &mut out, max_out)?;
            }
            _ => return Err("reserved block type 3".into()),
        }
        if out.len() >= max_out {
            out.truncate(max_out);
            return Ok(Inflated {
                data: out,
                complete: false,
            });
        }
        if bfinal == 1 {
            return Ok(Inflated {
                data: out,
                complete: true,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Bit reader (LSB-first, as DEFLATE packs its bits)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
    bit: u8,
}

impl<'a> BitReader<'a> {
    fn bits(&mut self, n: u8) -> Result<u32, String> {
        let mut v = 0u32;
        for i in 0..n {
            let byte = *self
                .b
                .get(self.pos)
                .ok_or("unexpected end of compressed data")?;
            let bit = (byte >> self.bit) & 1;
            v |= (bit as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(v)
    }

    fn align_byte(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() {
            return Err("unexpected end of compressed data".into());
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

// ---------------------------------------------------------------------------
// Huffman tables (canonical, decoded with the counts/offsets method)
// ---------------------------------------------------------------------------

const MAX_BITS: usize = 15;

struct Huffman {
    /// count[len] = number of codes with that bit length.
    count: [u16; MAX_BITS + 1],
    /// Symbols sorted by (length, symbol) — canonical order.
    symbols: Vec<u16>,
}

impl Huffman {
    /// Build from per-symbol code lengths (0 = symbol unused).
    fn new(lengths: &[u8]) -> Result<Huffman, String> {
        let mut count = [0u16; MAX_BITS + 1];
        for &l in lengths {
            if l as usize > MAX_BITS {
                return Err("code length exceeds 15".into());
            }
            count[l as usize] += 1;
        }
        if count[0] as usize == lengths.len() {
            return Err("empty Huffman code".into());
        }
        // Reject over-subscribed codes (an incomplete code is tolerated: some
        // real encoders emit one for the distance tree of tiny streams).
        let mut left: i32 = 1;
        for &cnt in &count[1..] {
            left <<= 1;
            left -= cnt as i32;
            if left < 0 {
                return Err("over-subscribed Huffman code".into());
            }
        }
        let mut offsets = [0u16; MAX_BITS + 1];
        for l in 1..MAX_BITS {
            offsets[l + 1] = offsets[l] + count[l];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offsets[l as usize] as usize] = sym as u16;
                offsets[l as usize] += 1;
            }
        }
        Ok(Huffman { count, symbols })
    }

    fn decode(&self, r: &mut BitReader) -> Result<u16, String> {
        let mut code: u32 = 0;
        let mut first: u32 = 0;
        let mut index: u32 = 0;
        for len in 1..=MAX_BITS {
            code |= r.bits(1)?;
            let cnt = self.count[len] as u32;
            if code < first + cnt {
                return Ok(self.symbols[(index + code - first) as usize]);
            }
            index += cnt;
            first = (first + cnt) << 1;
            code <<= 1;
        }
        Err("invalid Huffman code".into())
    }
}

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lit = [0u8; 288];
    lit[0..144].fill(8);
    lit[144..256].fill(9);
    lit[256..280].fill(7);
    lit[280..288].fill(8);
    let dist = [5u8; 30];
    (
        Huffman::new(&lit).expect("fixed literal table is well-formed"),
        Huffman::new(&dist).expect("fixed distance table is well-formed"),
    )
}

/// Order in which code lengths for the code-length alphabet are stored.
const CLC_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn dynamic_tables(r: &mut BitReader) -> Result<(Huffman, Huffman), String> {
    let hlit = r.bits(5)? as usize + 257;
    let hdist = r.bits(5)? as usize + 1;
    let hclen = r.bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err("dynamic header declares too many codes".into());
    }
    let mut clc_lengths = [0u8; 19];
    for &idx in CLC_ORDER.iter().take(hclen) {
        clc_lengths[idx] = r.bits(3)? as u8;
    }
    let clc = Huffman::new(&clc_lengths)?;

    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < lengths.len() {
        let sym = clc.decode(r)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err("repeat code with no previous length".into());
                }
                let prev = lengths[i - 1];
                let n = r.bits(2)? as usize + 3;
                fill_run(&mut lengths, &mut i, prev, n)?;
            }
            17 => {
                let n = r.bits(3)? as usize + 3;
                fill_run(&mut lengths, &mut i, 0, n)?;
            }
            18 => {
                let n = r.bits(7)? as usize + 11;
                fill_run(&mut lengths, &mut i, 0, n)?;
            }
            _ => return Err("invalid code-length symbol".into()),
        }
    }
    let lit = Huffman::new(&lengths[..hlit])?;
    let dist = Huffman::new(&lengths[hlit..])?;
    Ok((lit, dist))
}

fn fill_run(lengths: &mut [u8], i: &mut usize, value: u8, n: usize) -> Result<(), String> {
    if *i + n > lengths.len() {
        return Err("code-length run overflows the table".into());
    }
    lengths[*i..*i + n].fill(value);
    *i += n;
    Ok(())
}

// ---------------------------------------------------------------------------
// Block bodies
// ---------------------------------------------------------------------------

fn stored_block(r: &mut BitReader, out: &mut Vec<u8>, max_out: usize) -> Result<(), String> {
    r.align_byte();
    let hdr = r.take(4)?;
    let len = u16::from_le_bytes([hdr[0], hdr[1]]) as usize;
    let nlen = u16::from_le_bytes([hdr[2], hdr[3]]);
    if nlen != !(len as u16) {
        return Err("stored block LEN/NLEN mismatch".into());
    }
    let data = r.take(len)?;
    let want = max_out.saturating_sub(out.len()).min(len);
    out.extend_from_slice(&data[..want]);
    Ok(())
}

/// Extra-bit tables for length codes 257..=285 (RFC 1951 §3.2.5).
const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn huffman_block(
    r: &mut BitReader,
    lit: &Huffman,
    dist: &Huffman,
    out: &mut Vec<u8>,
    max_out: usize,
) -> Result<(), String> {
    loop {
        let sym = lit.decode(r)?;
        match sym {
            0..=255 => {
                out.push(sym as u8);
                if out.len() >= max_out {
                    return Ok(());
                }
            }
            256 => return Ok(()), // end of block
            257..=285 => {
                let idx = (sym - 257) as usize;
                let len = LEN_BASE[idx] as usize + r.bits(LEN_EXTRA[idx])? as usize;
                let dsym = dist.decode(r)? as usize;
                if dsym >= 30 {
                    return Err("invalid distance symbol".into());
                }
                let d = DIST_BASE[dsym] as usize + r.bits(DIST_EXTRA[dsym])? as usize;
                if d > out.len() {
                    return Err("back-reference before start of output".into());
                }
                for _ in 0..len {
                    let byte = out[out.len() - d];
                    out.push(byte);
                    if out.len() >= max_out {
                        return Ok(());
                    }
                }
            }
            _ => return Err("invalid literal/length symbol".into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — the compressed vectors were produced once with a reference zlib
// (raw streams, wbits=-15) and are embedded verbatim, so the tests stay
// deterministic and fully offline.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// zlib level 6 output for b"peek" — a single fixed-Huffman block.
    const FIXED_PEEK: &[u8] = &[0x2b, 0x48, 0x4d, 0xcd, 0x06, 0x00];

    /// zlib level 6 output for b"tensor " * 40 — fixed Huffman with LZ77
    /// back-references (280 bytes collapse to 13).
    const REPEAT_TENSOR: &[u8] = &[
        0x2b, 0x49, 0xcd, 0x2b, 0xce, 0x2f, 0x52, 0x28, 0x19, 0xa5, 0x50, 0x29, 0x00,
    ];

    /// zlib level 9 output for `lcg_text(300)` — a dynamic-Huffman block
    /// (BTYPE=2 verified at generation time).
    const DYNAMIC_LCG300: &[u8] = &[
        0x0d, 0x8f, 0x4b, 0x0a, 0xc0, 0x20, 0x0c, 0x05, 0xaf, 0x92, 0x0b, 0x54, 0x8c, 0xbf, 0xd6,
        0xd3, 0x48, 0xc1, 0x07, 0x05, 0xc1, 0x06, 0x93, 0x8d, 0xb7, 0x6f, 0x77, 0x6f, 0x35, 0x33,
        0x8f, 0x46, 0x6b, 0x6f, 0xa0, 0xd4, 0xb2, 0x10, 0xcf, 0xa3, 0xab, 0x51, 0x3a, 0xe8, 0x66,
        0x05, 0x9c, 0x37, 0x49, 0xa9, 0xc2, 0x32, 0x23, 0xb6, 0xb0, 0xb4, 0xf4, 0x69, 0x40, 0x17,
        0x08, 0x03, 0xff, 0x12, 0x24, 0x1a, 0x0c, 0x9d, 0x82, 0x60, 0xcc, 0xfb, 0x5d, 0x95, 0xf0,
        0xb0, 0x14, 0x01, 0xf7, 0x08, 0xca, 0xea, 0x33, 0xb2, 0xf7, 0x07, 0x6e, 0x94, 0x5c, 0xfc,
        0x29, 0x2e, 0xbd, 0x05, 0xa5, 0x8a, 0xdd, 0xfa, 0x9c, 0x97, 0xfe, 0x2a, 0x74, 0x24, 0x90,
        0x61, 0xb8, 0x09, 0x8e, 0x1e, 0x87, 0x46, 0xda, 0xe8, 0xd5, 0xda, 0xde, 0x38, 0xef, 0x32,
        0x1d, 0xed, 0x75, 0xfc, 0xac, 0x62, 0xeb, 0x0a, 0x4c, 0x71, 0x59, 0x7d, 0x87, 0xdb, 0x0f,
        0xc4, 0x57, 0x54, 0x96, 0x2c, 0x6a, 0x31, 0x2d, 0x35, 0xaf, 0x74, 0x05, 0x19, 0x50, 0x79,
        0x0a, 0x2e, 0xa1, 0xed, 0x4c, 0xfe, 0x90, 0x10, 0x6d, 0x1f, 0x7f, 0xe4, 0x26, 0xca, 0xe0,
        0x74, 0xd7, 0xea, 0x33, 0xa9, 0x84, 0x89, 0x47, 0x11, 0x03, 0xdb, 0xff, 0xe6, 0xcd, 0x36,
        0x9f, 0xd3, 0x93, 0x3f, 0xb7, 0x6c, 0x1b, 0xb4, 0x3c, 0x48, 0xe6, 0x90, 0x16, 0x08, 0xd2,
        0x9c, 0x5c, 0x85, 0xfb, 0xed, 0xfb, 0x07,
    ];

    /// The plain text DYNAMIC_LCG300 decompresses to, regenerated here with
    /// the same 64-bit LCG used at vector-generation time.
    fn lcg_text(n: usize) -> Vec<u8> {
        const ALPHA: &[u8] = b"tensorpeek shape dtype 0123456789._-";
        let mut s: u64 = 0x243F_6A88_85A3_08D3;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ALPHA[((s >> 33) as usize) % ALPHA.len()]
            })
            .collect()
    }

    /// Wrap `data` in valid DEFLATE stored blocks (BTYPE=0).
    fn stored_stream(data: &[u8]) -> Vec<u8> {
        crate::builder::deflate_stored(data)
    }

    #[test]
    fn fixed_huffman_literals() {
        let r = inflate_prefix(FIXED_PEEK, 1 << 16).unwrap();
        assert_eq!(r.data, b"peek");
        assert!(r.complete);
    }

    #[test]
    fn fixed_huffman_with_back_references() {
        let r = inflate_prefix(REPEAT_TENSOR, 1 << 16).unwrap();
        assert_eq!(r.data, b"tensor ".repeat(40));
        assert!(r.complete);
    }

    #[test]
    fn dynamic_huffman_block() {
        let r = inflate_prefix(DYNAMIC_LCG300, 1 << 16).unwrap();
        assert_eq!(r.data, lcg_text(300));
        assert!(r.complete);
    }

    #[test]
    fn stored_blocks_round_trip() {
        let plain: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let r = inflate_prefix(&stored_stream(&plain), plain.len() + 10).unwrap();
        assert_eq!(r.data, plain);
        assert!(r.complete);
    }

    #[test]
    fn prefix_budget_stops_early_in_literals_and_mid_match_copy() {
        // Budget inside a literal run.
        let r = inflate_prefix(DYNAMIC_LCG300, 10).unwrap();
        assert_eq!(r.data, lcg_text(300)[..10]);
        assert!(!r.complete, "must report an incomplete decode");
        // Budget landing inside an LZ77 copy of "tensor " repeats.
        let r = inflate_prefix(REPEAT_TENSOR, 25).unwrap();
        assert_eq!(r.data, &b"tensor ".repeat(40)[..25]);
        assert!(!r.complete);
    }

    #[test]
    fn corrupt_streams_are_rejected() {
        // BTYPE=0, LEN=4 but NLEN is not its complement.
        let bad = [0x01, 0x04, 0x00, 0x00, 0x00, b'a', b'b', b'c', b'd'];
        assert!(inflate_prefix(&bad, 100).unwrap_err().contains("LEN/NLEN"));
        // First byte 0b00000111: BFINAL=1, BTYPE=3 (reserved).
        assert!(inflate_prefix(&[0x07], 100)
            .unwrap_err()
            .contains("reserved"));
        // Empty input is an error, not a panic.
        assert!(inflate_prefix(&[], 100).is_err());
    }

    #[test]
    fn distance_before_output_start_is_rejected() {
        // Fixed block whose first symbol is a match — nothing to copy from.
        // Symbol 257 (len 3) is code 0b0000001, then distance code 0.
        // Build it bit by bit: BFINAL=1, BTYPE=01, then the codes.
        let mut bits: Vec<u8> = vec![1, 1, 0]; // header (LSB first)
        bits.extend([0, 0, 0, 0, 0, 0, 1]); // literal code for 257 (MSB-first 0000001)
        bits.extend([0, 0, 0, 0, 0]); // distance code 0 (5 bits)
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            bytes[i / 8] |= b << (i % 8);
        }
        let err = inflate_prefix(&bytes, 100).unwrap_err();
        assert!(err.contains("back-reference"), "got: {err}");
    }

    #[test]
    fn truncated_stream_reports_end_of_data() {
        let err = inflate_prefix(&DYNAMIC_LCG300[..40], 1 << 16).unwrap_err();
        assert!(err.contains("end of compressed data"), "got: {err}");
    }
}
