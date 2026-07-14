//! NumPy `.npz` parser. An npz file is a ZIP archive of `.npy` members, so
//! this module is a purpose-built ZIP reader: it locates the end-of-central-
//! directory record (ZIP64-aware), walks the central directory, and pulls
//! just enough of each member — decompressing a bounded prefix with the
//! in-repo DEFLATE decoder when the archive was written by
//! `np.savez_compressed` — to parse every member's npy header.

use std::io::{Read, Seek, SeekFrom};

use crate::inflate::inflate_prefix;
use crate::json::Json;
use crate::npy;
use crate::report::{PeekError, Report, Tensor};

const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4b50;
const CDIR_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

/// A ZIP comment can be up to 65535 bytes, so the EOCD may sit that far from
/// the end of the file.
const EOCD_SCAN: u64 = 22 + 65535;
/// Central directories beyond this are not real-world npz files.
const MAX_CDIR: u64 = 64 * 1024 * 1024;
/// Compressed bytes budget per member when hunting for the npy header.
const MAX_COMP_READ: u64 = 4 * 1024 * 1024;

struct Member {
    name: String,
    method: u16,
    flags: u16,
    comp_size: u64,
    uncomp_size: u64,
    local_offset: u64,
}

pub fn parse<R: Read + Seek>(r: &mut R, file_len: u64, path: &str) -> Result<Report, PeekError> {
    let mut report = Report::new(path, "npz", file_len);

    // --- 1. end of central directory -----------------------------------------
    let scan_len = EOCD_SCAN.min(file_len);
    let mut tail = vec![0u8; scan_len as usize];
    r.seek(SeekFrom::Start(file_len - scan_len))?;
    r.read_exact(&mut tail)?;
    let eocd_pos = find_eocd(&tail).ok_or(PeekError(
        "not a zip archive (no end-of-central-directory record)".into(),
    ))?;
    let eocd = &tail[eocd_pos..];
    let mut entries = u16le(eocd, 10) as u64;
    let mut cd_size = u32le(eocd, 12) as u64;
    let mut cd_offset = u32le(eocd, 16) as u64;
    let eocd_file_pos = file_len - scan_len + eocd_pos as u64;

    // --- 2. ZIP64 (numpy writes it for archives or members over 4 GiB) -------
    let needs_zip64 = entries == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF;
    let mut zip64 = false;
    if needs_zip64 {
        if eocd_file_pos < 20 {
            return Err(PeekError(
                "ZIP64 markers present but no ZIP64 locator fits".into(),
            ));
        }
        let mut loc = [0u8; 20];
        r.seek(SeekFrom::Start(eocd_file_pos - 20))?;
        r.read_exact(&mut loc)?;
        if u32le(&loc, 0) != EOCD64_LOCATOR_SIG {
            return Err(PeekError(
                "ZIP64 locator record is missing or corrupt".into(),
            ));
        }
        let eocd64_pos = u64le(&loc, 8);
        if eocd64_pos.saturating_add(56) > file_len {
            return Err(PeekError(
                "ZIP64 end-of-central-directory offset is out of bounds".into(),
            ));
        }
        let mut e64 = [0u8; 56];
        r.seek(SeekFrom::Start(eocd64_pos))?;
        r.read_exact(&mut e64)?;
        if u32le(&e64, 0) != EOCD64_SIG {
            return Err(PeekError(
                "ZIP64 end-of-central-directory record is corrupt".into(),
            ));
        }
        entries = u64le(&e64, 32);
        cd_size = u64le(&e64, 40);
        cd_offset = u64le(&e64, 48);
        zip64 = true;
    }

    // --- 3. central directory --------------------------------------------------
    if cd_size > MAX_CDIR || cd_offset.saturating_add(cd_size) > file_len {
        return Err(PeekError(format!(
            "central directory ({cd_size} bytes at offset {cd_offset}) does not fit the file"
        )));
    }
    if entries > cd_size / 46 {
        return Err(PeekError(format!(
            "central directory declares {entries} entries, which cannot fit in {cd_size} bytes"
        )));
    }
    let mut cd = vec![0u8; cd_size as usize];
    r.seek(SeekFrom::Start(cd_offset))?;
    r.read_exact(&mut cd)?;

    let mut members = Vec::with_capacity(entries as usize);
    let mut pos = 0usize;
    for i in 0..entries {
        if pos + 46 > cd.len() || u32le(&cd, pos) != CDIR_SIG {
            return Err(PeekError(format!(
                "central directory entry #{i} is corrupt"
            )));
        }
        let flags = u16le(&cd, pos + 8);
        let method = u16le(&cd, pos + 10);
        let mut comp_size = u32le(&cd, pos + 20) as u64;
        let mut uncomp_size = u32le(&cd, pos + 24) as u64;
        let name_len = u16le(&cd, pos + 28) as usize;
        let extra_len = u16le(&cd, pos + 30) as usize;
        let comment_len = u16le(&cd, pos + 32) as usize;
        let mut local_offset = u32le(&cd, pos + 42) as u64;
        let end = pos + 46 + name_len + extra_len + comment_len;
        if end > cd.len() {
            return Err(PeekError(format!(
                "central directory entry #{i} runs past the directory"
            )));
        }
        let name = String::from_utf8_lossy(&cd[pos + 46..pos + 46 + name_len]).into_owned();

        // ZIP64 extra field 0x0001: 64-bit values, present only for the
        // fields that hold the 0xFFFFFFFF sentinel, in a fixed order.
        let mut extra = &cd[pos + 46 + name_len..pos + 46 + name_len + extra_len];
        while extra.len() >= 4 {
            let id = u16le(extra, 0);
            let sz = u16le(extra, 2) as usize;
            if extra.len() < 4 + sz {
                break;
            }
            if id == 0x0001 {
                let mut f = &extra[4..4 + sz];
                for target in [&mut uncomp_size, &mut comp_size, &mut local_offset] {
                    if *target == 0xFFFF_FFFF && f.len() >= 8 {
                        *target = u64le(f, 0);
                        f = &f[8..];
                    }
                }
            }
            extra = &extra[4 + sz..];
        }
        members.push(Member {
            name,
            method,
            flags,
            comp_size,
            uncomp_size,
            local_offset,
        });
        pos = end;
    }

    // --- 4. per-member npy headers ----------------------------------------------
    let mut stored = 0u64;
    let mut deflated = 0u64;
    let mut other_files: Vec<Json> = Vec::new();
    let mut data_bytes = 0u64;
    for m in &members {
        data_bytes = data_bytes.saturating_add(m.comp_size);
        match m.method {
            0 => stored += 1,
            8 => deflated += 1,
            other => {
                report.problems.push(format!(
                    "member '{}' uses unsupported compression method {other}",
                    m.name
                ));
                continue;
            }
        }
        if !m.name.ends_with(".npy") {
            other_files.push(Json::Str(m.name.clone()));
            continue;
        }
        if m.flags & 0x1 != 0 {
            report
                .problems
                .push(format!("member '{}' is encrypted", m.name));
            continue;
        }
        match member_header(r, file_len, m) {
            Ok(h) => {
                for p in &h.problems {
                    report.problems.push(format!("member '{}': {p}", m.name));
                }
                let expected_total = h.data_bytes().map(|d| d + h.header_total);
                if let Some(total) = expected_total {
                    if total != m.uncomp_size {
                        report.problems.push(format!(
                            "member '{}': npy header promises {total} bytes but the archive stores {} — corrupt or truncated member",
                            m.name, m.uncomp_size
                        ));
                    }
                }
                let name = m.name.strip_suffix(".npy").unwrap_or(&m.name);
                let bytes = h
                    .data_bytes()
                    .unwrap_or(m.uncomp_size.saturating_sub(h.header_total));
                let mut t = Tensor::new(name, &h.dtype, h.shape.clone(), h.header_total, bytes);
                t.extra.push(("member".into(), Json::Str(m.name.clone())));
                t.extra.push((
                    "compression".into(),
                    Json::Str(if m.method == 8 { "deflate" } else { "stored" }.into()),
                ));
                t.extra
                    .push(("compressed_bytes".into(), Json::from(m.comp_size)));
                report.tensors.push(t);
            }
            Err(e) => report.problems.push(format!("member '{}': {e}", m.name)),
        }
    }

    report.header_bytes = file_len.saturating_sub(cd_offset);
    report.data_bytes = data_bytes;
    report.details = Json::Obj(vec![
        ("members".into(), Json::Int(members.len() as i128)),
        ("zip64".into(), Json::Bool(zip64)),
        ("stored".into(), Json::Int(stored as i128)),
        ("deflated".into(), Json::Int(deflated as i128)),
        ("other_files".into(), Json::Arr(other_files)),
    ]);
    Ok(report)
}

/// Read just enough of one member to parse its npy header.
fn member_header<R: Read + Seek>(
    r: &mut R,
    file_len: u64,
    m: &Member,
) -> Result<npy::NpyHeader, PeekError> {
    if m.local_offset.saturating_add(30) > file_len {
        return Err(PeekError("local header offset is out of bounds".into()));
    }
    let mut lh = [0u8; 30];
    r.seek(SeekFrom::Start(m.local_offset))?;
    r.read_exact(&mut lh)?;
    if u32le(&lh, 0) != LOCAL_SIG {
        return Err(PeekError("local file header signature mismatch".into()));
    }
    let name_len = u16le(&lh, 26) as u64;
    let extra_len = u16le(&lh, 28) as u64;
    let data_off = m.local_offset + 30 + name_len + extra_len;
    let want = (npy::MAX_PREFIX as u64).min(m.uncomp_size.max(12));

    let prefix = match m.method {
        0 => {
            let n = want.min(m.comp_size).min(file_len.saturating_sub(data_off));
            let mut buf = vec![0u8; n as usize];
            r.seek(SeekFrom::Start(data_off))?;
            r.read_exact(&mut buf)?;
            buf
        }
        8 => {
            let n = m
                .comp_size
                .min(MAX_COMP_READ)
                .min(file_len.saturating_sub(data_off));
            let mut comp = vec![0u8; n as usize];
            r.seek(SeekFrom::Start(data_off))?;
            r.read_exact(&mut comp)?;
            let inflated = inflate_prefix(&comp, want as usize)
                .map_err(|e| PeekError(format!("deflate stream is corrupt: {e}")))?;
            inflated.data
        }
        other => return Err(PeekError(format!("unsupported compression method {other}"))),
    };
    npy::parse_header(&prefix)
}

fn find_eocd(tail: &[u8]) -> Option<usize> {
    if tail.len() < 22 {
        return None;
    }
    // Scan backwards: the EOCD closest to the end is the real one, and its
    // comment length must reach exactly to EOF.
    for pos in (0..=tail.len() - 22).rev() {
        if u32le(tail, pos) == EOCD_SIG {
            let comment_len = u16le(tail, pos + 20) as usize;
            if pos + 22 + comment_len == tail.len() {
                return Some(pos);
            }
        }
    }
    None
}

fn u16le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn u64le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{self, ZipWriter};
    use std::io::Cursor;

    fn parse_bytes(b: &[u8]) -> Result<Report, PeekError> {
        parse(&mut Cursor::new(b), b.len() as u64, "test.npz")
    }

    /// Raw-deflate stream produced once by a reference zlib for an npy
    /// payload `<f4`, shape (2, 3), 24 zero data bytes (152 bytes total).
    const DEFLATED_NPY_F32_2X3: &[u8] = &[
        0x9b, 0xec, 0x17, 0xea, 0x1b, 0x10, 0xc9, 0xc8, 0x50, 0xc6, 0x50, 0xad, 0x9e, 0x92, 0x5a,
        0x9c, 0x5c, 0xa4, 0x6e, 0xa5, 0xa0, 0x6e, 0x93, 0x66, 0xa2, 0xae, 0xa3, 0xa0, 0x9e, 0x96,
        0x5f, 0x54, 0x52, 0x94, 0x98, 0x17, 0x9f, 0x5f, 0x94, 0x92, 0x0a, 0x12, 0x77, 0x4b, 0xcc,
        0x29, 0x4e, 0x05, 0x8a, 0x17, 0x67, 0x24, 0x16, 0xa4, 0x02, 0xf9, 0x1a, 0x46, 0x3a, 0x0a,
        0xc6, 0x9a, 0x3a, 0x0a, 0xb5, 0x0a, 0x64, 0x03, 0x2e, 0x06, 0x1c, 0x00, 0x00,
    ];

    #[test]
    fn parses_a_stored_archive_with_two_arrays() {
        let bytes = builder::npz(&[
            ("weights.npy", builder::npy("<f4", false, &[4, 4]), false),
            ("bias.npy", builder::npy("<f4", false, &[4]), false),
        ]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors.len(), 2);
        assert_eq!(
            r.tensors[0].name, "weights",
            "member names lose the .npy suffix"
        );
        assert_eq!(r.tensors[0].shape, [4, 4]);
        assert_eq!(r.tensors[1].bytes, 16);
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn parses_a_deflated_member_via_the_in_repo_inflater() {
        let bytes = builder::npz(&[("arr_0.npy", builder::npy("<i8", false, &[100]), true)]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].dtype, "i64");
        assert_eq!(r.tensors[0].shape, [100]);
        let comp = r.tensors[0]
            .extra
            .iter()
            .find(|(k, _)| k == "compression")
            .unwrap();
        assert_eq!(comp.1.as_str(), Some("deflate"));
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn parses_a_member_deflated_by_a_reference_zlib() {
        // The compressed stream uses real Huffman coding, not our
        // stored-block writer — this proves interop with numpy's output.
        let mut z = ZipWriter::new();
        z.add_precompressed("arr_0.npy", DEFLATED_NPY_F32_2X3, 152, 0x1826_b378);
        let bytes = z.finish(false);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.tensors[0].dtype, "f32");
        assert_eq!(r.tensors[0].shape, [2, 3]);
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn details_count_members_and_compression_kinds() {
        let bytes = builder::npz(&[
            ("a.npy", builder::npy("<f4", false, &[2]), false),
            ("b.npy", builder::npy("<f4", false, &[2]), true),
            ("readme.txt", b"hello".to_vec(), false),
        ]);
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.details.get("members").and_then(Json::as_int), Some(3));
        assert_eq!(r.details.get("stored").and_then(Json::as_int), Some(2));
        assert_eq!(r.details.get("deflated").and_then(Json::as_int), Some(1));
        let others = r.details.get("other_files").and_then(Json::as_arr).unwrap();
        assert_eq!(others[0].as_str(), Some("readme.txt"));
        assert_eq!(r.tensors.len(), 2, "the .txt member is not a tensor");
    }

    #[test]
    fn zip64_archives_are_read() {
        let mut z = ZipWriter::new();
        z.add_stored("big.npy", &builder::npy("<u1", false, &[64]));
        let bytes = z.finish(true); // force ZIP64 records
        let r = parse_bytes(&bytes).unwrap();
        assert_eq!(r.details.get("zip64"), Some(&Json::Bool(true)));
        assert_eq!(r.tensors[0].shape, [64]);
        assert!(r.problems.is_empty(), "problems: {:?}", r.problems);
    }

    #[test]
    fn member_size_mismatch_is_flagged() {
        // The npy header inside promises 100 i64s but the stored member is
        // physically shorter: np.savez never writes this; a corrupt file does.
        let mut member = builder::npy("<i8", false, &[100]);
        member.truncate(member.len() - 40);
        let bytes = builder::npz(&[("weights.npy", member, false)]);
        let r = parse_bytes(&bytes).unwrap();
        assert!(
            r.problems
                .iter()
                .any(|p| p.contains("promises") && p.contains("weights.npy")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn encrypted_and_exotically_compressed_members_are_problems() {
        let mut z = ZipWriter::new();
        z.add_stored("secret.npy", &builder::npy("<f4", false, &[2]));
        z.entries[0].flags = 0x1;
        let bytes = z.finish(false);
        let r = parse_bytes(&bytes).unwrap();
        assert!(r.tensors.is_empty());
        assert!(r.problems.iter().any(|p| p.contains("encrypted")));

        let mut z = ZipWriter::new();
        z.add_stored("w.npy", &builder::npy("<f4", false, &[2]));
        z.entries[0].method = 14; // LZMA
        let bytes = z.finish(false);
        let r = parse_bytes(&bytes).unwrap();
        assert!(
            r.problems.iter().any(|p| p.contains("method 14")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn a_corrupt_deflate_stream_is_a_member_problem() {
        let mut z = ZipWriter::new();
        z.add_precompressed("bad.npy", &[0x07, 0xff, 0xff], 152, 0);
        let bytes = z.finish(false);
        let r = parse_bytes(&bytes).unwrap();
        assert!(
            r.problems
                .iter()
                .any(|p| p.contains("deflate stream is corrupt")),
            "problems: {:?}",
            r.problems
        );
    }

    #[test]
    fn hostile_archives_are_fatal() {
        // No EOCD record at all.
        let err = parse_bytes(&[0u8; 100]).unwrap_err();
        assert!(err.0.contains("no end-of-central-directory"));
        // An entry count that cannot fit in the central directory.
        let bytes = builder::npz(&[("a.npy", builder::npy("<f4", false, &[2]), false)]);
        let mut corrupt = bytes.clone();
        let eocd = corrupt.len() - 22;
        corrupt[eocd + 10] = 0xEE; // entries: 0xEEEE, way beyond cd_size/46
        corrupt[eocd + 11] = 0xEE;
        let err = parse_bytes(&corrupt).unwrap_err();
        assert!(err.0.contains("cannot fit"), "got: {err}");
    }
}
