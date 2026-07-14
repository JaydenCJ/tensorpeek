# tensorpeek output schema and per-format notes

`tensorpeek inspect` emits one JSON object per file (an array when several
files are given). The schema is a compatibility surface: keys are only ever
added, never renamed or removed.

## Top-level keys

| Key | Type | Meaning |
|---|---|---|
| `file` | string | The path as given on the command line |
| `format` | string | `safetensors`, `gguf`, `npy` or `npz` |
| `file_bytes` | int | Size of the file on disk |
| `header_bytes` | int | Bytes occupied by header/index structures |
| `data_bytes` | int | Bytes of tensor data the header promises¹ |
| `tensor_count` | int | Number of tensors (unaffected by `--filter`) |
| `parameters` | int | Total element count across all tensors |
| `<format>` | object | Format-specific details (see below) |
| `metadata` | object | File-level metadata, omitted when empty |
| `tensors` | array | Omitted with `--no-tensors`; narrowed by `--filter` |
| `problems` | array | Non-fatal irregularities; omitted when empty |
| `error` | string | Only in batch output, for a file that failed to parse |

¹ For npz, `data_bytes` is the sum of *compressed* member payloads (what is
physically in the archive); per-member uncompressed sizes are on the tensors.

Each entry of `tensors` has `name`, `dtype`, `shape`, `numel`, `offset`
(within the data region it lives in) and `bytes`, plus format extras.

## Problems vs. errors

A file that cannot be interpreted at all (bad magic, malformed header,
implausible counts) is an **error**: stderr message, exit code 1. A file
whose header parses but does not add up — truncated data section, trailing
bytes, dtype/offset size mismatch, unknown dtype — gets **`problems`**
entries and still exits 0, because an inspector should describe what it
sees. `--strict` turns any problem into exit code 1 for CI gates.

## safetensors

- Detection: no magic; an 8-byte little-endian header length that fits the
  file, followed by `{`, is treated as safetensors (extension as fallback).
- `header_bytes` = 8 + JSON header length. The 100 MB header cap of the
  reference implementation is enforced.
- Details object: `{"header_json_bytes": N}`.
- Dtypes are canonicalized to lowercase (`F32` → `f32`); unknown dtypes are
  kept verbatim and flagged in `problems`.
- Expected byte spans are recomputed from dtype × shape and compared with
  `data_offsets`; mismatches, truncation and trailing bytes are `problems`.

## GGUF

- Versions 2 and 3, little-endian. Big-endian files are rejected with a
  specific hint; v1 is rejected as obsolete.
- `header_bytes` = the aligned data-section start (header plus padding);
  alignment comes from `general.alignment` (validated; falls back to 32).
- Details object: `version`, `alignment`, `data_start`, `architecture`.
- Tensor sizes are computed from the ggml type table (block length × block
  bytes), so a truncated data section is reported with the exact number of
  missing bytes without reading any tensor data.
- **Shape order caveat:** shapes are reported exactly as stored, i.e. in
  ggml element order with `ne0` (fastest-varying) first. A tensor that a
  Python framework saves as `[vocab, hidden]` appears as `[hidden, vocab]`
  in GGUF. tensorpeek does not silently reorder either representation.
- Metadata arrays longer than `--array-limit` (default 16) are summarized
  as `{"$array": {"elem": ..., "len": N, "first": [...]}}` so tokenizer
  vocabularies do not drown the report; `--full-arrays` lifts the limit.

## npy

- Format versions 1.0–3.0. The header dict is parsed by a purpose-built
  Python-literal parser (strings, booleans, ints, tuples, lists).
- The single array is unnamed: `tensors[0].name` is `""` (rendered `-` by
  `ls`).
- Details object: `version`, raw `descr`, `byte_order`
  (`little`/`big`/`native`/`none`), `fortran_order`, `itemsize`.
- Structured (record) dtypes report `dtype: "struct"` with the computed
  record size, including subarray shapes and nested field lists.

## npz

- A ZIP archive of `.npy` members. tensorpeek reads the end-of-central-
  directory record (ZIP64-aware, including 0xFFFFFFFF sentinel resolution
  via extra fields), walks the central directory, and parses each member's
  npy header.
- Members written by `np.savez_compressed` are handled by a bounded,
  in-repo raw-DEFLATE decoder that stops as soon as the npy header is
  available — a multi-gigabyte member costs a few KB of decompression.
- Tensor names are the member names minus `.npy`; extras per tensor:
  `member`, `compression` (`stored`/`deflate`), `compressed_bytes`.
- Details object: `members`, `zip64`, `stored`, `deflated`, `other_files`
  (non-`.npy` members, which np.savez never writes but hand-made archives
  may contain).
- Encrypted members and exotic compression methods are reported as
  `problems`, not crashes; the npy header's promised size is cross-checked
  against the archive's recorded uncompressed size.

## Reading strategy

Every parser reads only what it needs: the safetensors JSON header, the
GGUF metadata + tensor-info section, the first ≤64 KiB of an npy file, and
for npz the central directory plus a bounded prefix of each member. Tensor
data is never loaded, so inspecting a 40 GB checkpoint takes milliseconds.
