# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added

- Unified JSON report schema across all four formats: `file`, `format`, `file_bytes`, `header_bytes`, `data_bytes`, `tensor_count`, `parameters`, per-format details, `metadata`, `tensors[]` (name, dtype, shape, numel, offset, bytes) and non-fatal `problems[]`.
- safetensors parser: 8-byte header length with the 100 MB format cap, JSON header via an in-repo parser (exact integers up to i128, surrogate pairs, depth cap), `__metadata__`, per-tensor dtype/shape/offset validation, byte-precise truncation and trailing-byte diagnostics.
- GGUF parser (v2/v3, little-endian): all 13 metadata value types, duplicate-key and non-UTF-8 notes, `general.alignment` validation with safe fallback, the ggml type table (F32..MXFP4 block geometry) for exact tensor sizes, expected-layout arithmetic with missing-byte counts, big-endian detection hint, and hostile-input guards (count plausibility, string/array length pre-checks, nesting depth caps). Large metadata arrays are summarized via `--array-limit` (default 16) with `--full-arrays` to lift it.
- npy parser: format versions 1.0-3.0, a purpose-built Python-literal parser for the header dict, simple and structured (record) descrs with computed itemsizes, byte order, `fortran_order`, scalar shapes, truncation diagnostics.
- npz parser: an in-repo ZIP reader (end-of-central-directory scan, ZIP64 records and extra fields, per-member local headers) that extracts every member's npy header — including from `np.savez_compressed` archives, using a bounded raw-DEFLATE decoder (stored, fixed and dynamic Huffman blocks) written on `std` alone.
- CLI: `inspect` (default command; pretty or `--compact` JSON, single object or array, `--no-tensors`, glob `--filter`), `ls` (aligned human table with parameter/byte humanization), `formats`, `--as` detection override, `--strict` gating, and CI-friendly exit codes (0 ok / 1 parse failure or strict problems / 2 usage or I/O error).
- Magic-based format detection with a safetensors plausibility heuristic and extension fallback.
- Spec-exact fixture writers for all four formats (`tensorpeek::builder`), a fixture-generator example, and a shape-gate CI example script.
- Test suite: 77 unit tests, 11 CLI integration tests, and `scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/tensorpeek/releases/tag/v0.1.0
