# Contributing to tensorpeek

Thanks for your interest in improving tensorpeek. Issues, discussions and pull requests are all welcome.

## Getting started

Prerequisites: Rust 1.75 or newer (stable toolchain).

```bash
git clone https://github.com/JaydenCJ/tensorpeek.git
cd tensorpeek
cargo build
cargo test
bash scripts/smoke.sh
```

`scripts/smoke.sh` writes real fixture files in all four formats with the in-repo builders and drives the CLI end to end — JSON output, tables, filtering, truncation detection, `--strict` gating and the documented exit codes. It finishes in well under a minute and must print `SMOKE OK`.

## Before you open a pull request

1. `cargo fmt` — formatting is enforced.
2. `cargo clippy --all-targets -- -D warnings` — clippy must be clean.
3. `cargo test` — 77 unit tests and 11 CLI integration tests must pass.
4. `bash scripts/smoke.sh` — the smoke test must print `SMOKE OK`.
5. Add tests for behavior changes. Every parser (`safetensors`, `gguf`, `npy`, `npz`, `inflate`, `json`) is a pure module that reads from byte slices or `Read` implementations; please keep it that way.

## Ground rules

- Keep dependencies at zero. tensorpeek parses binary formats with `std` alone — even DEFLATE decompression is in-repo; adding any dependency needs a very strong justification in the PR description.
- No network calls ever, no telemetry. tensorpeek reads local files and writes to stdout — nothing else.
- The JSON output schema is a compatibility surface: never rename or remove an existing key, only add new ones (users' `jq` pipelines reference them).
- New format support needs: magic-based detection in `sniff`, a parser producing the unified `Report`, a fixture writer in `builder`, and both positive and hostile-input tests.
- Code comments and doc comments are written in English.

## Reporting bugs

Please include the `tensorpeek --version` output, the full `tensorpeek inspect --compact` output for the affected file, and — since tensor files are huge — just the header region if you can share it (`head -c 1048576 model.gguf > header.bin`). The header is all tensorpeek ever reads, so it is enough to reproduce any report.

## Security

tensorpeek parses untrusted binary input, so parser bugs (out-of-memory on crafted counts, panics on malformed records, runaway decompression) are security-relevant. Please do not open a public issue for those; use GitHub's private vulnerability reporting on this repository instead.
