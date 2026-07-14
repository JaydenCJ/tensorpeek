//! End-to-end tests against the compiled binary: real fixture files are
//! written with `tensorpeek::builder`, the CLI runs as a child process, and
//! stdout/exit codes are asserted. Everything happens in a per-test temp
//! directory; nothing touches the network.

use std::path::PathBuf;
use std::process::{Command, Output};

use tensorpeek::builder::{self, GgufBuilder};
use tensorpeek::json::Json;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_tensorpeek")
}

/// A unique scratch directory under the target dir; cleaned up on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("tensorpeek-test-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> String {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("binary runs")
}

fn stdout_json(out: &Output) -> Json {
    Json::parse(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "stdout is not JSON ({e}): {}",
            String::from_utf8_lossy(&out.stdout)
        )
    })
}

#[test]
fn version_and_help() {
    let out = run(&["--version"]);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        format!("tensorpeek {}", env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(out.status.code(), Some(0));

    let out = run(&["--help"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("COMMANDS:"), "help must list commands");
    assert!(
        text.contains("EXIT CODES:"),
        "help must document exit codes"
    );
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn inspect_safetensors_reports_the_full_schema() {
    let s = Scratch::new("st");
    let file = s.write(
        "model.safetensors",
        &builder::safetensors(
            &[
                ("embed.weight", "F32", &[32, 8]),
                ("fc1.weight", "F16", &[8, 16]),
            ],
            &[("format", "pt")],
        ),
    );
    let out = run(&["inspect", &file]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j = stdout_json(&out);
    assert_eq!(j.get("format").and_then(Json::as_str), Some("safetensors"));
    assert_eq!(j.get("tensor_count").and_then(Json::as_int), Some(2));
    assert_eq!(
        j.get("parameters").and_then(Json::as_int),
        Some(32 * 8 + 8 * 16)
    );
    assert_eq!(
        j.get("metadata")
            .and_then(|m| m.get("format"))
            .and_then(Json::as_str),
        Some("pt")
    );
    let t0 = &j.get("tensors").and_then(Json::as_arr).unwrap()[0];
    assert_eq!(t0.get("name").and_then(Json::as_str), Some("embed.weight"));
    assert_eq!(t0.get("dtype").and_then(Json::as_str), Some("f32"));
}

#[test]
fn gguf_metadata_filter_and_array_limit() {
    let s = Scratch::new("gguf");
    let file = s.write("m.gguf", &GgufBuilder::demo().build());
    let out = run(&["inspect", "--filter", "blk.*", "--array-limit", "4", &file]);
    assert_eq!(out.status.code(), Some(0));
    let j = stdout_json(&out);
    assert_eq!(
        j.get("gguf")
            .and_then(|d| d.get("architecture"))
            .and_then(Json::as_str),
        Some("llama")
    );
    let tensors = j.get("tensors").and_then(Json::as_arr).unwrap();
    assert_eq!(tensors.len(), 2, "filter must keep only blk.* tensors");
    assert_eq!(
        j.get("tensor_count").and_then(Json::as_int),
        Some(4),
        "counts stay unfiltered"
    );
    let tokens = j
        .get("metadata")
        .and_then(|m| m.get("tokenizer.ggml.tokens"))
        .unwrap();
    assert_eq!(
        tokens
            .get("$array")
            .and_then(|a| a.get("len"))
            .and_then(Json::as_int),
        Some(12),
        "long arrays are summarized at --array-limit 4"
    );
}

#[test]
fn implicit_inspect_with_compact_output_is_single_line_json() {
    let s = Scratch::new("compact");
    let file = s.write("a.npy", &builder::npy("<i8", true, &[5]));
    // No `inspect` keyword: the first argument is a file, so inspect is implied.
    let out = run(&["--compact", &file]);
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(text.trim().lines().count(), 1);
    let j = stdout_json(&out);
    assert_eq!(j.get("format").and_then(Json::as_str), Some("npy"));
    assert_eq!(
        j.get("npy").and_then(|d| d.get("fortran_order")),
        Some(&Json::Bool(true))
    );
}

#[test]
fn npz_members_become_named_tensors() {
    let s = Scratch::new("npz");
    let file = s.write(
        "w.npz",
        &builder::npz(&[
            ("weights.npy", builder::npy("<f4", false, &[4, 4]), true),
            ("bias.npy", builder::npy("<f4", false, &[4]), false),
        ]),
    );
    let out = run(&["inspect", &file]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j = stdout_json(&out);
    assert_eq!(
        j.get("npz")
            .and_then(|d| d.get("members"))
            .and_then(Json::as_int),
        Some(2)
    );
    let tensors = j.get("tensors").and_then(Json::as_arr).unwrap();
    assert_eq!(
        tensors[0].get("name").and_then(Json::as_str),
        Some("weights")
    );
    assert_eq!(
        tensors[0].get("compression").and_then(Json::as_str),
        Some("deflate")
    );
}

#[test]
fn multi_file_inspect_emits_an_array_with_error_objects() {
    let s = Scratch::new("multi");
    let good = s.write("good.npy", &builder::npy("<f4", false, &[2]));
    let bad = s.write("bad.gguf", b"GGUFxxxxthis is not a real header");
    let out = run(&["inspect", &good, &bad]);
    assert_eq!(out.status.code(), Some(1), "one failed parse must exit 1");
    let j = stdout_json(&out);
    let arr = j.as_arr().expect("multiple files produce a JSON array");
    assert_eq!(arr.len(), 2);
    assert!(arr[0].get("error").is_none());
    assert!(
        arr[1].get("error").is_some(),
        "the broken file gets an error object"
    );
}

#[test]
fn ls_renders_an_aligned_table_with_a_summary() {
    let s = Scratch::new("ls");
    let file = s.write(
        "m.safetensors",
        &builder::safetensors(&[("fc.weight", "F32", &[64, 64])], &[]),
    );
    let out = run(&["ls", &file]);
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("1 tensor · 4.1 K params"),
        "summary line: {text}"
    );
    assert!(
        text.contains("NAME") && text.contains("DTYPE"),
        "table header: {text}"
    );
    assert!(text.contains("64×64"), "shape column: {text}");
}

#[test]
fn strict_turns_truncation_problems_into_exit_1() {
    let s = Scratch::new("strict");
    let mut bytes = builder::safetensors(&[("w", "F32", &[100])], &[]);
    bytes.truncate(bytes.len() - 40);
    let file = s.write("cut.safetensors", &bytes);

    let out = run(&["inspect", &file]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "problems alone do not fail without --strict"
    );
    let j = stdout_json(&out);
    assert!(j.get("problems").is_some());

    let out = run(&["inspect", "--strict", &file]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("40 missing"));
}

#[test]
fn forced_format_overrides_detection() {
    let s = Scratch::new("forced");
    // A GGUF header inside a file with a lying extension, read via --as.
    let file = s.write("model.bin", &GgufBuilder::demo().build());
    let out = run(&["inspect", "--as", "gguf", "--no-tensors", &file]);
    assert_eq!(out.status.code(), Some(0));
    let j = stdout_json(&out);
    assert_eq!(j.get("format").and_then(Json::as_str), Some("gguf"));
    assert!(j.get("tensors").is_none(), "--no-tensors drops the list");
    assert_eq!(j.get("tensor_count").and_then(Json::as_int), Some(4));
}

#[test]
fn documented_exit_codes_for_usage_io_and_content_errors() {
    let s = Scratch::new("errs");
    // Unreadable input and usage mistakes exit 2.
    let out = run(&["inspect", &s.path("does-not-exist.gguf")]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot open"));
    let out = run(&["--bogus-flag", "x"]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(&["inspect"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no input files"));
    // Content that no parser recognizes exits 1 and hints at --as.
    let file = s.write(
        "mystery.bin",
        &[0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0],
    );
    let out = run(&["inspect", &file]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--as"));
}

#[test]
fn formats_command_documents_all_four() {
    let out = run(&["formats"]);
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8_lossy(&out.stdout);
    for f in ["safetensors", "gguf", "npy", "npz"] {
        assert!(text.contains(f), "missing {f}");
    }
}
