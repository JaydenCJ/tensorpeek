//! Command-line front end: argument parsing, format dispatch, output and
//! the documented exit codes (0 = ok, 1 = parse failure or `--strict`
//! problems, 2 = usage error or unreadable file).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::json::Json;
use crate::render;
use crate::report::Report;
use crate::sniff::{self, Format};
use crate::{gguf, npy, npz, safetensors, VERSION};

const DEFAULT_ARRAY_LIMIT: usize = 16;

#[derive(Debug, PartialEq)]
enum Cmd {
    Inspect,
    Ls,
    Formats,
    Help,
    Version,
}

#[derive(Debug)]
struct Opts {
    command: Cmd,
    files: Vec<String>,
    compact: bool,
    no_tensors: bool,
    filter: Option<String>,
    array_limit: usize,
    strict: bool,
    forced: Option<Format>,
}

pub fn run(args: &[String]) -> i32 {
    let opts = match parse_args(args) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("tensorpeek: {msg}");
            eprintln!("try 'tensorpeek --help'");
            return 2;
        }
    };
    match opts.command {
        Cmd::Help => {
            out_line(&help_text());
            0
        }
        Cmd::Version => {
            out_line(&format!("tensorpeek {VERSION}"));
            0
        }
        Cmd::Formats => {
            out_line(formats_text());
            0
        }
        Cmd::Inspect => run_inspect(&opts),
        Cmd::Ls => run_ls(&opts),
    }
}

/// Write to stdout, tolerating a closed pipe: `tensorpeek … | grep -q` and
/// `| head` close the read end early, and that must not be a panic.
fn out_line(text: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "{text}");
}

fn out_raw(text: &str) {
    use std::io::Write;
    let _ = write!(std::io::stdout(), "{text}");
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        command: Cmd::Inspect,
        files: Vec::new(),
        compact: false,
        no_tensors: false,
        filter: None,
        array_limit: DEFAULT_ARRAY_LIMIT,
        strict: false,
        forced: None,
    };
    if args.is_empty() {
        opts.command = Cmd::Help;
        return Ok(opts);
    }
    let mut rest = args;
    match args[0].as_str() {
        "inspect" => rest = &args[1..],
        "ls" => {
            opts.command = Cmd::Ls;
            rest = &args[1..];
        }
        "formats" => {
            opts.command = Cmd::Formats;
            rest = &args[1..];
        }
        "help" => {
            opts.command = Cmd::Help;
            return Ok(opts);
        }
        // Anything else is a file (or an option) for the implicit `inspect`.
        _ => {}
    }

    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (arg.as_str(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            inline
                .clone()
                .or_else(|| it.next().cloned())
                .ok_or(format!("{name} needs a value"))
        };
        match flag {
            // Flags without a value must not swallow one silently:
            // `--strict=false` would otherwise still mean `--strict`.
            "--compact" | "--no-tensors" | "--strict" | "--full-arrays" | "--help"
            | "--version"
                if inline.is_some() =>
            {
                return Err(format!("{flag} does not take a value"))
            }
            "--compact" => opts.compact = true,
            "--no-tensors" => opts.no_tensors = true,
            "--strict" => opts.strict = true,
            "--full-arrays" => opts.array_limit = usize::MAX,
            "--filter" => opts.filter = Some(value("--filter")?),
            "--array-limit" => {
                let v = value("--array-limit")?;
                opts.array_limit = v.parse().map_err(|_| {
                    format!("--array-limit needs a non-negative integer, got '{v}'")
                })?;
            }
            "--as" => {
                let v = value("--as")?;
                opts.forced = Some(Format::from_name(&v).ok_or(format!(
                    "unknown format '{v}' (expected safetensors, gguf, npy or npz)"
                ))?);
            }
            "--help" | "-h" => {
                opts.command = Cmd::Help;
                return Ok(opts);
            }
            "--version" | "-V" => {
                opts.command = Cmd::Version;
                return Ok(opts);
            }
            f if f.starts_with('-') && f.len() > 1 => return Err(format!("unknown option '{f}'")),
            _ => opts.files.push(arg.clone()),
        }
    }

    if matches!(opts.command, Cmd::Inspect | Cmd::Ls) && opts.files.is_empty() {
        return Err("no input files given".into());
    }
    if opts.command == Cmd::Formats && !opts.files.is_empty() {
        return Err("'formats' takes no arguments".into());
    }
    Ok(opts)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn run_inspect(opts: &Opts) -> i32 {
    let mut worst = 0i32;
    let mut docs: Vec<Json> = Vec::with_capacity(opts.files.len());
    for path in &opts.files {
        match inspect_file(path, opts.forced, opts.array_limit) {
            Ok(report) => {
                if opts.strict && !report.problems.is_empty() {
                    for p in &report.problems {
                        eprintln!("tensorpeek: {path}: problem: {p}");
                    }
                    worst = worst.max(1);
                }
                docs.push(report.to_json(!opts.no_tensors, opts.filter.as_deref()));
            }
            Err((code, msg)) => {
                eprintln!("tensorpeek: {path}: {msg}");
                docs.push(Json::Obj(vec![
                    ("file".into(), Json::Str(path.clone())),
                    ("error".into(), Json::Str(msg)),
                ]));
                worst = worst.max(code);
            }
        }
    }
    let doc = if docs.len() == 1 {
        docs.into_iter().next().unwrap()
    } else {
        Json::Arr(docs)
    };
    out_line(&if opts.compact {
        doc.compact()
    } else {
        doc.pretty()
    });
    worst
}

fn run_ls(opts: &Opts) -> i32 {
    let mut worst = 0i32;
    for (i, path) in opts.files.iter().enumerate() {
        match inspect_file(path, opts.forced, opts.array_limit) {
            Ok(report) => {
                if i > 0 {
                    out_line("");
                }
                out_raw(&render::tensor_table(&report, opts.filter.as_deref()));
                if opts.strict && !report.problems.is_empty() {
                    worst = worst.max(1);
                }
            }
            Err((code, msg)) => {
                eprintln!("tensorpeek: {path}: {msg}");
                worst = worst.max(code);
            }
        }
    }
    worst
}

/// Open, detect and parse one file. The error carries the exit-code class:
/// 2 for unreadable input, 1 for content that cannot be parsed.
fn inspect_file(
    path: &str,
    forced: Option<Format>,
    array_limit: usize,
) -> Result<Report, (i32, String)> {
    let mut file = File::open(path).map_err(|e| (2, format!("cannot open: {e}")))?;
    let meta = file
        .metadata()
        .map_err(|e| (2, format!("cannot stat: {e}")))?;
    if meta.is_dir() {
        return Err((2, "is a directory".into()));
    }
    let file_len = meta.len();

    let mut prefix = [0u8; 16];
    let mut got = 0usize;
    while got < prefix.len() {
        match file.read(&mut prefix[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) => return Err((2, format!("cannot read: {e}"))),
        }
    }
    let format = forced.or_else(|| sniff::detect(&prefix[..got], file_len, path)).ok_or((
        1,
        "unrecognized format (not safetensors, GGUF, npy or npz; use --as to override detection)"
            .to_string(),
    ))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| (2, format!("cannot seek: {e}")))?;

    let result = match format {
        Format::Safetensors => safetensors::parse(&mut file, file_len, path),
        Format::Gguf => gguf::parse(&mut file, file_len, path, array_limit),
        Format::Npy => npy::parse(&mut file, file_len, path),
        Format::Npz => npz::parse(&mut file, file_len, path),
    };
    result.map_err(|e| (1, e.0))
}

// ---------------------------------------------------------------------------
// Static text
// ---------------------------------------------------------------------------

fn help_text() -> String {
    format!(
        "tensorpeek {VERSION} — safetensors / GGUF / npy / npz header inspector

USAGE:
    tensorpeek [COMMAND] [OPTIONS] <FILE>...

COMMANDS:
    inspect     Report file headers as JSON (the default command)
    ls          Human-readable tensor table
    formats     List supported formats and how they are detected
    help        Show this message

OPTIONS:
    --compact          Single-line JSON output
    --no-tensors       Omit the tensor list (counts are kept)
    --filter <GLOB>    Only list tensors whose name matches, e.g. 'blk.*.weight'
                       (comma separates alternatives; * and ? wildcards)
    --array-limit <N>  Summarize GGUF metadata arrays longer than N (default {DEFAULT_ARRAY_LIMIT})
    --full-arrays      Keep every element of GGUF metadata arrays
    --strict           Exit 1 when a file has problems (truncation, size mismatches)
    --as <FORMAT>      Skip detection: safetensors | gguf | npy | npz
    -h, --help         Show this message
    -V, --version      Show the version

EXIT CODES:
    0   every file parsed (and, with --strict, no problems were found)
    1   a file could not be parsed, or --strict found problems
    2   usage error or unreadable input"
    )
}

fn formats_text() -> &'static str {
    "FORMAT       DETECTED BY                                WHAT IS READ
safetensors  8-byte LE header length + '{' (no magic)   JSON header: dtype, shape, data_offsets, __metadata__
gguf         magic 'GGUF', versions 2-3 little-endian   metadata KVs, tensor infos, alignment, expected layout
npy          magic '\\x93NUMPY', versions 1.0-3.0        header dict: descr, fortran_order, shape
npz          ZIP signature 'PK' (ZIP64 supported)       central directory + every member's npy header

Only header regions are read; tensor data is never loaded.
Detection order: magic bytes, then the safetensors heuristic, then the
file extension. Force a format with --as <FORMAT>."
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn implicit_inspect_collects_files_and_flags() {
        let o = parse_args(&sv(&["model.gguf", "--compact", "--filter=blk.*", "b.npy"])).unwrap();
        assert_eq!(o.command, Cmd::Inspect);
        assert_eq!(o.files, ["model.gguf", "b.npy"]);
        assert!(o.compact);
        assert_eq!(o.filter.as_deref(), Some("blk.*"));
    }

    #[test]
    fn option_values_work_inline_and_separated() {
        let o = parse_args(&sv(&[
            "ls",
            "--filter",
            "x?",
            "--array-limit=3",
            "--as",
            "gguf",
            "f",
        ]))
        .unwrap();
        assert_eq!(o.command, Cmd::Ls);
        assert_eq!(o.filter.as_deref(), Some("x?"));
        assert_eq!(o.array_limit, 3);
        assert_eq!(o.forced, Some(Format::Gguf));
    }

    #[test]
    fn usage_errors_are_reported() {
        assert!(parse_args(&sv(&["--bogus", "f"]))
            .unwrap_err()
            .contains("unknown option"));
        assert!(parse_args(&sv(&["inspect"]))
            .unwrap_err()
            .contains("no input files"));
        assert!(parse_args(&sv(&["--as", "onnx", "f"]))
            .unwrap_err()
            .contains("unknown format"));
        assert!(parse_args(&sv(&["--filter"]))
            .unwrap_err()
            .contains("needs a value"));
        // A boolean flag must reject an inline value instead of silently
        // treating `--strict=false` as `--strict`.
        assert!(parse_args(&sv(&["--strict=false", "f"]))
            .unwrap_err()
            .contains("does not take a value"));
    }
}
