//! Generate small, deterministic fixture files in all four formats using
//! tensorpeek's built-in spec-exact writers — handy for trying the CLI
//! without downloading a real checkpoint.
//!
//! Usage: cargo run --example gen_fixtures -- <output-dir>

use std::path::Path;

use tensorpeek::builder::{self, GgufBuilder};

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: gen_fixtures <output-dir>");
        std::process::exit(2);
    });
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).expect("create output dir");

    let mut wrote = Vec::new();
    let mut put = |name: &str, bytes: Vec<u8>| {
        std::fs::write(dir.join(name), &bytes).expect("write fixture");
        wrote.push(format!("{name} ({} bytes)", bytes.len()));
    };

    // A small MLP-style safetensors checkpoint with metadata.
    put(
        "model.safetensors",
        builder::safetensors(
            &[
                ("embed.weight", "F32", &[32, 8]),
                ("fc1.weight", "F16", &[8, 16]),
                ("fc1.bias", "F16", &[16]),
            ],
            &[("format", "pt"), ("producer", "gen_fixtures")],
        ),
    );

    // A llama-flavored GGUF file: realistic keys, a tiny vocabulary,
    // q8_0 / q4_0 / f32 tensors.
    put("model.gguf", GgufBuilder::demo().build());

    // A plain npy array and an npz archive with one stored and one
    // deflate-compressed member.
    put("embedding.npy", builder::npy("<f4", false, &[512, 64]));
    put(
        "weights.npz",
        builder::npz(&[
            ("weights.npy", builder::npy("<f4", false, &[4, 4]), false),
            ("bias.npy", builder::npy("<i8", false, &[100]), true),
        ]),
    );

    // The classic broken upload: a safetensors file missing its tail.
    let mut cut = builder::safetensors(&[("w", "F32", &[256])], &[]);
    cut.truncate(cut.len() - 100);
    put("truncated.safetensors", cut);

    // Something that is no tensor file at all.
    put(
        "not-a-tensor.bin",
        b"\xde\xad\xbe\xefjust some junk bytes".to_vec(),
    );

    println!("wrote {} fixtures to {}:", wrote.len(), dir.display());
    for w in &wrote {
        println!("  {w}");
    }
}
