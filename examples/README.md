# tensorpeek examples

Two runnable examples, both offline and deterministic.

## gen_fixtures.rs

Generates one file per supported format with tensorpeek's built-in
spec-exact writers: a safetensors checkpoint with metadata, a
llama-flavored GGUF file (q8_0 / q4_0 / f32 tensors, tiny vocabulary), a
plain npy array, an npz archive with one stored and one deflate-compressed
member, a truncated safetensors file, and a junk file that is no tensor
format at all.

```bash
cargo run --example gen_fixtures -- /tmp/tensorpeek-fixtures
cargo run -- inspect /tmp/tensorpeek-fixtures/model.gguf
cargo run -- ls /tmp/tensorpeek-fixtures/*.safetensors
```

| File | Expected outcome |
|---|---|
| `model.safetensors` | 3 tensors, `format=pt` metadata, no problems |
| `model.gguf` | GGUF v3, `llama` architecture, 4 tensors |
| `embedding.npy` | one unnamed `f32` tensor, shape 512×64 |
| `weights.npz` | 2 members, one `stored`, one `deflate` |
| `truncated.safetensors` | `problems` reports exactly 100 missing bytes |
| `not-a-tensor.bin` | exit 1, "unrecognized format" with an `--as` hint |

## shape-gate.sh

Shows `tensorpeek inspect --strict --compact --filter` as a CI gate: assert
that a checkpoint contains a tensor with the expected name, dtype and shape
using nothing but tensorpeek and `grep` — no Python, no framework.

```bash
cargo run --example gen_fixtures -- /tmp/tensorpeek-fixtures
PATH="$PWD/target/debug:$PATH" \
  bash examples/shape-gate.sh /tmp/tensorpeek-fixtures/model.safetensors \
  embed.weight f32 '[32,8]'
```

The fixture writers are seeded by fixed presets, so their output is
byte-identical on every machine.
