#!/usr/bin/env bash
# Smoke test: builds tensorpeek, generates real fixture files in all four
# formats with the in-repo writers, then drives the CLI end to end — JSON
# reports, tables, filtering, truncation detection, --strict gating and the
# documented exit codes. Self-contained: temp dirs only, no network.
set -euo pipefail

cd "$(dirname "$0")/.."

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "[smoke] building..."
cargo build --quiet
BIN=target/debug/tensorpeek

WORK=$(mktemp -d "${TMPDIR:-/tmp}/tensorpeek-smoke.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# --- 1. version/help sanity -------------------------------------------------
"$BIN" --version | grep -q '^tensorpeek 0\.1\.0$' || fail "--version mismatch"
"$BIN" --help | grep -q 'COMMANDS:' || fail "--help missing sections"

# --- 2. generate fixtures with the in-repo writers ---------------------------
echo "[smoke] generating fixtures"
cargo run --quiet --example gen_fixtures -- "$WORK" > /dev/null

# --- 3. safetensors: full JSON report ----------------------------------------
echo "[smoke] safetensors"
"$BIN" inspect "$WORK/model.safetensors" > "$WORK/st.json"
grep -q '"format": "safetensors"' "$WORK/st.json" || fail "st format missing"
grep -q '"tensor_count": 3' "$WORK/st.json" || fail "st tensor count wrong"
grep -q '"parameters": 400' "$WORK/st.json" || fail "st parameter count wrong"
grep -q '"producer": "gen_fixtures"' "$WORK/st.json" || fail "st metadata missing"
grep -q '"problems"' "$WORK/st.json" && fail "clean st file reported problems"

# --- 4. gguf: metadata, dtypes, array summarization ---------------------------
echo "[smoke] gguf"
"$BIN" inspect "$WORK/model.gguf" > "$WORK/gguf.json"
grep -q '"architecture": "llama"' "$WORK/gguf.json" || fail "gguf architecture missing"
grep -q '"dtype": "q8_0"' "$WORK/gguf.json" || fail "gguf quant dtype missing"
grep -q '"alignment": 32' "$WORK/gguf.json" || fail "gguf alignment missing"
"$BIN" inspect --array-limit 4 "$WORK/model.gguf" | grep -q '"\$array"' \
  || fail "array summarization marker missing"

# --- 5. npy / npz (incl. the deflate-compressed member) -----------------------
echo "[smoke] npy / npz"
"$BIN" inspect "$WORK/embedding.npy" | grep -q '"shape": \[' || fail "npy shape missing"
"$BIN" inspect "$WORK/weights.npz" > "$WORK/npz.json"
grep -q '"name": "bias"' "$WORK/npz.json" || fail "npz member name wrong"
grep -q '"compression": "deflate"' "$WORK/npz.json" || fail "deflated member not parsed"

# --- 6. ls table and --filter --------------------------------------------------
echo "[smoke] ls / filter"
"$BIN" ls "$WORK/model.gguf" > "$WORK/ls.out"
grep -q 'NAME' "$WORK/ls.out" || fail "ls table header missing"
grep -q '4 tensors' "$WORK/ls.out" || fail "ls summary wrong"
grep -q '64×256' "$WORK/ls.out" || fail "ls shape column wrong"
"$BIN" inspect --compact --filter 'blk.*' "$WORK/model.gguf" \
  | grep -q '"name":"blk.0.attn_norm.weight"' || fail "--filter did not match"

# --- 7. truncation: reported, then gated by --strict ---------------------------
echo "[smoke] truncation / --strict"
"$BIN" inspect "$WORK/truncated.safetensors" > "$WORK/cut.json" \
  || fail "truncated file must still parse without --strict"
grep -q '100 missing' "$WORK/cut.json" || fail "missing byte count wrong"
if "$BIN" inspect --strict "$WORK/truncated.safetensors" > /dev/null 2>&1; then
  fail "--strict did not gate a truncated file"
fi

# --- 8. exit codes ---------------------------------------------------------------
echo "[smoke] exit codes"
set +e
"$BIN" inspect "$WORK/not-a-tensor.bin" > /dev/null 2>&1; [ $? -eq 1 ] \
  || { set -e; fail "unrecognized content should exit 1"; }
"$BIN" inspect "$WORK/does-not-exist.gguf" > /dev/null 2>&1; [ $? -eq 2 ] \
  || { set -e; fail "missing file should exit 2"; }
"$BIN" --bogus-flag x 2> /dev/null; [ $? -eq 2 ] \
  || { set -e; fail "unknown flag should exit 2"; }
set -e

# --- 9. multi-file batch: array output with per-file errors ---------------------
echo "[smoke] batch"
"$BIN" inspect --compact "$WORK/model.gguf" "$WORK/not-a-tensor.bin" \
  > "$WORK/batch.json" 2> /dev/null && fail "batch with a bad file should exit 1"
head -c 1 "$WORK/batch.json" | grep -q '\[' || fail "batch output is not an array"
grep -q '"error"' "$WORK/batch.json" || fail "batch missing the error object"

# --- 10. formats command + the shape-gate example --------------------------------
echo "[smoke] formats / shape gate"
"$BIN" formats | grep -q 'safetensors' || fail "formats listing incomplete"
PATH="$PWD/target/debug:$PATH" bash examples/shape-gate.sh \
  "$WORK/model.safetensors" embed.weight f32 '[32,8]' > /dev/null \
  || fail "shape gate rejected a matching tensor"
if PATH="$PWD/target/debug:$PATH" bash examples/shape-gate.sh \
  "$WORK/model.safetensors" embed.weight f32 '[8,32]' > /dev/null 2>&1; then
  fail "shape gate accepted a wrong shape"
fi

echo "SMOKE OK"
