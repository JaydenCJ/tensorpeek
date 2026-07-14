#!/usr/bin/env bash
# A CI shape gate with no Python in sight: assert that a checkpoint contains
# a tensor with the expected name, dtype and shape before shipping it.
#
#   bash examples/shape-gate.sh model.safetensors embed.weight f32 '[32,8]'
#
# Exits 0 when the tensor matches, 1 otherwise. Uses only tensorpeek + grep,
# so it runs in the slimmest container. Also demonstrates --strict: a
# truncated file fails the gate even when the shape matches.
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "usage: shape-gate.sh <file> <tensor-name> <dtype> <shape-json>" >&2
  exit 2
fi
FILE=$1 NAME=$2 DTYPE=$3 SHAPE=$4

# --strict: truncation or size mismatches fail the gate outright.
JSON=$(tensorpeek inspect --strict --compact --filter "$NAME" "$FILE")

WANT="\"name\":\"$NAME\",\"dtype\":\"$DTYPE\",\"shape\":$SHAPE"
if echo "$JSON" | grep -qF "$WANT"; then
  echo "shape gate OK: $NAME is $DTYPE $SHAPE"
else
  echo "shape gate FAILED: expected $NAME $DTYPE $SHAPE" >&2
  echo "  got: $JSON" >&2
  exit 1
fi
