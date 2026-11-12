#!/usr/bin/env bash
# Fetch the parts of SemBench the harness uses, at a pinned commit.
set -euo pipefail

COMMIT=c4b2bcff3a5907148f504cfb6faea4c6d3f90e8f
DIR="$(cd "$(dirname "$0")" && pwd)/.sembench"

URL=https://github.com/SemBench/SemBench.git

[ -d "$DIR/.git" ] || git init -q "$DIR"
git -C "$DIR" remote get-url origin >/dev/null 2>&1 || git -C "$DIR" remote add origin "$URL"
# Results the harness wrote on earlier runs; some shadow tracked files.
rm -rf "$DIR"/files/*/raw_results "$DIR"/files/*/metrics
git -C "$DIR" sparse-checkout set --no-cone \
  /src/evaluator/ /src/scenario/movie/evaluation/ \
  /files/movie/data/ /files/movie/query/ /LICENSE-MIT /LICENSE-APACHE
git -C "$DIR" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
# The evaluator rewrites tracked ground-truth files; discard that on a refetch.
git -C "$DIR" checkout -q -f FETCH_HEAD
echo "SemBench $COMMIT in $DIR"
