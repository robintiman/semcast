#!/usr/bin/env bash
# Fetch the parts of SemBench the harness uses, at a pinned commit.
set -euo pipefail

COMMIT=c4b2bcff3a5907148f504cfb6faea4c6d3f90e8f
DIR="$(cd "$(dirname "$0")" && pwd)/.sembench"

if [ ! -d "$DIR/.git" ]; then
  git init -q "$DIR"
  git -C "$DIR" remote add origin https://github.com/SemBench/SemBench.git
fi
git -C "$DIR" sparse-checkout set --no-cone \
  /src/evaluator/ /src/scenario/movie/evaluation/ \
  /files/movie/data/ /files/movie/query/ /LICENSE-MIT /LICENSE-APACHE
git -C "$DIR" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$DIR" checkout -q FETCH_HEAD
echo "SemBench $COMMIT in $DIR"
