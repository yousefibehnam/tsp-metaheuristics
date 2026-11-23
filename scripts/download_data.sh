#!/usr/bin/env bash
# Downloads four symmetric TSP instances and their optimal tours from TSPLIB into data/tsplib/
# and verifies SHA-256 checksums.
#
#   berlin52   52 locations in Berlin (Groetschel)               optimum  7542
#   eil51      51-city problem (Christofides/Eilon)              optimum   426
#   st70       70-city problem (Smith/Thompson)                  optimum   675
#   kroA100    100-city problem A (Krolak/Felts/Nelson)          optimum 21282
#
# Source: TSPLIB, https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
# Cite: G. Reinelt, TSPLIB - A Traveling Salesman Problem Library, ORSA Journal on Computing 3(4),
#       376-384, 1991. https://doi.org/10.1287/ijoc.3.4.376
#
# Usage: scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/data/tsplib"
BASE="https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# file  sha256 of the .gz archive
fetch() {
  local file="$1" sha="$2"
  local target="$DIR/$file" archive="$DIR/$file.gz"
  if [[ -f "$target" ]]; then
    echo "already present: $file"
    return
  fi
  curl -fsSL --retry 3 -o "$archive" "$BASE/$file.gz"
  local actual
  actual="$(sha256_of "$archive")"
  if [[ "$actual" != "$sha" ]]; then
    echo "error: SHA-256 mismatch for $file.gz (expected $sha, got $actual)" >&2
    rm -f "$archive"
    exit 1
  fi
  gunzip -f "$archive"
  echo "downloaded $file"
}

mkdir -p "$DIR"
fetch berlin52.tsp 5bd7da24ec487b0d0bab1d68c423310297865d0770b31e79e6c99ef7b264b832
fetch berlin52.opt.tour e11a963d3feb97b59e7f092f13bac12f5c7100841c3e550f6a922e275f409898
fetch eil51.tsp 7176f94c7ca15c242fa4e9d418c19f326c17695ea1d6b2ea2d3caf8c9d172bbe
fetch eil51.opt.tour 291912b6ebbea12b675a3b733772c42c62b611244d74532b37825ff158896d75
fetch st70.tsp 7b6d70948114e9406dd3a4cc2fc2329aeda71e70b7c7327db23ad27aa2086087
fetch st70.opt.tour 49ba624411b615cdf19836b47100e58a983afc7291b29bcb0df2cbf79f11f669
fetch kroA100.tsp 016cb494dcd766cc06531c12a9fa79bd348cb21f2b2bbb9edb864951e8de7aa5
fetch kroA100.opt.tour 884ef503fb9649b662fd598860cacc2038061152a71df2bb5b9597dfc69d1727
