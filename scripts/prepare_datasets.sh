#!/usr/bin/env bash
# ----------------------------------------------------------------
# prepare_datasets.sh – Extract the 20 paper benchmark datasets
# ----------------------------------------------------------------
# The bundle ships a single zip archive
#   data/dense_benchmark_v3/paper_datasets.zip   (~22 MB)
# containing the minimal files needed by the loader for the 20
# datasets used in the paper:
#
#   <Dataset>/fold{0..4}/trainval.npz
#   <Dataset>/fold{0..4}/test.npz
#
# This script extracts the archive in place so that the resulting
# directory layout is exactly what `hgemo.cli` expects:
#
#   data/dense_benchmark_v3/<Dataset>/foldK/{trainval,test}.npz
#
# Re-running the script is idempotent: existing files are kept
# (use --force to overwrite).
# ----------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

DATA_DIR="data/dense_benchmark_v3"
ARCHIVE="${DATA_DIR}/paper_datasets.zip"

FORCE=0
if [[ "${1:-}" == "--force" || "${1:-}" == "-f" ]]; then
    FORCE=1
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: archive not found: $ARCHIVE" >&2
    echo "       Make sure you cloned the repo with the data file present." >&2
    exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: 'unzip' is required but not installed." >&2
    echo "       On Debian/Ubuntu: sudo apt-get install unzip" >&2
    exit 1
fi

echo "=== H-GEMO dataset preparation ==="
echo "Archive: $ARCHIVE"
echo "Target : $DATA_DIR/"
echo ""

if [[ $FORCE -eq 1 ]]; then
    echo "[--force] Overwriting existing files."
    unzip -o -q "$ARCHIVE" -d "$DATA_DIR"
else
    # -n = never overwrite existing files (idempotent)
    unzip -n -q "$ARCHIVE" -d "$DATA_DIR"
fi

echo ""
echo "Extracted datasets:"
for d in "$DATA_DIR"/*/; do
    name="$(basename "$d")"
    [[ "$name" == "__pycache__" ]] && continue
    n_folds=$(find "$d" -mindepth 1 -maxdepth 1 -type d -name 'fold*' | wc -l)
    printf "  %-18s %d folds\n" "$name" "$n_folds"
done

echo ""
echo "=== Done. You can now run any of the launch scripts. ==="
