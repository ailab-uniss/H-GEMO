#!/usr/bin/env bash
# ----------------------------------------------------------------
# launch_bench.sh  –  Run the main 20-dataset benchmark (Table 1)
# ----------------------------------------------------------------
# Usage:
#   bash scripts/launch_bench.sh                  # all 20 datasets
#   bash scripts/launch_bench.sh Emotions Yeast   # subset
#
# Each (dataset, fold) pair produces a separate run directory under
#   runs/repro/main_bench/<DATASET>/fold_<K>/
# ----------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# The 20 benchmark datasets used in the paper (Table 1)
ALL_DATASETS=(
    Arts Business Computers Education Emotions
    Enron Entertain Foodtruck Genbase Health
    Medical Recreation Reference Scene Science
    Slashdot Social Water-quality Yeast Yelp
)

# If user passes dataset names as arguments, use those; otherwise all 20
if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

N_FOLDS=5
CONFIG="configs/main_bench.yaml"

echo "=== H-GEMO main benchmark ==="
echo "Datasets: ${DATASETS[*]}"
echo "Folds:    0..$(( N_FOLDS - 1 ))"
echo ""

for DS in "${DATASETS[@]}"; do
    for FOLD in $(seq 0 $(( N_FOLDS - 1 ))); do
        OUT_DIR="runs/repro/main_bench/${DS}"
        echo "[$(date +%H:%M:%S)] Running ${DS}  fold ${FOLD} …"
        python -m hgemo.cli run \
            --config "$CONFIG" \
            --fold-idx "$FOLD" \
            --override \
                "dataset.name=${DS}" \
                "logging.out_dir=${OUT_DIR}"
    done
done

echo ""
echo "=== All main benchmark runs complete ==="
