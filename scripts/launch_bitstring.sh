#!/usr/bin/env bash
# ----------------------------------------------------------------
# launch_bitstring.sh  –  Run bitstring ablation (Table 2)
# ----------------------------------------------------------------
# Usage:
#   bash scripts/launch_bitstring.sh                  # all 20 datasets
#   bash scripts/launch_bitstring.sh Emotions Yeast   # subset
# ----------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

ALL_DATASETS=(
    Arts Business Computers Education Emotions
    Enron Entertain Foodtruck Genbase Health
    Medical Recreation Reference Scene Science
    Slashdot Social Water-quality Yeast Yelp
)

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

N_FOLDS=5
CONFIG="configs/bitstring_ablation.yaml"

echo "=== Bitstring ablation ==="
echo "Datasets: ${DATASETS[*]}"
echo ""

for DS in "${DATASETS[@]}"; do
    for FOLD in $(seq 0 $(( N_FOLDS - 1 ))); do
        OUT_DIR="runs/repro/bitstring/${DS}"
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
echo "=== All bitstring ablation runs complete ==="
