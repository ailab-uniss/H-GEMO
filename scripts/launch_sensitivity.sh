#!/usr/bin/env bash
# ----------------------------------------------------------------
# launch_sensitivity.sh  –  Run all sensitivity analyses (Figs 4-6)
# ----------------------------------------------------------------
# Usage:
#   bash scripts/launch_sensitivity.sh              # all variants
#   bash scripts/launch_sensitivity.sh pop_size_20  # one variant
#
# The 9 pilot datasets are used for sensitivity, as in the paper.
# ----------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# 9 sensitivity-pilot datasets (same as the paper)
PILOT_DATASETS=(
    Emotions Enron Genbase Medical Scene
    Slashdot Water-quality Yeast Yelp
)

# All sensitivity config files
ALL_CONFIGS=(
    configs/sensitivity/pop_size_20.yaml
    configs/sensitivity/pop_size_30.yaml
    configs/sensitivity/pop_size_80.yaml
    configs/sensitivity/bs_pop20.yaml
    configs/sensitivity/bs_pop30.yaml
    configs/sensitivity/bs_pop50.yaml
    configs/sensitivity/bs_pop80.yaml
    configs/sensitivity/routing_0.2_0.4_0.4.yaml
    configs/sensitivity/routing_0.2_0.6_0.2.yaml
    configs/sensitivity/routing_0.6_0.2_0.2.yaml
    configs/sensitivity/topm_ratio_005.yaml
    configs/sensitivity/topm_ratio_010.yaml
    configs/sensitivity/topm_ratio_020.yaml
    configs/sensitivity/topm_ratio_025.yaml
)

# If user passes config basenames, filter
if [[ $# -gt 0 ]]; then
    CONFIGS=()
    for arg in "$@"; do
        found="configs/sensitivity/${arg}.yaml"
        if [[ -f "$found" ]]; then
            CONFIGS+=("$found")
        else
            echo "WARNING: config not found: $found"
        fi
    done
else
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

N_FOLDS=5

echo "=== H-GEMO sensitivity analysis ==="
echo "Configs:  ${#CONFIGS[@]}"
echo "Datasets: ${PILOT_DATASETS[*]}"
echo ""

for CFG in "${CONFIGS[@]}"; do
    CFG_NAME="$(basename "$CFG" .yaml)"
    echo "--- Config: ${CFG_NAME} ---"
    for DS in "${PILOT_DATASETS[@]}"; do
        for FOLD in $(seq 0 $(( N_FOLDS - 1 ))); do
            OUT_DIR="runs/repro/sensitivity/${CFG_NAME}/${DS}"
            echo "[$(date +%H:%M:%S)] ${CFG_NAME} / ${DS} / fold ${FOLD}"
            python -m hgemo.cli run \
                --config "$CFG" \
                --fold-idx "$FOLD" \
                --override \
                    "dataset.name=${DS}" \
                    "logging.out_dir=${OUT_DIR}"
        done
    done
done

echo ""
echo "=== All sensitivity runs complete ==="
