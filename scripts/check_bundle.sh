#!/usr/bin/env bash
# ----------------------------------------------------------------
# check_bundle.sh  –  Verify the reproducibility bundle is intact
# ----------------------------------------------------------------
# Usage:  bash scripts/check_bundle.sh
#
# Checks:
#   1. All expected files exist
#   2. Python can import the hgemo package
#   3. YAML configs parse correctly
#   4. Smoke test runs (optional, with --smoke flag)
# ----------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PASS=0
FAIL=0

ok()   { echo "  ✓ $1"; PASS=$(( PASS + 1 )); }
fail() { echo "  ✗ $1"; FAIL=$(( FAIL + 1 )); }

echo "=== Bundle integrity check ==="
echo ""

# --- 1. File existence ---
echo "[1/4] Checking required files…"
REQUIRED_FILES=(
    hgemo/__init__.py
    hgemo/cli.py
    hgemo/config.py
    hgemo/datasets.py
    hgemo/experiment.py
    hgemo/genotypes.py
    hgemo/logging_utils.py
    hgemo/metrics.py
    hgemo/ml_eval.py
    hgemo/mlknn_impl.py
    hgemo/npz_format.py
    hgemo/nsga2.py
    hgemo/utils.py
    configs/main_bench.yaml
    configs/bitstring_ablation.yaml
    configs/smoke.yaml
    requirements.txt
    README.md
)

for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        ok "$f"
    else
        fail "$f  (MISSING)"
    fi
done

# --- 2. Python imports ---
echo ""
echo "[2/4] Checking Python imports…"
MODULES=(
    hgemo.config
    hgemo.cli
    hgemo.datasets
    hgemo.metrics
    hgemo.nsga2
    hgemo.mlknn_impl
    hgemo.genotypes
    hgemo.ml_eval
    hgemo.experiment
)
for mod in "${MODULES[@]}"; do
    if python -c "import $mod" 2>/dev/null; then
        ok "import $mod"
    else
        fail "import $mod"
    fi
done

# --- 3. YAML parsing ---
echo ""
echo "[3/4] Checking YAML configs…"
for cfg in configs/*.yaml configs/sensitivity/*.yaml; do
    if python -c "import yaml; yaml.safe_load(open('$cfg'))" 2>/dev/null; then
        ok "$cfg"
    else
        fail "$cfg  (PARSE ERROR)"
    fi
done

# --- 4. Optional smoke test ---
echo ""
if [[ "${1:-}" == "--smoke" ]]; then
    echo "[4/4] Running smoke test (this may take a few minutes)…"
    if python -m hgemo.cli run --config configs/smoke.yaml --fold-idx 0; then
        ok "smoke test passed"
    else
        fail "smoke test FAILED"
    fi
else
    echo "[4/4] Smoke test skipped (pass --smoke to enable)"
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[[ $FAIL -eq 0 ]] && echo "All checks passed ✓" || exit 1
