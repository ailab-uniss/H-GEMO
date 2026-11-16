#!/usr/bin/env bash
# ----------------------------------------------------------------
# check_bundle.sh  –  Verify the installation
# ----------------------------------------------------------------
# Usage:  bash scripts/check_bundle.sh [--demo]
#
# Checks that the hgemo package imports and that the YAML configs
# parse; with --demo it also runs the quick demo on Emotions.
# ----------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PASS=0
FAIL=0
ok()   { echo "  ✓ $1"; PASS=$(( PASS + 1 )); }
fail() { echo "  ✗ $1"; FAIL=$(( FAIL + 1 )); }

echo "[1/3] Python imports…"
for mod in hgemo.config hgemo.cli hgemo.datasets hgemo.metrics hgemo.nsga2 \
           hgemo.mlknn_impl hgemo.genotypes hgemo.ml_eval hgemo.experiment; do
    if python -c "import $mod" 2>/dev/null; then ok "import $mod"; else fail "import $mod"; fi
done

echo "[2/3] YAML configs…"
for cfg in configs/*.yaml; do
    if python -c "import yaml; yaml.safe_load(open('$cfg'))" 2>/dev/null; then ok "$cfg"; else fail "$cfg"; fi
done

if [[ "${1:-}" == "--demo" ]]; then
    echo "[3/3] Demo run (Emotions, fold 0)…"
    if python -m hgemo.cli run --config configs/demo.yaml --fold-idx 0; then
        ok "demo run"
    else
        fail "demo run"
    fi
else
    echo "[3/3] Demo run skipped (pass --demo to enable)"
fi

echo ""
echo "=== ${PASS} passed, ${FAIL} failed ==="
[[ $FAIL -eq 0 ]] || exit 1
