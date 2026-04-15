#!/usr/bin/env bash
# ----------------------------------------------------------------
# create_venv.sh  –  Create a fresh virtual-env and install deps
# ----------------------------------------------------------------
# Usage:  bash scripts/create_venv.sh [venv_dir]
#   venv_dir  (optional, default: .venv)
#
# The script will:
#   1. Create a Python 3.10+ virtual environment
#   2. Install PyTorch (CPU or CUDA, user's choice)
#   3. Install the remaining requirements
# ----------------------------------------------------------------
set -euo pipefail

VENV_DIR="${1:-.venv}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Creating venv in ${VENV_DIR} ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

echo ""
echo "Install PyTorch with CUDA support? [y/N]"
read -r CUDA_CHOICE
if [[ "${CUDA_CHOICE,,}" == "y" ]]; then
    echo "=== Installing PyTorch (CUDA 12.1) ==="
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    echo "=== Installing PyTorch (CPU) ==="
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "=== Installing remaining requirements ==="
pip install -r "$ROOT_DIR/requirements.txt"

echo ""
echo "Done.  Activate with:  source ${VENV_DIR}/bin/activate"
