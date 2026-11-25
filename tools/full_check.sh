#!/usr/bin/env bash
# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
# Full check on a machine that has the private SAI compiler checkout: every test, then the pool x board
# regression. SAI runs in place from SAI_REPO (bin/sai); nothing is cloned, installed or copied from it.
#   SAI_REPO=/path/to/mcu_edge_ai SAI_MD6605_GCC=/path/to/riscv-none-elf-gcc tools/full_check.sh
set -euo pipefail
: "${SAI_REPO:?set SAI_REPO to your SAI compiler checkout}"
cd "$(dirname "$0")/.."
export SAI_REPO
before=$(git -C "$SAI_REPO" status --porcelain --ignored | sha256sum)
bin/sai version
python -m pytest -q tests -rs
SAI_PLATFORM_DATA="${SAI_PLATFORM_DATA:-$(mktemp -d)}" python -m backend.regression
after=$(git -C "$SAI_REPO" status --porcelain --ignored | sha256sum)
[ "$before" = "$after" ] && echo "SAI checkout untouched" || { echo "WARNING: the SAI checkout changed"; exit 1; }
