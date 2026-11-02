#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -euo pipefail

# ANSI Color Codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}  NeuralOps v2.0.2: One-Click GPU Automation & Verification Suite    ${NC}"
echo -e "${CYAN}======================================================================${NC}"

# 1. Hardware & Acceleration Detection
echo -e "\n${YELLOW}[1/6] Detecting GPU Hardware & Driver Configuration...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    ACCEL="cuda"
else
    echo -e "${YELLOW}Notice: nvidia-smi not detected. Checking PyTorch hardware discovery...${NC}"
    ACCEL=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')")
fi
echo -e "${GREEN}✓ Target Execution Backend: ${ACCEL^^}${NC}"

# 2. Package Installation & Verification
echo -e "\n${YELLOW}[2/6] Verifying NeuralOps Editable Installation...${NC}"
python3 -m pip install -e . --no-deps
python3 -c "import neuralops; print(f'✓ NeuralOps v{neuralops.__version__} loaded successfully.')"

# 3. Canonical Registry Integrity Check
echo -e "\n${YELLOW}[3/6] Verifying Architectural Registry Invariants...${NC}"
python3 -c "
from neuralops import ATTENTION_REGISTRY
from neuralops.compression import COMPRESSOR_REGISTRY
attns = len(ATTENTION_REGISTRY)
comps = len(COMPRESSOR_REGISTRY)
print(f'✓ Attentions Registered : {attns} / 131')
print(f'✓ Compressors Registered: {comps} / 72')
assert attns == 131, f'Expected 131 attentions, got {attns}'
assert comps == 72, f'Expected 72 compressors, got {comps}'
"

# 4. Run Unified GPU Master Benchmark Suite
echo -e "\n${YELLOW}[4/6] Executing Master GPU Benchmark Suite & Persisting Telemetry...${NC}"
mkdir -p results results/tables academic/paper/tables academic/paper/arxiv/figures

python3 scripts/benchmarks/gpu_master_suite.py \
    --category all \
    --device "${ACCEL}" \
    --export

# 5. Smart GPU Trainer Execution
echo -e "\n${YELLOW}[5/6] Running AutoMax GPU Adaptive Trainer...${NC}"
python3 scripts/benchmarks/smart_gpu_trainer.py

# 6. Generate LaTeX Paper Tables & Publication Figures
echo -e "\n${YELLOW}[6/6] Generating Camera-Ready Paper Tables & Figures...${NC}"
python3 scripts/export_paper_tables.py
python3 scripts/generate_arxiv_figures.py

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}✓ ALL TASKS COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}  • Empirical SQLite Ledger : results/master_ledger.sqlite${NC}"
echo -e "${GREEN}  • JSONL Stream            : results/benchmarks_latest.jsonl${NC}"
echo -e "${GREEN}  • CSV Summary             : results/benchmarks_latest.csv${NC}"
echo -e "${GREEN}  • LaTeX Paper Tables      : academic/paper/tables/${NC}"
echo -e "${GREEN}  • Vector Figures (PDF+PNG): academic/paper/arxiv/figures/${NC}"
echo -e "${GREEN}======================================================================${NC}"
