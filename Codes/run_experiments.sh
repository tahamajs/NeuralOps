#!/bin/bash
# run_experiments.sh
# Complete experiment runner for SVD-Lin

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 SVD-Lin Experiment Runner                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"

# Default values
MODEL=${1:-"bert-base-uncased"}
TASKS=${2:-"cola sst2 mrpc"}
ATTENTION=${3:-"svd"}
RANK=${4:-128}
EPOCHS=${5:-3}
BATCH_SIZE=${6:-16}

echo -e "\n📋 Configuration:"
echo -e "   Model: $MODEL"
echo -e "   Tasks: $TASKS"
echo -e "   Attention: $ATTENTION"
echo -e "   Rank: $RANK"
echo -e "   Epochs: $EPOCHS"
echo -e "   Batch size: $BATCH_SIZE"

# Run experiment
echo -e "\n${YELLOW}🚀 Running experiment...${NC}"

python main.py \
    --model_name "$MODEL" \
    --tasks $TASKS \
    --attention_type "$ATTENTION" \
    --rank "$RANK" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --visualize \
    --generate_report \
    --verbose

echo -e "\n${GREEN}✅ Experiment completed!${NC}"