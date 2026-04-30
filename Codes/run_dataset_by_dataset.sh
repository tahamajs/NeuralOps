#!/bin/bash
# run_dataset_by_dataset.sh
# Run all methods on each GLUE dataset separately (dataset by dataset)
# Updated: Added Fisher-GeLoRA, DisLoRA, multi-seed support, memory optimizations

set -e  # Stop on error (comment out if you want to continue on errors)

# Disable network requests (use local cache only)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# General settings
MODEL="bert-base-uncased"
EPOCHS=3
MAX_SEQ_LEN=128
WARMUP_RATIO=0.1
MAX_GRAD_NORM=1.0

# Multi-seed settings (set to 1 for quick testing, 3-5 for final experiments)
NUM_SEEDS=1
BASE_SEED=42

# List of datasets (tasks) - in order of execution
TASKS=("cola" "mrpc" "rte" "stsb" "sst2")

# List of methods (in any order)
METHODS=(
    # "baseline"
    # "lora"
    # "dora"
    # "dislora"
    # "fisher_lora"
    "fisher_gelora"
    "rdrop"
    "r3f"
    "mixout"
    "llrd"
    "lora+rdrop"
)

# Output root directory
OUTPUT_ROOT="./experiments"

echo "============================================================"
echo "Starting GLUE Experiments (Dataset by Dataset)"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Seeds: $NUM_SEEDS (starting from $BASE_SEED)"
echo "Tasks: ${TASKS[@]}"
echo "Methods: ${METHODS[@]}"
echo "============================================================"

# Function to clean MPS cache
clean_cache() {
    python -c "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else None" 2>/dev/null || true
}

# Loop over each dataset (task)
for TASK in "${TASKS[@]}"; do
    echo " "
    echo "########################################"
    echo "##### DATASET: $TASK #####"
    echo "########################################"
    
    # Loop over each method
    for METHOD in "${METHODS[@]}"; do
        echo " "
        echo "========== METHOD: $METHOD on $TASK =========="
        
        # Method-specific hyperparameters
        case $METHOD in
            baseline)
                BATCH=16
                LR=2e-5
                GRAD_ACC=4
                EXTRA=""
                ;;
            lora)
                BATCH=16
                LR=5e-5
                GRAD_ACC=4
                EXTRA="--lora_r 8 --lora_alpha 16 --lora_dropout 0.1"
                ;;
            dora)
                BATCH=16
                LR=5e-5
                GRAD_ACC=4
                EXTRA="--lora_r 8 --lora_alpha 16 --lora_dropout 0.1"
                ;;
            dislora)
                BATCH=16
                LR=3e-4
                GRAD_ACC=4
                EXTRA="--lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --dislora_beta 0.5 --dislora_capacity_factor 1.0"
                ;;
            fisher_lora)
                BATCH=16
                LR=3e-4
                GRAD_ACC=4
                EXTRA="--lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --fisher_num_batches 20 --fisher_sample_size 200"
                ;;
            fisher_gelora)
                BATCH=16
                LR=1e-4
                GRAD_ACC=4
                EXTRA="--lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --gelora_margin 1.0 --gelora_min_rank 2 --gelora_max_rank 32 --fisher_scale_max 2.0 --fisher_num_batches 20 --fisher_sample_size 200"
                ;;
            rdrop)
                BATCH=8
                LR=2e-5
                GRAD_ACC=4
                EXTRA="--rdrop_alpha 5.0"
                ;;
            r3f)
                BATCH=8
                LR=2e-5
                GRAD_ACC=4
                EXTRA="--r3f_epsilon 1e-5"
                ;;
            mixout)
                BATCH=16
                LR=2e-5
                GRAD_ACC=4
                EXTRA="--mixout_p 0.5"
                ;;
            llrd)
                BATCH=16
                LR=2e-5
                GRAD_ACC=4
                EXTRA="--llrd_decay 0.95"
                ;;
            lora+rdrop)
                BATCH=8
                LR=1e-4
                GRAD_ACC=4
                EXTRA="--lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --rdrop_alpha 5.0"
                ;;
            *)
                BATCH=16
                LR=2e-5
                GRAD_ACC=4
                EXTRA=""
                ;;
        esac
        
        echo ">>> Running $METHOD on $TASK (batch=$BATCH, acc=$GRAD_ACC, lr=$LR, seeds=$NUM_SEEDS)"
        
        # Build the command with multi-seed support
        CMD="python run_glue_svd.py \
            --method $METHOD \
            --tasks $TASK \
            --model_name_or_path $MODEL \
            --output_root $OUTPUT_ROOT \
            --num_train_epochs $EPOCHS \
            --per_device_train_batch_size $BATCH \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps $GRAD_ACC \
            --learning_rate $LR \
            --max_seq_length $MAX_SEQ_LEN \
            --warmup_ratio $WARMUP_RATIO \
            --max_grad_norm $MAX_GRAD_NORM \
            --save_total_limit 1 \
            --num_seeds $NUM_SEEDS \
            --seed $BASE_SEED"
        
        # Add gradient checkpointing for memory-intensive methods
        if [[ "$METHOD" == "rdrop" || "$METHOD" == "r3f" || "$METHOD" == "fisher_gelora" || "$METHOD" == "lora+rdrop" ]]; then
            CMD="$CMD --gradient_checkpointing"
        fi
        
        # Add method-specific extra arguments
        if [ -n "$EXTRA" ]; then
            CMD="$CMD $EXTRA"
        fi
        
        echo ">> $CMD"
        
        # Run the command with error handling
        if eval $CMD; then
            echo "✅ SUCCESS: $METHOD on $TASK"
        else
            echo "❌ FAILED: $METHOD on $TASK (continuing with next method)"
            # Optional: uncomment to stop on failure
            # exit 1
        fi
        
        # Clear MPS cache after each method (free memory)
        clean_cache
        
        echo ">>> Finished $METHOD on $TASK"
        echo "---"
        sleep 2  # Brief pause to ensure memory cleanup
    done
    
    # Full cleanup after completing all methods for a dataset
    clean_cache
    
    # Optional: Clean old checkpoints to save disk space after each dataset
    echo "Cleaning old checkpoints to save disk space..."
    find "$OUTPUT_ROOT" -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true
    
    echo "########################################"
    echo "##### Completed ALL methods for $TASK #####"
    echo "########################################"
done

echo "============================================================"
echo "All experiments completed successfully."
echo "Results saved in $OUTPUT_ROOT"
echo "============================================================"

# Final summary
echo ""
echo "📊 EXPERIMENT SUMMARY"
echo "============================================================"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Seeds per run: $NUM_SEEDS"
echo "Total combinations: $((${#TASKS[@]} * ${#METHODS[@]}))"
echo ""
echo "Datasets processed: ${TASKS[@]}"
echo "Methods attempted: ${METHODS[@]}"
echo "============================================================"

# Optional: Generate quick result summary if results directory exists
if [ -d "$OUTPUT_ROOT" ]; then
    echo ""
    echo "📁 Results directory: $OUTPUT_ROOT"
    echo "Latest run:"
    ls -lt "$OUTPUT_ROOT" | head -5 || true
fi