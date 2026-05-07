#!/usr/bin/env bash
# ============================================================================
# Experiment Orchestrator – Single Dataset, Full Logging & Visualization
# Runs: attention x compressor x method x seed (all combinations)
# Saves: attention maps, loss curves, model diagrams, CSV summary
# ============================================================================

set -e   # stop on error (remove if you want to continue after failures)

# ---------- Configuration (edit these) ----------
DATASET="sst2"                        # only one dataset
ATTENTIONS=("token_compressed" "aegis_v4" "gqa" "full")
COMPRESSORS=("tt" "none")            # "none" means no compressor
METHODS=("baseline" "lora")           # methods to try
SEEDS=(42)                            # seeds (you can add more)

# Base command (add --save_visuals to save pictures)
BASE_CMD="python main.py --save_visuals"

# Output directories
EXPERIMENT_BASE="./experiments"
REPORT_CSV="$EXPERIMENT_BASE/summary_$DATASET.csv"
mkdir -p "$EXPERIMENT_BASE"

# ---------- Helper: log to CSV ----------
log_to_csv() {
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$timestamp,$DATASET,$1,$2,$3,$4,$5,$6" >> "$REPORT_CSV"
}

# ---------- Run a single experiment ----------
run_experiment() {
    local attention="$1"
    local compressor="$2"
    local method="$3"
    local seed="$4"

    local exp_name="${attention}_${compressor}_${method}_${DATASET}_seed${seed}"
    local output_dir="$EXPERIMENT_BASE/$exp_name"
    mkdir -p "$output_dir"

    echo ""
    echo "=============================================="
    echo "RUNNING: $exp_name"
    echo "Start time: $(date)"
    echo "=============================================="

    # Build command
    cmd="$BASE_CMD"
    cmd="$cmd --attn_type $attention"
    cmd="$cmd --compressor $compressor"
    cmd="$cmd --method $method"
    cmd="$cmd --task $DATASET"
    cmd="$cmd --seed $seed"
    cmd="$cmd --output_dir $EXPERIMENT_BASE"
    cmd="$cmd --exp_name $exp_name"

    echo "Command: $cmd"
    # Run and capture stdout+stderr into a log file
    $cmd 2>&1 | tee "$output_dir/run.log"

    # Extract final accuracy and loss from the log (look for "eval_accuracy" and "eval_loss")
    accuracy=$(grep -E "eval_accuracy" "$output_dir/run.log" | tail -1 | grep -oP "[0-9.]+" | head -1 || echo "NA")
    loss=$(grep -E "eval_loss" "$output_dir/run.log" | tail -1 | grep -oP "[0-9.]+" | head -1 || echo "NA")

    echo "Finished: accuracy=$accuracy, loss=$loss"
    log_to_csv "$attention" "$compressor" "$method" "$seed" "$accuracy" "$loss"
    echo "Log saved to $output_dir/run.log"
    echo "=============================================="
}

# ---------- Main loop (all combinations) ----------
# Create CSV header if not exists
if [ ! -f "$REPORT_CSV" ]; then
    echo "timestamp,dataset,attention,compressor,method,seed,accuracy,loss" > "$REPORT_CSV"
fi

# Run experiments
for method in "${METHODS[@]}"; do
    for attention in "${ATTENTIONS[@]}"; do
        for compressor in "${COMPRESSORS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_experiment "$attention" "$compressor" "$method" "$seed"
            done
        done
    done
done

echo ""
echo "========== ALL EXPERIMENTS COMPLETED =========="
echo "Summary CSV: $REPORT_CSV"
echo "Individual experiment folders: $EXPERIMENT_BASE"

# ---------- Generate summary plot (optional) ----------
if command -v python &> /dev/null; then
    python -c "
import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = '$REPORT_CSV'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    # Average accuracy per attention, compressor, method
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    pivot = df.pivot_table(index=['attention', 'compressor'], columns='method', values='accuracy', aggfunc='mean')
    if not pivot.empty:
        pivot.plot(kind='bar', figsize=(12,6))
        plt.title(f'Average Accuracy on {df.iloc[0][\"dataset\"]}')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('$EXPERIMENT_BASE/summary_plot.png')
        print('Summary plot saved to $EXPERIMENT_BASE/summary_plot.png')
" || echo "Python plot generation skipped (install pandas/matplotlib if needed)."
fi

echo "Done."    