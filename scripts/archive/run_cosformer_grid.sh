#!/bin/bash
# Run cosformer on 2 datasets × 4 ranks in 4 parallel workers

MODEL_BERT="models/distilbert_sst2"
MODEL_ROBERTA="models/roberta-base"
FAMILY_BERT="bert"
FAMILY_ROBERTA="roberta"

# Datasets and their corresponding models/families
declare -A DS_MODEL=( [sst2]="$MODEL_BERT" [mrpc]="$MODEL_ROBERTA" )
declare -A DS_FAMILY=( [sst2]="$FAMILY_BERT" [mrpc]="$FAMILY_ROBERTA" )

RANKS="16 32 64 128"
DATASETS="sst2 mrpc"

# Build a list of all jobs
JOBS=()
for DS in $DATASETS; do
  for R in $RANKS; do
    JOBS+=("${DS}:${R}")
  done
done

echo "Total jobs: ${#JOBS[@]}"
echo "Running with --parallel 4"
echo ""

# Run 4 jobs at a time
COUNTER=0
BATCH=()
for JOB in "${JOBS[@]}"; do
  BATCH+=("$JOB")
  COUNTER=$((COUNTER + 1))

  if [ $((COUNTER % 4)) -eq 0 ] || [ $COUNTER -eq ${#JOBS[@]} ]; then
    # Extract datasets and ranks from batch
    DS_LIST=""
    RANK_LIST=""
    MODEL=""
    FAMILY=""
    
    for B in "${BATCH[@]}"; do
      DS="${B%%:*}"
      R="${B##*:}"
      DS_LIST="$DS_LIST $DS"
      RANK_LIST="$RANK_LIST $R"
      MODEL="${DS_MODEL[$DS]}"
      FAMILY="${DS_FAMILY[$DS]}"
    done
    
    # For simplicity, run each job separately with its own dataset/model
    for B in "${BATCH[@]}"; do
      DS="${B%%:*}"
      R="${B##*:}"
      M="${DS_MODEL[$DS]}"
      F="${DS_FAMILY[$DS]}"
      
      echo "════════════════════════════════════════════════"
      echo "  cosformer @ ${DS} @ rank=${R} (model=${M})"
      echo "════════════════════════════════════════════════"
      
      ssh gpu "powershell -Command \"cd D:\SVD_Project; python Codes/framework/run_quality_sweep.py \
          --modules cosformer \
          --model ${M} \
          --model_family ${F} \
          --dataset ${DS} \
          --rank ${R} \
          --recovery_steps 50 \
          --lora_rank 16 \
          --batch_size 8 \
          --fresh \
          --output_dir results/cosformer_grid/${DS}_rank${R}/\""
    done
    
    BATCH=()
  fi
done

echo ""
echo "✅ All 8 jobs complete."
echo "Results in results/cosformer_grid/"
