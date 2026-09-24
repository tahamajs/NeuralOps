#!/bin/bash
mkdir -p logs/cosformer_grid

echo "════════ SST-2 @ 4 ranks in parallel ════════"
for RANK in 16 32 64 128; do
    echo "Launching SST-2 @ rank=$RANK ..."
    (
        ssh gpu "powershell -Command \"cd D:\SVD_Project; python Codes/framework/run_quality_sweep.py --modules cosformer --model models/distilbert_sst2 --model_family bert --dataset sst2 --rank ${RANK} --recovery_steps 50 --lora_rank 16 --batch_size 8 --fresh --output_dir results/cosformer_grid/sst2_rank${RANK}/\""
    ) > logs/cosformer_grid/sst2_rank${RANK}.log 2>&1 &
done
wait
echo "✅ SST-2 done."

echo ""
echo "════════ MRPC @ 4 ranks in parallel ════════"
for RANK in 16 32 64 128; do
    echo "Launching MRPC @ rank=$RANK ..."
    (
        ssh gpu "powershell -Command \"cd D:\SVD_Project; python Codes/framework/run_quality_sweep.py --modules cosformer --model models/roberta-base --model_family roberta --dataset mrpc --rank ${RANK} --recovery_steps 50 --lora_rank 16 --batch_size 4 --fresh --output_dir results/cosformer_grid/mrpc_rank${RANK}/\""
    ) > logs/cosformer_grid/mrpc_rank${RANK}.log 2>&1 &
done
wait
echo "✅ MRPC done."

echo ""
echo "All 8 configs complete."
echo "Logs: logs/cosformer_grid/"
