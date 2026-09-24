#!/usr/bin/env bash
cd /d/SVD_Project || exit 1

echo "Launching Job 1: SSM Scaling..."
nohup /d/SVD_Project/scripts/run_job1_ssm.sh > /d/SVD_Project/logs/seqlen_groupC.log 2>&1 &

echo "Launching Job 2: Linformer batch=16..."
nohup /d/SVD_Project/scripts/run_job2_linformer.sh > /d/SVD_Project/logs/linformer.log 2>&1 &

echo "Launching Job 3: DistilBERT Rank Sweep..."
nohup /d/SVD_Project/scripts/run_job3_distilbert.sh > /d/SVD_Project/logs/rank_sweep_distilbert.log 2>&1 &

echo "Launching Job 4: RoBERTa Rank Sweep..."
nohup /d/SVD_Project/scripts/run_job4_roberta.sh > /d/SVD_Project/logs/rank_sweep_roberta.log 2>&1 &

echo "Launching Job 5: GPT-2 Rank Sweep..."
nohup /d/SVD_Project/scripts/run_job5_gpt2.sh > /d/SVD_Project/logs/rank_sweep_gpt2.log 2>&1 &

echo "All 5 jobs launched in background."
ps -W | grep python || true
