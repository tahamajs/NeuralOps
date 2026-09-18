@echo off
cd /d D:\SVD_Project
python -u run_failed_ppl.py > ppl_attention_sweep.log 2>&1
python -u summarize_ppl.py >> ppl_attention_sweep.log 2>&1
