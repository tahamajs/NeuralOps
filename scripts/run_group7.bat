@echo off
cd /d D:\svd_project
set PYTHONPATH=D:\svd_project;D:\svd_project\Codes\framework
echo [LAUNCH] Starting Group 7 sweep: mamba, primal, sdrm, spectral_mamba
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --model models\distilbert_mnli_full --dataset mnli --dataset_dir Codes\framework\data\glue_data\mnli --modules mamba,primal,sdrm,spectral_mamba --recovery_steps 100 --parallel 1 --batch_size 16 --no_skip > logs\step_d_group7.log 2>&1
echo [COMPLETE] Exit code: %ERRORLEVEL%
