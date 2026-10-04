@echo off
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set PYTHONPATH=.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

:loop
echo Running KD Sweep with BS=16 to maximize GPU...
python scripts\run_kd_reuse_model.py --recovery_steps 500 --batch_size 16
if %ERRORLEVEL% NEQ 0 (
    if %ERRORLEVEL% EQU 999 goto end
    echo Hard crash detected. Restarting script to skip the bad module...
    goto loop
)
:end
echo Sweep finished successfully!
