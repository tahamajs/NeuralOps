@echo off
setlocal

echo ===============================================================================
echo  NeuralOps: Full End-to-End Benchmark Suite (Windows / CUDA Host)
echo ===============================================================================

set "PYTHON_EXE=C:\Users\Hadi Schumacher\anaconda3\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

set "PYTHONPATH=%~dp0;%~dp0Codes;%PYTHONPATH%"
cd /d "%~dp0"

echo [*] Phase 1: Launching Adaptive Efficiency Sweep (720 combinations)...
"%PYTHON_EXE%" -u Codes\framework\two_phase_benchmark.py --mode efficiency --precision fp16 --batch_size 128 --seq_len 128

echo [*] Phase 2: Launching Multi-Seed Statistical Quality Sweep...
"%PYTHON_EXE%" -u Codes\framework\run_quality_sweep.py --models "bert,roberta,distilbert,t5" --datasets "sst2,mrpc,cola,stsb" --seeds "42,123,777" --epochs 3 --max_steps 20 --precision fp16

echo ===============================================================================
echo  Full End-to-End Benchmark Completed Successfully!
echo ===============================================================================
exit /b 0
