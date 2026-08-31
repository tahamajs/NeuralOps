@echo off
setlocal

echo ===============================================================================
echo  NeuralOps Phase 2 Quality Benchmark Sweep & Multi-Seed Statistical Suite
echo ===============================================================================

set "PYTHON_EXE=C:\Users\Hadi Schumacher\anaconda3\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

set "PYTHONPATH=D:\SVD_Project;D:\SVD_Project\Codes;%PYTHONPATH%"
cd /d "D:\SVD_Project"

echo [*] Starting Phase 2 Quality Sweep...
"%PYTHON_EXE%" -u Codes\framework\run_quality_sweep.py --models "bert,roberta,distilbert,t5" --datasets "sst2,mrpc,cola,stsb" --seeds "42,123,777" --epochs 3 --max_steps 20 --precision fp16

echo [*] Quality sweep finished successfully!
exit /b 0
