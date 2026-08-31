@echo off
setlocal

cd /d D:\SVD_Project
set PYTHONPATH=Codes;%PYTHONPATH%

python -u retest_broken_modules.py > D:\SVD_Project\retest_broken.log 2>&1

endlocal
