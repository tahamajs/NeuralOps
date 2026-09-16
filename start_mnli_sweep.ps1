Set-Location -Path "D:\SVD_Project"
& "C:\Users\Hadi Schumacher\anaconda3\python.exe" -u run_mnli_multi_class_sweep.py --models distilbert,roberta,gpt2 --batch_size 32 --recovery_steps 15 *> run_mnli_sweep.log
