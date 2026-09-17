Set-Location -Path "D:\SVD_Project"
& "C:\Users\Hadi Schumacher\anaconda3\python.exe" -u run_agnews_multi_class_sweep.py --models distilbert,roberta,gpt2 --batch_size 64 --recovery_steps 25 *> run_agnews_sweep.log
