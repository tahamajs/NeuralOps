import subprocess

commands = [
    "python -u Codes/framework/run_quality_sweep.py --modules none --model models/roberta-base --dataset mnli --finetune_mode full --learning_rate 2e-5 --finetune_steps 20000 --batch_size 32 --seq_len 128 --seed 42 --output_dir models/base_checkpoints/roberta_mnli",
    "python -u Codes/framework/run_quality_sweep.py --modules none --model models/gpt2 --dataset ag_news --finetune_mode full --learning_rate 2e-5 --finetune_steps 10000 --batch_size 32 --seq_len 128 --seed 42 --output_dir models/base_checkpoints/gpt2_agnews",
    "python -u Codes/framework/run_quality_sweep.py --modules none --model models/gpt2 --dataset mnli --finetune_mode full --learning_rate 2e-5 --finetune_steps 20000 --batch_size 32 --seq_len 128 --seed 42 --output_dir models/base_checkpoints/gpt2_mnli"
]

for cmd in commands:
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)
