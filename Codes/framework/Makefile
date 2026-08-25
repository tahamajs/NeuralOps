PYTHONPATH := $(shell cd ../..; pwd)/Codes
export PYTHONPATH

.PHONY: prepare fast suite report list test smoke

prepare:
	python3 data/prepare_datasets.py --tasks sst2,mrpc,qnli,cola --save_dir ./prepared_benchmark_data

fast:
	python3 launch_gpu_server.py --mode fast --precision bf16 --batch_size 16

suite:
	bash run_on_gpu_server.sh suite

report:
	python3 auto_run.py --report

list:
	python3 main.py list

test:
	PYTHONPATH=../../Codes pytest tests/ -v --timeout=120

smoke:
	PYTHONPATH=../../Codes python3 check_framework_sync.py
