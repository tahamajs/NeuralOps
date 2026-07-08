from django.core.management.base import BaseCommand
from gpu_benchmark.models import BenchmarkSuite, BenchmarkTest

SUITES = [
    {
        "name": "DLPerf Standard v1.0",
        "suite_type": "training",
        "description": "Vast.ai-style composite deep learning benchmark",
        "version": "1.0",
        "tests": [
            {
                "name": "ResNet-50 Training",
                "category": "cnn",
                "model_name": "resnet50",
                "dataset_name": "ImageNet",
                "metric_name": "throughput",
                "metric_direction": "maximize",
                "default_batch_sizes": [32, 64, 128, 256],
                "default_precisions": ["fp32", "fp16"],
                "weight": 1.0,
            },
            {
                "name": "BERT-Large Training",
                "category": "transformer",
                "model_name": "bert-large-uncased",
                "dataset_name": "Wikipedia",
                "metric_name": "throughput",
                "metric_direction": "maximize",
                "default_batch_sizes": [16, 32, 64],
                "default_precisions": ["fp32", "fp16"],
                "weight": 1.5,
            },
            {
                "name": "GPT-2 Training",
                "category": "llm",
                "model_name": "gpt2",
                "dataset_name": "OpenWebText",
                "metric_name": "throughput",
                "metric_direction": "maximize",
                "default_batch_sizes": [4, 8, 16],
                "default_precisions": ["fp32", "fp16"],
                "weight": 2.0,
            },
        ],
    },
    {
        "name": "DLPerf Inference v1.0",
        "suite_type": "inference",
        "description": "ML inference throughput and latency benchmark",
        "version": "1.0",
        "tests": [
            {
                "name": "ResNet-50 Inference",
                "category": "cnn",
                "model_name": "resnet50",
                "dataset_name": "ImageNet",
                "metric_name": "throughput",
                "metric_direction": "maximize",
                "default_batch_sizes": [1, 8, 32, 64, 128, 256],
                "default_precisions": ["fp32", "fp16"],
                "weight": 1.0,
            },
            {
                "name": "BERT-Base Inference",
                "category": "transformer",
                "model_name": "bert-base-uncased",
                "dataset_name": "SQuAD",
                "metric_name": "throughput",
                "metric_direction": "maximize",
                "default_batch_sizes": [1, 8, 32, 64],
                "default_precisions": ["fp32", "fp16"],
                "weight": 1.0,
            },
            {
                "name": "Llama-2-7B Inference",
                "category": "llm",
                "model_name": "llama-2-7b",
                "dataset_name": "custom",
                "metric_name": "tokens_per_sec",
                "metric_direction": "maximize",
                "default_batch_sizes": [1, 4, 8, 16],
                "default_precisions": ["fp16", "bf16"],
                "weight": 2.5,
            },
        ],
    },
    {
        "name": "GPU Stress Test v1.0",
        "suite_type": "stress",
        "description": "Memory bandwidth, compute stress, and thermal tests",
        "version": "1.0",
        "tests": [
            {
                "name": "Memory Bandwidth Test",
                "category": "synthetic",
                "model_name": "cuda_bandwidth",
                "dataset_name": "N/A",
                "metric_name": "throughput",
                "metric_direction": "maximize",
                "default_batch_sizes": [1],
                "default_precisions": ["fp32"],
                "weight": 0.5,
            },
            {
                "name": "GEMM Compute Test",
                "category": "synthetic",
                "model_name": "cuda_gemm",
                "dataset_name": "N/A",
                "metric_name": "gflops",
                "metric_direction": "maximize",
                "default_batch_sizes": [1],
                "default_precisions": ["fp32", "fp16"],
                "weight": 0.5,
            },
        ],
    },
]


class Command(BaseCommand):
    help = "Seed benchmark suites and tests"

    def handle(self, *args, **options):
        for suite_data in SUITES:
            tests_data = suite_data.pop("tests", [])
            suite, created = BenchmarkSuite.objects.get_or_create(
                name=suite_data["name"], defaults=suite_data
            )
            for test_data in tests_data:
                BenchmarkTest.objects.get_or_create(
                    suite=suite, name=test_data["name"], defaults=test_data
                )
            status = "Created" if created else "Updated"
            self.stdout.write(
                f"{status} suite: {suite.name} with {len(tests_data)} tests"
            )
