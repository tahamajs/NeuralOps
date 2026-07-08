from rest_framework import serializers

from .models import (BenchmarkResult, BenchmarkRun, BenchmarkSuite,
                     BenchmarkTest, ListingBenchmarkScore)


class BenchmarkTestSerializer(serializers.ModelSerializer):
    suite_name = serializers.CharField(source="suite.name", read_only=True)

    class Meta:
        model = BenchmarkTest
        fields = [
            "id",
            "suite",
            "suite_name",
            "name",
            "category",
            "description",
            "model_name",
            "dataset_name",
            "metric_name",
            "metric_direction",
            "default_batch_sizes",
            "default_precisions",
            "warmup_iterations",
            "benchmark_iterations",
            "timeout_seconds",
            "docker_image",
            "weight",
            "is_active",
            "created_at",
        ]


class BenchmarkSuiteSerializer(serializers.ModelSerializer):
    tests = BenchmarkTestSerializer(many=True, read_only=True)
    test_count = serializers.SerializerMethodField()

    class Meta:
        model = BenchmarkSuite
        fields = [
            "id",
            "name",
            "suite_type",
            "description",
            "version",
            "is_active",
            "tests",
            "test_count",
            "created_at",
        ]

    def get_test_count(self, obj):
        return obj.tests.count()


class BenchmarkResultSerializer(serializers.ModelSerializer):
    test_name = serializers.CharField(source="test.name", read_only=True)
    test_category = serializers.CharField(source="test.category", read_only=True)

    class Meta:
        model = BenchmarkResult
        fields = [
            "id",
            "run",
            "test",
            "test_name",
            "test_category",
            "listing",
            "batch_size",
            "precision",
            "metric_value",
            "metric_std_dev",
            "peak_memory_mb",
            "gpu_utilization_pct",
            "cpu_utilization_pct",
            "power_watts",
            "is_valid",
            "error_message",
            "normalized_score",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class BenchmarkRunSerializer(serializers.ModelSerializer):
    listing_gpu_model = serializers.CharField(
        source="listing.gpu_model", read_only=True
    )
    suite_name = serializers.CharField(source="suite.name", read_only=True)
    results = BenchmarkResultSerializer(many=True, read_only=True)
    composite_score = serializers.FloatField(read_only=True)
    duration_seconds = serializers.FloatField(read_only=True)

    class Meta:
        model = BenchmarkRun
        fields = [
            "id",
            "listing",
            "listing_gpu_model",
            "suite",
            "suite_name",
            "initiated_by",
            "status",
            "batch_sizes",
            "precisions",
            "error_message",
            "started_at",
            "completed_at",
            "duration_seconds",
            "composite_score",
            "results",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "initiated_by",
            "status",
            "error_message",
            "started_at",
            "completed_at",
            "duration_seconds",
            "composite_score",
            "created_at",
        ]


class ListingBenchmarkScoreSerializer(serializers.ModelSerializer):
    listing_gpu_model = serializers.CharField(
        source="listing.gpu_model", read_only=True
    )
    provider_username = serializers.CharField(
        source="listing.provider.username", read_only=True
    )

    class Meta:
        model = ListingBenchmarkScore
        fields = [
            "id",
            "listing",
            "listing_gpu_model",
            "provider_username",
            "composite_score",
            "training_score",
            "inference_score",
            "llm_score",
            "best_throughput",
            "best_latency_ms",
            "best_tokens_per_sec",
            "total_runs",
            "updated_at",
        ]
        read_only_fields = ["id", "total_runs", "updated_at"]
