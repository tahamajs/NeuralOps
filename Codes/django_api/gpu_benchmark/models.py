"""
GPU Benchmark Models — MLPerf-style standardized performance testing.

Architecture inspired by:
  - Vast.ai DLPerf: single composite score for cross-GPU comparison
  - MLPerf Training: time-to-accuracy for standard models (ResNet, BERT, GPT)
  - MLPerf Inference: throughput and latency benchmarks
  - NVIDIA DCGM Diagnostics: memory, stress, and interconnect tests

Taxonomy:
    BenchmarkSuite → BenchmarkTest → BenchmarkRun → BenchmarkResult → BenchmarkScore
"""
import uuid
from decimal import Decimal
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


class BenchmarkSuite(models.Model):
    """A named collection of benchmark tests (e.g., 'MLPerf Training v4.0', 'Vast DLPerf')."""
    SUITE_TYPES = [
        ('training', 'Training'),
        ('inference', 'Inference'),
        ('hpc', 'HPC / Scientific'),
        ('storage', 'Storage I/O'),
        ('stress', 'Stress Test'),
        ('custom', 'Custom'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    suite_type = models.CharField(max_length=20, choices=SUITE_TYPES, default='training')
    description = models.TextField(blank=True)
    version = models.CharField(max_length=20, default='1.0')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['suite_type', 'name']

    def __str__(self):
        return f"{self.name} v{self.version}"


class BenchmarkTest(models.Model):
    """A single benchmark test within a suite (e.g., 'ResNet-50 Training', 'BERT Inference')."""
    TEST_CATEGORIES = [
        ('cnn', 'CNN / Vision'),
        ('transformer', 'Transformer / NLP'),
        ('llm', 'Large Language Model'),
        ('recommendation', 'Recommendation System'),
        ('scientific', 'Scientific Computing'),
        ('synthetic', 'Synthetic / Stress'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    suite = models.ForeignKey(BenchmarkSuite, on_delete=models.CASCADE, related_name='tests')
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=30, choices=TEST_CATEGORIES, default='cnn')
    description = models.TextField(blank=True)

    # The model to benchmark
    model_name = models.CharField(max_length=200, help_text="e.g., 'resnet50', 'bert-base-uncased'")
    dataset_name = models.CharField(max_length=200, help_text="e.g., 'ImageNet', 'SQuAD v2'")

    # What we measure
    metric_name = models.CharField(max_length=50, default='throughput',
                                    choices=[('throughput','Throughput (samples/sec)'),
                                             ('latency','Latency (ms)'),
                                             ('time_to_accuracy','Time to Accuracy (sec)'),
                                             ('tokens_per_sec','Tokens/sec'),
                                             ('gflops','GFLOPs')])
    metric_direction = models.CharField(max_length=10, default='maximize',
                                         choices=[('maximize','Maximize'),('minimize','Minimize')])

    # Test parameters
    default_batch_sizes = models.JSONField(default=list,
                                            help_text="e.g., [1, 8, 16, 32, 64]")
    default_precisions = models.JSONField(default=list,
                                           help_text="e.g., ['fp32', 'fp16', 'bf16']")
    warmup_iterations = models.PositiveIntegerField(default=10)
    benchmark_iterations = models.PositiveIntegerField(default=50)
    timeout_seconds = models.PositiveIntegerField(default=600)
    docker_image = models.CharField(max_length=300, blank=True,
                                     help_text="Pre-built Docker image for this benchmark")

    # Weight in composite score
    weight = models.FloatField(default=1.0, validators=[MinValueValidator(0.0)])

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['suite', 'name']]
        ordering = ['suite', 'category', 'name']

    def __str__(self):
        return f"{self.name} ({self.suite.name})"


class BenchmarkRun(models.Model):
    """A single execution of a benchmark suite against a GPU listing."""
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('timed_out', 'Timed Out'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    listing = models.ForeignKey(
        'gpu_marketplace.GPUListing',
        on_delete=models.CASCADE,
        related_name='benchmark_runs'
    )
    suite = models.ForeignKey(BenchmarkSuite, on_delete=models.CASCADE, related_name='runs')
    initiated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='benchmark_runs'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    batch_sizes = models.JSONField(default=list, blank=True)
    precisions = models.JSONField(default=list, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['listing', '-created_at']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"Run {self.id.hex[:8]} on {self.listing.gpu_model} ({self.status})"

    @property
    def duration_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def composite_score(self):
        """Return the DLPerf-style composite score for this run."""
        results = self.results.filter(is_valid=True)
        if not results.exists():
            return None
        return self._compute_composite(results)

    def _compute_composite(self, results):
        """Weighted geometric mean of normalized scores."""
        import math
        total_weight = Decimal('0')
        weighted_sum = Decimal('0')
        for r in results:
            test = r.test
            if r.normalized_score is not None:
                weighted_sum += Decimal(str(r.normalized_score)) * Decimal(str(test.weight))
                total_weight += Decimal(str(test.weight))
        if total_weight == 0:
            return None
        return round(float(weighted_sum / total_weight), 2)


class BenchmarkResult(models.Model):
    """The result of a single benchmark test within a run."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(BenchmarkRun, on_delete=models.CASCADE, related_name='results')
    test = models.ForeignKey(BenchmarkTest, on_delete=models.CASCADE, related_name='results')
    listing = models.ForeignKey(
        'gpu_marketplace.GPUListing',
        on_delete=models.CASCADE,
        related_name='benchmark_results'
    )

    # Configuration
    batch_size = models.PositiveIntegerField()
    precision = models.CharField(max_length=10, default='fp32')

    # Raw metrics
    metric_value = models.FloatField(null=True, blank=True)
    metric_std_dev = models.FloatField(null=True, blank=True)
    peak_memory_mb = models.FloatField(null=True, blank=True)
    gpu_utilization_pct = models.FloatField(null=True, blank=True)
    cpu_utilization_pct = models.FloatField(null=True, blank=True)
    power_watts = models.FloatField(null=True, blank=True)

    # Validation
    is_valid = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)

    # Normalized score (0-100) for composite scoring
    normalized_score = models.FloatField(null=True, blank=True,
                                          validators=[MinValueValidator(0), MaxValueValidator(100)])

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['run', 'test', 'batch_size', 'precision']]
        ordering = ['test__category', 'test__name', 'batch_size']

    def __str__(self):
        return f"{self.test.name} @ b{self.batch_size} {self.precision}: {self.metric_value}"


class ListingBenchmarkScore(models.Model):
    """
    The current composite benchmark score for a GPU listing.
    Updated after every completed benchmark run. Like Vast.ai's DLPerf score.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    listing = models.OneToOneField(
        'gpu_marketplace.GPUListing',
        on_delete=models.CASCADE,
        related_name='benchmark_score'
    )
    # Composite DLPerf-style score (0-100 scale)
    composite_score = models.FloatField(null=True, blank=True)
    # Sub-scores per category
    training_score = models.FloatField(null=True, blank=True)
    inference_score = models.FloatField(null=True, blank=True)
    llm_score = models.FloatField(null=True, blank=True)
    # Best individual results
    best_throughput = models.FloatField(null=True, blank=True)
    best_latency_ms = models.FloatField(null=True, blank=True)
    best_tokens_per_sec = models.FloatField(null=True, blank=True)
    # Metadata
    last_run = models.ForeignKey(
        BenchmarkRun, on_delete=models.SET_NULL, null=True, blank=True
    )
    total_runs = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-composite_score']

    def __str__(self):
        return f"Score for {self.listing.gpu_model}: {self.composite_score}"

    def recalculate(self):
        """Recalculate the composite score from all historical results."""
        results = BenchmarkResult.objects.filter(
            listing=self.listing, is_valid=True
        ).select_related('test__suite')

        if not results.exists():
            self.composite_score = None
            self.save()
            return

        # Best result per test (max throughput or min latency)
        best_per_test = {}
        for r in results:
            key = f"{r.test_id}_{r.batch_size}_{r.precision}"
            if key not in best_per_test or (
                (r.test.metric_direction == 'maximize' and r.metric_value > best_per_test[key].metric_value)
                or (r.test.metric_direction == 'minimize' and r.metric_value < best_per_test[key].metric_value)
            ):
                best_per_test[key] = r

        # Category scores
        cats = {'cnn': [], 'transformer': [], 'llm': [], 'recommendation': [], 'scientific': []}
        for r in best_per_test.values():
            cats.get(r.test.category, []).append(r.normalized_score or r.metric_value or 0)

        self.composite_score = self._weighted_avg(best_per_test.values())
        self.training_score = self._avg(cats.get('cnn', []) + cats.get('transformer', []))
        self.inference_score = self._avg(cats.get('cnn', []))
        self.llm_score = self._avg(cats.get('llm', []))

        # Best individual metrics
        for r in best_per_test.values():
            if r.test.metric_name == 'throughput':
                self.best_throughput = max(self.best_throughput or 0, r.metric_value or 0)
            if r.test.metric_name == 'latency':
                self.best_latency_ms = min(self.best_latency_ms or float('inf'), r.metric_value or float('inf'))
            if r.test.metric_name == 'tokens_per_sec':
                self.best_tokens_per_sec = max(self.best_tokens_per_sec or 0, r.metric_value or 0)

        self.save()

    @staticmethod
    def _weighted_avg(results):
        total_w = Decimal('0'); total_v = Decimal('0')
        for r in results:
            w = Decimal(str(r.test.weight))
            v = Decimal(str(r.normalized_score or r.metric_value or 0))
            total_v += v * w; total_w += w
        return round(float(total_v / total_w), 2) if total_w else None

    @staticmethod
    def _avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else None