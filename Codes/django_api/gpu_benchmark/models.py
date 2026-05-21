# gpu_benchmark/models.py
class GPUBenchmark(models.Model):
    listing = models.ForeignKey('gpu_marketplace.GPUListing', on_delete=models.CASCADE,
                                related_name='benchmarks')
    benchmark_type = models.CharField(max_length=50, default='mlperf')
    tf32_tflops = models.FloatField(null=True)
    memory_bandwidth_gbs = models.FloatField(null=True)
    inference_latency_ms = models.FloatField(null=True)
    training_throughput = models.FloatField(null=True)  # samples/sec
    verified_by = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='verified_benchmarks')
    created_at = models.DateTimeField(auto_now_add=True)