# gpu_marketplace/models.py

from django.db import models
from django.conf import settings

class GPUListing(models.Model):
    """A GPU resource offered for rent."""
    provider = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                 related_name='gpu_listings')
    gpu_model = models.CharField(max_length=100)          # e.g., 'RTX 4090', 'A100'
    vram_gb = models.IntegerField()
    cpu_cores = models.IntegerField(default=8)
    ram_gb = models.IntegerField(default=32)
    storage_gb = models.IntegerField(default=100)

    # Pricing (per‑hour basis)
    base_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)   # in USD or stablecoin
    min_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)
    current_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)

    # Status & availability
    STATUS_CHOICES = [
        ('online', 'Online'),
        ('offline', 'Offline'),
        ('busy', 'Busy'),
        ('reserved', 'Reserved'),
        ('maintenance', 'Maintenance'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='online')

    total_hours_rented = models.FloatField(default=0)
    total_earnings_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    uptime_percentage = models.FloatField(default=100.0)
    reputation_score = models.FloatField(default=5.0)  # 0‑10 scale

    # Futures / scheduling
    available_from = models.DateTimeField(null=True, blank=True)
    available_until = models.DateTimeField(null=True, blank=True)
    max_continuous_hours = models.IntegerField(default=72)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class ComputeOrder(models.Model):
    """A buy or sell order in the order book."""
    ORDER_TYPES = [('buy', 'Buy'), ('sell', 'Sell')]
    ORDER_MODES = [('market', 'Market'), ('limit', 'Limit'), ('auction', 'Auction'), ('future', 'Future')]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='compute_orders')
    order_type = models.CharField(max_length=10, choices=ORDER_TYPES)
    order_mode = models.CharField(max_length=10, choices=ORDER_MODES)

    # For buy orders: requirements
    gpu_model_preference = models.CharField(max_length=100, blank=True)
    min_vram_gb = models.IntegerField(null=True, blank=True)
    max_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    required_hours = models.IntegerField()

    start_window = models.DateTimeField(null=True, blank=True)
    deadline = models.DateTimeField(null=True, blank=True)

    STATUS_CHOICES = [
        ('open', 'Open'),
        ('partially_filled', 'Partially Filled'),
        ('filled', 'Filled'),
        ('canceled', 'Canceled'),
        ('expired', 'Expired'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')

    # Matching result
    matched_listing = models.ForeignKey(GPUListing, on_delete=models.SET_NULL, null=True, blank=True)
    final_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

class ComputeJob(models.Model):
    """A running or completed compute job."""
    order = models.ForeignKey(ComputeOrder, on_delete=models.CASCADE, related_name='jobs')
    listing = models.ForeignKey(GPUListing, on_delete=models.CASCADE, related_name='jobs')

    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('verifying', 'Verifying'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('disputed', 'Disputed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')

    docker_image = models.CharField(max_length=300)
    command = models.TextField()
    input_data_url = models.URLField(blank=True)
    output_data_url = models.URLField(null=True, blank=True)

    price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)
    hours_consumed = models.FloatField(default=0)
    total_cost_usd = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)

    verification_proof = models.ForeignKey('gpu_validator.ProofOfCompute', on_delete=models.SET_NULL,
                                           null=True, blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)