"""
GPU Marketplace — Core Models
Inspired by: Vast.ai (offer/instance model), ACE (spot market), Aethir (checker nodes)

Taxonomy:
    GPUModel → GPUListing (Offer) → ComputeOrder → ComputeJob (Instance) → JobEvent

Key concepts:
    - GPUListing: A GPU resource listed for rent (like Vast "offer")
    - ComputeOrder: A buy/sell order in the order book (like exchange order)
    - ComputeJob: A running/completed compute instance (like Vast "instance")
    - Bid: A price bid for auction-mode orders
    - PricingRule: Dynamic pricing configuration
    - GPUModelSpec: Reference specs for known GPU models
"""

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


# ═══════════════════════════════════════════════════════════════
# 1. GPU Model Reference Specs
# ═══════════════════════════════════════════════════════════════
class GPUModelSpec(models.Model):
    """Reference specifications for known GPU models (RTX 4090, A100, H100, etc.)."""

    name = models.CharField(max_length=100, unique=True)
    manufacturer = models.CharField(max_length=50, default="NVIDIA")
    vram_gb = models.IntegerField()
    architecture = models.CharField(max_length=50, blank=True)
    tf32_tflops = models.FloatField(null=True, blank=True)
    fp16_tflops = models.FloatField(null=True, blank=True)
    memory_bandwidth_gbs = models.FloatField(null=True, blank=True)
    tdp_watts = models.IntegerField(null=True, blank=True)
    market_price_usd = models.DecimalField(
        max_digits=8, decimal_places=2, null=True, blank=True
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["manufacturer", "name"]

    def __str__(self):
        return f"{self.name} ({self.vram_gb} GB)"


# ═══════════════════════════════════════════════════════════════
# 2. GPU Listing — The Offer (like Vast "offer" / "ask")
# ═══════════════════════════════════════════════════════════════
class GPUListing(models.Model):
    """A GPU resource listed for rent by a provider."""

    LISTING_TYPES = [
        ("on_demand", "On-Demand (guaranteed)"),
        ("spot", "Spot (interruptible)"),
        ("reserved", "Reserved (long-term)"),
    ]
    STATUS_CHOICES = [
        ("online", "Online — Ready"),
        ("offline", "Offline"),
        ("busy", "Busy — Computing"),
        ("reserved", "Reserved (Future)"),
        ("maintenance", "Maintenance"),
    ]
    VERIFICATION_LEVELS = [
        ("unverified", "Unverified"),
        ("benchmark_verified", "Benchmark Verified"),
        ("tee_verified", "TEE Verified"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    provider = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="gpu_listings"
    )
    gpu_model_spec = models.ForeignKey(
        GPUModelSpec,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="listings",
    )
    gpu_model = models.CharField(
        max_length=100, help_text="e.g., 'RTX 4090', 'A100-SXM4-80GB'"
    )
    gpu_count = models.PositiveIntegerField(default=1)
    vram_gb = models.IntegerField(help_text="VRAM per GPU in GB")
    cpu_cores = models.IntegerField(default=8)
    ram_gb = models.IntegerField(default=32)
    storage_gb = models.IntegerField(default=100)
    bandwidth_mbps = models.IntegerField(
        default=1000, help_text="Network bandwidth in Mbps"
    )
    location = models.CharField(
        max_length=200, blank=True, help_text="Data center / region"
    )
    country_code = models.CharField(max_length=2, blank=True)

    # Pricing
    listing_type = models.CharField(
        max_length=20, choices=LISTING_TYPES, default="on_demand"
    )
    base_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)
    min_price_per_hour = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )
    current_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)
    storage_price_per_gb_hour = models.DecimalField(
        max_digits=8, decimal_places=6, default=0
    )

    # Status & verification
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="online")
    verification_level = models.CharField(
        max_length=30, choices=VERIFICATION_LEVELS, default="unverified"
    )

    # Scores (like Vast DLPerf + Reliability)
    dlperf_score = models.FloatField(
        null=True, blank=True, help_text="Benchmark throughput score"
    )
    reliability_score = models.FloatField(default=60.0, help_text="0–100, starts at 60")
    uptime_percentage = models.FloatField(default=100.0)
    reputation_score = models.FloatField(default=5.0, help_text="0–10 scale")

    # Stats
    total_hours_rented = models.FloatField(default=0)
    total_earnings_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    completed_jobs = models.PositiveIntegerField(default=0)
    failed_jobs = models.PositiveIntegerField(default=0)
    total_rentals = models.PositiveIntegerField(default=0)

    # Scheduling
    available_from = models.DateTimeField(null=True, blank=True)
    available_until = models.DateTimeField(null=True, blank=True)
    max_rental_hours = models.PositiveIntegerField(
        default=72, help_text="Max continuous rental hours"
    )
    min_rental_hours = models.PositiveIntegerField(default=0)

    # Container
    docker_image_whitelist = models.JSONField(
        default=list, blank=True, help_text="Allowed Docker images"
    )
    supports_ssh = models.BooleanField(default=True)
    supports_jupyter = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-reliability_score", "-dlperf_score"]
        indexes = [
            models.Index(fields=["status", "gpu_model"]),
            models.Index(fields=["current_price_per_hour"]),
            models.Index(fields=["reliability_score"]),
        ]

    def __str__(self):
        return f"{self.gpu_model} ×{self.gpu_count} @ ${self.current_price_per_hour}/hr ({self.provider.username})"

    def update_dynamic_price(self):
        """Adjust price based on demand/supply factors."""
        base = float(self.base_price_per_hour)
        online_count = (
            GPUListing.objects.filter(gpu_model=self.gpu_model, status="online").count()
            or 1
        )
        demand_factor = ComputeOrder.objects.filter(
            gpu_model_preference__iexact=self.gpu_model, order_type="buy", status="open"
        ).count() / max(online_count, 1)
        factor = 1.0 + min(demand_factor * 0.2, 1.0)  # Max +100%
        new_price = max(float(self.min_price_per_hour or base * 0.5), base * factor)
        self.current_price_per_hour = round(new_price, 4)
        self.save(update_fields=["current_price_per_hour"])


# ═══════════════════════════════════════════════════════════════
# 3. Compute Order — The Order Book Entry
# ═══════════════════════════════════════════════════════════════
class ComputeOrder(models.Model):
    """A buy or sell order in the order book (like exchange order)."""

    ORDER_TYPES = [("buy", "Buy (need GPU)"), ("sell", "Sell (offer GPU)")]
    ORDER_MODES = [
        ("market", "Market — best available price"),
        ("limit", "Limit — specific price"),
        ("auction", "Auction — bidding"),
        ("future", "Future — reserve ahead"),
    ]
    STATUS_CHOICES = [
        ("open", "Open"),
        ("partially_filled", "Partially Filled"),
        ("filled", "Filled"),
        ("canceled", "Canceled"),
        ("expired", "Expired"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="compute_orders",
    )
    order_type = models.CharField(max_length=10, choices=ORDER_TYPES)
    order_mode = models.CharField(max_length=10, choices=ORDER_MODES, default="limit")

    # Requirements (for buy orders)
    gpu_model_preference = models.CharField(max_length=100, blank=True)
    min_vram_gb = models.IntegerField(null=True, blank=True)
    min_gpu_count = models.PositiveIntegerField(default=1)
    max_price_per_hour = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )
    required_hours = models.PositiveIntegerField(default=1)

    # Docker requirements
    docker_image = models.CharField(max_length=300, blank=True)
    command = models.TextField(blank=True)
    input_data_url = models.URLField(blank=True, max_length=1000)

    # Timing
    start_window = models.DateTimeField(null=True, blank=True)
    deadline = models.DateTimeField(null=True, blank=True)

    # Status & matching
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="open")
    matched_listing = models.ForeignKey(
        GPUListing,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="matched_orders",
    )
    final_price_per_hour = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    filled_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["order_type", "status"]),
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.order_type.upper()} {self.gpu_model_preference} @ ≤${self.max_price_per_hour}/hr ({self.status})"


# ═══════════════════════════════════════════════════════════════
# 4. Bid — Auction Bids
# ═══════════════════════════════════════════════════════════════
class Bid(models.Model):
    """A price bid for auction-mode orders."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    order = models.ForeignKey(
        ComputeOrder, on_delete=models.CASCADE, related_name="bids"
    )
    bidder = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="gpu_bids"
    )
    price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)
    listing = models.ForeignKey(
        GPUListing, on_delete=models.CASCADE, related_name="bids"
    )
    status = models.CharField(
        max_length=20,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("accepted", "Accepted"),
            ("rejected", "Rejected"),
        ],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["order", "listing", "bidder"]]
        ordering = ["-price_per_hour"]

    def __str__(self):
        return f"Bid ${self.price_per_hour}/hr on {self.order_id}"


# ═══════════════════════════════════════════════════════════════
# 5. Compute Job — The Running Instance (like Vast "instance")
# ═══════════════════════════════════════════════════════════════
class ComputeJob(models.Model):
    """A running or completed compute job on a rented GPU."""

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("provisioning", "Provisioning"),
        ("running", "Running"),
        ("paused", "Paused"),
        ("verifying", "Verifying"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("preempted", "Preempted"),
        ("disputed", "Disputed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    order = models.ForeignKey(
        ComputeOrder, on_delete=models.CASCADE, related_name="jobs"
    )
    listing = models.ForeignKey(
        GPUListing, on_delete=models.CASCADE, related_name="jobs"
    )
    consumer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="gpu_jobs"
    )
    provider = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="provided_jobs"
    )

    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")

    # Configuration
    docker_image = models.CharField(max_length=300)
    command = models.TextField()
    env_vars = models.JSONField(default=dict, blank=True)
    input_data_url = models.URLField(blank=True, max_length=1000)
    output_data_url = models.URLField(null=True, blank=True, max_length=1000)

    # Financials
    price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)
    hours_consumed = models.FloatField(default=0)
    total_cost_usd = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )
    storage_cost_usd = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    # Verification
    verification_proof = models.TextField(
        blank=True, help_text="SHA-256 hash of outputs"
    )
    tee_attestation = models.TextField(blank=True, help_text="TEE attestation report")
    verified_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="verified_jobs",
    )
    verified_at = models.DateTimeField(null=True, blank=True)

    # Timing
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["consumer", "-created_at"]),
            models.Index(fields=["provider", "-created_at"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"Job {self.id.hex[:8]} — {self.status} on {self.listing.gpu_model}"

    @property
    def duration_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        if self.started_at:
            return (timezone.now() - self.started_at).total_seconds()
        return 0

    def mark_completed(self, output_url=""):
        self.status = "completed"
        self.completed_at = timezone.now()
        self.hours_consumed = round(self.duration_seconds / 3600, 4)
        self.total_cost_usd = round(
            float(self.price_per_hour) * self.hours_consumed
            + float(self.storage_cost_usd),
            2,
        )
        if output_url:
            self.output_data_url = output_url
        self.save()

    def mark_failed(self, error_message=""):
        self.status = "failed"
        self.completed_at = timezone.now()
        self.hours_consumed = round(self.duration_seconds / 3600, 4)
        self.total_cost_usd = round(
            float(self.price_per_hour) * self.hours_consumed
            + float(self.storage_cost_usd),
            2,
        )
        self.save()


# ═══════════════════════════════════════════════════════════════
# 6. Job Event — Audit Trail
# ═══════════════════════════════════════════════════════════════
class JobEvent(models.Model):
    """An event in the lifecycle of a compute job."""

    EVENT_TYPES = [
        ("created", "Job Created"),
        ("matched", "Matched to Listing"),
        ("provisioning", "Provisioning Started"),
        ("running", "Job Started"),
        ("heartbeat", "Heartbeat"),
        ("paused", "Paused"),
        ("resumed", "Resumed"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("preempted", "Preempted"),
        ("disputed", "Disputed"),
        ("verified", "Verified"),
        ("payment_released", "Payment Released"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(ComputeJob, on_delete=models.CASCADE, related_name="events")
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    description = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.event_type}] Job {self.job_id}"


# ═══════════════════════════════════════════════════════════════
# 7. Pricing Rule — Dynamic Pricing Engine
# ═══════════════════════════════════════════════════════════════
class PricingRule(models.Model):
    """A dynamic pricing configuration for a GPU model."""

    gpu_model = models.CharField(max_length=100, unique=True)
    base_price_per_hour = models.DecimalField(max_digits=10, decimal_places=4)
    demand_multiplier = models.FloatField(
        default=0.2, help_text="Max % increase per unit demand"
    )
    supply_multiplier = models.FloatField(
        default=0.1, help_text="Decrease % per unit supply"
    )
    floor_price = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )
    ceiling_price = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )
    is_active = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Pricing for {self.gpu_model}"


# ═══════════════════════════════════════════════════════════════
# 8. Favorite Listing — Watchlist
# ═══════════════════════════════════════════════════════════════
class FavoriteListing(models.Model):
    """A user's saved/favorited GPU listing."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="favorite_listings",
    )
    listing = models.ForeignKey(
        GPUListing, on_delete=models.CASCADE, related_name="favorited_by"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "listing"]]


# gpu_marketplace/models.py – ADD BELOW EXISTING MODELS


class TradeOrder(models.Model):
    """A buy or sell order placed on the trading exchange."""

    ORDER_SIDE = [("buy", "Buy"), ("sell", "Sell")]
    ORDER_TYPE = [("limit", "Limit"), ("market", "Market")]
    STATUS = [
        ("open", "Open"),
        ("filled", "Filled"),
        ("partially_filled", "Partially Filled"),
        ("canceled", "Canceled"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="trade_orders"
    )
    gpu_model = models.CharField(max_length=100)
    side = models.CharField(max_length=10, choices=ORDER_SIDE)
    order_type = models.CharField(max_length=10, choices=ORDER_TYPE, default="limit")
    quantity = models.PositiveIntegerField(help_text="Number of GPU-hours")
    price = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Price per GPU-hour (required for limit orders)",
    )
    filled_quantity = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS, default="open")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["gpu_model", "status"]),
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.side.upper()} {self.quantity}h {self.gpu_model} @ {self.price}"


class Trade(models.Model):
    """A completed trade between a buyer and seller."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    gpu_model = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=4)
    quantity = models.PositiveIntegerField()
    buyer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="trades_bought"
    )
    seller = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="trades_sold"
    )
    buy_order = models.ForeignKey(
        TradeOrder,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="trades_as_buyer",
    )
    sell_order = models.ForeignKey(
        TradeOrder,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="trades_as_seller",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["gpu_model", "-created_at"]),
        ]


class Candlestick(models.Model):
    """OHLCV candlestick data for a GPU model (1-minute intervals)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    gpu_model = models.CharField(max_length=100)
    interval_start = models.DateTimeField(db_index=True)
    open_price = models.DecimalField(max_digits=10, decimal_places=4)
    high_price = models.DecimalField(max_digits=10, decimal_places=4)
    low_price = models.DecimalField(max_digits=10, decimal_places=4)
    close_price = models.DecimalField(max_digits=10, decimal_places=4)
    volume = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = [["gpu_model", "interval_start"]]
        ordering = ["interval_start"]


class Portfolio(models.Model):
    """A user's holdings of GPU-hours for a specific GPU model."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="portfolios"
    )
    gpu_model = models.CharField(max_length=100)
    quantity = models.IntegerField(
        default=0, help_text="Net GPU-hours held (positive = long)"
    )

    class Meta:
        unique_together = [["user", "gpu_model"]]

    def __str__(self):
        return f"{self.user.username} holds {self.quantity}h of {self.gpu_model}"
