from rest_framework import serializers
from .models import DashboardWidget

class DashboardWidgetSerializer(serializers.ModelSerializer):
    class Meta:
        model = DashboardWidget
        fields = '__all__'
        read_only_fields = ['id', 'user', 'created_at']


# web_dashboard/serializers.py – append after the existing DashboardWidgetSerializer

from rest_framework import serializers

# ── GPU Marketplace Serializers (for consumer dashboard) ─────────

class GPUListingSerializer(serializers.Serializer):
    """Lightweight listing for the consumer marketplace."""
    id = serializers.UUIDField()
    gpu_model = serializers.CharField()
    gpu_count = serializers.IntegerField()
    vram_gb = serializers.IntegerField()
    cpu_cores = serializers.IntegerField()
    ram_gb = serializers.IntegerField()
    storage_gb = serializers.IntegerField()
    location = serializers.CharField()
    listing_type = serializers.CharField()
    base_price_per_hour = serializers.DecimalField(max_digits=10, decimal_places=4)
    current_price_per_hour = serializers.DecimalField(max_digits=10, decimal_places=4)
    reliability_score = serializers.FloatField()
    dlperf_score = serializers.FloatField()
    total_rentals = serializers.IntegerField()
    provider = serializers.CharField(source='provider.username')
    provider_id = serializers.IntegerField(source='provider.id')


class ComputeOrderSerializer(serializers.Serializer):
    id = serializers.UUIDField(read_only=True)
    order_type = serializers.CharField()
    order_mode = serializers.CharField()
    gpu_model_preference = serializers.CharField(required=False, allow_blank=True)
    min_vram_gb = serializers.IntegerField(required=False)
    max_price_per_hour = serializers.DecimalField(max_digits=10, decimal_places=4, required=False)
    required_hours = serializers.IntegerField(required=False)
    docker_image = serializers.CharField(required=False, allow_blank=True)
    command = serializers.CharField(required=False, allow_blank=True)
    status = serializers.CharField(read_only=True)


class ComputeJobSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    status = serializers.CharField()
    listing_gpu_model = serializers.CharField(source='listing.gpu_model')
    price_per_hour = serializers.DecimalField(max_digits=10, decimal_places=4)
    hours_consumed = serializers.FloatField()
    total_cost_usd = serializers.DecimalField(max_digits=10, decimal_places=2)
    started_at = serializers.DateTimeField()
    completed_at = serializers.DateTimeField()


class EscrowBalanceSerializer(serializers.Serializer):
    balance_usd = serializers.DecimalField(max_digits=10, decimal_places=2)
    locked_balance_usd = serializers.DecimalField(max_digits=10, decimal_places=2)


class InvoiceSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    invoice_number = serializers.CharField()
    status = serializers.CharField()
    total_usd = serializers.DecimalField(max_digits=10, decimal_places=2)
    due_date = serializers.DateField()
    created_at = serializers.DateTimeField()