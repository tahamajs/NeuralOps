from django.db.models import Avg
from rest_framework import serializers

from .models import (Bid, ComputeJob, ComputeOrder, FavoriteListing,
                     GPUListing, GPUModelSpec, JobEvent, PricingRule)


class GPUModelSpecSerializer(serializers.ModelSerializer):
    listing_count = serializers.SerializerMethodField()
    avg_price = serializers.SerializerMethodField()

    class Meta:
        model = GPUModelSpec
        fields = [
            "id",
            "name",
            "manufacturer",
            "vram_gb",
            "architecture",
            "tf32_tflops",
            "fp16_tflops",
            "memory_bandwidth_gbs",
            "tdp_watts",
            "market_price_usd",
            "listing_count",
            "avg_price",
        ]

    def get_listing_count(self, obj):
        return obj.listings.filter(status="online").count()

    def get_avg_price(self, obj):
        avg = obj.listings.filter(status="online").aggregate(
            a=Avg("current_price_per_hour")
        )["a"]
        return round(float(avg), 4) if avg else None


class GPUListingSerializer(serializers.ModelSerializer):
    provider_username = serializers.CharField(
        source="provider.username", read_only=True
    )
    provider_reputation = serializers.SerializerMethodField()
    gpu_model_spec_detail = GPUModelSpecSerializer(
        source="gpu_model_spec", read_only=True
    )

    class Meta:
        model = GPUListing
        fields = [
            "id",
            "provider",
            "provider_username",
            "provider_reputation",
            "gpu_model_spec",
            "gpu_model_spec_detail",
            "gpu_model",
            "gpu_count",
            "vram_gb",
            "cpu_cores",
            "ram_gb",
            "storage_gb",
            "bandwidth_mbps",
            "location",
            "country_code",
            "listing_type",
            "base_price_per_hour",
            "min_price_per_hour",
            "current_price_per_hour",
            "storage_price_per_gb_hour",
            "status",
            "verification_level",
            "dlperf_score",
            "reliability_score",
            "uptime_percentage",
            "reputation_score",
            "total_hours_rented",
            "total_earnings_usd",
            "completed_jobs",
            "failed_jobs",
            "total_rentals",
            "available_from",
            "available_until",
            "max_rental_hours",
            "supports_ssh",
            "supports_jupyter",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "provider",
            "provider_username",
            "provider_reputation",
            "gpu_model_spec_detail",
            "status",
            "dlperf_score",
            "reliability_score",
            "uptime_percentage",
            "reputation_score",
            "total_hours_rented",
            "total_earnings_usd",
            "completed_jobs",
            "failed_jobs",
            "total_rentals",
            "created_at",
            "updated_at",
        ]

    def get_provider_reputation(self, obj):
        try:
            return obj.provider.reputation_score.provider_score
        except:
            return 5.0


class GPUListingCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = GPUListing
        fields = [
            "gpu_model",
            "gpu_count",
            "vram_gb",
            "cpu_cores",
            "ram_gb",
            "storage_gb",
            "bandwidth_mbps",
            "location",
            "country_code",
            "listing_type",
            "base_price_per_hour",
            "min_price_per_hour",
            "max_rental_hours",
            "min_rental_hours",
            "supports_ssh",
            "supports_jupyter",
            "available_from",
            "available_until",
        ]


class ComputeOrderSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    bid_count = serializers.SerializerMethodField()

    class Meta:
        model = ComputeOrder
        fields = [
            "id",
            "user",
            "username",
            "order_type",
            "order_mode",
            "gpu_model_preference",
            "min_vram_gb",
            "min_gpu_count",
            "max_price_per_hour",
            "required_hours",
            "docker_image",
            "command",
            "input_data_url",
            "start_window",
            "deadline",
            "status",
            "matched_listing",
            "final_price_per_hour",
            "bid_count",
            "created_at",
            "updated_at",
            "filled_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "status",
            "matched_listing",
            "final_price_per_hour",
            "filled_at",
            "created_at",
            "updated_at",
        ]

    def get_bid_count(self, obj):
        return obj.bids.count()


class ComputeJobSerializer(serializers.ModelSerializer):
    consumer_username = serializers.CharField(
        source="consumer.username", read_only=True
    )
    provider_username = serializers.CharField(
        source="provider.username", read_only=True
    )
    listing_gpu_model = serializers.CharField(
        source="listing.gpu_model", read_only=True
    )
    events = serializers.SerializerMethodField()

    class Meta:
        model = ComputeJob
        fields = [
            "id",
            "order",
            "listing",
            "listing_gpu_model",
            "consumer",
            "consumer_username",
            "provider",
            "provider_username",
            "status",
            "docker_image",
            "command",
            "env_vars",
            "input_data_url",
            "output_data_url",
            "price_per_hour",
            "hours_consumed",
            "total_cost_usd",
            "storage_cost_usd",
            "verification_proof",
            "tee_attestation",
            "verified_by",
            "verified_at",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
            "events",
        ]
        read_only_fields = [
            "id",
            "consumer",
            "provider",
            "listing",
            "status",
            "hours_consumed",
            "total_cost_usd",
            "verified_by",
            "verified_at",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
        ]

    def get_events(self, obj):
        return JobEventSerializer(obj.events.all()[:20], many=True).data


class JobEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobEvent
        fields = ["id", "job", "event_type", "description", "metadata", "created_at"]


class BidSerializer(serializers.ModelSerializer):
    bidder_username = serializers.CharField(source="bidder.username", read_only=True)
    listing_gpu_model = serializers.CharField(
        source="listing.gpu_model", read_only=True
    )

    class Meta:
        model = Bid
        fields = [
            "id",
            "order",
            "bidder",
            "bidder_username",
            "price_per_hour",
            "listing",
            "listing_gpu_model",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "bidder", "status", "created_at"]


class PricingRuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = PricingRule
        fields = "__all__"


class FavoriteListingSerializer(serializers.ModelSerializer):
    listing_detail = GPUListingSerializer(source="listing", read_only=True)

    class Meta:
        model = FavoriteListing
        fields = ["id", "user", "listing", "listing_detail", "created_at"]
        read_only_fields = ["id", "user", "created_at"]
