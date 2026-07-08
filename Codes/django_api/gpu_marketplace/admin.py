from django.contrib import admin

from .models import *


@admin.register(GPUModelSpec)
class GPUModelSpecAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "manufacturer",
        "vram_gb",
        "tf32_tflops",
        "market_price_usd",
    )
    list_filter = ("manufacturer",)
    search_fields = ("name",)


@admin.register(GPUListing)
class GPUListingAdmin(admin.ModelAdmin):
    list_display = (
        "gpu_model",
        "provider",
        "status",
        "current_price_per_hour",
        "reliability_score",
        "total_earnings_usd",
        "created_at",
    )
    list_filter = ("status", "listing_type", "verification_level", "gpu_model")
    search_fields = ("provider__username", "gpu_model", "location")
    readonly_fields = (
        "total_hours_rented",
        "total_earnings_usd",
        "completed_jobs",
        "failed_jobs",
        "total_rentals",
    )


@admin.register(ComputeOrder)
class ComputeOrderAdmin(admin.ModelAdmin):
    list_display = (
        "order_type",
        "order_mode",
        "gpu_model_preference",
        "status",
        "max_price_per_hour",
        "user",
        "created_at",
    )
    list_filter = ("order_type", "order_mode", "status")


@admin.register(ComputeJob)
class ComputeJobAdmin(admin.ModelAdmin):
    list_display = (
        "id_short",
        "status",
        "consumer",
        "provider",
        "listing",
        "price_per_hour",
        "total_cost_usd",
        "created_at",
    )
    list_filter = ("status",)
    search_fields = ("consumer__username", "provider__username")
    readonly_fields = ("hours_consumed", "total_cost_usd")

    def id_short(self, obj):
        return obj.id.hex[:8]


@admin.register(Bid)
class BidAdmin(admin.ModelAdmin):
    list_display = ("order", "bidder", "price_per_hour", "status", "created_at")


@admin.register(JobEvent)
class JobEventAdmin(admin.ModelAdmin):
    list_display = ("job", "event_type", "created_at")
    list_filter = ("event_type",)


@admin.register(PricingRule)
class PricingRuleAdmin(admin.ModelAdmin):
    list_display = (
        "gpu_model",
        "base_price_per_hour",
        "demand_multiplier",
        "is_active",
    )


@admin.register(FavoriteListing)
class FavoriteListingAdmin(admin.ModelAdmin):
    list_display = ("user", "listing", "created_at")
