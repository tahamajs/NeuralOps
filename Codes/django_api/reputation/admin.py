from django.contrib import admin

from .models import (Recommendation, ReputationEvent, ReputationScore,
                     ReputationSnapshot, ReputationWeight)


@admin.register(ReputationWeight)
class ReputationWeightAdmin(admin.ModelAdmin):
    list_display = (
        "OTHER_READS",
        "FULL_TEXT_READS",
        "RECOMMENDATIONS",
        "CITATIONS",
        "updated_at",
    )


@admin.register(ReputationScore)
class ReputationScoreAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "total_score",
        "percentile",
        "total_papers",
        "total_citations",
        "updated_at",
    )
    search_fields = ("user__username",)
    readonly_fields = ("total_score", "percentile", "updated_at", "created_at")


@admin.register(ReputationEvent)
class ReputationEventAdmin(admin.ModelAdmin):
    list_display = ("user", "event_type", "weighted_value", "paper", "created_at")
    list_filter = ("event_type",)
    search_fields = ("user__username",)


@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ("user", "paper", "score", "is_dismissed", "created_at")
    list_filter = ("is_dismissed",)


@admin.register(ReputationSnapshot)
class ReputationSnapshotAdmin(admin.ModelAdmin):
    list_display = ("user", "total_score", "percentile", "recorded_at")
    list_filter = ("recorded_at",)
