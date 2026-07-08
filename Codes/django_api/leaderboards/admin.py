# leaderboards/admin.py
from django.contrib import admin

from .models import *


@admin.register(BenchmarkTask)
class BenchmarkTaskAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "created_at")
    list_filter = ("category",)
    search_fields = ("name",)
    prepopulated_fields = {"slug": ("name",)}


@admin.register(BenchmarkDataset)
class BenchmarkDatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "task", "train_size", "test_size")
    list_filter = ("task",)
    search_fields = ("name",)
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Metric)
class MetricAdmin(admin.ModelAdmin):
    list_display = ("name", "direction", "is_primary", "unit")
    list_filter = ("direction", "is_primary")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Benchmark)
class BenchmarkAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "status",
        "visibility",
        "created_by",
        "submission_count",
        "created_at",
    )
    list_filter = ("status", "visibility")
    search_fields = ("name", "description")
    prepopulated_fields = {"slug": ("name",)}
    filter_horizontal = ("collaborators",)


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "benchmark",
        "user",
        "status",
        "composite_score",
        "created_at",
    )
    list_filter = ("status", "benchmark")
    search_fields = ("model_name", "user__username")


@admin.register(Vote)
class VoteAdmin(admin.ModelAdmin):
    list_display = ("submission", "user", "direction", "created_at")


@admin.register(Badge)
class BadgeAdmin(admin.ModelAdmin):
    list_display = ("name", "icon", "benchmark")


@admin.register(UserBadge)
class UserBadgeAdmin(admin.ModelAdmin):
    list_display = ("user", "badge", "awarded_at")


@admin.register(BenchmarkFollow)
class BenchmarkFollowAdmin(admin.ModelAdmin):
    list_display = ("user", "benchmark", "notify_on_new_submission", "created_at")
