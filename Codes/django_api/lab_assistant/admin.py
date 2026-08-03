from django.contrib import admin

from .models import (
    AIModelFeedback,
    AnnotationGuideline,
    AnnotationProject,
    AnnotationReview,
    DataTask,
)


@admin.register(AnnotationProject)
class AnnotationProjectAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "owner",
        "task_type",
        "status",
        "total_tasks",
        "completed_tasks",
        "created_at",
    )
    list_filter = ("task_type", "status")
    search_fields = ("name", "owner__username")


@admin.register(DataTask)
class DataTaskAdmin(admin.ModelAdmin):
    list_display = (
        "project",
        "task_index",
        "status",
        "annotated_by",
        "ai_confidence",
        "annotated_at",
    )
    list_filter = ("status",)
    search_fields = ("project__name",)


@admin.register(AnnotationReview)
class AnnotationReviewAdmin(admin.ModelAdmin):
    list_display = ("task", "reviewer", "status", "created_at")
    list_filter = ("status",)


@admin.register(AnnotationGuideline)
class AnnotationGuidelineAdmin(admin.ModelAdmin):
    list_display = ("project", "created_at", "updated_at")


@admin.register(AIModelFeedback)
class AIModelFeedbackAdmin(admin.ModelAdmin):
    list_display = ("task", "is_correct", "created_at")
    list_filter = ("is_correct",)
