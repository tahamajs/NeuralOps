from django.contrib import admin

from .models import *


@admin.register(ReviewRequest)
class ReviewRequestAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "author",
        "submission_type",
        "status",
        "visibility",
        "created_at",
    )
    list_filter = ("status", "submission_type", "visibility")
    search_fields = ("title", "author__username")


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = (
        "review_request",
        "reviewer",
        "overall_score",
        "status",
        "created_at",
    )
    list_filter = ("status", "is_anonymous")


@admin.register(ReviewTemplate)
class ReviewTemplateAdmin(admin.ModelAdmin):
    list_display = ("name", "submission_type", "is_default", "usage_count")


@admin.register(ReviewBadge)
class ReviewBadgeAdmin(admin.ModelAdmin):
    list_display = ("name", "icon", "required_reviews")


@admin.register(ReviewerBadge)
class ReviewerBadgeAdmin(admin.ModelAdmin):
    list_display = ("reviewer", "badge", "awarded_at")
