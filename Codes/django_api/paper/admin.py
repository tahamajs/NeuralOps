from django.contrib import admin

from .models import PaperCompile, PaperFile, PaperProject, PaperTemplate


class PaperFileInline(admin.TabularInline):
    model = PaperFile
    extra = 0
    readonly_fields = ["version", "created_at", "updated_at"]
    fields = ["filename", "is_binary", "version", "created_at"]


class PaperCompileInline(admin.TabularInline):
    model = PaperCompile
    extra = 0
    readonly_fields = [
        "status",
        "compiler",
        "error_count",
        "warning_count",
        "started_at",
        "finished_at",
        "created_at",
    ]
    fields = ["status", "compiler", "error_count", "warning_count", "started_at"]


@admin.register(PaperProject)
class PaperProjectAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "owner",
        "compiler",
        "file_count",
        "last_compiled_at",
        "updated_at",
    ]
    list_filter = ["compiler", "created_at"]
    search_fields = ["name", "description", "owner__username"]
    readonly_fields = ["id", "version", "created_at", "updated_at"]
    inlines = [PaperFileInline, PaperCompileInline]


@admin.register(PaperFile)
class PaperFileAdmin(admin.ModelAdmin):
    list_display = ["filename", "project", "version", "updated_at"]
    list_filter = ["project"]
    search_fields = ["filename", "project__name"]
    readonly_fields = ["id", "version", "created_at", "updated_at"]


@admin.register(PaperCompile)
class PaperCompileAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "project",
        "status",
        "compiler",
        "error_count",
        "warning_count",
        "created_at",
    ]
    list_filter = ["status", "compiler"]
    readonly_fields = [
        "id",
        "status",
        "log",
        "error_count",
        "warning_count",
        "started_at",
        "finished_at",
        "created_at",
    ]


@admin.register(PaperTemplate)
class PaperTemplateAdmin(admin.ModelAdmin):
    list_display = ["name", "category", "is_public", "usage_count", "created_at"]
    list_filter = ["category", "is_public"]
    search_fields = ["name", "description"]
