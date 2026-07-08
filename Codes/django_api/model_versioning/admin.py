from django.contrib import admin

from .models import (ModelAccessControl, ModelApproval, ModelBranch,
                     ModelCommit, ModelLineage, ModelSecurityScan,
                     ModelUsageLog, ModelVersion, ModelVersionDiff)


# ----------------------------- Inlines -----------------------------
class ModelVersionDiffInline(admin.TabularInline):
    model = ModelVersionDiff
    fk_name = "from_version"
    extra = 0
    fields = ("to_version", "diff_size_mb", "original_size_saved_mb")
    readonly_fields = fields


class ModelLineageInline(admin.TabularInline):
    model = ModelLineage
    fk_name = "child"
    extra = 0
    fields = ("parent", "relationship_type", "notes")
    readonly_fields = fields


class ModelAccessControlInline(admin.TabularInline):
    model = ModelAccessControl
    extra = 0
    fields = ("user", "access_level", "granted_at", "expires_at")
    readonly_fields = ("granted_at",)


class ModelSecurityScanInline(admin.TabularInline):
    model = ModelSecurityScan
    extra = 0
    fields = ("scan_type", "status", "severity", "passed")
    readonly_fields = fields


class ModelUsageLogInline(admin.TabularInline):
    model = ModelUsageLog
    extra = 0
    fields = ("user", "action", "created_at")
    readonly_fields = fields


class ModelCommitInline(admin.TabularInline):
    model = ModelCommit
    fk_name = "branch"
    extra = 0
    fields = ("version", "parent", "merge_parent", "message")
    readonly_fields = ("created_at",)


# ----------------------------- Admins ------------------------------
@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "run",
        "version_number",
        "version",
        "artifact_size_mb",
        "compressed_size_mb",
        "storage_tier",
        "is_production",
        "is_published_to_hub",
        "validation_passed",
        "created_at",
    )
    list_filter = (
        "storage_tier",
        "is_production",
        "is_published_to_hub",
        "validation_passed",
        "compression_method",
    )
    search_fields = ("version", "run__name", "hub_repo_url")
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = [
        ModelVersionDiffInline,
        ModelLineageInline,
        ModelAccessControlInline,
        ModelSecurityScanInline,
        ModelUsageLogInline,
    ]
    fieldsets = (
        (
            None,
            {
                "fields": (
                    "id",
                    "run",
                    "version",
                    "version_number",
                    "is_production",
                    "is_published_to_hub",
                    "hub_repo_url",
                )
            },
        ),
        (
            "Artifact",
            {
                "fields": (
                    "artifact_path",
                    "compressed_artifact_path",
                    "artifact_size_mb",
                    "compressed_size_mb",
                    "compression_method",
                    "compression_config",
                )
            },
        ),
        (
            "Metrics & Validation",
            {
                "fields": (
                    "metrics",
                    "validation_passed",
                    "git_commit",
                    "config_snapshot_id",
                )
            },
        ),
        ("Storage", {"fields": ("storage_tier",)}),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
            },
        ),
    )


@admin.register(ModelVersionDiff)
class ModelVersionDiffAdmin(admin.ModelAdmin):
    list_display = ("id", "from_version", "to_version", "diff_size_mb", "created_at")
    search_fields = ("from_version__version", "to_version__version")
    readonly_fields = ("id", "created_at")


@admin.register(ModelLineage)
class ModelLineageAdmin(admin.ModelAdmin):
    list_display = ("id", "parent", "child", "relationship_type", "created_at")
    list_filter = ("relationship_type",)
    search_fields = ("parent__version", "child__version")
    readonly_fields = ("id", "created_at")


@admin.register(ModelApproval)
class ModelApprovalAdmin(admin.ModelAdmin):
    list_display = ("id", "version", "status", "requested_by", "reviewer", "created_at")
    list_filter = ("status",)
    search_fields = ("version__version", "requested_by__username")
    readonly_fields = ("created_at", "updated_at")


@admin.register(ModelAccessControl)
class ModelAccessControlAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "version",
        "user",
        "access_level",
        "granted_by",
        "granted_at",
        "expires_at",
    )
    list_filter = ("access_level",)
    search_fields = ("user__username", "version__version")
    readonly_fields = ("granted_at",)


@admin.register(ModelSecurityScan)
class ModelSecurityScanAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "version",
        "scan_type",
        "status",
        "severity",
        "passed",
        "scanned_at",
    )
    list_filter = ("scan_type", "status", "severity")
    search_fields = ("version__version",)
    readonly_fields = ("scanned_at",)


@admin.register(ModelUsageLog)
class ModelUsageLogAdmin(admin.ModelAdmin):
    list_display = ("id", "version", "user", "action", "created_at")
    list_filter = ("action",)
    search_fields = ("user__username", "version__version")
    readonly_fields = ("created_at",)


@admin.register(ModelBranch)
class ModelBranchAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "root_version", "created_by", "created_at")
    search_fields = ("name", "root_version__version")


@admin.register(ModelCommit)
class ModelCommitAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "branch",
        "version",
        "parent",
        "merge_parent",
        "created_by",
        "created_at",
    )
    search_fields = ("branch__name", "version__version")
