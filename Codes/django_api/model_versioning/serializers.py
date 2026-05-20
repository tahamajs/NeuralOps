from rest_framework import serializers
from .models import ModelVersion, ModelVersionDiff, ModelLineage

class ModelVersionSerializer(serializers.ModelSerializer):
    # ── Read‑only related fields ──────────────────────────────────
    run_id = serializers.UUIDField(source='run.id', read_only=True)
    run_name = serializers.CharField(source='run.name', read_only=True)

    # ── Computed counts ───────────────────────────────────────────
    diffs_from_count = serializers.SerializerMethodField()
    diffs_to_count = serializers.SerializerMethodField()
    children_count = serializers.SerializerMethodField()

    # ── Human‑readable size ───────────────────────────────────────
    size_mb_display = serializers.SerializerMethodField()
    compressed_size_mb_display = serializers.SerializerMethodField()
    compression_ratio = serializers.SerializerMethodField()

    # ── Timestamps ─────────────────────────────────────────────────
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)

    class Meta:
        model = ModelVersion
        fields = [
            'id',
            'run_id',
            'run_name',
            'version',
            'version_number',
            'artifact_path',
            'compressed_artifact_path',
            'compression_method',
            'compression_config',
            'artifact_size_mb',
            'compressed_size_mb',
            'size_mb_display',
            'compressed_size_mb_display',
            'compression_ratio',
            'storage_tier',
            'is_published_to_hub',
            'hub_repo_url',
            'metrics',
            'git_commit',
            'config_snapshot_id',
            'validation_passed',
            'is_production',
            'diffs_from_count',
            'diffs_to_count',
            'children_count',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'run_id',
            'run_name',
            'created_at',
            'updated_at',
            'diffs_from_count',
            'diffs_to_count',
            'children_count',
            'size_mb_display',
            'compressed_size_mb_display',
            'compression_ratio',
        ]

    def get_diffs_from_count(self, obj):
        return obj.diffs_from.count()

    def get_diffs_to_count(self, obj):
        return obj.diffs_to.count()

    def get_children_count(self, obj):
        return obj.parents.count()  # related_name='parents' from ModelLineage child FK

    def get_size_mb_display(self, obj):
        if obj.artifact_size_mb < 1:
            return f"{obj.artifact_size_mb * 1024:.1f} KB"
        return f"{obj.artifact_size_mb:.1f} MB"

    def get_compressed_size_mb_display(self, obj):
        if not obj.compressed_size_mb:
            return None
        if obj.compressed_size_mb < 1:
            return f"{obj.compressed_size_mb * 1024:.1f} KB"
        return f"{obj.compressed_size_mb:.1f} MB"

    def get_compression_ratio(self, obj):
        if obj.artifact_size_mb and obj.compressed_size_mb:
            return round(obj.compressed_size_mb / obj.artifact_size_mb, 3)
        return None


class ModelVersionDiffSerializer(serializers.ModelSerializer):
    from_version_number = serializers.IntegerField(source='from_version.version_number', read_only=True)
    to_version_number = serializers.IntegerField(source='to_version.version_number', read_only=True)

    class Meta:
        model = ModelVersionDiff
        fields = [
            'id',
            'from_version',
            'to_version',
            'from_version_number',
            'to_version_number',
            'diff_artifact',
            'diff_size_mb',
            'original_size_saved_mb',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']