# paper/serializers.py
from rest_framework import serializers

from .models import (PaperComment, PaperCompile, PaperFile, PaperProject,
                     PaperShareLink, PaperTemplate, PaperVersion)


class PaperFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaperFile
        fields = [
            "id",
            "filename",
            "content",
            "binary",
            "is_binary",
            "version",
            "extension",
            "is_tex",
            "is_image",
            "is_bib",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "version",
            "extension",
            "is_tex",
            "is_image",
            "is_bib",
            "created_at",
            "updated_at",
        ]


class PaperCompileSerializer(serializers.ModelSerializer):
    initiated_by_username = serializers.CharField(
        source="initiated_by.username", read_only=True
    )

    class Meta:
        model = PaperCompile
        fields = [
            "id",
            "project",
            "initiated_by",
            "initiated_by_username",
            "status",
            "compiler",
            "root_file",
            "pdf",
            "log",
            "error_count",
            "warning_count",
            "started_at",
            "finished_at",
            "created_at",
            "duration",
        ]
        read_only_fields = [
            "id",
            "status",
            "pdf",
            "log",
            "error_count",
            "warning_count",
            "started_at",
            "finished_at",
            "created_at",
        ]


class PaperProjectListSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    file_count = serializers.IntegerField(read_only=True)
    last_compile_status = serializers.SerializerMethodField()

    class Meta:
        model = PaperProject
        fields = [
            "id",
            "name",
            "description",
            "owner",
            "owner_username",
            "root_file",
            "compiler",
            "version",
            "file_count",
            "last_compile_status",
            "created_at",
            "updated_at",
            "last_compiled_at",
        ]
        read_only_fields = [
            "id",
            "version",
            "owner",
            "created_at",
            "updated_at",
            "last_compiled_at",
        ]

    def get_last_compile_status(self, obj):
        last = obj.last_compile
        return last.status if last else None


class PaperProjectDetailSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    collaborator_usernames = serializers.SerializerMethodField()
    files = PaperFileSerializer(many=True, read_only=True)
    last_compiles = serializers.SerializerMethodField()

    class Meta:
        model = PaperProject
        fields = [
            "id",
            "name",
            "description",
            "owner",
            "owner_username",
            "collaborators",
            "collaborator_usernames",
            "root_file",
            "compiler",
            "draft_mode",
            "stop_on_first_error",
            "version",
            "git_remote",
            "git_branch",
            "files",
            "last_compiles",
            "created_at",
            "updated_at",
            "last_compiled_at",
        ]
        read_only_fields = [
            "id",
            "version",
            "owner",
            "files",
            "created_at",
            "updated_at",
            "last_compiled_at",
        ]

    def get_collaborator_usernames(self, obj):
        return [u.username for u in obj.collaborators.all()]

    def get_last_compiles(self, obj):
        compiles = obj.compiles.order_by("-created_at")[:5]
        return PaperCompileSerializer(compiles, many=True).data


class PaperTemplateSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)

    class Meta:
        model = PaperTemplate
        fields = [
            "id",
            "name",
            "description",
            "category",
            "files",
            "is_public",
            "author",
            "author_username",
            "usage_count",
            "created_at",
        ]
        read_only_fields = ["id", "author", "usage_count", "created_at"]


class CompileRequestSerializer(serializers.Serializer):
    compiler = serializers.ChoiceField(
        choices=["pdflatex", "xelatex", "lualatex"], required=False, default="pdflatex"
    )
    draft_mode = serializers.BooleanField(required=False, default=False)
    stop_on_first_error = serializers.BooleanField(required=False, default=False)


class FileCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaperFile
        fields = ["filename", "content"]


class PaperCommentSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)

    class Meta:
        model = PaperComment
        fields = [
            "id",
            "project",
            "file",
            "author",
            "author_username",
            "line_start",
            "line_end",
            "content",
            "is_resolved",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "author", "created_at", "updated_at"]


class PaperVersionSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True
    )

    class Meta:
        model = PaperVersion
        fields = [
            "id",
            "project",
            "version_number",
            "files_snapshot",
            "created_by",
            "created_by_username",
            "message",
            "created_at",
        ]
        read_only_fields = ["id", "created_by", "created_at"]


class PaperShareLinkSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaperShareLink
        fields = [
            "id",
            "project",
            "token",
            "password",
            "expires_at",
            "is_active",
            "view_count",
            "max_views",
            "created_at",
            "public_url",
        ]
        read_only_fields = ["id", "token", "view_count", "created_at", "public_url"]
