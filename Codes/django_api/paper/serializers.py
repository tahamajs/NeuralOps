from rest_framework import serializers
from .models import PaperProject, PaperFile, PaperCompile, PaperTemplate


class PaperFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaperFile
        fields = [
            'id', 'filename', 'content', 'binary', 'is_binary',
            'version', 'extension', 'is_tex', 'is_image', 'is_bib',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'version', 'extension', 'is_tex',
                            'is_image', 'is_bib', 'created_at', 'updated_at']


class PaperCompileSerializer(serializers.ModelSerializer):
    initiated_by_username = serializers.CharField(
        source='initiated_by.username', read_only=True
    )

    class Meta:
        model = PaperCompile
        fields = [
            'id', 'project', 'initiated_by', 'initiated_by_username',
            'status', 'compiler', 'root_file',
            'pdf', 'log', 'error_count', 'warning_count',
            'started_at', 'finished_at', 'created_at', 'duration',
        ]
        read_only_fields = ['id', 'status', 'pdf', 'log', 'error_count',
                            'warning_count', 'started_at', 'finished_at',
                            'created_at']


class PaperProjectListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views."""
    owner_username = serializers.CharField(source='owner.username', read_only=True)
    file_count = serializers.IntegerField(read_only=True)
    last_compile_status = serializers.SerializerMethodField()

    class Meta:
        model = PaperProject
        fields = [
            'id', 'name', 'description', 'owner', 'owner_username',
            'root_file', 'compiler', 'version',
            'file_count', 'last_compile_status',
            'created_at', 'updated_at', 'last_compiled_at',
        ]
        read_only_fields = ['id', 'version', 'owner', 'created_at',
                            'updated_at', 'last_compiled_at']

    def get_last_compile_status(self, obj):
        last = obj.last_compile
        return last.status if last else None


class PaperProjectDetailSerializer(serializers.ModelSerializer):
    """Full serializer including files."""
    owner_username = serializers.CharField(source='owner.username', read_only=True)
    collaborator_usernames = serializers.SerializerMethodField()
    files = PaperFileSerializer(many=True, read_only=True)
    last_compiles = serializers.SerializerMethodField()

    class Meta:
        model = PaperProject
        fields = [
            'id', 'name', 'description', 'owner', 'owner_username',
            'collaborators', 'collaborator_usernames',
            'root_file', 'compiler', 'draft_mode', 'stop_on_first_error',
            'version', 'git_remote', 'git_branch',
            'files', 'last_compiles',
            'created_at', 'updated_at', 'last_compiled_at',
        ]
        read_only_fields = ['id', 'version', 'owner', 'files',
                            'created_at', 'updated_at', 'last_compiled_at']

    def get_collaborator_usernames(self, obj):
        return [u.username for u in obj.collaborators.all()]

    def get_last_compiles(self, obj):
        compiles = obj.compiles.order_by('-created_at')[:5]
        return PaperCompileSerializer(compiles, many=True).data


class PaperTemplateSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(
        source='author.username', read_only=True
    )

    class Meta:
        model = PaperTemplate
        fields = [
            'id', 'name', 'description', 'category',
            'files', 'is_public', 'author', 'author_username',
            'usage_count', 'created_at',
        ]
        read_only_fields = ['id', 'author', 'usage_count', 'created_at']


class CompileRequestSerializer(serializers.Serializer):
    compiler = serializers.ChoiceField(
        choices=['pdflatex', 'xelatex', 'lualatex'],
        required=False,
        default='pdflatex',
    )
    draft_mode = serializers.BooleanField(required=False, default=False)
    stop_on_first_error = serializers.BooleanField(required=False, default=False)


class FileCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaperFile
        fields = ['filename', 'content']