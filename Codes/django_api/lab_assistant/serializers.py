from rest_framework import serializers

from .models import (AIModelFeedback, AnnotationGuideline, AnnotationProject,
                     AnnotationReview, DataTask)


class AnnotationProjectListSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    task_count = serializers.SerializerMethodField()
    completion_pct = serializers.SerializerMethodField()

    class Meta:
        model = AnnotationProject
        fields = [
            "id",
            "name",
            "description",
            "task_type",
            "owner",
            "owner_username",
            "status",
            "total_tasks",
            "completed_tasks",
            "task_count",
            "completion_pct",
            "labels",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "owner",
            "status",
            "total_tasks",
            "completed_tasks",
            "created_at",
            "updated_at",
        ]

    def get_task_count(self, obj):
        return obj.tasks.count()

    def get_completion_pct(self, obj):
        if obj.total_tasks == 0:
            return 0
        return round(obj.completed_tasks / obj.total_tasks * 100, 1)


class AnnotationProjectDetailSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    annotator_usernames = serializers.SerializerMethodField()

    class Meta:
        model = AnnotationProject
        fields = "__all__"
        read_only_fields = [
            "id",
            "owner",
            "total_tasks",
            "completed_tasks",
            "created_at",
            "updated_at",
        ]

    def get_annotator_usernames(self, obj):
        return [u.username for u in obj.annotators.all()]


class DataTaskSerializer(serializers.ModelSerializer):
    project_name = serializers.CharField(source="project.name", read_only=True)
    annotated_by_username = serializers.CharField(
        source="annotated_by.username", read_only=True
    )

    class Meta:
        model = DataTask
        fields = [
            "id",
            "project",
            "project_name",
            "data",
            "ai_predictions",
            "ai_confidence",
            "annotations",
            "annotated_by",
            "annotated_by_username",
            "status",
            "task_index",
            "is_ground_truth",
            "agreement_score",
            "created_at",
            "updated_at",
            "annotated_at",
        ]
        read_only_fields = [
            "id",
            "annotated_by",
            "annotated_by_username",
            "status",
            "created_at",
            "updated_at",
            "annotated_at",
        ]


class AnnotationReviewSerializer(serializers.ModelSerializer):
    reviewer_username = serializers.CharField(
        source="reviewer.username", read_only=True
    )

    class Meta:
        model = AnnotationReview
        fields = [
            "id",
            "task",
            "reviewer",
            "reviewer_username",
            "status",
            "comment",
            "created_at",
        ]
        read_only_fields = ["id", "reviewer", "created_at"]


class AnnotationGuidelineSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnnotationGuideline
        fields = ["id", "project", "content", "examples", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]


class AIModelFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIModelFeedback
        fields = [
            "id",
            "task",
            "ai_prediction",
            "human_correction",
            "is_correct",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]
