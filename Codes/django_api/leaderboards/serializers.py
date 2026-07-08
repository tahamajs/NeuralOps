# leaderboards/serializers.py
from django.db.models import F, Window
from django.db.models.functions import DenseRank
from rest_framework import serializers

from .models import *


class MetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = Metric
        fields = "__all__"


class BenchmarkTaskSerializer(serializers.ModelSerializer):
    dataset_count = serializers.SerializerMethodField()

    class Meta:
        model = BenchmarkTask
        fields = [
            "id",
            "name",
            "slug",
            "description",
            "category",
            "dataset_count",
            "created_at",
        ]

    def get_dataset_count(self, obj):
        return obj.datasets.count()


class BenchmarkDatasetSerializer(serializers.ModelSerializer):
    task_name = serializers.CharField(source="task.name", read_only=True)

    class Meta:
        model = BenchmarkDataset
        fields = "__all__"


class ScoreSerializer(serializers.ModelSerializer):
    metric_name = serializers.CharField(source="metric.name", read_only=True)
    dataset_name = serializers.CharField(source="dataset.name", read_only=True)

    class Meta:
        model = Score
        fields = [
            "id",
            "metric",
            "metric_name",
            "dataset",
            "dataset_name",
            "value",
            "is_public",
            "created_at",
        ]


class SubmissionSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    scores = ScoreSerializer(many=True, read_only=True)
    rank = serializers.SerializerMethodField()
    vote_summary = serializers.SerializerMethodField()

    class Meta:
        model = Submission
        fields = [
            "id",
            "benchmark",
            "user",
            "username",
            "team_name",
            "model_name",
            "model_description",
            "training_run",
            "architecture",
            "parameters_million",
            "hardware_used",
            "code_url",
            "paper_url",
            "huggingface_model_id",
            "status",
            "error_message",
            "public_score",
            "private_score",
            "composite_score",
            "rank",
            "scores",
            "vote_summary",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "status",
            "public_score",
            "private_score",
            "composite_score",
            "rank",
            "created_at",
            "updated_at",
        ]

    def get_rank(self, obj):
        return obj.rank

    def get_vote_summary(self, obj):
        ups = obj.votes.filter(direction="up").count()
        downs = obj.votes.filter(direction="down").count()
        return {"upvotes": ups, "downvotes": downs, "score": ups - downs}


class BenchmarkListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views."""

    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True
    )
    task_names = serializers.SerializerMethodField()
    primary_metric_name = serializers.CharField(
        source="primary_metric.name", read_only=True, default=""
    )
    submission_count = serializers.IntegerField(read_only=True)
    is_open = serializers.BooleanField(read_only=True)

    class Meta:
        model = Benchmark
        fields = [
            "id",
            "name",
            "slug",
            "description",
            "created_by",
            "created_by_username",
            "task_names",
            "primary_metric_name",
            "status",
            "visibility",
            "submission_count",
            "is_open",
            "opens_at",
            "closes_at",
            "created_at",
            "updated_at",
        ]

    def get_task_names(self, obj):
        return [t.name for t in obj.tasks.all()]


class BenchmarkDetailSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True
    )
    tasks_detail = BenchmarkTaskSerializer(source="tasks", many=True, read_only=True)
    datasets_detail = BenchmarkDatasetSerializer(
        source="datasets", many=True, read_only=True
    )
    metrics_detail = MetricSerializer(source="metrics", many=True, read_only=True)
    primary_metric_name = serializers.CharField(
        source="primary_metric.name", read_only=True, default=""
    )
    submission_count = serializers.IntegerField(read_only=True)
    is_open = serializers.BooleanField(read_only=True)
    leaderboard = serializers.SerializerMethodField()

    class Meta:
        model = Benchmark
        fields = [
            "id",
            "name",
            "slug",
            "description",
            "created_by",
            "created_by_username",
            "collaborators",
            "tasks",
            "tasks_detail",
            "datasets",
            "datasets_detail",
            "metrics",
            "metrics_detail",
            "primary_metric",
            "primary_metric_name",
            "status",
            "visibility",
            "submission_limit_per_user",
            "public_split_ratio",
            "opens_at",
            "closes_at",
            "final_results_published",
            "is_open",
            "submission_count",
            "leaderboard",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_by", "status", "created_at", "updated_at"]

    def get_leaderboard(self, obj):
        """Return the current leaderboard (top 50 ranked submissions)."""
        primary_metric = obj.primary_metric
        submissions = (
            obj.submissions.filter(status="completed")
            .annotate(
                rank=Window(
                    expression=DenseRank(), order_by=F("composite_score").desc()
                )
            )
            .order_by("rank")[:50]
        )
        return [
            {
                "rank": s.rank,
                "model_name": s.model_name,
                "username": s.user.username,
                "team_name": s.team_name or s.user.username,
                "composite_score": s.composite_score,
                "public_score": s.public_score,
                "parameters_million": s.parameters_million,
                "submission_id": str(s.id),
                "badges": [
                    b.badge.name for b in UserBadge.objects.filter(submission=s)
                ],
            }
            for s in submissions
        ]


class VoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vote
        fields = ["id", "submission", "user", "direction", "created_at"]
        read_only_fields = ["id", "user", "created_at"]


class BadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Badge
        fields = "__all__"


class UserBadgeSerializer(serializers.ModelSerializer):
    badge_detail = BadgeSerializer(source="badge", read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = UserBadge
        fields = [
            "id",
            "user",
            "username",
            "badge",
            "badge_detail",
            "submission",
            "awarded_at",
        ]
