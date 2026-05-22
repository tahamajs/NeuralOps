from rest_framework import serializers
from .models import DailyActivity, Streak, WeeklySummary


class DailyActivitySerializer(serializers.ModelSerializer):
    class Meta:
        model = DailyActivity
        fields = [
            'date', 'total_count',
            'paper_reads', 'experiments_run', 'notes_created',
            'papers_written', 'code_commits', 'benchmark_submissions',
            'discussion_posts',
        ]


class HeatmapDataSerializer(serializers.Serializer):
    """Serializer for the frontend heatmap – returns date and count for a year."""
    date = serializers.DateField()
    count = serializers.IntegerField()


class StreakSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Streak
        fields = ['id', 'user', 'username', 'current_streak', 'longest_streak', 'last_active_date', 'updated_at']
        read_only_fields = ['id', 'user', 'username', 'current_streak', 'longest_streak', 'last_active_date', 'updated_at']


class WeeklySummarySerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = WeeklySummary
        fields = [
            'id', 'user', 'username', 'week_start', 'total_activities',
            'paper_reads', 'experiments_run', 'notes_created', 'papers_written',
            'most_active_day', 'summary_text', 'created_at',
        ]
        read_only_fields = ['id', 'created_at']