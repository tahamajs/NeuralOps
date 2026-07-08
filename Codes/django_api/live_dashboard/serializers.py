from rest_framework import serializers

from .models import Dashboard, DashboardWidget, SavedSQLQuery


class DashboardWidgetSerializer(serializers.ModelSerializer):
    class Meta:
        model = DashboardWidget
        fields = "__all__"
        read_only_fields = ["id", "dashboard", "created_at"]


class DashboardSerializer(serializers.ModelSerializer):
    widgets = DashboardWidgetSerializer(many=True, read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = Dashboard
        fields = [
            "id",
            "user",
            "username",
            "name",
            "description",
            "is_public",
            "widgets",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "username",
            "widgets",
            "created_at",
            "updated_at",
        ]


class SavedSQLQuerySerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = SavedSQLQuery
        fields = [
            "id",
            "user",
            "username",
            "name",
            "sql",
            "description",
            "is_public",
            "created_at",
        ]
        read_only_fields = ["id", "user", "username", "created_at"]
