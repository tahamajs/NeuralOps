from rest_framework import serializers
from .models import DashboardWidget

class DashboardWidgetSerializer(serializers.ModelSerializer):
    class Meta:
        model = DashboardWidget
        fields = '__all__'
        read_only_fields = ['id', 'user', 'created_at']