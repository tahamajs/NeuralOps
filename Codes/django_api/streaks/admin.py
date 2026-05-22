from django.contrib import admin
from .models import DailyActivity, Streak, WeeklySummary

admin.site.register(DailyActivity)
admin.site.register(Streak)
admin.site.register(WeeklySummary)