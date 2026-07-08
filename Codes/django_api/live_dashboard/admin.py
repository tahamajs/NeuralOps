from django.contrib import admin

from .models import Dashboard, DashboardWidget, SavedSQLQuery

admin.site.register(Dashboard)
admin.site.register(DashboardWidget)
admin.site.register(SavedSQLQuery)
