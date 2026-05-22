from django.contrib import admin
from .models import MarketplaceQuery, ComparisonSet, ComplianceCheck, SmartRecommendation

admin.site.register(MarketplaceQuery)
admin.site.register(ComparisonSet)
admin.site.register(ComplianceCheck)
admin.site.register(SmartRecommendation)