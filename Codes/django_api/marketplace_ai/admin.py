from django.contrib import admin

from .models import (ComparisonSet, ComplianceCheck, MarketplaceQuery,
                     SmartRecommendation)

admin.site.register(MarketplaceQuery)
admin.site.register(ComparisonSet)
admin.site.register(ComplianceCheck)
admin.site.register(SmartRecommendation)
