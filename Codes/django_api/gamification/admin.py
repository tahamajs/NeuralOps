from django.contrib import admin
from .models import Badge, UserBadge, ContributionEvent, UserStreak, LeaderboardSnapshot, AchievementUnlock, PremiumFeature

admin.site.register(Badge)
admin.site.register(UserBadge)
admin.site.register(ContributionEvent)
admin.site.register(UserStreak)
admin.site.register(LeaderboardSnapshot)
admin.site.register(AchievementUnlock)
admin.site.register(PremiumFeature)