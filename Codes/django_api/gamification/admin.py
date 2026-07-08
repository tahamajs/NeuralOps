from django.contrib import admin

from .models import (AchievementUnlock, Badge, ContributionEvent,
                     LeaderboardSnapshot, PremiumFeature, UserBadge,
                     UserStreak)

admin.site.register(Badge)
admin.site.register(UserBadge)
admin.site.register(ContributionEvent)
admin.site.register(UserStreak)
admin.site.register(LeaderboardSnapshot)
admin.site.register(AchievementUnlock)
admin.site.register(PremiumFeature)
