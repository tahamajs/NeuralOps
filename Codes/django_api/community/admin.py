from django.contrib import admin

from .models import *

admin.site.register(Tag)
admin.site.register(Question)
admin.site.register(Answer)
admin.site.register(Comment)
admin.site.register(Vote)
admin.site.register(Bookmark)
admin.site.register(Bounty)
admin.site.register(ReputationEvent)
admin.site.register(TrustLevel)
admin.site.register(Badge)
admin.site.register(UserBadge)
admin.site.register(Privilege)
admin.site.register(ActivityLog)
admin.site.register(DigestSubscription)
