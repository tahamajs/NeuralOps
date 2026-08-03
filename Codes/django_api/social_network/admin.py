from django.contrib import admin

from .models import (
    ActivityFeed,
    Discussion,
    DiscussionReply,
    Follow,
    GroupMembership,
    PublicationAlert,
    ResearchGroup,
)

admin.site.register(Follow)
admin.site.register(ResearchGroup)
admin.site.register(GroupMembership)
admin.site.register(Discussion)
admin.site.register(DiscussionReply)
admin.site.register(ActivityFeed)
admin.site.register(PublicationAlert)
