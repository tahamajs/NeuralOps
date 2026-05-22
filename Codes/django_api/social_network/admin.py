from django.contrib import admin
from .models import Follow, ResearchGroup, GroupMembership, Discussion, DiscussionReply, ActivityFeed, PublicationAlert

admin.site.register(Follow)
admin.site.register(ResearchGroup)
admin.site.register(GroupMembership)
admin.site.register(Discussion)
admin.site.register(DiscussionReply)
admin.site.register(ActivityFeed)
admin.site.register(PublicationAlert)