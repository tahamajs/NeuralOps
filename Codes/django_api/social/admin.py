from django.contrib import admin
from .models import Connection, ResearchGroup, GroupMembership, Discussion, DiscussionReply, ActivityFeed, PublicationAlert

admin.site.register(Connection)
admin.site.register(ResearchGroup)
admin.site.register(GroupMembership)
admin.site.register(Discussion)
admin.site.register(DiscussionReply)
admin.site.register(ActivityFeed)
admin.site.register(PublicationAlert)