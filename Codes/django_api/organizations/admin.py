from django.contrib import admin
from .models import *
for m in [Organization, Role, Membership, Team, TeamMembership, Invitation,
           ResourceGroup, AuditLog, Plan, Subscription]:
    admin.site.register(m)
