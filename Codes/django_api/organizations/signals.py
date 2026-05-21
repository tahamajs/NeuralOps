# signals.py
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Membership, Organization, Team, TeamMembership, AuditLog

@receiver(post_save, sender=Membership)
def update_org_member_count(sender, instance, created, **kwargs):
    if created:
        org = instance.organization
        org.member_count = org.memberships.count()
        org.save(update_fields=['member_count'])

@receiver(post_delete, sender=Membership)
def decrease_org_member_count(sender, instance, **kwargs):
    org = instance.organization
    org.member_count = org.memberships.count()
    org.save(update_fields=['member_count'])

@receiver(post_save, sender=Team)
def update_org_team_count(sender, instance, created, **kwargs):
    if created:
        org = instance.organization
        org.team_count = org.teams.count()
        org.save(update_fields=['team_count'])