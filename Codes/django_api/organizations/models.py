# organizations/models.py
"""
Complete Organizations & Teams System — Multi-Tenant SaaS for Research Labs

Architecture inspired by:
  - GitHub Organizations: Owner/Admin/Member roles, team hierarchy, audit log
  - GitLab Groups: hierarchical subgroups, Guest→Reporter→Dev→Maintainer→Owner
  - HuggingFace Resource Groups: fine‑grained repo‑level access control
  - Slack Unified Grid: multi‑workspace, shared DB with tenant isolation
  - WandB Teams: Organization → Team → Project scoping
  - Linear: sub‑teams, private teams, team‑specific workflows
  - Discord: role hierarchy (server > category > channel), override permissions
  - Notion Teamspaces: Open/Closed/Private visibility, inherited permissions
"""
import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator


# ═══════════════════════════════════════════════════════════════
# 1. ORGANIZATION — the top‑level tenant
# ═══════════════════════════════════════════════════════════════
class Organization(models.Model):
    """Top‑level tenant — a company, research lab, or university group."""
    TIER_CHOICES = [
        ('free', 'Free'),
        ('pro', 'Pro'),
        ('team', 'Team'),
        ('enterprise', 'Enterprise'),
    ]
    VISIBILITY_CHOICES = [
        ('public', 'Public'),
        ('private', 'Private'),
        ('hidden', 'Hidden'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=200, unique=True)
    display_name = models.CharField(max_length=300, blank=True)
    description = models.TextField(blank=True)
    logo = models.ImageField(upload_to='org_logos/', blank=True, null=True)
    website = models.URLField(blank=True)

    # Billing & tier
    tier = models.CharField(max_length=20, choices=TIER_CHOICES, default='free')
    max_members = models.PositiveIntegerField(default=10)
    max_teams = models.PositiveIntegerField(default=3)
    max_resource_groups = models.PositiveIntegerField(default=0, help_text="0 = disabled, enterprise only")
    storage_limit_gb = models.PositiveIntegerField(default=5)
    billing_email = models.EmailField(blank=True)

    # Visibility & access
    visibility = models.CharField(max_length=20, choices=VISIBILITY_CHOICES, default='private')
    require_sso = models.BooleanField(default=False)
    sso_provider = models.CharField(max_length=50, blank=True,
                                     help_text="'okta', 'azure', 'google', 'saml'")
    sso_metadata = models.JSONField(default=dict, blank=True)
    allowed_email_domains = models.JSONField(default=list, blank=True,
                                              help_text="e.g., ['university.edu', 'company.com']")

    # Stats
    member_count = models.PositiveIntegerField(default=0)
    team_count = models.PositiveIntegerField(default=0)
    experiment_count = models.PositiveIntegerField(default=0)

    # Soft delete
    is_active = models.BooleanField(default=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['tier']),
        ]

    def __str__(self):
        return self.display_name or self.name

    @property
    def owners(self):
        return self.memberships.filter(role='owner').select_related('user')

    @property
    def is_enterprise(self):
        return self.tier == 'enterprise'

    def soft_delete(self):
        self.is_active = False
        self.deleted_at = timezone.now()
        self.save(update_fields=['is_active', 'deleted_at'])


# ═══════════════════════════════════════════════════════════════
# 2. ROLE — configurable role definitions (Enterprise RBAC)
# ═══════════════════════════════════════════════════════════════
class Role(models.Model):
    """
    Configurable role definition — like GitHub custom roles.
    Built‑in: Owner, Admin, Member, Guest.
    Enterprise can define custom roles.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name='roles')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    is_builtin = models.BooleanField(default=False)
    permissions = models.JSONField(default=dict, blank=True,
                                    help_text="e.g., {'can_manage_members': true, 'can_create_teams': true}")

    class Meta:
        unique_together = [['organization', 'name']]
        ordering = ['organization', 'name']

    def __str__(self):
        return f"{self.name} @ {self.organization.name}"


# ═══════════════════════════════════════════════════════════════
# 3. MEMBERSHIP — user ↔ organization with a role
# ═══════════════════════════════════════════════════════════════
class Membership(models.Model):
    """A user's membership in an organization with a specific role."""
    BUILTIN_ROLES = [
        ('owner', 'Owner'),
        ('admin', 'Admin'),
        ('member', 'Member'),
        ('guest', 'Guest'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='org_memberships')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name='memberships')
    role = models.CharField(max_length=20, choices=BUILTIN_ROLES, default='member')
    custom_role = models.ForeignKey(Role, on_delete=models.SET_NULL, null=True, blank=True)
    is_default = models.BooleanField(default=False,
                                      help_text="The default org when a user has multiple")
    joined_at = models.DateTimeField(auto_now_add=True)
    invited_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                   null=True, blank=True, related_name='sent_invites')

    class Meta:
        unique_together = [['user', 'organization']]
        ordering = ['organization', 'user']
        indexes = [
            models.Index(fields=['user', 'organization']),
            models.Index(fields=['organization', 'role']),
        ]

    def __str__(self):
        return f"{self.user.username} → {self.organization.name} ({self.role})"

    @property
    def effective_permissions(self):
        if self.custom_role:
            return self.custom_role.permissions
        return BUILTIN_ROLE_PERMISSIONS.get(self.role, {})

    @property
    def is_owner(self): return self.role == 'owner'
    @property
    def is_admin(self): return self.role in ('owner', 'admin')
    @property
    def is_member(self): return self.role in ('owner', 'admin', 'member')


BUILTIN_ROLE_PERMISSIONS = {
    'owner': {
        'can_manage_members': True, 'can_manage_teams': True,
        'can_manage_billing': True, 'can_delete_org': True,
        'can_create_resource_groups': True, 'can_manage_sso': True,
        'can_view_audit_log': True, 'can_manage_roles': True,
        'can_manage_all_experiments': True, 'can_manage_all_papers': True,
    },
    'admin': {
        'can_manage_members': True, 'can_manage_teams': True,
        'can_manage_billing': False, 'can_delete_org': False,
        'can_create_resource_groups': True, 'can_manage_sso': False,
        'can_view_audit_log': True, 'can_manage_roles': False,
        'can_manage_all_experiments': True, 'can_manage_all_papers': True,
    },
    'member': {
        'can_manage_members': False, 'can_manage_teams': False,
        'can_manage_billing': False, 'can_delete_org': False,
        'can_create_resource_groups': False, 'can_manage_sso': False,
        'can_view_audit_log': False, 'can_manage_roles': False,
        'can_manage_all_experiments': False, 'can_manage_all_papers': False,
    },
    'guest': {
        'can_manage_members': False, 'can_manage_teams': False,
        'can_manage_billing': False, 'can_delete_org': False,
        'can_create_resource_groups': False, 'can_manage_sso': False,
        'can_view_audit_log': False, 'can_manage_roles': False,
        'can_manage_all_experiments': False, 'can_manage_all_papers': False,
    },
}


# ═══════════════════════════════════════════════════════════════
# 4. TEAM — a group within an organization (like GitHub team)
# ═══════════════════════════════════════════════════════════════
class Team(models.Model):
    """A named group within an organization — like GitHub Teams or Linear Teams."""
    VISIBILITY_CHOICES = [
        ('visible', 'Visible'),
        ('secret', 'Secret'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name='teams')
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200)
    description = models.TextField(blank=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True,
                               related_name='children',
                               help_text="Parent team for hierarchy (Linear‑style sub‑teams)")
    visibility = models.CharField(max_length=20, choices=VISIBILITY_CHOICES, default='visible')
    members = models.ManyToManyField(settings.AUTH_USER_MODEL, through='TeamMembership',
                                     related_name='teams')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['organization', 'slug']]
        ordering = ['organization', 'name']

    def __str__(self):
        return f"{self.organization.name}/{self.name}"

    @property
    def member_count(self):
        return self.team_memberships.count()


class TeamMembership(models.Model):
    """Membership of a user in a specific team."""
    TEAM_ROLES = [
        ('maintainer', 'Maintainer'),
        ('member', 'Member'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='team_memberships')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='team_memberships')
    role = models.CharField(max_length=20, choices=TEAM_ROLES, default='member')
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['team', 'user']]


# ═══════════════════════════════════════════════════════════════
# 5. INVITATION — pending invitation to join an org/team
# ═══════════════════════════════════════════════════════════════
class Invitation(models.Model):
    """Pending invitation to join an organization or team."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name='invitations')
    team = models.ForeignKey(Team, on_delete=models.CASCADE, null=True, blank=True,
                             related_name='invitations')
    email = models.EmailField()
    invited_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                   related_name='sent_org_invites')
    role = models.CharField(max_length=20, choices=Membership.BUILTIN_ROLES, default='member')
    token = models.CharField(max_length=64, unique=True)
    status = models.CharField(max_length=20, default='pending',
                               choices=[('pending','Pending'),('accepted','Accepted'),
                                        ('expired','Expired'),('revoked','Revoked')])
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    @property
    def is_expired(self):
        return timezone.now() > self.expires_at


# ═══════════════════════════════════════════════════════════════
# 6. RESOURCE GROUP — fine‑grained access (HuggingFace‑style)
# ═══════════════════════════════════════════════════════════════
class ResourceGroup(models.Model):
    """
    A group of resources (experiments, papers, models) with shared access.
    Like HuggingFace Resource Groups — enterprise feature.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name='resource_groups')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    teams = models.ManyToManyField(Team, blank=True, related_name='resource_groups')
    members = models.ManyToManyField(settings.AUTH_USER_MODEL, blank=True,
                                     related_name='resource_groups')
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['organization', 'name']]
        ordering = ['organization', 'name']

    def __str__(self):
        return f"RG: {self.organization.name}/{self.name}"


# ═══════════════════════════════════════════════════════════════
# 7. AUDIT LOG — compliance & activity tracking
# ═══════════════════════════════════════════════════════════════
class AuditLog(models.Model):
    """Immutable audit trail for organization compliance."""
    ACTION_CHOICES = [
        ('org_created', 'Organization Created'),
        ('org_updated', 'Organization Updated'),
        ('org_deleted', 'Organization Deleted'),
        ('member_added', 'Member Added'),
        ('member_removed', 'Member Removed'),
        ('member_role_changed', 'Member Role Changed'),
        ('team_created', 'Team Created'),
        ('team_deleted', 'Team Deleted'),
        ('team_member_added', 'Team Member Added'),
        ('team_member_removed', 'Team Member Removed'),
        ('invitation_sent', 'Invitation Sent'),
        ('invitation_accepted', 'Invitation Accepted'),
        ('rg_created', 'Resource Group Created'),
        ('rg_member_added', 'Resource Group Member Added'),
        ('sso_configured', 'SSO Configured'),
        ('billing_changed', 'Billing Changed'),
        ('permission_changed', 'Permission Changed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE,
                                     related_name='audit_logs')
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                              null=True, blank=True, related_name='audit_actions')
    action = models.CharField(max_length=30, choices=ACTION_CHOICES)
    target_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                    null=True, blank=True, related_name='audit_targets')
    description = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['organization', '-created_at']),
            models.Index(fields=['actor', '-created_at']),
        ]

    def __str__(self):
        return f"[{self.action}] {self.actor} @ {self.organization.name}"


# ═══════════════════════════════════════════════════════════════
# 8. BILLING — invoice & plan tracking
# ═══════════════════════════════════════════════════════════════
class Plan(models.Model):
    """A billing plan definition."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    tier = models.CharField(max_length=20, choices=Organization.TIER_CHOICES, default='free')
    price_monthly_cents = models.PositiveIntegerField(default=0)
    price_yearly_cents = models.PositiveIntegerField(default=0)
    max_members = models.PositiveIntegerField(default=10)
    max_teams = models.PositiveIntegerField(default=3)
    max_resource_groups = models.PositiveIntegerField(default=0)
    storage_limit_gb = models.PositiveIntegerField(default=5)
    includes_sso = models.BooleanField(default=False)
    includes_audit_log = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.tier})"


class Subscription(models.Model):
    """An organization's active subscription to a plan."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.OneToOneField(Organization, on_delete=models.CASCADE,
                                        related_name='subscription')
    plan = models.ForeignKey(Plan, on_delete=models.SET_NULL, null=True)
    status = models.CharField(max_length=20, default='active',
                               choices=[('active','Active'),('canceled','Canceled'),
                                        ('past_due','Past Due'),('trial','Trial')])
    trial_ends_at = models.DateTimeField(null=True, blank=True)
    current_period_start = models.DateTimeField(null=True, blank=True)
    current_period_end = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.organization.name} → {self.plan.name if self.plan else 'Free'}"
    

class LabProfile(models.Model):
    """Public‑facing lab profile."""
    organization = models.OneToOneField(Organization, on_delete=models.CASCADE, related_name='lab_profile')
    banner_image = models.ImageField(upload_to='lab_banners/', blank=True)
    research_focus = models.TextField(blank=True)
    publications_count = models.PositiveIntegerField(default=0)
    active_members_count = models.PositiveIntegerField(default=0)
    total_compute_hours = models.FloatField(default=0)
    showcase_experiments = models.ManyToManyField('experiment.TrainingRun', blank=True, related_name='lab_showcases')
    showcase_papers = models.ManyToManyField('paper.PaperProject', blank=True, related_name='lab_showcases')
    is_public = models.BooleanField(default=False)

class LabMetrics(models.Model):
    """Aggregated metrics for a lab."""
    lab = models.OneToOneField(LabProfile, on_delete=models.CASCADE, related_name='metrics')
    weekly_active_users = models.PositiveIntegerField(default=0)
    monthly_experiments = models.PositiveIntegerField(default=0)
    total_gpu_hours = models.FloatField(default=0)
    avg_reputation = models.FloatField(default=0)
    updated_at = models.DateTimeField(auto_now=True)