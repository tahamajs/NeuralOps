from django.urls import path
from . import views

urlpatterns = [
    # ── Organizations ─────────────────────────────────────────
    path('orgs/', views.organization_list, name='org_list'),
    path('orgs/<uuid:org_id>/', views.organization_detail, name='org_detail'),

    # ── Members ───────────────────────────────────────────────
    path('orgs/<uuid:org_id>/members/', views.member_list, name='member_list'),
    path('orgs/<uuid:org_id>/members/<int:user_id>/role/', views.change_member_role, name='change_member_role'),
    path('orgs/<uuid:org_id>/members/<int:user_id>/remove/', views.remove_member, name='remove_member'),
    path('orgs/<uuid:org_id>/leave/', views.leave_organization, name='leave_org'),

    # ── Invitations ───────────────────────────────────────────
    path('orgs/<uuid:org_id>/invite/', views.invite_member, name='invite_member'),
    path('invitations/', views.my_invitations, name='my_invitations'),
    path('invitations/<uuid:invitation_id>/accept/', views.accept_invitation, name='accept_invitation'),

    # ── Teams ─────────────────────────────────────────────────
    path('orgs/<uuid:org_id>/teams/', views.team_list, name='team_list'),
    path('orgs/<uuid:org_id>/teams/<uuid:team_id>/', views.team_detail, name='team_detail'),
    path('orgs/<uuid:org_id>/teams/<uuid:team_id>/members/', views.add_team_member, name='add_team_member'),
    path('orgs/<uuid:org_id>/teams/<uuid:team_id>/members/<int:user_id>/', views.remove_team_member, name='remove_team_member'),

    # ── Experiments ───────────────────────────────────────────
    path('orgs/<uuid:org_id>/experiments/', views.org_experiments, name='org_experiments'),

    # ── Audit Log ─────────────────────────────────────────────
    path('orgs/<uuid:org_id>/audit-log/', views.audit_log, name='audit_log'),

    # ── Resource Groups (Enterprise) ──────────────────────────
    path('orgs/<uuid:org_id>/resource-groups/', views.resource_group_list, name='resource_group_list'),

    # ── Roles ─────────────────────────────────────────────────
    path('orgs/<uuid:org_id>/roles/', views.role_list, name='role_list'),

    # ── Billing ───────────────────────────────────────────────
    path('orgs/<uuid:org_id>/subscription/', views.subscription_detail, name='subscription_detail'),
    path('plans/', views.plan_list, name='plan_list'),
]