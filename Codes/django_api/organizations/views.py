# organizations/views.py
"""Organizations REST API — complete multi‑tenant team management."""

import logging
import secrets
from datetime import timedelta

from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


def _get_org(org_id):
    return get_object_or_404(Organization, id=org_id, is_active=True)


def _get_membership(org, user):
    return get_object_or_404(org.memberships, user=user)


def _can(user, org, perm):
    return _get_membership(org, user).effective_permissions.get(perm, False)


def _can_or_403(user, org, perm):
    if not _can(user, org, perm):
        from django.core.exceptions import PermissionDenied

        raise PermissionDenied(f"Missing permission: {perm}")


def _log(
    org, actor, action, target_user=None, description="", metadata=None, request=None
):
    AuditLog.objects.create(
        organization=org,
        actor=actor,
        action=action,
        target_user=target_user,
        description=description,
        metadata=metadata or {},
        ip_address=request.META.get("REMOTE_ADDR") if request else None,
    )


# ═══════════════════════════════════════════════════════════════
# ORGANIZATIONS — Create, List, Detail, Update, Delete
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def organization_list(request):
    if request.method == "GET":
        qs = Organization.objects.filter(is_active=True).annotate(
            mcount=Count("memberships")
        )
        # Show orgs the user belongs to, plus public ones
        qs = qs.filter(
            Q(memberships__user=request.user) | Q(visibility="public")
        ).distinct()
        qs = qs.order_by("name")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            OrganizationListSerializer(
                page, many=True, context={"request": request}
            ).data
        )

    elif request.method == "POST":
        serializer = OrganizationDetailSerializer(data=request.data)
        if serializer.is_valid():
            org = serializer.save()
            Membership.objects.create(
                user=request.user, organization=org, role="owner", is_default=True
            )
            org.member_count = 1
            org.save(update_fields=["member_count"])
            _log(org, request.user, "org_created", request=request)
            return Response(
                OrganizationDetailSerializer(org, context={"request": request}).data,
                status=201,
            )
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def organization_detail(request, org_id):
    org = _get_org(org_id)
    if request.method == "GET":
        return Response(
            OrganizationDetailSerializer(org, context={"request": request}).data
        )
    elif request.method == "PATCH":
        _can_or_403(request.user, org, "can_manage_members")
        serializer = OrganizationDetailSerializer(org, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            _log(org, request.user, "org_updated", request=request)
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        _can_or_403(request.user, org, "can_delete_org")
        org.soft_delete()
        _log(org, request.user, "org_deleted", request=request)
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# MEMBERS — List, Add, Remove, Change Role
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def member_list(request, org_id):
    org = _get_org(org_id)
    qs = org.memberships.select_related("user").order_by("-role", "user__username")
    role_filter = request.query_params.get("role")
    if role_filter:
        qs = qs.filter(role=role_filter)
    return Response(MembershipSerializer(qs, many=True).data)


@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def change_member_role(request, org_id, user_id):
    org = _get_org(org_id)
    _can_or_403(request.user, org, "can_manage_members")
    membership = _get_membership(org, user_id=user_id)
    new_role = request.data.get("role")
    if new_role not in dict(Membership.BUILTIN_ROLES):
        return Response({"error": "Invalid role"}, status=400)
    old_role = membership.role
    membership.role = new_role
    membership.save(update_fields=["role"])
    _log(
        org,
        request.user,
        "member_role_changed",
        target_user=membership.user,
        description=f"{old_role} → {new_role}",
        request=request,
    )
    return Response(MembershipSerializer(membership).data)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def remove_member(request, org_id, user_id):
    org = _get_org(org_id)
    _can_or_403(request.user, org, "can_manage_members")
    membership = _get_membership(org, user_id=user_id)
    if membership.role == "owner" and org.memberships.filter(role="owner").count() <= 1:
        return Response({"error": "Cannot remove the last owner"}, status=400)
    membership.delete()
    org.member_count = org.memberships.count()
    org.save(update_fields=["member_count"])
    _log(
        org,
        request.user,
        "member_removed",
        target_user=membership.user,
        request=request,
    )
    return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def leave_organization(request, org_id):
    org = _get_org(org_id)
    membership = _get_membership(org, request.user)
    if membership.role == "owner" and org.memberships.filter(role="owner").count() <= 1:
        return Response(
            {"error": "Cannot leave as last owner. Transfer ownership first."},
            status=400,
        )
    membership.delete()
    org.member_count = org.memberships.count()
    org.save(update_fields=["member_count"])
    _log(
        org,
        request.user,
        "member_removed",
        target_user=request.user,
        description="Self-removed",
        request=request,
    )
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# INVITATIONS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def invite_member(request, org_id):
    org = _get_org(org_id)
    _can_or_403(request.user, org, "can_manage_members")
    email = request.data.get("email")
    role = request.data.get("role", "member")
    if not email:
        return Response({"error": "email required"}, status=400)
    if org.member_count >= org.max_members:
        return Response({"error": "Member limit reached"}, status=400)

    from django.contrib.auth.models import User

    existing_user = User.objects.filter(email=email).first()
    if existing_user and org.memberships.filter(user=existing_user).exists():
        return Response({"error": "User is already a member"}, status=400)

    token = secrets.token_urlsafe(32)
    invitation = Invitation.objects.create(
        organization=org,
        email=email,
        invited_by=request.user,
        role=role,
        token=token,
        expires_at=timezone.now() + timedelta(days=7),
    )
    _log(
        org,
        request.user,
        "invitation_sent",
        description=f"Invited {email} as {role}",
        request=request,
    )
    # TODO: send email with invitation link
    return Response({"token": token, "status": "invited"}, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_invitations(request):
    qs = Invitation.objects.filter(
        email=request.user.email, status="pending", expires_at__gt=timezone.now()
    ).order_by("-created_at")
    return Response(InvitationSerializer(qs, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def accept_invitation(request, invitation_id):
    invitation = get_object_or_404(
        Invitation, id=invitation_id, email=request.user.email
    )
    if invitation.is_expired:
        return Response({"error": "Invitation expired"}, status=400)
    org = invitation.organization
    if org.member_count >= org.max_members:
        return Response({"error": "Organization is full"}, status=400)
    Membership.objects.create(user=request.user, organization=org, role=invitation.role)
    org.member_count = org.memberships.count()
    org.save(update_fields=["member_count"])
    invitation.status = "accepted"
    invitation.save(update_fields=["status"])
    _log(org, request.user, "invitation_accepted", request=request)
    return Response({"status": "accepted", "organization": org.name})


# ═══════════════════════════════════════════════════════════════
# TEAMS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def team_list(request, org_id):
    org = _get_org(org_id)
    if request.method == "GET":
        qs = org.teams.annotate(mcount=Count("team_memberships"))
        if not _can(request.user, org, "can_manage_all_experiments"):
            qs = qs.filter(
                Q(visibility="visible") | Q(team_memberships__user=request.user)
            ).distinct()
        return Response(TeamSerializer(qs, many=True).data)
    elif request.method == "POST":
        _can_or_403(request.user, org, "can_manage_teams")
        if org.teams.count() >= org.max_teams:
            return Response({"error": "Team limit reached"}, status=400)
        serializer = TeamSerializer(data=request.data)
        if serializer.is_valid():
            team = serializer.save(organization=org)
            TeamMembership.objects.create(
                team=team, user=request.user, role="maintainer"
            )
            org.team_count = org.teams.count()
            org.save(update_fields=["team_count"])
            _log(
                org,
                request.user,
                "team_created",
                description=f"Team: {team.name}",
                request=request,
            )
            return Response(TeamDetailSerializer(team).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def team_detail(request, org_id, team_id):
    org = _get_org(org_id)
    team = get_object_or_404(Team, id=team_id, organization=org)
    if request.method == "GET":
        return Response(TeamDetailSerializer(team).data)
    elif request.method == "PATCH":
        if not team.team_memberships.filter(
            user=request.user, role="maintainer"
        ).exists():
            _can_or_403(request.user, org, "can_manage_teams")
        serializer = TeamSerializer(team, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            _log(
                org,
                request.user,
                "team_updated",
                description=f"Team: {team.name}",
                request=request,
            )
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        _can_or_403(request.user, org, "can_manage_teams")
        team.delete()
        org.team_count = org.teams.count()
        org.save(update_fields=["team_count"])
        _log(
            org,
            request.user,
            "team_deleted",
            description=f"Team: {team.name}",
            request=request,
        )
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def add_team_member(request, org_id, team_id):
    org = _get_org(org_id)
    team = get_object_or_404(Team, id=team_id, organization=org)
    user_id = request.data.get("user_id")
    user = get_object_or_404(User, id=user_id)
    if not org.memberships.filter(user=user).exists():
        return Response(
            {"error": "User must be a member of the organization first"}, status=400
        )
    tm, created = TeamMembership.objects.get_or_create(team=team, user=user)
    _log(
        org,
        request.user,
        "team_member_added",
        target_user=user,
        description=f"Team: {team.name}",
        request=request,
    )
    return Response({"status": "added"}, status=201 if created else 200)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def remove_team_member(request, org_id, team_id, user_id):
    org = _get_org(org_id)
    team = get_object_or_404(Team, id=team_id, organization=org)
    TeamMembership.objects.filter(team=team, user_id=user_id).delete()
    _log(
        org,
        request.user,
        "team_member_removed",
        target_user_id=user_id,
        description=f"Team: {team.name}",
        request=request,
    )
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# ORG EXPERIMENTS / DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def org_experiments(request, org_id):
    org = _get_org(org_id)
    from jobscheduler.models import Job

    qs = Job.objects.filter(user__org_memberships__organization=org).select_related(
        "user"
    )
    status_filter = request.query_params.get("status")
    if status_filter:
        qs = qs.filter(status=status_filter)
    qs = qs.order_by("-created_at")[:50]
    return Response(
        [
            {
                "id": str(j.id),
                "name": j.name,
                "status": j.status,
                "username": j.user.username,
                "created_at": j.created_at,
            }
            for j in qs
        ]
    )


# ═══════════════════════════════════════════════════════════════
# AUDIT LOG
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def audit_log(request, org_id):
    org = _get_org(org_id)
    _can_or_403(request.user, org, "can_view_audit_log")
    qs = org.audit_logs.order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(AuditLogSerializer(page, many=True).data)


# ═══════════════════════════════════════════════════════════════
# RESOURCE GROUPS (Enterprise)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def resource_group_list(request, org_id):
    org = _get_org(org_id)
    if request.method == "GET":
        return Response(
            ResourceGroupSerializer(org.resource_groups.all(), many=True).data
        )
    elif request.method == "POST":
        _can_or_403(request.user, org, "can_create_resource_groups")
        serializer = ResourceGroupSerializer(data=request.data)
        if serializer.is_valid():
            rg = serializer.save(organization=org, created_by=request.user)
            _log(
                org,
                request.user,
                "rg_created",
                description=f"RG: {rg.name}",
                request=request,
            )
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# ROLES
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def role_list(request, org_id):
    org = _get_org(org_id)
    if request.method == "GET":
        return Response(RoleSerializer(org.roles.all(), many=True).data)
    elif request.method == "POST":
        _can_or_403(request.user, org, "can_manage_roles")
        serializer = RoleSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(organization=org)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# SUBSCRIPTION
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def subscription_detail(request, org_id):
    org = _get_org(org_id)
    try:
        sub = org.subscription
        return Response(SubscriptionSerializer(sub).data)
    except:
        return Response({"plan": "free", "status": "active"})


# ═══════════════════════════════════════════════════════════════
# PLANS (public)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def plan_list(request):
    return Response(
        [
            {
                "name": p.name,
                "tier": p.tier,
                "price_monthly_cents": p.price_monthly_cents,
                "max_members": p.max_members,
                "max_teams": p.max_teams,
                "includes_sso": p.includes_sso,
                "includes_audit_log": p.includes_audit_log,
            }
            for p in Plan.objects.filter(is_active=True)
        ]
    )
