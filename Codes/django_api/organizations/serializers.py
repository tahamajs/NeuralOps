# organizations/serializers.py
from rest_framework import serializers

from .models import *


class OrganizationListSerializer(serializers.ModelSerializer):
    owner_count = serializers.SerializerMethodField()
    my_role = serializers.SerializerMethodField()

    class Meta:
        model = Organization
        fields = [
            "id",
            "name",
            "slug",
            "display_name",
            "description",
            "logo",
            "tier",
            "visibility",
            "member_count",
            "team_count",
            "experiment_count",
            "owner_count",
            "my_role",
            "created_at",
        ]

    def get_owner_count(self, obj):
        return obj.memberships.filter(role="owner").count()

    def get_my_role(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            m = obj.memberships.filter(user=request.user).first()
            return m.role if m else None
        return None


class OrganizationDetailSerializer(serializers.ModelSerializer):
    owners = serializers.SerializerMethodField()
    my_role = serializers.SerializerMethodField()
    my_permissions = serializers.SerializerMethodField()
    subscription_status = serializers.SerializerMethodField()

    class Meta:
        model = Organization
        fields = "__all__"
        read_only_fields = [
            "id",
            "member_count",
            "team_count",
            "experiment_count",
            "created_at",
            "updated_at",
        ]

    def get_owners(self, obj):
        return [
            {"id": m.user.id, "username": m.user.username}
            for m in obj.memberships.filter(role="owner").select_related("user")
        ]

    def get_my_role(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            m = obj.memberships.filter(user=request.user).first()
            return m.role if m else None
        return None

    def get_my_permissions(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            m = obj.memberships.filter(user=request.user).first()
            return m.effective_permissions if m else {}
        return {}

    def get_subscription_status(self, obj):
        try:
            return obj.subscription.status if obj.subscription else "free"
        except:
            return "free"


class MembershipSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    email = serializers.CharField(source="user.email", read_only=True)
    avatar_url = serializers.SerializerMethodField()

    class Meta:
        model = Membership
        fields = [
            "id",
            "user",
            "username",
            "email",
            "avatar_url",
            "role",
            "custom_role",
            "is_default",
            "joined_at",
            "invited_by",
        ]
        read_only_fields = ["id", "user", "joined_at", "invited_by"]

    def get_avatar_url(self, obj):
        try:
            return obj.user.profile.avatar_url
        except:
            return None


class TeamSerializer(serializers.ModelSerializer):
    member_count = serializers.IntegerField(read_only=True)
    parent_name = serializers.CharField(
        source="parent.name", read_only=True, default=""
    )

    class Meta:
        model = Team
        fields = [
            "id",
            "organization",
            "name",
            "slug",
            "description",
            "parent",
            "parent_name",
            "visibility",
            "member_count",
            "created_at",
        ]


class TeamDetailSerializer(serializers.ModelSerializer):
    members = serializers.SerializerMethodField()

    class Meta:
        model = Team
        fields = [
            "id",
            "organization",
            "name",
            "slug",
            "description",
            "parent",
            "visibility",
            "members",
            "created_at",
        ]

    def get_members(self, obj):
        return [
            {
                "id": tm.user.id,
                "username": tm.user.username,
                "role": tm.role,
                "avatar_url": getattr(
                    getattr(tm.user, "profile", None), "avatar_url", None
                ),
            }
            for tm in obj.team_memberships.select_related("user")
        ]


class InvitationSerializer(serializers.ModelSerializer):
    invited_by_username = serializers.CharField(
        source="invited_by.username", read_only=True
    )

    class Meta:
        model = Invitation
        fields = [
            "id",
            "organization",
            "team",
            "email",
            "invited_by",
            "invited_by_username",
            "role",
            "status",
            "expires_at",
            "created_at",
        ]
        read_only_fields = ["id", "invited_by", "status", "created_at"]


class ResourceGroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResourceGroup
        fields = "__all__"
        read_only_fields = ["id", "organization", "created_by", "created_at"]


class AuditLogSerializer(serializers.ModelSerializer):
    actor_username = serializers.CharField(
        source="actor.username", read_only=True, default=""
    )
    target_username = serializers.CharField(
        source="target_user.username", read_only=True, default=""
    )

    class Meta:
        model = AuditLog
        fields = [
            "id",
            "organization",
            "actor",
            "actor_username",
            "action",
            "target_user",
            "target_username",
            "description",
            "metadata",
            "ip_address",
            "created_at",
        ]


class RoleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Role
        fields = "__all__"
        read_only_fields = ["id", "organization", "is_builtin"]


class SubscriptionSerializer(serializers.ModelSerializer):
    plan_name = serializers.CharField(source="plan.name", read_only=True, default="")

    class Meta:
        model = Subscription
        fields = "__all__"
        read_only_fields = ["id", "organization", "created_at", "updated_at"]
