from rest_framework import serializers

from .models import *


class ReputationScoreSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ReputationScore
        fields = [
            "id",
            "user",
            "username",
            "provider_score",
            "consumer_score",
            "total_jobs_completed",
            "total_jobs_failed",
            "total_disputes_lost",
            "total_disputes_won",
            "last_updated",
            "created_at",
        ]
        read_only_fields = ["id", "user", "username", "last_updated", "created_at"]


class StakeSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    can_withdraw = serializers.BooleanField(read_only=True)

    class Meta:
        model = Stake
        fields = [
            "id",
            "user",
            "username",
            "amount_usd",
            "slashable_amount_usd",
            "status",
            "unbonding_started_at",
            "can_withdraw",
            "listing",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "username",
            "status",
            "can_withdraw",
            "created_at",
            "updated_at",
        ]


class SlashingConditionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SlashingCondition
        fields = "__all__"


class SlashingEventSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    condition_name = serializers.CharField(source="condition.name", read_only=True)

    class Meta:
        model = SlashingEvent
        fields = [
            "id",
            "user",
            "username",
            "condition",
            "condition_name",
            "stake",
            "job",
            "dispute",
            "amount_slashed_usd",
            "reason",
            "recipient",
            "created_at",
        ]
        read_only_fields = ["id", "user", "username", "created_at"]


class DisputeSerializer(serializers.ModelSerializer):
    raised_by_username = serializers.CharField(
        source="raised_by.username", read_only=True
    )
    against_username = serializers.CharField(source="against.username", read_only=True)
    jury_count = serializers.SerializerMethodField()
    vote_summary = serializers.SerializerMethodField()

    class Meta:
        model = Dispute
        fields = [
            "id",
            "job",
            "raised_by",
            "raised_by_username",
            "against",
            "against_username",
            "reason",
            "evidence",
            "status",
            "required_jurors",
            "resolution",
            "resolution_reason",
            "dispute_fee_usd",
            "jury_reward_usd",
            "jury_count",
            "vote_summary",
            "created_at",
            "resolved_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "raised_by",
            "status",
            "resolution",
            "created_at",
            "resolved_at",
            "updated_at",
        ]

    def get_jury_count(self, obj):
        return obj.jury_members.count()

    def get_vote_summary(self, obj):
        votes = obj.jury_votes.all()
        return {
            "consumer_wins": votes.filter(vote="consumer_wins").count(),
            "provider_wins": votes.filter(vote="provider_wins").count(),
            "total": votes.count(),
        }


class JuryVoteSerializer(serializers.ModelSerializer):
    juror_username = serializers.CharField(source="juror.user.username", read_only=True)

    class Meta:
        model = JuryVote
        fields = [
            "id",
            "dispute",
            "juror",
            "juror_username",
            "vote",
            "reasoning",
            "is_majority",
            "created_at",
        ]
        read_only_fields = ["id", "juror", "is_majority", "created_at"]


class ReputationBadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReputationBadge
        fields = "__all__"


class UserReputationBadgeSerializer(serializers.ModelSerializer):
    badge_detail = ReputationBadgeSerializer(source="badge", read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = UserReputationBadge
        fields = ["id", "user", "username", "badge", "badge_detail", "awarded_at"]
        read_only_fields = ["id", "awarded_at"]


class TrustEdgeSerializer(serializers.ModelSerializer):
    source_username = serializers.CharField(source="source.username", read_only=True)
    target_username = serializers.CharField(source="target.username", read_only=True)

    class Meta:
        model = TrustEdge
        fields = [
            "id",
            "graph",
            "source",
            "source_username",
            "target",
            "target_username",
            "trust_score",
            "derivation",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class EigenTrustResultSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = EigenTrustResult
        fields = [
            "id",
            "graph",
            "user",
            "username",
            "global_trust_score",
            "rank",
            "computed_at",
        ]
        read_only_fields = ["id", "computed_at"]
