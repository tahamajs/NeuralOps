"""Reputation REST API — scores, staking, disputes, jury, badges, trust graph."""

import logging

from django.db.models import Q, Sum
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import *
from .scoring import build_reputation_graph, compute_eigentrust
from .serializers import *

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# REPUTATION SCORES
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def score_detail(request, username):
    """View a user's reputation score by username."""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user = get_object_or_404(User, username=username)
    score, _ = ReputationScore.objects.get_or_create(user=user)
    badges = UserReputationBadge.objects.filter(user=user).select_related("badge")
    data = ReputationScoreSerializer(score).data
    data["badges"] = UserReputationBadgeSerializer(badges, many=True).data
    return Response(data)


@api_view(["GET"])
@permission_classes([AllowAny])
def reputation_leaderboard(request):
    """Global reputation leaderboard — top providers."""
    qs = ReputationScore.objects.filter(provider_score__gte=0).order_by(
        "-provider_score"
    )
    role = request.query_params.get("role", "provider")
    if role == "consumer":
        qs = qs.order_by("-consumer_score")
    limit = min(int(request.query_params.get("limit", 50)), 100)
    results = []
    for i, score in enumerate(qs[:limit], 1):
        results.append(
            {
                "rank": i,
                "username": score.user.username,
                "provider_score": score.provider_score,
                "consumer_score": score.consumer_score,
                "total_jobs": score.total_jobs_completed,
                "success_rate": round(
                    (score.total_jobs_completed - score.total_jobs_failed)
                    / max(score.total_jobs_completed, 1)
                    * 100,
                    1,
                ),
            }
        )
    return Response({"results": results, "total": qs.count()})


# ═══════════════════════════════════════════════════════════════
# STAKING
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def stake_tokens(request):
    """Stake tokens as collateral (like Lium's Collateral Contract)."""
    amount = request.data.get("amount_usd")
    if not amount:
        return Response({"error": "amount_usd required"}, status=400)
    try:
        amount = Decimal(str(amount))
        if amount < Decimal("10.00"):
            return Response({"error": "Minimum stake is $10.00"}, status=400)
    except:
        return Response({"error": "Invalid amount"}, status=400)

    # Verify user has funds in escrow
    try:
        from gpu_escrow.models import EscrowAccount

        account = EscrowAccount.objects.get(user=request.user)
        if account.balance_usd < amount:
            return Response({"error": "Insufficient escrow balance"}, status=400)
        account.withdraw(amount, "Staked as collateral")
    except ImportError:
        pass  # Skip escrow check if not installed

    stake = Stake.objects.create(
        user=request.user, amount_usd=amount, slashable_amount_usd=amount
    )
    return Response(StakeSerializer(stake).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_stakes(request):
    """List the authenticated user's stakes."""
    stakes = Stake.objects.filter(user=request.user).order_by("-created_at")
    return Response(StakeSerializer(stakes, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_unbonding(request, stake_id):
    """Start the unbonding period for a stake (EigenLayer-style safety delay)."""
    stake = get_object_or_404(Stake, id=stake_id, user=request.user)
    if stake.status != "active":
        return Response({"error": f"Stake is {stake.status}, not active"}, status=400)
    stake.start_unbonding()
    return Response(StakeSerializer(stake).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def withdraw_stake(request, stake_id):
    """Withdraw a stake after the unbonding period."""
    stake = get_object_or_404(Stake, id=stake_id, user=request.user)
    if not stake.can_withdraw():
        remaining = (
            Stake.UNBONDING_PERIOD_DAYS
            - (timezone.now() - stake.unbonding_started_at).days
        )
        return Response(
            {
                "error": f"Unbonding period not complete. {max(0, remaining)} days remaining."
            },
            status=400,
        )
    stake.status = "withdrawn"
    stake.save(update_fields=["status"])
    # Return funds to escrow
    try:
        from gpu_escrow.models import EscrowAccount

        account, _ = EscrowAccount.objects.get_or_create(user=request.user)
        account.deposit(stake.amount_usd, "Stake withdrawal")
    except ImportError:
        pass
    return Response(StakeSerializer(stake).data)


# ═══════════════════════════════════════════════════════════════
# SLASHING EVENTS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def slashing_history(request):
    """List slashing events for the authenticated user."""
    qs = SlashingEvent.objects.filter(user=request.user).order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        SlashingEventSerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def slashing_condition_list(request):
    """List all active slashing conditions."""
    return Response(
        SlashingConditionSerializer(
            SlashingCondition.objects.filter(is_active=True), many=True
        ).data
    )


# ═══════════════════════════════════════════════════════════════
# DISPUTES — Kleros-style jury arbitration
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def raise_dispute(request, job_id):
    """Raise a dispute for a completed compute job."""
    from gpu_marketplace.models import ComputeJob

    job = get_object_or_404(ComputeJob, id=job_id)

    if job.status not in ("completed", "failed", "disputed"):
        return Response(
            {"error": "Job must be completed or failed to dispute"}, status=400
        )
    if request.user not in (job.consumer, job.provider):
        return Response({"error": "You must be a party to this job"}, status=403)
    if hasattr(job, "dispute"):
        return Response({"error": "A dispute already exists for this job"}, status=400)

    against = job.provider if request.user == job.consumer else job.consumer

    dispute = Dispute.objects.create(
        job=job,
        raised_by=request.user,
        against=against,
        reason=request.data.get("reason", ""),
        evidence=request.data.get("evidence", {}),
    )
    # Charge dispute fee
    try:
        from gpu_escrow.models import EscrowAccount

        account = EscrowAccount.objects.get(user=request.user)
        account.withdraw(dispute.dispute_fee_usd, f"Dispute fee for job {job.id}")
    except:
        pass

    return Response(DisputeSerializer(dispute).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dispute_detail(request, dispute_id):
    """Get dispute details including jury and votes."""
    dispute = get_object_or_404(Dispute, id=dispute_id)
    data = DisputeSerializer(dispute).data
    data["jury_members"] = [
        {"username": jm.user.username, "has_voted": jm.has_voted}
        for jm in dispute.jury_members.all()
    ]
    data["votes"] = JuryVoteSerializer(dispute.jury_votes.all(), many=True).data
    return Response(data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_disputes(request):
    """List disputes where the current user is involved."""
    qs = Dispute.objects.filter(
        Q(raised_by=request.user) | Q(against=request.user)
    ).order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(DisputeSerializer(page, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def select_jury(request, dispute_id):
    """Trigger jury selection for a dispute (Kleros-style random draw)."""
    dispute = get_object_or_404(Dispute, id=dispute_id)
    if dispute.status not in ("open", "evidence_gathering"):
        return Response(
            {"error": f"Cannot select jury in status: {dispute.status}"}, status=400
        )
    if not request.user.is_staff:
        return Response({"error": "Only staff can trigger jury selection"}, status=403)
    jurors = dispute.select_jury()
    return Response({"jurors_selected": len(jurors), "status": dispute.status})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def cast_jury_vote(request, dispute_id):
    """Cast a vote as a juror."""
    dispute = get_object_or_404(Dispute, id=dispute_id, status="voting")
    juror = get_object_or_404(JuryMember, dispute=dispute, user=request.user)
    if juror.has_voted:
        return Response({"error": "Already voted"}, status=400)

    vote_choice = request.data.get("vote")
    if vote_choice not in ("consumer_wins", "provider_wins"):
        return Response(
            {"error": "vote must be 'consumer_wins' or 'provider_wins'"}, status=400
        )

    JuryVote.objects.create(
        dispute=dispute,
        juror=juror,
        vote=vote_choice,
        reasoning=request.data.get("reasoning", ""),
    )
    juror.has_voted = True
    juror.save(update_fields=["has_voted"])

    # Check if we can tally
    dispute.tally_votes()
    return Response(DisputeSerializer(dispute).data)


# ═══════════════════════════════════════════════════════════════
# BADGES
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def badge_list(request):
    return Response(
        ReputationBadgeSerializer(ReputationBadge.objects.all(), many=True).data
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_badges(request):
    badges = UserReputationBadge.objects.filter(user=request.user).select_related(
        "badge"
    )
    return Response(UserReputationBadgeSerializer(badges, many=True).data)


# ═══════════════════════════════════════════════════════════════
# TRUST GRAPH — EigenTrust
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def compute_trust_graph(request):
    """Build a reputation graph and compute EigenTrust scores."""
    if not request.user.is_staff:
        return Response({"error": "Staff only"}, status=403)
    graph = build_reputation_graph()
    results = compute_eigentrust(graph)
    return Response(
        {
            "graph_id": str(graph.id),
            "users_ranked": len(results),
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def trust_ranking(request):
    """Return the most recent EigenTrust ranking."""
    latest = EigenTrustResult.objects.order_by("-computed_at").first()
    if not latest:
        return Response({"error": "No trust graph computed yet"}, status=404)
    qs = EigenTrustResult.objects.filter(graph=latest.graph).order_by("rank")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        EigenTrustResultSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_reputation_dashboard(request):
    """Full reputation dashboard for the authenticated user."""
    score, _ = ReputationScore.objects.get_or_create(user=request.user)
    stakes = Stake.objects.filter(user=request.user, status__in=["active", "unbonding"])
    disputes = Dispute.objects.filter(
        Q(raised_by=request.user) | Q(against=request.user)
    )
    badges = UserReputationBadge.objects.filter(user=request.user).select_related(
        "badge"
    )
    slashing = SlashingEvent.objects.filter(user=request.user)

    return Response(
        {
            "provider_score": score.provider_score,
            "consumer_score": score.consumer_score,
            "total_jobs": score.total_jobs_completed,
            "success_rate": round(
                (score.total_jobs_completed - score.total_jobs_failed)
                / max(score.total_jobs_completed, 1)
                * 100,
                1,
            ),
            "active_stakes": float(stakes.aggregate(s=Sum("amount_usd"))["s"] or 0),
            "open_disputes": disputes.filter(
                status__in=["open", "evidence_gathering", "jury_selection", "voting"]
            ).count(),
            "disputes_won": disputes.filter(
                Q(raised_by=request.user, resolution="consumer_wins")
                | Q(against=request.user, resolution="provider_wins")
            ).count(),
            "total_slashed": float(
                slashing.aggregate(s=Sum("amount_slashed_usd"))["s"] or 0
            ),
            "badges": UserReputationBadgeSerializer(badges, many=True).data,
        }
    )
