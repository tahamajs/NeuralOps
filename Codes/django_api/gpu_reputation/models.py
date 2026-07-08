"""
GPU Reputation Models — Trust Layer with Staking, Slashing, and Dispute Resolution

Architecture inspired by:
  - EigenLayer: programmable slashing, Operator Sets, Unique Stake,
    redistribution of slashed funds, safety delays
  - Bittensor/Lium: collateral staking, Yuma Consensus, validator scoring,
    miner reputation weighted by stake
  - Kleros: random jury selection from staked token holders, supermajority
    voting, economic incentives for honest jurors
  - Render Network: reputation scores drive resource allocation,
    higher reputation → more concurrent jobs
  - OpenRank/EigenTrust: verifiable graph-based reputation algorithm,
    pre-trusted peers, PageRank-style eigenvector computation
  - Akash: security deposits, downtime slashing (0.01%), double-sign (5%)

Taxonomy:
    ReputationScore → Stake → OperatorSet → SlashingCondition → SlashingEvent
    Dispute → Jury → JuryVote → DisputeResolution
    ReputationGraph → TrustEdge → EigenTrustResult
"""

import uuid
from decimal import Decimal

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


# ═══════════════════════════════════════════════════════════════
# 1. REPUTATION SCORE — Per-user dual-role score
# ═══════════════════════════════════════════════════════════════
class ReputationScore(models.Model):
    """
    A user's reputation as both a provider (GPU supplier) and consumer (buyer).
    Like Render Network's reputation: higher scores → more concurrent jobs.

    Scores are on a 0–100 scale, start at 50, and evolve with every completed job.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reputation_score",
    )
    # Provider score: how reliable you are as a GPU supplier
    provider_score = models.FloatField(
        default=50.0, validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    # Consumer score: how reliable you are as a GPU buyer
    consumer_score = models.FloatField(
        default=50.0, validators=[MinValueValidator(0), MaxValueValidator(100)]
    )

    # Stats
    total_jobs_completed = models.PositiveIntegerField(default=0)
    total_jobs_failed = models.PositiveIntegerField(default=0)
    total_disputes_lost = models.PositiveIntegerField(default=0)
    total_disputes_won = models.PositiveIntegerField(default=0)

    # Time-weighted (recent activity matters more)
    last_updated = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-provider_score", "-consumer_score"]

    def __str__(self):
        return f"{self.user.username}: P:{self.provider_score:.1f} C:{self.consumer_score:.1f}"

    def update_provider_score(self, job_success: bool):
        """Update provider score with exponential moving average."""
        alpha = 0.15  # weight for new observation
        obs = 100.0 if job_success else 0.0
        self.provider_score = round(alpha * obs + (1 - alpha) * self.provider_score, 2)
        self.total_jobs_completed += 1
        if not job_success:
            self.total_jobs_failed += 1
        self.save()

    def update_consumer_score(self, payment_on_time: bool, dispute_outcome: str = None):
        """Update consumer score."""
        alpha = 0.10
        if dispute_outcome == "lost":
            obs = 0.0
        elif payment_on_time:
            obs = 100.0
        else:
            obs = 50.0
        self.consumer_score = round(alpha * obs + (1 - alpha) * self.consumer_score, 2)
        self.save()


# ═══════════════════════════════════════════════════════════════
# 2. STAKE — Collateral deposited by providers
# ═══════════════════════════════════════════════════════════════
class Stake(models.Model):
    """
    Collateral staked by a provider (like Lium's Collateral Contract).
    Funds are locked and can be slashed for misbehavior.
    Inspired by EigenLayer: Unique Stake per Operator Set.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="stakes"
    )
    amount_usd = models.DecimalField(max_digits=12, decimal_places=2)
    # How much of this stake is currently slashable
    slashable_amount_usd = models.DecimalField(
        max_digits=12, decimal_places=2, default=0
    )

    STATUS_CHOICES = [
        ("active", "Active"),
        ("unbonding", "Unbonding"),  # EigenLayer-style safety delay
        ("withdrawn", "Withdrawn"),
        ("slashed", "Slashed"),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    unbonding_started_at = models.DateTimeField(null=True, blank=True)
    # EigenLayer safety delay: 7 days before funds can be withdrawn
    UNBONDING_PERIOD_DAYS = 7

    # Link to listing (optional: stake can be per-listing)
    listing = models.ForeignKey(
        "gpu_marketplace.GPUListing",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="stakes",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "status"]),
        ]

    def __str__(self):
        return f"{self.user.username} staked ${self.amount_usd} ({self.status})"

    def start_unbonding(self):
        """Begin the unbonding period (EigenLayer-style safety delay)."""
        self.status = "unbonding"
        self.unbonding_started_at = timezone.now()
        self.save(update_fields=["status", "unbonding_started_at"])

    def can_withdraw(self):
        """Check if the unbonding period has elapsed."""
        if self.status != "unbonding":
            return False
        if not self.unbonding_started_at:
            return False
        return (
            timezone.now() - self.unbonding_started_at
        ).days >= self.UNBONDING_PERIOD_DAYS

    @property
    def slashable(self):
        """Return the amount that can be slashed (like EigenLayer's slashable stake)."""
        if self.status == "active":
            return self.amount_usd
        return Decimal("0")


# ═══════════════════════════════════════════════════════════════
# 3. SLASHING CONDITION — EigenLayer-style programmable rules
# ═══════════════════════════════════════════════════════════════
class SlashingCondition(models.Model):
    """
    A configurable slashing rule (like EigenLayer AVS Slashing Conditions).
    Defines what behavior triggers slashing and how much is slashed.

    Examples:
    - Downtime > 1 hour: slash 0.01% of stake
    - Failed job: slash 5% of stake
    - Fraudulent output: slash 50% of stake
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    condition_type = models.CharField(
        max_length=50,
        choices=[
            ("downtime", "Downtime"),
            ("job_failure", "Job Failure"),
            ("fraud", "Fraudulent Output"),
            ("dispute_lost", "Dispute Lost"),
            ("sla_violation", "SLA Violation"),
            ("double_sign", "Double Sign"),
            ("custom", "Custom"),
        ],
    )
    slash_fraction = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Fraction of stake to slash (0.0–1.0). 0.0001 = 0.01% for downtime.",
    )
    # Burn or redistribute? (EigenLayer redistribution feature)
    redistribution = models.BooleanField(
        default=False,
        help_text="If True, slashed funds go to affected party instead of being burned",
    )
    redistribution_recipient = models.CharField(
        max_length=50,
        default="platform",
        choices=[
            ("platform", "Platform Treasury"),
            ("victim", "Affected Party"),
            ("jury", "Jury Pool"),
            ("burn", "Burn"),
        ],
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["condition_type", "name"]

    def __str__(self):
        return f"{self.name} ({self.slash_fraction*100:.2f}%)"


class SlashingEvent(models.Model):
    """
    A recorded slashing event (like EigenLayer's slashId).
    When a provider violates a slashing condition, their stake is reduced.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="slashing_events",
    )
    condition = models.ForeignKey(
        SlashingCondition, on_delete=models.SET_NULL, null=True
    )
    stake = models.ForeignKey(
        Stake, on_delete=models.SET_NULL, null=True, related_name="slashing_events"
    )
    job = models.ForeignKey(
        "gpu_marketplace.ComputeJob",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="slashing_events",
    )
    dispute = models.ForeignKey(
        "Dispute",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="slashing_events",
    )
    amount_slashed_usd = models.DecimalField(max_digits=12, decimal_places=2)
    reason = models.TextField()
    # Where did the slashed funds go?
    recipient = models.CharField(
        max_length=50,
        default="burn",
        choices=[
            ("burn", "Burned"),
            ("platform", "Platform Treasury"),
            ("victim", "Affected Party"),
            ("jury", "Jury Pool"),
        ],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self):
        return f"Slash: ${self.amount_slashed_usd} from {self.user.username} — {self.reason[:50]}"


# ═══════════════════════════════════════════════════════════════
# 4. DISPUTE — Kleros-style jury arbitration
# ═══════════════════════════════════════════════════════════════
class Dispute(models.Model):
    """
    A formal dispute between a consumer and provider.
    Inspired by Kleros: random jury selection, supermajority voting, economic incentives.
    """

    STATUS_CHOICES = [
        ("open", "Open"),
        ("evidence_gathering", "Evidence Gathering"),
        ("jury_selection", "Jury Selection"),
        ("voting", "Voting"),
        ("resolved", "Resolved"),
        ("appealed", "Appealed"),
        ("withdrawn", "Withdrawn"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.OneToOneField(
        "gpu_marketplace.ComputeJob", on_delete=models.CASCADE, related_name="dispute"
    )
    raised_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="disputes_raised",
    )
    against = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="disputes_against",
    )
    reason = models.TextField()
    evidence = models.JSONField(
        default=dict,
        blank=True,
        help_text="Links to logs, screenshots, verification data",
    )

    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="open")
    required_jurors = models.PositiveIntegerField(default=5)

    # Resolution
    resolution = models.CharField(
        max_length=30,
        null=True,
        blank=True,
        choices=[
            ("consumer_wins", "Consumer Wins"),
            ("provider_wins", "Provider Wins"),
            ("split", "Split Decision"),
            ("tie", "Tie"),
        ],
    )
    resolution_reason = models.TextField(blank=True)

    # Kleros-style: fees and rewards
    dispute_fee_usd = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal("5.00"),
        help_text="Fee paid by dispute initiator",
    )
    jury_reward_usd = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal("1.00"),
        help_text="Reward per juror for honest voting",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "-created_at"]),
            models.Index(fields=["raised_by", "-created_at"]),
        ]

    def __str__(self):
        return f"Dispute for Job {self.job_id.hex[:8]} — {self.status}"

    def select_jury(self):
        """Kleros-style random jury selection from staked users."""
        from .jury import select_jurors

        jurors = select_jurors(self, self.required_jurors)
        for juror_user in jurors:
            JuryMember.objects.get_or_create(dispute=self, user=juror_user)
        self.status = "voting"
        self.save(update_fields=["status"])
        return jurors

    def tally_votes(self):
        """Count jury votes and determine resolution (supermajority = 2/3)."""
        votes = self.jury_votes.all()
        consumer_wins = votes.filter(vote="consumer_wins").count()
        provider_wins = votes.filter(vote="provider_wins").count()
        total = votes.count()

        if total < self.required_jurors:
            return None  # Not enough votes

        supermajority_threshold = int(self.required_jurors * 2 / 3)

        if consumer_wins >= supermajority_threshold:
            self.resolution = "consumer_wins"
        elif provider_wins >= supermajority_threshold:
            self.resolution = "provider_wins"
        elif consumer_wins == provider_wins:
            self.resolution = "tie"
        else:
            self.resolution = "split"
        self.status = "resolved"
        self.resolved_at = timezone.now()
        self.save()

        # Reward jurors who voted with the majority
        for v in votes:
            if v.vote == self.resolution:
                v.is_majority = True
                v.save(update_fields=["is_majority"])
                # Credit juror
                _credit_juror(v.user, self.jury_reward_usd)

        # Apply slashing to loser
        self._apply_slashing()

        return self.resolution

    def _apply_slashing(self):
        """Apply slashing to the losing party based on dispute outcome."""
        if self.resolution == "consumer_wins":
            # Provider loses → slash provider's stake
            _slash_user(self.against, "dispute_lost", self)
        elif self.resolution == "provider_wins":
            # Consumer loses → lower consumer score
            score, _ = ReputationScore.objects.get_or_create(user=self.raised_by)
            score.total_disputes_lost += 1
            score.update_consumer_score(payment_on_time=False, dispute_outcome="lost")


class JuryMember(models.Model):
    """A user selected as a juror for a dispute (Kleros-style)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dispute = models.ForeignKey(
        Dispute, on_delete=models.CASCADE, related_name="jury_members"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="jury_duties"
    )
    has_voted = models.BooleanField(default=False)
    selected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["dispute", "user"]]
        ordering = ["selected_at"]

    def __str__(self):
        return f"Juror {self.user.username} for Dispute {self.dispute_id.hex[:8]}"


class JuryVote(models.Model):
    """A juror's vote in a dispute."""

    VOTE_CHOICES = [
        ("consumer_wins", "Consumer Wins"),
        ("provider_wins", "Provider Wins"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dispute = models.ForeignKey(
        Dispute, on_delete=models.CASCADE, related_name="jury_votes"
    )
    juror = models.ForeignKey(
        JuryMember, on_delete=models.CASCADE, related_name="votes"
    )
    vote = models.CharField(max_length=20, choices=VOTE_CHOICES)
    reasoning = models.TextField(blank=True)
    is_majority = models.BooleanField(
        default=False, help_text="True if this vote aligned with the final resolution"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["dispute", "juror"]]

    def __str__(self):
        return f"{self.juror.user.username} voted {self.vote}"


# ═══════════════════════════════════════════════════════════════
# 5. TRUST GRAPH — OpenRank/EigenTrust-style reputation graph
# ═══════════════════════════════════════════════════════════════
class ReputationGraph(models.Model):
    """
    A snapshot of the reputation graph (like OpenRank's Eigentrust input).
    Nodes = users, Edges = TrustEdge (pairwise trust scores).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, default="global")
    computed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-computed_at"]

    def __str__(self):
        return f"Reputation Graph '{self.name}' at {self.computed_at}"


class TrustEdge(models.Model):
    """
    A directed trust edge between two users.
    source trusts target with a score from 0.0 (no trust) to 1.0 (full trust).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    graph = models.ForeignKey(
        ReputationGraph, on_delete=models.CASCADE, related_name="edges"
    )
    source = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="trust_given"
    )
    target = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="trust_received",
    )
    trust_score = models.FloatField(
        default=0.5, validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    # How this trust edge was derived
    derivation = models.CharField(
        max_length=50,
        default="direct",
        choices=[
            ("direct", "Direct Interaction"),
            ("benchmark", "Benchmark Score"),
            ("jury", "Jury Participation"),
            ("staking", "Staking Relationship"),
        ],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["graph", "source", "target"]]

    def __str__(self):
        return (
            f"{self.source.username} → {self.target.username}: {self.trust_score:.2f}"
        )


class EigenTrustResult(models.Model):
    """
    The result of running the EigenTrust algorithm on a reputation graph.
    Produces a global trust score for each user (0.0–1.0).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    graph = models.ForeignKey(
        ReputationGraph, on_delete=models.CASCADE, related_name="results"
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    global_trust_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    rank = models.PositiveIntegerField(null=True)
    computed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["graph", "user"]]
        ordering = ["-global_trust_score"]

    def __str__(self):
        return (
            f"{self.user.username}: {self.global_trust_score:.4f} (rank #{self.rank})"
        )


# ═══════════════════════════════════════════════════════════════
# 6. REPUTATION BADGE — Achievement badges
# ═══════════════════════════════════════════════════════════════
class ReputationBadge(models.Model):
    """Achievement badges awarded based on reputation milestones."""

    TIERS = [
        ("bronze", "Bronze"),
        ("silver", "Silver"),
        ("gold", "Gold"),
        ("platinum", "Platinum"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    tier = models.CharField(max_length=20, choices=TIERS, default="bronze")
    icon = models.CharField(max_length=10, default="🏅")
    min_provider_score = models.FloatField(null=True, blank=True)
    min_consumer_score = models.FloatField(null=True, blank=True)
    min_jobs_completed = models.PositiveIntegerField(null=True, blank=True)
    max_disputes_lost = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["tier", "name"]

    def __str__(self):
        return f"{self.icon} {self.name} ({self.tier})"


class UserReputationBadge(models.Model):
    """A badge awarded to a user."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reputation_badges",
    )
    badge = models.ForeignKey(ReputationBadge, on_delete=models.CASCADE)
    awarded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "badge"]]

    def __str__(self):
        return f"{self.badge.name} → {self.user.username}"


# ═══════════════════════════════════════════════════════════════
# 7. HELPERS
# ═══════════════════════════════════════════════════════════════
def _credit_juror(user, amount: Decimal):
    """Credit a juror for honest voting."""
    try:
        from gpu_escrow.models import EscrowAccount

        account, _ = EscrowAccount.objects.get_or_create(user=user)
        account.deposit(amount, "Jury reward for honest voting")
    except ImportError:
        pass


def _slash_user(user, condition_type: str, dispute=None):
    """Apply slashing to a user's stake."""
    try:
        condition = SlashingCondition.objects.filter(
            condition_type=condition_type, is_active=True
        ).first()
        if not condition:
            return

        stake = Stake.objects.filter(user=user, status="active").first()
        if not stake:
            return

        slash_amount = stake.amount_usd * Decimal(str(condition.slash_fraction))
        stake.amount_usd -= slash_amount
        stake.save(update_fields=["amount_usd"])

        SlashingEvent.objects.create(
            user=user,
            condition=condition,
            stake=stake,
            dispute=dispute,
            amount_slashed_usd=slash_amount,
            reason=f"Slashed for {condition.name}",
            recipient=condition.redistribution_recipient,
        )
    except Exception:
        pass
