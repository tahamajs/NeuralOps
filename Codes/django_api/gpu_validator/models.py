import uuid
from decimal import Decimal

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


class ValidatorNode(models.Model):
    """
    A user who has registered as a validator (Checker Node).
    They must stake a minimum amount to participate.
    """

    STATUS_CHOICES = [
        ("active", "Active"),
        ("inactive", "Inactive"),
        ("slashed", "Slashed"),
        ("banned", "Banned"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="validator_node",
    )
    staked_amount_usd = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=Decimal("0.00"),
        help_text="Amount staked as collateral",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")

    # Stats
    total_verifications = models.PositiveIntegerField(default=0)
    correct_verifications = models.PositiveIntegerField(default=0)
    incorrect_verifications = models.PositiveIntegerField(default=0)
    accuracy_score = models.FloatField(
        default=100.0, validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    total_rewards_earned = models.DecimalField(
        max_digits=12, decimal_places=2, default=Decimal("0.00")
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-accuracy_score", "-total_verifications"]

    def __str__(self):
        return f"Validator {self.user.username} (acc: {self.accuracy_score:.1f}%)"

    @property
    def is_active(self):
        return self.status == "active"

    def update_accuracy(self):
        total = self.correct_verifications + self.incorrect_verifications
        if total > 0:
            self.accuracy_score = round((self.correct_verifications / total) * 100, 2)
            self.save(update_fields=["accuracy_score"])
        if self.accuracy_score < 50.0 and total >= 10:
            self.status = "slashed"
            self.save(update_fields=["status"])


class VerificationTask(models.Model):
    """
    A verification task assigned to multiple validators when a ComputeJob completes.
    Consensus is reached when a majority of validators agree.
    """

    STATUS_CHOICES = [
        ("pending", "Pending Assignment"),
        ("in_progress", "In Progress"),
        ("consensus_reached", "Consensus Reached"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.OneToOneField(
        "gpu_marketplace.ComputeJob",
        on_delete=models.CASCADE,
        related_name="verification_task",
    )
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="pending")
    required_validators = models.PositiveIntegerField(
        default=3, help_text="Number of validators required for consensus"
    )
    consensus_result = models.CharField(
        max_length=20,
        choices=[("valid", "Valid"), ("invalid", "Invalid")],
        null=True,
        blank=True,
    )
    consensus_reached_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Verify Job {self.job_id.hex[:8]} ({self.status})"

    def check_consensus(self):
        """Check if consensus has been reached."""
        votes = self.verifications.values_list("result", flat=True)
        valid_count = sum(1 for v in votes if v == "valid")
        invalid_count = len(votes) - valid_count
        if len(votes) >= self.required_validators:
            if valid_count > invalid_count:
                self.consensus_result = "valid"
            else:
                self.consensus_result = "invalid"
            self.status = "consensus_reached"
            self.consensus_reached_at = timezone.now()
            self.save()
            return True
        return False


class Verification(models.Model):
    """
    A single validator's verification result for a task.
    """

    RESULT_CHOICES = [("valid", "Valid"), ("invalid", "Invalid")]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(
        VerificationTask, on_delete=models.CASCADE, related_name="verifications"
    )
    validator = models.ForeignKey(
        ValidatorNode, on_delete=models.CASCADE, related_name="verifications"
    )
    result = models.CharField(max_length=10, choices=RESULT_CHOICES)
    confidence = models.FloatField(
        default=1.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    evidence = models.TextField(blank=True, help_text="Any evidence or notes")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["task", "validator"]]
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.validator.user.username} → {self.result} (Task {self.task_id})"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # After saving, check if consensus is reached
        self.task.check_consensus()


class ProofOfCompute(models.Model):
    """
    Cryptographic proof that a job executed correctly.
    This can be a hash of outputs, a TEE attestation, or validator signatures.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.OneToOneField(
        "gpu_marketplace.ComputeJob",
        on_delete=models.CASCADE,
        related_name="proof_of_compute",
    )
    # Hash-based proof: SHA-256 of (job_id + outputs + timestamp)
    hash_signature = models.CharField(max_length=256, blank=True)
    # TEE attestation report (if available)
    tee_attestation = models.TextField(blank=True)
    # All validator signatures aggregated
    validator_signatures = models.JSONField(default=dict, blank=True)
    verified_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-verified_at"]

    def __str__(self):
        return f"Proof for Job {self.job_id.hex[:8]}"
