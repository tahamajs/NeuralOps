# gpu_escrow/models.py
import uuid
from decimal import Decimal

from django.conf import settings
from django.db import models, transaction


class EscrowAccount(models.Model):
    """
    A user's escrow wallet.
    - balance_usd: available funds (can be withdrawn)
    - locked_balance_usd: funds held in escrow for ongoing jobs
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="escrow_account",
    )
    balance_usd = models.DecimalField(
        max_digits=12, decimal_places=2, default=Decimal("0.00")
    )
    locked_balance_usd = models.DecimalField(
        max_digits=12, decimal_places=2, default=Decimal("0.00")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["user__username"]

    def __str__(self):
        return f"Escrow for {self.user.username}: ${self.balance_usd} (${self.locked_balance_usd} locked)"

    @property
    def total_balance_usd(self):
        """Total funds (available + locked)."""
        return self.balance_usd + self.locked_balance_usd

    @transaction.atomic
    def deposit(self, amount_usd: Decimal, description: str = ""):
        """Add funds to the account."""
        self.balance_usd += amount_usd
        self.save(update_fields=["balance_usd", "updated_at"])
        EscrowTransaction.objects.create(
            account=self,
            tx_type="deposit",
            amount_usd=amount_usd,
            description=description,
        )

    @transaction.atomic
    def withdraw(self, amount_usd: Decimal, description: str = ""):
        """Remove available funds from the account."""
        if amount_usd > self.balance_usd:
            raise ValueError("Insufficient available balance")
        self.balance_usd -= amount_usd
        self.save(update_fields=["balance_usd", "updated_at"])
        EscrowTransaction.objects.create(
            account=self,
            tx_type="withdrawal",
            amount_usd=-amount_usd,  # negative to indicate withdrawal
            description=description,
        )

    @transaction.atomic
    def lock_funds(self, amount_usd: Decimal, job=None):
        """Move funds from available to locked for a job."""
        if amount_usd > self.balance_usd:
            raise ValueError("Insufficient available balance to lock")
        self.balance_usd -= amount_usd
        self.locked_balance_usd += amount_usd
        self.save(update_fields=["balance_usd", "locked_balance_usd", "updated_at"])
        EscrowTransaction.objects.create(
            account=self,
            tx_type="lock",
            amount_usd=amount_usd,
            job=job,
        )

    @transaction.atomic
    def release_funds(self, amount_usd: Decimal, to_account: "EscrowAccount", job=None):
        """Release locked funds to another account (provider)."""
        if amount_usd > self.locked_balance_usd:
            raise ValueError("Insufficient locked balance to release")
        self.locked_balance_usd -= amount_usd
        self.save(update_fields=["locked_balance_usd", "updated_at"])
        to_account.balance_usd += amount_usd
        to_account.save(update_fields=["balance_usd", "updated_at"])
        EscrowTransaction.objects.create(
            account=self,  # consumer's account (funds leave)
            tx_type="release",
            amount_usd=amount_usd,
            job=job,
            counterparty_account=to_account,
        )

    @transaction.atomic
    def refund_locked_funds(self, amount_usd: Decimal, job=None):
        """Return locked funds back to available balance (refund)."""
        if amount_usd > self.locked_balance_usd:
            raise ValueError("Insufficient locked balance to refund")
        self.locked_balance_usd -= amount_usd
        self.balance_usd += amount_usd
        self.save(update_fields=["balance_usd", "locked_balance_usd", "updated_at"])
        EscrowTransaction.objects.create(
            account=self,
            tx_type="refund",
            amount_usd=amount_usd,
            job=job,
        )


class EscrowTransaction(models.Model):
    """
    A record of every movement of funds in the escrow system.
    Positive amount_usd = funds added to account; negative = removed.
    For lock/release/refund, amount_usd is positive and the direction
    is determined by tx_type and counterparty.
    """

    TX_TYPES = [
        ("deposit", "Deposit"),
        ("withdrawal", "Withdrawal"),
        ("lock", "Lock for Job"),
        ("release", "Release to Provider"),
        ("refund", "Refund to Consumer"),
        ("penalty", "Penalty / Slashing"),
        ("reward", "Reward"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(
        EscrowAccount, on_delete=models.CASCADE, related_name="transactions"
    )
    tx_type = models.CharField(max_length=20, choices=TX_TYPES)
    amount_usd = models.DecimalField(max_digits=12, decimal_places=2)
    description = models.TextField(blank=True)

    # For lock/release/refund, link to the job and maybe the counterparty account
    job = models.ForeignKey(
        "gpu_marketplace.ComputeJob",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="escrow_transactions",
    )
    counterparty_account = models.ForeignKey(
        EscrowAccount,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="incoming_transactions",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["account", "-created_at"]),
            models.Index(fields=["job"]),
            models.Index(fields=["tx_type"]),
        ]

    def __str__(self):
        return f"{self.tx_type} of ${self.amount_usd} for {self.account.user.username}"

    @property
    def is_credit(self):
        """True if this transaction adds to the user's total balance."""
        return self.tx_type in ("deposit", "release", "refund", "reward")
