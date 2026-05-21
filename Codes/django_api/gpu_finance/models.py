"""
GPU Finance Models — Commodities‑Style Spot Market, Futures, Derivatives, and Billing

Architecture inspired by:
  - CME Group Compute Futures (2026): cash‑settled contracts on GPU lease rate indices
  - ACE (Agentic Capacity Exchange): spot market + futures + options for GPU compute
  - GAIB: tokenized GPU yields, USDai stablecoin, debt/equity/hybrid financing
  - Akash: escrow with ACT (USD‑pegged), automatic per‑block provider payouts
  - Vast.ai: Stripe autobilling, invoice history, host payouts
  - io.net: USD‑denominated payouts, dynamic incentive engine

Taxonomy:
    ComputeIndex → FuturesContract → FuturesPosition → FuturesSettlement
    Invoice → InvoiceLineItem → Payout → PaymentMethod
    BillingPlan → Subscription → UsageMeter → UsageRecord
    DerivativeOption → OptionPosition
"""
import uuid
from decimal import Decimal
from django.db import models, transaction
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


# ═══════════════════════════════════════════════════════════════
# 1. COMPUTE INDEX — The "Benchmark Price" (like WTI / Brent)
# ═══════════════════════════════════════════════════════════════
class ComputeIndex(models.Model):
    """
    A daily benchmark index tracking GPU lease rates for a specific GPU model.
    Like CME Group's Silicon Data GPU benchmark index — the underlying
    reference price for futures contracts.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    gpu_model = models.CharField(max_length=100, unique=True,
                                  help_text="e.g., 'H100-SXM-80GB', 'A100-SXM4-40GB'")
    index_name = models.CharField(max_length=200, unique=True,
                                   help_text="e.g., 'NEURALOPS-H100-IDX'")

    # Current index value (USD per GPU-hour)
    current_value = models.DecimalField(max_digits=10, decimal_places=4)
    previous_value = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    daily_change_pct = models.FloatField(null=True)

    # Aggregation metadata
    data_points = models.PositiveIntegerField(default=0,
                                               help_text="Number of listings used in this calculation")
    sample_std_dev = models.FloatField(null=True)

    # Timestamp
    calculated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['gpu_model']

    def __str__(self):
        return f"{self.index_name}: ${self.current_value}/hr"

    def calculate(self):
        """Recalculate the index from live marketplace listings."""
        from gpu_marketplace.models import GPUListing
        listings = GPUListing.objects.filter(
            gpu_model__iexact=self.gpu_model, status='online'
        )
        if not listings.exists():
            self.current_value = Decimal('0')
            self.data_points = 0
            self.save()
            return

        prices = [l.current_price_per_hour for l in listings]
        self.previous_value = self.current_value
        self.current_value = sum(prices) / len(prices)
        self.data_points = len(prices)
        if self.previous_value and self.previous_value > 0:
            self.daily_change_pct = round(
                float((self.current_value - self.previous_value) / self.previous_value) * 100, 2
            )
        self.save()


class IndexHistory(models.Model):
    """Historical daily snapshots of a compute index for charting."""
    index = models.ForeignKey(ComputeIndex, on_delete=models.CASCADE, related_name='history')
    value = models.DecimalField(max_digits=10, decimal_places=4)
    data_points = models.PositiveIntegerField(default=0)
    recorded_at = models.DateField(auto_now_add=True)

    class Meta:
        unique_together = [['index', 'recorded_at']]
        ordering = ['-recorded_at']


# ═══════════════════════════════════════════════════════════════
# 2. FUTURES CONTRACTS — Standardized GPU compute derivatives
# ═══════════════════════════════════════════════════════════════
class FuturesContract(models.Model):
    """
    A standardized futures contract for GPU compute hours.
    Inspired by CME Group's compute futures: cash‑settled, based on benchmark index,
    with monthly/quarterly expiries.

    Contract specifications:
    - Underlying: ComputeIndex for a specific GPU model
    - Contract size: e.g., 100 GPU-hours
    - Settlement: Cash (no physical delivery of compute)
    - Expiry: Monthly / Quarterly
    """
    CONTRACT_MONTHS = [
        ('F', 'January'), ('G', 'February'), ('H', 'March'),
        ('J', 'April'), ('K', 'May'), ('M', 'June'),
        ('N', 'July'), ('Q', 'August'), ('U', 'September'),
        ('V', 'October'), ('X', 'November'), ('Z', 'December'),
    ]
    STATUS_CHOICES = [
        ('open', 'Open for Trading'),
        ('settling', 'Settling'),
        ('settled', 'Settled'),
        ('expired', 'Expired'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    index = models.ForeignKey(ComputeIndex, on_delete=models.CASCADE, related_name='futures')
    contract_code = models.CharField(max_length=20, unique=True,
                                      help_text="e.g., 'H100-F-2026-06'")
    contract_month = models.CharField(max_length=1, choices=CONTRACT_MONTHS)
    contract_year = models.PositiveIntegerField()
    contract_size_hours = models.PositiveIntegerField(default=100,
                                                       help_text="GPU-hours per contract")
    settlement_price = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')

    # Trading stats
    open_interest = models.PositiveIntegerField(default=0)
    total_volume = models.PositiveIntegerField(default=0)
    day_high = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    day_low = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    last_price = models.DecimalField(max_digits=10, decimal_places=4, null=True)

    # Dates
    first_notice_date = models.DateField(null=True)
    last_trading_date = models.DateField(null=True)
    settlement_date = models.DateField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['index', 'contract_month', 'contract_year']]
        ordering = ['contract_year', 'contract_month']

    def __str__(self):
        return f"{self.contract_code} ({self.status})"

    @property
    def is_expired(self):
        if self.last_trading_date and timezone.now().date() > self.last_trading_date:
            return True
        return False

    def settle(self, final_price: Decimal):
        """Cash‑settle all open positions at final_price."""
        self.settlement_price = final_price
        self.status = 'settled'
        self.save()
        for pos in self.positions.filter(status='open'):
            pos.settle(final_price)


class FuturesPosition(models.Model):
    """A trader's position (long or short) in a futures contract."""
    POSITION_TYPES = [('long', 'Long'), ('short', 'Short')]
    STATUS_CHOICES = [('open', 'Open'), ('closed', 'Closed'), ('settled', 'Settled')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    contract = models.ForeignKey(FuturesContract, on_delete=models.CASCADE,
                                  related_name='positions')
    trader = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                               related_name='futures_positions')
    position_type = models.CharField(max_length=10, choices=POSITION_TYPES)
    quantity = models.PositiveIntegerField(default=1)
    entry_price = models.DecimalField(max_digits=10, decimal_places=4)
    exit_price = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')

    # P&L
    realized_pnl_usd = models.DecimalField(max_digits=12, decimal_places=2, null=True)
    margin_locked_usd = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    opened_at = models.DateTimeField(auto_now_add=True)
    closed_at = models.DateTimeField(null=True)

    class Meta:
        ordering = ['-opened_at']
        indexes = [
            models.Index(fields=['trader', 'status']),
            models.Index(fields=['contract', 'status']),
        ]

    def __str__(self):
        return f"{self.trader.username} {self.position_type} {self.quantity}x {self.contract.contract_code}"

    def settle(self, final_price: Decimal):
        """Cash‑settle this position."""
        self.exit_price = final_price
        self.status = 'settled'
        multiplier = 1 if self.position_type == 'long' else -1
        self.realized_pnl_usd = (
            (final_price - self.entry_price) * self.quantity * self.contract.contract_size_hours * multiplier
        )
        self.closed_at = timezone.now()
        self.save()
        # Update trader's escrow balance
        if self.realized_pnl_usd:
            _credit_user(self.trader, self.realized_pnl_usd, f"Futures P&L: {self.contract.contract_code}")


class FuturesOrder(models.Model):
    """A limit/market order for a futures contract (exchange‑style)."""
    ORDER_TYPES = [('buy', 'Buy'), ('sell', 'Sell')]
    ORDER_MODES = [('market', 'Market'), ('limit', 'Limit'), ('stop', 'Stop')]
    STATUS_CHOICES = [('open', 'Open'), ('filled', 'Filled'), ('canceled', 'Canceled')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    contract = models.ForeignKey(FuturesContract, on_delete=models.CASCADE, related_name='orders')
    trader = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                               related_name='futures_orders')
    order_type = models.CharField(max_length=10, choices=ORDER_TYPES)
    order_mode = models.CharField(max_length=10, choices=ORDER_MODES, default='limit')
    quantity = models.PositiveIntegerField(default=1)
    limit_price = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    stop_price = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    filled_quantity = models.PositiveIntegerField(default=0)
    avg_fill_price = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.order_type} {self.quantity}x {self.contract.contract_code} @ {self.limit_price}"


# ═══════════════════════════════════════════════════════════════
# 3. BILLING — Invoices, Payments, Payouts
# ═══════════════════════════════════════════════════════════════
class Invoice(models.Model):
    """A bill issued to a consumer for GPU usage."""
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('open', 'Open'),
        ('paid', 'Paid'),
        ('overdue', 'Overdue'),
        ('canceled', 'Canceled'),
        ('disputed', 'Disputed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='invoices')
    invoice_number = models.CharField(max_length=30, unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')

    # Totals
    subtotal_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    tax_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    amount_paid_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    # Dates
    period_start = models.DateField(null=True)
    period_end = models.DateField(null=True)
    due_date = models.DateField(null=True)
    paid_at = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"Invoice #{self.invoice_number} — {self.user.username}"

    def recalculate_totals(self):
        """Sum all line items and update totals."""
        items = self.line_items.all()
        self.subtotal_usd = sum(item.total_usd for item in items)
        self.tax_usd = self.subtotal_usd * Decimal('0.00')  # Tax placeholder
        self.total_usd = self.subtotal_usd + self.tax_usd
        self.save(update_fields=['subtotal_usd', 'tax_usd', 'total_usd'])


class InvoiceLineItem(models.Model):
    """A single charge on an invoice (GPU hours, storage, bandwidth, etc.)."""
    CHARGE_TYPES = [
        ('gpu_rental', 'GPU Rental'),
        ('storage', 'Storage'),
        ('bandwidth', 'Bandwidth'),
        ('benchmark', 'Benchmark Fee'),
        ('futures_settlement', 'Futures Settlement'),
        ('service_fee', 'Platform Service Fee'),
        ('credit', 'Credit'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    invoice = models.ForeignKey(Invoice, on_delete=models.CASCADE, related_name='line_items')
    charge_type = models.CharField(max_length=30, choices=CHARGE_TYPES)
    description = models.CharField(max_length=300)
    job = models.ForeignKey('gpu_marketplace.ComputeJob', on_delete=models.SET_NULL,
                            null=True, blank=True, related_name='invoice_items')
    quantity = models.DecimalField(max_digits=12, decimal_places=4, default=1)
    unit_price_usd = models.DecimalField(max_digits=10, decimal_places=4)
    total_usd = models.DecimalField(max_digits=12, decimal_places=2)

    class Meta:
        ordering = ['charge_type', 'description']

    def __str__(self):
        return f"{self.description}: ${self.total_usd}"


class Payout(models.Model):
    """A payment to a GPU provider for their earnings."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    METHOD_CHOICES = [
        ('stripe', 'Stripe'),
        ('paypal', 'PayPal'),
        ('wise', 'Wise'),
        ('crypto', 'Crypto (USDC)'),
        ('credits', 'Platform Credits'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    provider = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                 related_name='payouts')
    amount_usd = models.DecimalField(max_digits=12, decimal_places=2)
    fee_usd = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    net_amount_usd = models.DecimalField(max_digits=12, decimal_places=2)
    method = models.CharField(max_length=20, choices=METHOD_CHOICES, default='credits')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    reference = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Payout ${self.net_amount_usd} to {self.provider.username} ({self.status})"


class PaymentMethod(models.Model):
    """A user's saved payment method (card, crypto wallet, etc.)."""
    METHOD_TYPES = [
        ('credit_card', 'Credit Card'),
        ('debit_card', 'Debit Card'),
        ('crypto', 'Crypto Wallet'),
        ('bank_transfer', 'Bank Transfer'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='payment_methods')
    method_type = models.CharField(max_length=20, choices=METHOD_TYPES)
    is_default = models.BooleanField(default=False)
    is_verified = models.BooleanField(default=False)

    # Masked details (real sensitive data handled by Stripe / payment provider)
    last_four = models.CharField(max_length=4, blank=True)
    expiry_month = models.PositiveIntegerField(null=True)
    expiry_year = models.PositiveIntegerField(null=True)
    provider_token = models.CharField(max_length=200, blank=True,
                                       help_text="Stripe PaymentMethod ID or similar")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_default', '-created_at']

    def __str__(self):
        return f"{self.get_method_type_display()} *{self.last_four}"


# ═══════════════════════════════════════════════════════════════
# 4. SUBSCRIPTION — Usage‑Based Billing Plans
# ═══════════════════════════════════════════════════════════════
class BillingPlan(models.Model):
    """A usage‑based billing plan (free, pro, enterprise)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    tier = models.CharField(max_length=20, default='free',
                             choices=[('free','Free'),('pro','Pro'),('team','Team'),
                                      ('enterprise','Enterprise')])
    monthly_base_fee_usd = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    gpu_markup_pct = models.FloatField(default=5.0,
                                        help_text="Platform fee % added to GPU rental price")
    included_gpu_hours = models.PositiveIntegerField(default=0,
                                                      help_text="Free GPU hours per month")
    max_concurrent_jobs = models.PositiveIntegerField(default=2)
    priority_support = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['monthly_base_fee_usd']

    def __str__(self):
        return f"{self.name} (${self.monthly_base_fee_usd}/mo)"


class Subscription(models.Model):
    """A user's active subscription to a billing plan."""
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('canceled', 'Canceled'),
        ('past_due', 'Past Due'),
        ('trial', 'Trial'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name='billing_subscription')
    plan = models.ForeignKey(BillingPlan, on_delete=models.SET_NULL, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='trial')
    trial_ends_at = models.DateTimeField(null=True)
    current_period_start = models.DateTimeField(null=True)
    current_period_end = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} → {self.plan.name if self.plan else 'Free'}"


class UsageRecord(models.Model):
    """A metered usage record for a consumer (GPU hours consumed)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='usage_records')
    job = models.ForeignKey('gpu_marketplace.ComputeJob', on_delete=models.SET_NULL,
                            null=True, blank=True, related_name='usage_records')
    gpu_model = models.CharField(max_length=100)
    hours_consumed = models.FloatField()
    cost_usd = models.DecimalField(max_digits=12, decimal_places=2)
    platform_fee_usd = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-recorded_at']
        indexes = [
            models.Index(fields=['user', '-recorded_at']),
        ]

    def __str__(self):
        return f"{self.user.username}: {self.hours_consumed}h on {self.gpu_model} — ${self.cost_usd}"


# ═══════════════════════════════════════════════════════════════
# 5. HELPERS
# ═══════════════════════════════════════════════════════════════
def _credit_user(user, amount: Decimal, description: str = ""):
    """Credit a user's escrow account with funds from futures P&L."""
    try:
        from gpu_escrow.models import EscrowAccount
        account, _ = EscrowAccount.objects.get_or_create(user=user)
        account.deposit(amount, description)
    except ImportError:
        pass