# gpu_escrow/models.py
from django.db import models
from django.conf import settings

class EscrowAccount(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name='escrow_account')
    balance_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    locked_balance_usd = models.DecimalField(max_digits=12, decimal_places=2, default=0)

class EscrowTransaction(models.Model):
    TX_TYPES = [
        ('deposit', 'Deposit'),
        ('lock', 'Lock'),
        ('release', 'Release'),
        ('refund', 'Refund'),
        ('penalty', 'Penalty'),
    ]
    job = models.ForeignKey('gpu_marketplace.ComputeJob', on_delete=models.CASCADE,
                            related_name='escrow_txs')
    tx_type = models.CharField(max_length=20, choices=TX_TYPES)
    amount_usd = models.DecimalField(max_digits=10, decimal_places=2)
    from_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                  related_name='escrow_out')
    to_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name='escrow_in', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)