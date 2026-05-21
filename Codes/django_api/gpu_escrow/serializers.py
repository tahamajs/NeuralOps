# gpu_escrow/serializers.py
from rest_framework import serializers
from .models import EscrowAccount, EscrowTransaction


class EscrowAccountSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = EscrowAccount
        fields = [
            'id', 'user', 'username',
            'balance_usd', 'locked_balance_usd',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'user', 'username', 'balance_usd',
                            'locked_balance_usd', 'created_at', 'updated_at']


class EscrowTransactionSerializer(serializers.ModelSerializer):
    account_username = serializers.CharField(source='account.user.username', read_only=True)
    counterparty_username = serializers.CharField(
        source='counterparty_account.user.username', read_only=True, default=None
    )
    job_id = serializers.CharField(source='job.id', read_only=True, default=None)

    class Meta:
        model = EscrowTransaction
        fields = [
            'id', 'account', 'account_username', 'tx_type',
            'amount_usd', 'description', 'job', 'job_id',
            'counterparty_account', 'counterparty_username',
            'created_at',
        ]
        read_only_fields = ['id', 'account', 'account_username', 'tx_type',
                            'job', 'job_id', 'counterparty_account',
                            'counterparty_username', 'created_at']


class DepositWithdrawSerializer(serializers.Serializer):
    amount_usd = serializers.DecimalField(max_digits=10, decimal_places=2, min_value=0.01)
    description = serializers.CharField(required=False, allow_blank=True, default='')