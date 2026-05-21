from rest_framework import serializers
from .models import *


class ComputeIndexSerializer(serializers.ModelSerializer):
    class Meta:
        model = ComputeIndex
        fields = ['id', 'gpu_model', 'index_name', 'current_value',
                  'previous_value', 'daily_change_pct', 'data_points',
                  'sample_std_dev', 'calculated_at']


class IndexHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = IndexHistory
        fields = ['value', 'data_points', 'recorded_at']


class FuturesContractSerializer(serializers.ModelSerializer):
    index_name = serializers.CharField(source='index.index_name', read_only=True)
    current_index_value = serializers.DecimalField(source='index.current_value',
                                                    max_digits=10, decimal_places=4, read_only=True)

    class Meta:
        model = FuturesContract
        fields = ['id', 'index', 'index_name', 'current_index_value',
                  'contract_code', 'contract_month', 'contract_year',
                  'contract_size_hours', 'settlement_price', 'status',
                  'open_interest', 'total_volume', 'day_high', 'day_low',
                  'last_price', 'first_notice_date', 'last_trading_date',
                  'settlement_date', 'created_at']
        read_only_fields = ['id', 'status', 'settlement_price', 'open_interest',
                            'total_volume', 'day_high', 'day_low', 'last_price',
                            'created_at']


class FuturesPositionSerializer(serializers.ModelSerializer):
    trader_username = serializers.CharField(source='trader.username', read_only=True)
    contract_code = serializers.CharField(source='contract.contract_code', read_only=True)

    class Meta:
        model = FuturesPosition
        fields = ['id', 'contract', 'contract_code', 'trader', 'trader_username',
                  'position_type', 'quantity', 'entry_price', 'exit_price',
                  'status', 'realized_pnl_usd', 'margin_locked_usd',
                  'opened_at', 'closed_at']
        read_only_fields = ['id', 'trader', 'status', 'exit_price',
                            'realized_pnl_usd', 'opened_at', 'closed_at']


class FuturesOrderSerializer(serializers.ModelSerializer):
    trader_username = serializers.CharField(source='trader.username', read_only=True)
    contract_code = serializers.CharField(source='contract.contract_code', read_only=True)

    class Meta:
        model = FuturesOrder
        fields = ['id', 'contract', 'contract_code', 'trader', 'trader_username',
                  'order_type', 'order_mode', 'quantity', 'limit_price',
                  'stop_price', 'status', 'filled_quantity', 'avg_fill_price',
                  'created_at']
        read_only_fields = ['id', 'trader', 'status', 'filled_quantity',
                            'avg_fill_price', 'created_at']


class InvoiceLineItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = InvoiceLineItem
        fields = ['id', 'invoice', 'charge_type', 'description', 'job',
                  'quantity', 'unit_price_usd', 'total_usd']


class InvoiceSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    line_items = InvoiceLineItemSerializer(many=True, read_only=True)
    balance_due = serializers.SerializerMethodField()

    class Meta:
        model = Invoice
        fields = ['id', 'user', 'username', 'invoice_number', 'status',
                  'subtotal_usd', 'tax_usd', 'total_usd', 'amount_paid_usd',
                  'balance_due', 'period_start', 'period_end', 'due_date',
                  'paid_at', 'line_items', 'created_at']
        read_only_fields = ['id', 'user', 'username', 'status', 'subtotal_usd',
                            'tax_usd', 'total_usd', 'amount_paid_usd',
                            'paid_at', 'created_at']

    def get_balance_due(self, obj):
        return obj.total_usd - obj.amount_paid_usd


class PayoutSerializer(serializers.ModelSerializer):
    provider_username = serializers.CharField(source='provider.username', read_only=True)

    class Meta:
        model = Payout
        fields = ['id', 'provider', 'provider_username', 'amount_usd',
                  'fee_usd', 'net_amount_usd', 'method', 'status',
                  'reference', 'created_at', 'completed_at']
        read_only_fields = ['id', 'provider', 'status', 'net_amount_usd',
                            'created_at', 'completed_at']


class PaymentMethodSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentMethod
        fields = ['id', 'user', 'method_type', 'is_default', 'is_verified',
                  'last_four', 'expiry_month', 'expiry_year', 'created_at']
        read_only_fields = ['id', 'user', 'created_at']


class BillingPlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = BillingPlan
        fields = '__all__'


class SubscriptionSerializer(serializers.ModelSerializer):
    plan_name = serializers.CharField(source='plan.name', read_only=True)
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Subscription
        fields = ['id', 'user', 'username', 'plan', 'plan_name', 'status',
                  'trial_ends_at', 'current_period_start', 'current_period_end',
                  'created_at']
        read_only_fields = ['id', 'user', 'username', 'created_at']


class UsageRecordSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = UsageRecord
        fields = ['id', 'user', 'username', 'job', 'gpu_model',
                  'hours_consumed', 'cost_usd', 'platform_fee_usd',
                  'recorded_at']
        read_only_fields = ['id', 'user', 'username', 'recorded_at']