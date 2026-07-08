# gpu_escrow/admin.py
from django.contrib import admin

from .models import EscrowAccount, EscrowTransaction


@admin.register(EscrowAccount)
class EscrowAccountAdmin(admin.ModelAdmin):
    list_display = ("user", "balance_usd", "locked_balance_usd", "created_at")
    search_fields = ("user__username",)
    readonly_fields = ("balance_usd", "locked_balance_usd", "created_at", "updated_at")


@admin.register(EscrowTransaction)
class EscrowTransactionAdmin(admin.ModelAdmin):
    list_display = ("account", "tx_type", "amount_usd", "job", "created_at")
    list_filter = ("tx_type",)
    search_fields = ("account__user__username", "job__id")
    readonly_fields = ("created_at",)
