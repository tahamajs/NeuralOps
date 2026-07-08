from django.contrib import admin

from .models import (ProofOfCompute, ValidatorNode, Verification,
                     VerificationTask)


@admin.register(ValidatorNode)
class ValidatorNodeAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "status",
        "staked_amount_usd",
        "accuracy_score",
        "total_verifications",
        "created_at",
    )
    list_filter = ("status",)
    search_fields = ("user__username",)
    readonly_fields = (
        "accuracy_score",
        "total_verifications",
        "correct_verifications",
        "incorrect_verifications",
        "total_rewards_earned",
    )


@admin.register(VerificationTask)
class VerificationTaskAdmin(admin.ModelAdmin):
    list_display = (
        "job",
        "status",
        "required_validators",
        "consensus_result",
        "created_at",
    )
    list_filter = ("status", "consensus_result")


@admin.register(Verification)
class VerificationAdmin(admin.ModelAdmin):
    list_display = ("task", "validator", "result", "confidence", "created_at")
    list_filter = ("result",)


@admin.register(ProofOfCompute)
class ProofOfComputeAdmin(admin.ModelAdmin):
    list_display = ("job", "verified_at")
    readonly_fields = ("verified_at",)
