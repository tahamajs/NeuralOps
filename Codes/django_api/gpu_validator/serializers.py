from rest_framework import serializers

from .models import ProofOfCompute, ValidatorNode, Verification, VerificationTask


class ValidatorNodeSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ValidatorNode
        fields = [
            "id",
            "user",
            "username",
            "staked_amount_usd",
            "status",
            "total_verifications",
            "correct_verifications",
            "incorrect_verifications",
            "accuracy_score",
            "total_rewards_earned",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "username",
            "status",
            "total_verifications",
            "correct_verifications",
            "incorrect_verifications",
            "accuracy_score",
            "total_rewards_earned",
            "created_at",
            "updated_at",
        ]


class VerificationTaskSerializer(serializers.ModelSerializer):
    job_id = serializers.CharField(source="job.id", read_only=True)
    verification_count = serializers.SerializerMethodField()
    valid_count = serializers.SerializerMethodField()
    invalid_count = serializers.SerializerMethodField()

    class Meta:
        model = VerificationTask
        fields = [
            "id",
            "job",
            "job_id",
            "status",
            "required_validators",
            "consensus_result",
            "consensus_reached_at",
            "verification_count",
            "valid_count",
            "invalid_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "job",
            "job_id",
            "status",
            "consensus_result",
            "consensus_reached_at",
            "created_at",
            "updated_at",
        ]

    def get_verification_count(self, obj):
        return obj.verifications.count()

    def get_valid_count(self, obj):
        return obj.verifications.filter(result="valid").count()

    def get_invalid_count(self, obj):
        return obj.verifications.filter(result="invalid").count()


class VerificationSerializer(serializers.ModelSerializer):
    validator_username = serializers.CharField(
        source="validator.user.username", read_only=True
    )

    class Meta:
        model = Verification
        fields = [
            "id",
            "task",
            "validator",
            "validator_username",
            "result",
            "confidence",
            "evidence",
            "created_at",
        ]
        read_only_fields = ["id", "validator", "validator_username", "created_at"]


class ProofOfComputeSerializer(serializers.ModelSerializer):
    job_id = serializers.CharField(source="job.id", read_only=True)

    class Meta:
        model = ProofOfCompute
        fields = [
            "id",
            "job",
            "job_id",
            "hash_signature",
            "tee_attestation",
            "validator_signatures",
            "verified_at",
        ]
        read_only_fields = ["id", "job", "job_id", "verified_at"]
