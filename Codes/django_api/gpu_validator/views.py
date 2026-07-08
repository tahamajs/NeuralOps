import logging

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import (ProofOfCompute, ValidatorNode, Verification,
                     VerificationTask)
from .serializers import (ProofOfComputeSerializer, ValidatorNodeSerializer,
                          VerificationSerializer, VerificationTaskSerializer)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# VALIDATOR NODE MANAGEMENT
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def register_validator(request):
    """Register the authenticated user as a validator (stake required)."""
    if hasattr(request.user, "validator_node"):
        return Response(
            {"error": "Already a validator"}, status=status.HTTP_400_BAD_REQUEST
        )

    stake_amount = request.data.get("stake_amount_usd")
    if not stake_amount:
        return Response(
            {"error": "stake_amount_usd required"}, status=status.HTTP_400_BAD_REQUEST
        )

    try:
        stake_amount = float(stake_amount)
        if stake_amount < 10.0:
            return Response(
                {"error": "Minimum stake is $10"}, status=status.HTTP_400_BAD_REQUEST
            )
    except ValueError:
        return Response(
            {"error": "Invalid stake amount"}, status=status.HTTP_400_BAD_REQUEST
        )

    # In production: verify the user actually has the funds in their escrow account
    node = ValidatorNode.objects.create(
        user=request.user, staked_amount_usd=stake_amount, status="active"
    )
    return Response(ValidatorNodeSerializer(node).data, status=status.HTTP_201_CREATED)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_validator_status(request):
    """Get the authenticated user's validator status."""
    if not hasattr(request.user, "validator_node"):
        return Response({"error": "Not a validator"}, status=status.HTTP_404_NOT_FOUND)
    return Response(ValidatorNodeSerializer(request.user.validator_node).data)


@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def update_validator(request):
    """Update the validator's stake or status."""
    node = get_object_or_404(ValidatorNode, user=request.user)

    if "staked_amount_usd" in request.data:
        new_stake = request.data["staked_amount_usd"]
        node.staked_amount_usd = new_stake
        node.save(update_fields=["staked_amount_usd"])

    if "status" in request.data:
        new_status = request.data["status"]
        if new_status in ("active", "inactive"):
            node.status = new_status
            node.save(update_fields=["status"])

    return Response(ValidatorNodeSerializer(node).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def validator_list(request):
    """List all active validators with their stats."""
    qs = ValidatorNode.objects.filter(status="active").order_by("-accuracy_score")
    search = request.query_params.get("q")
    if search:
        qs = qs.filter(user__username__icontains=search)
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ValidatorNodeSerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def validator_detail(request, node_id):
    """View a specific validator's details."""
    node = get_object_or_404(ValidatorNode, id=node_id)
    return Response(ValidatorNodeSerializer(node).data)


# ═══════════════════════════════════════════════════════════════
# VERIFICATION TASKS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def verification_task_list(request):
    """List all verification tasks."""
    qs = VerificationTask.objects.all().order_by("-created_at")
    status_filter = request.query_params.get("status")
    if status_filter:
        qs = qs.filter(status=status_filter)
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        VerificationTaskSerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def verification_task_detail(request, task_id):
    """Get details of a verification task including all validator results."""
    task = get_object_or_404(VerificationTask, id=task_id)
    data = VerificationTaskSerializer(task).data
    data["verifications"] = VerificationSerializer(
        task.verifications.all(), many=True
    ).data
    return Response(data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_verification_tasks(request):
    """List verification tasks assigned to the authenticated validator."""
    if not hasattr(request.user, "validator_node"):
        return Response({"error": "Not a validator"}, status=status.HTTP_404_NOT_FOUND)
    node = request.user.validator_node
    tasks = (
        VerificationTask.objects.filter(verifications__validator=node)
        .distinct()
        .order_by("-created_at")
    )
    paginator = StandardPagination()
    page = paginator.paginate_queryset(tasks, request)
    return paginator.get_paginated_response(
        VerificationTaskSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# SUBMIT VERIFICATION
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def submit_verification(request, task_id):
    """
    A validator submits their verification result for a task.
    Body: { "result": "valid" or "invalid", "confidence": 0.0-1.0, "evidence": "..." }
    """
    if not hasattr(request.user, "validator_node"):
        return Response({"error": "Not a validator"}, status=status.HTTP_403_FORBIDDEN)

    node = request.user.validator_node
    if not node.is_active:
        return Response(
            {"error": "Validator is not active"}, status=status.HTTP_403_FORBIDDEN
        )

    task = get_object_or_404(VerificationTask, id=task_id)

    if task.status not in ("pending", "in_progress"):
        return Response(
            {"error": f"Task is {task.status}, cannot accept new verifications"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    result = request.data.get("result")
    if result not in ("valid", "invalid"):
        return Response(
            {"error": "result must be 'valid' or 'invalid'"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    confidence = float(request.data.get("confidence", 1.0))
    evidence = request.data.get("evidence", "")

    # Create or update verification
    verification, created = Verification.objects.update_or_create(
        task=task,
        validator=node,
        defaults={
            "result": result,
            "confidence": confidence,
            "evidence": evidence,
        },
    )

    # Update task status
    if task.status == "pending":
        task.status = "in_progress"
        task.save(update_fields=["status"])

    # Update validator stats (will be finalized after consensus)
    node.total_verifications += 1
    node.save(update_fields=["total_verifications"])

    return Response(
        VerificationSerializer(verification).data,
        status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
    )


# ═══════════════════════════════════════════════════════════════
# PROOF OF COMPUTE
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def proof_of_compute_detail(request, job_id):
    """Get the proof of compute for a specific job."""
    proof = get_object_or_404(ProofOfCompute, job_id=job_id)
    return Response(ProofOfComputeSerializer(proof).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_proof(request, job_id):
    """
    Generate a proof of compute for a job (called by the provider).
    Body: { "outputs_hash": "...", "tee_attestation": "..." }
    """
    from gpu_marketplace.models import ComputeJob

    job = get_object_or_404(ComputeJob, id=job_id)

    if job.provider != request.user:
        return Response(
            {"error": "Only the provider can generate proof"},
            status=status.HTTP_403_FORBIDDEN,
        )

    outputs_hash = request.data.get("outputs_hash", "")
    tee_attestation = request.data.get("tee_attestation", "")

    proof, created = ProofOfCompute.objects.update_or_create(
        job=job,
        defaults={
            "hash_signature": outputs_hash,
            "tee_attestation": tee_attestation,
        },
    )
    return Response(
        ProofOfComputeSerializer(proof).data,
        status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
    )


# ═══════════════════════════════════════════════════════════════
# CONSENSUS RESOLUTION
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def finalize_consensus(request, task_id):
    """Manually trigger consensus check (or it auto-checks on each vote)."""
    task = get_object_or_404(VerificationTask, id=task_id)
    if task.status != "in_progress":
        return Response(
            {"error": "Task is not in progress"}, status=status.HTTP_400_BAD_REQUEST
        )
    reached = task.check_consensus()
    if reached:
        # Update validator correctness scores
        for v in task.verifications.all():
            if v.result == task.consensus_result:
                v.validator.correct_verifications += 1
            else:
                v.validator.incorrect_verifications += 1
            v.validator.update_accuracy()
        task.status = "completed"
        task.save(update_fields=["status"])
    return Response(VerificationTaskSerializer(task).data)
