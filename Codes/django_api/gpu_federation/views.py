"""Federation REST API — complete CRUD for clusters, nodes, tasks, and results."""

import logging

from django.db.models import Q
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .coordinator import FederationCoordinator
from .models import *
from .serializers import *
from .tasks import run_federation_task

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# CLUSTERS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def cluster_list(request):
    if request.method == "GET":
        qs = (
            FederationCluster.objects.filter(
                Q(owner=request.user) | Q(nodes__listing__provider=request.user)
            )
            .distinct()
            .order_by("-created_at")
        )
        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            FederationClusterListSerializer(page, many=True).data
        )
    elif request.method == "POST":
        serializer = FederationClusterDetailSerializer(data=request.data)
        if serializer.is_valid():
            cluster = serializer.save(owner=request.user)
            return Response(FederationClusterDetailSerializer(cluster).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def cluster_detail(request, cluster_id):
    cluster = get_object_or_404(FederationCluster, id=cluster_id)
    if request.method == "GET":
        return Response(FederationClusterDetailSerializer(cluster).data)
    elif request.method == "PATCH":
        if cluster.owner != request.user:
            return Response({"error": "Only the owner can edit"}, status=403)
        serializer = FederationClusterDetailSerializer(
            cluster, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if cluster.owner != request.user:
            return Response({"error": "Only the owner can delete"}, status=403)
        cluster.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def add_node(request, cluster_id):
    """Add a GPU listing to a cluster."""
    cluster = get_object_or_404(FederationCluster, id=cluster_id, owner=request.user)
    listing_id = request.data.get("listing_id")
    if not listing_id:
        return Response({"error": "listing_id required"}, status=400)

    from gpu_marketplace.models import GPUListing

    listing = get_object_or_404(GPUListing, id=listing_id)

    if cluster.nodes.count() >= cluster.max_nodes:
        return Response({"error": "Cluster has reached maximum node limit"}, status=400)

    node, created = FederationNode.objects.get_or_create(
        cluster=cluster, listing=listing, defaults={"status": "active"}
    )
    if created:
        cluster.recalculate_resources()
    return Response(FederationNodeSerializer(node).data, status=201 if created else 200)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def remove_node(request, cluster_id, node_id):
    cluster = get_object_or_404(FederationCluster, id=cluster_id, owner=request.user)
    node = get_object_or_404(FederationNode, id=node_id, cluster=cluster)
    node.delete()
    cluster.recalculate_resources()
    return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def node_heartbeat(request, cluster_id, node_id):
    """Receive heartbeat from a provider's node agent."""
    node = get_object_or_404(FederationNode, id=node_id, cluster_id=cluster_id)
    if request.user != node.listing.provider:
        return Response({"error": "Only the provider can heartbeat"}, status=403)
    node.last_heartbeat = timezone.now()
    node.save(update_fields=["last_heartbeat"])
    return Response({"status": "ok"})


# ═══════════════════════════════════════════════════════════════
# TASKS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def task_list(request, cluster_id):
    cluster = get_object_or_404(FederationCluster, id=cluster_id)
    if request.method == "GET":
        qs = cluster.tasks.order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            FederationTaskSerializer(page, many=True).data
        )
    elif request.method == "POST":
        serializer = FederationTaskSerializer(data=request.data)
        if serializer.is_valid():
            task = serializer.save(cluster=cluster, owner=request.user)
            return Response(FederationTaskSerializer(task).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def launch_task(request, task_id):
    """Launch a federation task (initialize and start running)."""
    task = get_object_or_404(FederationTask, id=task_id)
    if task.owner != request.user:
        return Response({"error": "Only the owner can launch"}, status=403)
    if task.status != "queued":
        return Response({"error": f"Task is already {task.status}"}, status=400)

    coordinator = FederationCoordinator(task)
    try:
        coordinator.initialize()
        # Trigger Celery task to actually run
        run_federation_task.delay(str(task.id))
        return Response(FederationTaskSerializer(task).data)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def task_detail(request, task_id):
    task = get_object_or_404(FederationTask, id=task_id)
    return Response(FederationTaskSerializer(task).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def task_cancel(request, task_id):
    task = get_object_or_404(FederationTask, id=task_id)
    if task.owner != request.user:
        return Response({"error": "Only the owner can cancel"}, status=403)
    task.status = "canceled"
    task.completed_at = timezone.now()
    task.save(update_fields=["status", "completed_at"])
    return Response(FederationTaskSerializer(task).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def task_result(request, task_id):
    task = get_object_or_404(FederationTask, id=task_id)
    if not hasattr(task, "result"):
        return Response({"error": "No results yet"}, status=404)
    return Response(FederationResultSerializer(task.result).data)


# ═══════════════════════════════════════════════════════════════
# PROVIDER DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def provider_federation_dashboard(request):
    """Dashboard for providers to see their participating nodes and earnings."""
    nodes = FederationNode.objects.filter(
        listing__provider=request.user
    ).select_related("cluster")

    data = []
    for node in nodes:
        data.append(
            {
                "node_id": str(node.id),
                "listing_id": str(node.listing.id),
                "gpu_model": node.listing.gpu_model,
                "cluster_name": node.cluster.name,
                "cluster_owner": node.cluster.owner.username,
                "status": node.status,
                "shard_start": node.shard_start,
                "shard_end": node.shard_end,
                "tasks_completed": node.task_shards.filter(status="completed").count(),
                "earnings_usd": "0.00",  # placeholder
            }
        )
    return Response(data)
