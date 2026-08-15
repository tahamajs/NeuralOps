from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *
from .tasks import *


# ── Inference Endpoints ──────────────────────────────────────
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def endpoint_list(request):
    if request.method == "GET":
        endpoints = InferenceEndpoint.objects.filter(owner=request.user)
        return Response(InferenceEndpointSerializer(endpoints, many=True).data)
    elif request.method == "POST":
        serializer = InferenceEndpointSerializer(data=request.data)
        if serializer.is_valid():
            endpoint = serializer.save(owner=request.user)
            deploy_inference_endpoint.delay(str(endpoint.id))
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def call_endpoint(request, endpoint_id):
    """Invoke a deployed endpoint."""
    endpoint = get_object_or_404(InferenceEndpoint, id=endpoint_id)
    if endpoint.owner != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    input_data = request.data.get("input", "")
    # Call the actual endpoint URL
    import requests

    try:
        resp = requests.post(endpoint.endpoint_url, json={"inputs": input_data})
        EndpointLog.objects.create(
            endpoint=endpoint,
            input_data=input_data,
            output_data=resp.text,
            latency_ms=resp.elapsed.total_seconds() * 1000,
            status_code=resp.status_code,
        )
        return Response(resp.json())
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# ── Pipelines ──────────────────────────────────────────────────
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_pipeline(request, pipeline_id):
    pipeline = get_object_or_404(Pipeline, id=pipeline_id)
    run = PipelineRun.objects.create(
        pipeline=pipeline,
        triggered_by=request.user,
        params=request.data.get("params", {}),
    )
    execute_pipeline.delay(str(run.id))
    return Response(PipelineRunSerializer(run).data, status=201)


# ── Feature Store ──────────────────────────────────────────────
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def feature_ingest(request):
    group = request.data.get("group")
    entity_id = request.data.get("entity_id")
    values = request.data.get("values", {})
    FeatureStore.ingest(group, entity_id, values)
    return Response({"status": "ingested"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def feature_serve(request):
    group = request.query_params.get("group")
    entity_ids = request.query_params.getlist("entity_ids")
    data = FeatureStore.serve(group, entity_ids)
    return Response(data)


# ── CT/CD ──────────────────────────────────────────────────────
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def trigger_ctcd(request, pipeline_id):
    ctcd = get_object_or_404(CTCDPipeline, id=pipeline_id)
    run = CTCDRun.objects.create(pipeline=ctcd, status="queued")
    run_ctcd_pipeline.delay(str(run.id))
    return Response({"run_id": str(run.id)}, status=201)


# ── Distillation ───────────────────────────────────────────────
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_distillation(request):
    serializer = DistillationJobSerializer(data=request.data)
    if serializer.is_valid():
        job = serializer.save(user=request.user, status="queued")
        run_distillation.delay(str(job.id))
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


# ── Multi‑Node Fine‑Tuning ─────────────────────────────────────
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_multi_node_training(request):
    serializer = MultiNodeTrainingSerializer(data=request.data)
    if serializer.is_valid():
        training = serializer.save(user=request.user, status="queued")
        launch_multi_node_training.delay(str(training.id))
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)
