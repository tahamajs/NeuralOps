from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from integrations.models import CloudConnection, SyncTask
from integrations.serializers import CloudConnectionSerializer, SyncTaskSerializer

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def connect_cloud(request):
    """Store OAuth tokens for a cloud provider."""
    serializer = CloudConnectionSerializer(data=request.data)
    if serializer.is_valid():
        conn = serializer.save(user=request.user)
        return Response(CloudConnectionSerializer(conn).data, status=201)
    return Response(serializer.errors, status=400)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_connections(request):
    conns = CloudConnection.objects.filter(user=request.user, is_active=True)
    return Response(CloudConnectionSerializer(conns, many=True).data)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def disconnect_cloud(request, conn_id):
    conn = get_object_or_404(CloudConnection, id=conn_id, user=request.user)
    conn.is_active = False
    conn.access_token = ""
    conn.save()
    return Response(status=204)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_files(request, conn_id):
    """List files in a connected cloud storage."""
    conn = get_object_or_404(CloudConnection, id=conn_id, user=request.user)
    path = request.query_params.get("path", "")
    provider = conn.provider

    if provider == "github":
        import requests

        headers = {"Authorization": f"token {conn.access_token}"}
        repo = conn.metadata.get("repo", "")
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        resp = requests.get(url, headers=headers)
        return Response(resp.json())
    elif provider == "s3":
        import boto3

        s3 = boto3.client(
            "s3",
            aws_access_key_id=conn.metadata.get("key"),
            aws_secret_access_key=conn.access_token,
        )
        bucket = conn.metadata.get("bucket", "")
        objects = s3.list_objects_v2(Bucket=bucket, Prefix=path)
        return Response(
            [{"key": o["Key"], "size": o["Size"]} for o in objects.get("Contents", [])]
        )
    return Response({"error": "Provider listing not implemented"}, status=501)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_sync(request, conn_id):
    """Start a file sync operation."""
    conn = get_object_or_404(CloudConnection, id=conn_id, user=request.user)
    task = SyncTask.objects.create(
        connection=conn,
        direction=request.data.get("direction", "pull"),
        source_path=request.data.get("source_path", ""),
        target_path=request.data.get("target_path", "/workspace"),
    )
    from integrations.tasks import run_sync_task

    run_sync_task.delay(str(task.id))
    return Response(SyncTaskSerializer(task).data, status=201)
