from django.shortcuts import render

# Create your views here.
# model_versioning/views.py
@api_view(['GET'])
def list_model_versions(request, run_id):
    """List all versions for a run."""
    run = get_object_or_404(Job, id=run_id)
    versions = run.model_versions.all()
    return Response(ModelVersionSerializer(versions, many=True).data)

@api_view(['POST'])
def publish_to_hub(request, version_id):
    """Publish a specific model version to Hugging Face Hub."""
    version = get_object_or_404(ModelVersion, id=version_id)
    publisher = HuggingFacePublisher(version)
    hub_url = publisher.publish()
    return Response({'hub_url': hub_url, 'status': 'published'})

@api_view(['GET'])
def download_version(request, version_id):
    """Download a model version artifact."""
    version = get_object_or_404(ModelVersion, id=version_id)
    return FileResponse(
        open(f"{version.artifact_path}model.pth", 'rb'),
        as_attachment=True,
        filename=f"model_v{version.version_number}.pth"
    )





# # Example: Move to cold storage after Hub publish
# def archive_local_artifact(version):
#     """Move local model to cold storage after Hub upload."""
#     import boto3
#     s3 = boto3.client('s3')
#     s3.upload_file(
#         f"{version.artifact_path}model.pth",
#         settings.MODEL_ARCHIVE_BUCKET,
#         f"run_{version.run.id}/v{version.version_number}/model.pth",
#         StorageClass='GLACIER'
#     )
#     shutil.rmtree(version.artifact_path)
#     version.storage_tier = 'cold'
#     version.save()