@api_view(['GET'])
@permission_classes([IsAuthenticated])
def protocol_marketplace(request):
    """Browse and search public protocols."""
    qs = ResearchProtocol.objects.filter(is_public=True)
    tags = request.query_params.get('tags')
    if tags: qs = qs.filter(tags__contains=tags.split(','))
    search = request.query_params.get('q')
    if search: qs = qs.filter(Q(title__icontains=search) | Q(description__icontains=search))
    return Response(ResearchProtocolSerializer(qs, many=True).data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def use_protocol(request, protocol_id):
    """Launch an experiment from a protocol (auto‑fills config)."""
    protocol = get_object_or_404(ResearchProtocol, id=protocol_id)
    # Create experiment from protocol steps
    config = {step['key']: step['value'] for step in protocol.steps if step.get('key')}
    experiment = Experiment.objects.create(name=f"From protocol: {protocol.title}")
    run = TrainingRun.objects.create(experiment=experiment, config=config, status='queued')
    protocol.usage_count += 1
    protocol.save()
    return Response({'run_id': str(run.id)}, status=201)