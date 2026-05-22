@api_view(['GET'])
@permission_classes([IsAuthenticated])
def project_knowledge_graph(request, project_id):
    """Return the knowledge graph for a project (nodes + edges)."""
    project = get_object_or_404(ResearchProject, id=project_id)
    nodes = []
    edges = []

    # Add notes as nodes
    for note in project.notes.all():
        nodes.append({'id': str(note.id), 'type': 'note', 'title': note.title or 'Untitled', 'group': 'notes'})
    # Add experiments
    for run in project.experiments.all():
        nodes.append({'id': str(run.id), 'type': 'experiment', 'title': run.name or 'Run', 'group': 'experiments'})
    # Add papers
    for paper in project.papers.all():
        nodes.append({'id': str(paper.id), 'type': 'paper', 'title': paper.name, 'group': 'papers'})

    # Add edges
    for edge in ProjectKnowledgeEdge.objects.filter(
        Q(source_id__in=[n['id'] for n in nodes]) | Q(target_id__in=[n['id'] for n in nodes])
    ):
        edges.append({'from': str(edge.source_id), 'to': str(edge.target_id), 'label': edge.relationship})

    return Response({'nodes': nodes, 'edges': edges})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_knowledge_edge(request, project_id):
    """Link two assets in a project."""
    project = get_object_or_404(ResearchProject, id=project_id)
    edge = ProjectKnowledgeEdge.objects.create(
        source_type=request.data['source_type'],
        source_id=request.data['source_id'],
        target_type=request.data['target_type'],
        target_id=request.data['target_id'],
        relationship=request.data.get('relationship', 'related'),
        created_by=request.user,
    )
    return Response({'edge_id': str(edge.id)}, status=201)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_agent_team(request, project_id):
    """Create a custom agent team and run a task."""
    team_config = request.data.get('team_config', {})
    task = request.data.get('task', '')
    team = AgentTeam(team_config, project_id)
    results = team.run_cycle(task)
    return Response(results)