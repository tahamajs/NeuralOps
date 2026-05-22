from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import *
from .literature import LiteratureSynthesizer
from .hypothesis import HypothesisGenerator
from .gap_analysis import GapAnalyzer
from .grant_matcher import GrantDiscoveryEngine
from .databases import LifeScienceHub

# ── Literature Synthesis ───────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def synthesize_literature(request):
    query = request.data.get('query', '')
    papers = request.data.get('papers', [])
    output_type = request.data.get('output_type', 'report')
    if not query:
        return Response({'error': 'query required'}, status=400)
    review = LiteratureSynthesizer.synthesize(query, papers, output_type)
    return Response(LiteratureReviewSerializer(review).data, status=201)

# ── Hypothesis Generation ─────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_hypotheses(request):
    domain = request.data.get('domain', '')
    papers = request.data.get('papers', [])
    num = request.data.get('num_hypotheses', 3)
    if not domain:
        return Response({'error': 'domain required'}, status=400)
    hypotheses = HypothesisGenerator.generate(domain, papers, request.user.id, num_hypotheses=num)
    return Response(GeneratedHypothesisSerializer(hypotheses, many=True).data, status=201)

# ── Database Hub ──────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_databases(request):
    return Response(LifeScienceHub.REGISTRY)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def search_database(request, db_key):
    query = request.data.get('query', '')
    limit = request.data.get('limit', 10)
    result = LifeScienceHub.search(db_key, query, limit)
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cross_database_search(request):
    query = request.data.get('query', '')
    databases = request.data.get('databases', None)
    result = LifeScienceHub.cross_database_search(query, databases)
    return Response(result)

# ── Gap Analysis ──────────────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_research_gaps(request):
    papers = request.data.get('papers', [])
    domain = request.data.get('domain', '')
    analysis = GapAnalyzer.analyze(papers, domain, request.user.id)
    return Response(ResearchGapAnalysisSerializer(analysis).data, status=201)

# ── Grant Discovery ──────────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def search_grants(request):
    keywords = request.data.get('keywords', [])
    results = GrantDiscoveryEngine.search_grants(keywords)
    return Response(results)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def match_grants(request):
    matches = GrantDiscoveryEngine.match_researcher(request.user.id)
    return Response(GrantMatchSerializer(matches, many=True).data)