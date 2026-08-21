from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ResearchStream

@api_view(['GET'])
def get_research_stream(request):
    data = list(ResearchStream.objects.values())
    return Response(data)
