from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import AttentionMetric

@api_view(['GET'])
def get_attention_metrics(request):
    data = list(AttentionMetric.objects.values())
    return Response(data)
