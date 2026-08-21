from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import SupportSolution

@api_view(['GET'])
def get_support_solutions(request):
    data = list(SupportSolution.objects.values())
    return Response(data)
