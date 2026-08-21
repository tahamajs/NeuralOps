from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import GpuTrading

@api_view(['GET'])
def get_gpu_trading(request):
    data = list(GpuTrading.objects.values())
    return Response(data)
