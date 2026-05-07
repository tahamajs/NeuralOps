import subprocess
import tempfile
import os
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.http import FileResponse

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def compile_latex(request):
    source = request.data.get('source', '')
    if not source:
        return Response({'error': 'No LaTeX source provided'}, status=400)

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, 'document.tex')
        pdf_path = os.path.join(tmpdir, 'document.pdf')

        with open(tex_path, 'w') as f:
            f.write(source)

        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_path],
                capture_output=True, text=True, timeout=30
            )

            if not os.path.exists(pdf_path):
                log_path = os.path.join(tmpdir, 'document.log')
                log_content = ''
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        log_content = f.read()
                return Response({
                    'error': 'Compilation failed',
                    'log': log_content,
                }, status=500)

            return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')

        except subprocess.TimeoutExpired:
            return Response({'error': 'Compilation timed out'}, status=500)
        except FileNotFoundError:
            return Response({'error': 'pdflatex not installed on server'}, status=500)