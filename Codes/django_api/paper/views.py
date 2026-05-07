"""
Paper App – REST API Views (Overleaf-style)

Endpoints:
- Project CRUD
- File CRUD
- Compilation
- Output serving (PDF, log)
- Templates
- Git sync / export
- Collaboration management
"""
import os
import zipfile
import tempfile
from io import BytesIO

from django.shortcuts import get_object_or_404
from django.http import FileResponse, HttpResponse
from django.db.models import Q
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import PaperProject, PaperFile, PaperCompile, PaperTemplate
from .serializers import (
    PaperProjectListSerializer,
    PaperProjectDetailSerializer,
    PaperFileSerializer,
    PaperCompileSerializer,
    PaperTemplateSerializer,
    CompileRequestSerializer,
    FileCreateSerializer,
)
from .compiler import LatexCompiler


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# PROJECT CRUD
# ═══════════════════════════════════════════════════════════════

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def project_list(request):
    """List all projects owned by or shared with the user, or create a new one."""
    if request.method == 'GET':
        projects = PaperProject.objects.filter(
            Q(owner=request.user) | Q(collaborators=request.user)
        ).distinct().select_related('owner').prefetch_related('compiles')

        # Filter by search query
        q = request.query_params.get('q')
        if q:
            projects = projects.filter(name__icontains=q)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(projects, request)
        serializer = PaperProjectListSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == 'POST':
        serializer = PaperProjectDetailSerializer(data=request.data)
        if serializer.is_valid():
            project = serializer.save(owner=request.user)
            # Optionally create from template
            template_id = request.data.get('template_id')
            if template_id:
                _apply_template(project, template_id)
            return Response(
                PaperProjectDetailSerializer(project).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def project_detail(request, project_id):
    """Retrieve, update, or delete a single paper project."""
    project = _get_accessible_project(project_id, request.user)

    if request.method == 'GET':
        serializer = PaperProjectDetailSerializer(project)
        return Response(serializer.data)

    elif request.method == 'PATCH':
        if project.owner != request.user:
            return Response(
                {"error": "Only the owner can edit the project"},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = PaperProjectDetailSerializer(
            project, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        if project.owner != request.user:
            return Response(
                {"error": "Only the owner can delete the project"},
                status=status.HTTP_403_FORBIDDEN,
            )
        project.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ═══════════════════════════════════════════════════════════════
# FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def file_list(request, project_id):
    """List all files in a project or upload a new file."""
    project = _get_accessible_project(project_id, request.user)

    if request.method == 'GET':
        files = project.files.all()
        serializer = PaperFileSerializer(files, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        if project.owner != request.user:
            return Response(
                {"error": "Only the owner can upload files"},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = FileCreateSerializer(data=request.data)
        if serializer.is_valid():
            filename = serializer.validated_data['filename']
            content = serializer.validated_data.get('content', '')

            # Update if exists, create otherwise
            file_obj, created = PaperFile.objects.update_or_create(
                project=project,
                filename=filename,
                defaults={'content': content},
            )
            if not created:
                file_obj.version += 1
                file_obj.save(update_fields=['version', 'content', 'updated_at'])
            project.version += 1
            project.save(update_fields=['version', 'updated_at'])
            return Response(
                PaperFileSerializer(file_obj).data,
                status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAuthenticated])
def file_detail(request, project_id, filename):
    """Get, update, or delete a specific file in a project."""
    project = _get_accessible_project(project_id, request.user)
    file_obj = get_object_or_404(PaperFile, project=project, filename=filename)

    if request.method == 'GET':
        serializer = PaperFileSerializer(file_obj)
        return Response(serializer.data)

    elif request.method == 'PUT':
        if project.owner != request.user:
            return Response(
                {"error": "Only the owner can edit files"},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = PaperFileSerializer(file_obj, data=request.data, partial=True)
        if serializer.is_valid():
            file_obj = serializer.save()
            file_obj.version += 1
            file_obj.save(update_fields=['version', 'content', 'updated_at'])
            project.version += 1
            project.save(update_fields=['version', 'updated_at'])
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        if project.owner != request.user:
            return Response(
                {"error": "Only the owner can delete files"},
                status=status.HTTP_403_FORBIDDEN,
            )
        file_obj.delete()
        project.version += 1
        project.save(update_fields=['version', 'updated_at'])
        return Response(status=status.HTTP_204_NO_CONTENT)


# ═══════════════════════════════════════════════════════════════
# COMPILATION (Like Overleaf CLSI)
# ═══════════════════════════════════════════════════════════════

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def compile_project(request, project_id):
    """
    Compile a LaTeX project.

    This is the main compilation endpoint – analogous to Overleaf's
    POST /project/:id/compile.

    Body (optional):
    {
        "compiler": "pdflatex",       // or "xelatex", "lualatex"
        "draft_mode": false,
        "stop_on_first_error": false
    }

    Response:
    {
        "compile_id": "...",
        "status": "completed",
        "pdf_url": "/api/paper/projects/<id>/output/pdf/",
        "log_url": "/api/paper/projects/<id>/output/log/",
        "error_count": 0,
        "warning_count": 2,
        "duration_seconds": 3.2
    }
    """
    project = _get_accessible_project(project_id, request.user)

    compile_serializer = CompileRequestSerializer(data=request.data)
    compile_serializer.is_valid(raise_exception=True)

    compiler = compile_serializer.validated_data.get(
        'compiler', project.compiler
    )
    draft_mode = compile_serializer.validated_data.get(
        'draft_mode', project.draft_mode
    )
    stop_on_first_error = compile_serializer.validated_data.get(
        'stop_on_first_error', project.stop_on_first_error
    )

    # Create compile record
    compile_obj = PaperCompile.objects.create(
        project=project,
        initiated_by=request.user,
        status='queued',
        compiler=compiler,
        draft_mode=draft_mode,
        root_file=project.root_file,
    )
    compile_obj.mark_running()

    # Build project files dict
    project_files = {}
    for f in project.files.all():
        if f.is_binary and f.binary:
            # Read binary content
            with f.binary.open('rb') as bf:
                project_files[f.filename] = bf.read()
        else:
            project_files[f.filename] = f.content

    # Run compilation
    lc = LatexCompiler(
        project_files=project_files,
        root_file=project.root_file,
        compiler=compiler,
        draft_mode=draft_mode,
        stop_on_first_error=stop_on_first_error,
        timeout=60,
    )

    result = lc.compile()

    if result['status'] == 'completed':
        compile_obj.mark_completed(
            pdf_path=result['pdf_path'],
            log_text=result['log'][:500000],  # limit log size
            errors=result['errors'],
            warnings=result['warnings'],
        )
        project.last_compiled_at = compile_obj.finished_at
        project.save(update_fields=['last_compiled_at'])
    else:
        compile_obj.mark_failed(
            log_text=result['log'][:500000] if result.get('log') else '',
            errors=result.get('errors', 0),
            warnings=result.get('warnings', 0),
        )

    serializer = PaperCompileSerializer(compile_obj)
    response_data = {
        **serializer.data,
        'pdf_url': f'/api/paper/projects/{project_id}/output/pdf/'
                   if result['status'] == 'completed' else None,
        'log_url': f'/api/paper/projects/{project_id}/output/log/',
        'status': result['status'],
    }
    return Response(response_data)


# ═══════════════════════════════════════════════════════════════
# OUTPUT FILES (PDF, Log, Aux)
# ═══════════════════════════════════════════════════════════════

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def output_pdf(request, project_id):
    """Serve the most recent compiled PDF."""
    project = _get_accessible_project(project_id, request.user)
    last_compile = project.last_compile
    if not last_compile or not last_compile.pdf:
        return Response(
            {"error": "No compiled PDF available. Compile the project first."},
            status=status.HTTP_404_NOT_FOUND,
        )
    return FileResponse(
        last_compile.pdf.open('rb'),
        content_type='application/pdf',
        filename=f'{project.name}.pdf',
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def output_log(request, project_id):
    """Return the compilation log as plain text."""
    project = _get_accessible_project(project_id, request.user)
    last_compile = project.last_compile
    if not last_compile:
        return Response(
            {"error": "No compilation exists yet."},
            status=status.HTTP_404_NOT_FOUND,
        )
    return HttpResponse(
        last_compile.log,
        content_type='text/plain; charset=utf-8',
    )


# ═══════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def template_list(request):
    """List available templates or create a new one."""
    if request.method == 'GET':
        templates = PaperTemplate.objects.filter(
            Q(is_public=True) | Q(author=request.user)
        )
        serializer = PaperTemplateSerializer(templates, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = PaperTemplateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(author=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def use_template(request, template_id):
    """Create a new project from a template."""
    project_name = request.data.get('name', 'New Project')
    project = PaperProject.objects.create(
        owner=request.user,
        name=project_name,
    )
    _apply_template(project, template_id)
    return Response(
        PaperProjectDetailSerializer(project).data,
        status=status.HTTP_201_CREATED,
    )


def _apply_template(project, template_id):
    """Copy template files into a new project."""
    try:
        template = PaperTemplate.objects.get(id=template_id)
    except PaperTemplate.DoesNotExist:
        return
    for filename, content in template.files.items():
        PaperFile.objects.create(
            project=project,
            filename=filename,
            content=content,
        )
    template.usage_count += 1
    template.save(update_fields=['usage_count'])


# ═══════════════════════════════════════════════════════════════
# EXPORT (ZIP download – Git-like export)
# ═══════════════════════════════════════════════════════════════

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def export_project(request, project_id):
    """Download entire project as a .zip file."""
    project = _get_accessible_project(project_id, request.user)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in project.files.all():
            zf.writestr(f.filename, f.content)
    buffer.seek(0)
    return FileResponse(
        buffer,
        content_type='application/zip',
        as_attachment=True,
        filename=f'{project.name}.zip',
    )


# ═══════════════════════════════════════════════════════════════
# COLLABORATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_collaborator(request, project_id):
    """Add a collaborator by username."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response(
            {"error": "Only the owner can add collaborators"},
            status=status.HTTP_403_FORBIDDEN,
        )
    username = request.data.get('username')
    if not username:
        return Response(
            {"error": "username required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    from django.contrib.auth import get_user_model
    User = get_user_model()
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response(
            {"error": "User not found"},
            status=status.HTTP_404_NOT_FOUND,
        )
    project.collaborators.add(user)
    return Response({
        "status": "added",
        "collaborators": [u.username for u in project.collaborators.all()],
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def remove_collaborator(request, project_id):
    """Remove a collaborator by username."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response(
            {"error": "Only the owner can remove collaborators"},
            status=status.HTTP_403_FORBIDDEN,
        )
    username = request.data.get('username')
    if not username:
        return Response(
            {"error": "username required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    from django.contrib.auth import get_user_model
    User = get_user_model()
    project.collaborators.remove(User.objects.get(username=username))
    return Response({
        "status": "removed",
        "collaborators": [u.username for u in project.collaborators.all()],
    })


# ═══════════════════════════════════════════════════════════════
# COMPILE HISTORY
# ═══════════════════════════════════════════════════════════════

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compile_history(request, project_id):
    """List all compilations for a project."""
    project = _get_accessible_project(project_id, request.user)
    compiles = project.compiles.order_by('-created_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(compiles, request)
    serializer = PaperCompileSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_accessible_project(project_id, user):
    """Get a project that the user owns or has been shared with."""
    project = get_object_or_404(PaperProject, id=project_id)
    if project.owner != user and not project.collaborators.filter(id=user.id).exists():
        # If not owner or collaborator, deny access
        from django.core.exceptions import PermissionDenied
        raise PermissionDenied("You do not have access to this project.")
    return project