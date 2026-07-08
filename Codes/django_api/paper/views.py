# paper/views.py
"""
Paper App – Complete REST API (Overleaf-style)
Covers: Project CRUD, File CRUD, Compilation, Output, Templates,
         Share Links, Git Sync, Comments, Snapshots, Collaboration,
         Binary Upload, Word Count, PDF Preview, Clone, Rename
"""

import os
import secrets
import subprocess
import tempfile
import zipfile
from datetime import timedelta
from io import BytesIO

from django.db.models import Q
from django.http import FileResponse, HttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .compiler import LatexCompiler
from .models import (PaperComment, PaperCompile, PaperFile, PaperProject,
                     PaperShareLink, PaperTemplate, PaperVersion)
from .serializers import *


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# PROJECT CRUD
# ═══════════════════════════════════════════════════════════════


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def project_list(request):
    if request.method == "GET":
        projects = (
            PaperProject.objects.filter(
                Q(owner=request.user) | Q(collaborators=request.user)
            )
            .distinct()
            .select_related("owner")
        )
        q = request.query_params.get("q")
        if q:
            projects = projects.filter(name__icontains=q)
        paginator = StandardPagination()
        page = paginator.paginate_queryset(projects, request)
        serializer = PaperProjectListSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)
    elif request.method == "POST":
        serializer = PaperProjectDetailSerializer(data=request.data)
        if serializer.is_valid():
            project = serializer.save(owner=request.user)
            template_id = request.data.get("template_id")
            if template_id:
                _apply_template(project, template_id)
            return Response(PaperProjectDetailSerializer(project).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def project_detail(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if request.method == "GET":
        return Response(PaperProjectDetailSerializer(project).data)
    elif request.method == "PATCH":
        if project.owner != request.user:
            return Response({"error": "Only the owner can edit"}, status=403)
        serializer = PaperProjectDetailSerializer(
            project, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if project.owner != request.user:
            return Response({"error": "Only the owner can delete"}, status=403)
        project.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# PROJECT CLONE
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def clone_project(request, project_id):
    """Clone a project (own or accessible)."""
    source = _get_accessible_project(project_id, request.user)
    new_name = request.data.get("name", f"{source.name} (copy)")
    clone = PaperProject.objects.create(
        owner=request.user,
        name=new_name,
        description=source.description,
        root_file=source.root_file,
        compiler=source.compiler,
        draft_mode=source.draft_mode,
        stop_on_first_error=source.stop_on_first_error,
    )
    for f in source.files.all():
        PaperFile.objects.create(project=clone, filename=f.filename, content=f.content)
    return Response(PaperProjectDetailSerializer(clone).data, status=201)


# ═══════════════════════════════════════════════════════════════
# FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def file_list(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if request.method == "GET":
        files = project.files.all()
        serializer = PaperFileSerializer(files, many=True)
        return Response(serializer.data)
    elif request.method == "POST":
        if project.owner != request.user:
            return Response({"error": "Only the owner can upload files"}, status=403)
        filename = request.data.get("filename", "")
        content = request.data.get("content", "")
        uploaded_file = request.FILES.get("binary")
        is_binary = uploaded_file is not None
        file_obj, created = PaperFile.objects.update_or_create(
            project=project,
            filename=filename,
            defaults={
                "content": content if not is_binary else "",
                "is_binary": is_binary,
            },
        )
        if is_binary and uploaded_file:
            file_obj.binary.save(uploaded_file.name, uploaded_file, save=True)
        if not created:
            file_obj.version += 1
            file_obj.save(update_fields=["version", "content", "updated_at"])
        project.version += 1
        project.save(update_fields=["version", "updated_at"])
        return Response(
            PaperFileSerializer(file_obj).data, status=201 if created else 200
        )


@api_view(["GET", "PUT", "DELETE"])
@permission_classes([IsAuthenticated])
def file_detail(request, project_id, filename):
    project = _get_accessible_project(project_id, request.user)
    file_obj = get_object_or_404(PaperFile, project=project, filename=filename)
    if request.method == "GET":
        return Response(PaperFileSerializer(file_obj).data)
    elif request.method == "PUT":
        if project.owner != request.user:
            return Response({"error": "Only the owner can edit files"}, status=403)
        old_content = file_obj.content
        serializer = PaperFileSerializer(file_obj, data=request.data, partial=True)
        if serializer.is_valid():
            file_obj = serializer.save()
            if file_obj.content != old_content:
                file_obj.version += 1
                file_obj.save(update_fields=["version", "content", "updated_at"])
            project.version += 1
            project.save(update_fields=["version", "updated_at"])
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if project.owner != request.user:
            return Response({"error": "Only the owner can delete files"}, status=403)
        file_obj.delete()
        project.version += 1
        project.save(update_fields=["version", "updated_at"])
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def rename_file(request, project_id, filename):
    """Rename a file within a project."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can rename files"}, status=403)
    file_obj = get_object_or_404(PaperFile, project=project, filename=filename)
    new_filename = request.data.get("new_filename", "")
    if not new_filename:
        return Response({"error": "new_filename required"}, status=400)
    if PaperFile.objects.filter(project=project, filename=new_filename).exists():
        return Response({"error": "File with that name already exists"}, status=400)
    file_obj.filename = new_filename
    file_obj.save(update_fields=["filename", "updated_at"])
    project.version += 1
    project.save(update_fields=["version", "updated_at"])
    return Response(PaperFileSerializer(file_obj).data)


# ═══════════════════════════════════════════════════════════════
# COMPILATION
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def compile_project(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    compile_serializer = CompileRequestSerializer(data=request.data)
    compile_serializer.is_valid(raise_exception=True)
    compiler = compile_serializer.validated_data.get("compiler", project.compiler)
    draft_mode = compile_serializer.validated_data.get("draft_mode", project.draft_mode)
    stop_on_first_error = compile_serializer.validated_data.get(
        "stop_on_first_error", project.stop_on_first_error
    )
    compile_obj = PaperCompile.objects.create(
        project=project,
        initiated_by=request.user,
        status="queued",
        compiler=compiler,
        draft_mode=draft_mode,
        root_file=project.root_file,
    )
    compile_obj.mark_running()
    project_files = {}
    for f in project.files.all():
        if f.is_binary and f.binary:
            with f.binary.open("rb") as bf:
                project_files[f.filename] = bf.read()
        else:
            project_files[f.filename] = f.content
    lc = LatexCompiler(
        project_files=project_files,
        root_file=project.root_file,
        compiler=compiler,
        draft_mode=draft_mode,
        stop_on_first_error=stop_on_first_error,
        timeout=60,
    )
    result = lc.compile()
    if result["status"] == "completed":
        compile_obj.mark_completed(
            result["pdf_path"],
            result["log"][:500000],
            result["errors"],
            result["warnings"],
        )
        project.last_compiled_at = compile_obj.finished_at
        project.save(update_fields=["last_compiled_at"])
    else:
        compile_obj.mark_failed(
            result.get("log", "")[:500000],
            result.get("errors", 0),
            result.get("warnings", 0),
        )
    serializer = PaperCompileSerializer(compile_obj)
    return Response(
        {
            **serializer.data,
            "pdf_url": (
                f"/api/paper/projects/{project_id}/output/pdf/"
                if result["status"] == "completed"
                else None
            ),
            "log_url": f"/api/paper/projects/{project_id}/output/log/",
            "status": result["status"],
        }
    )


# ═══════════════════════════════════════════════════════════════
# OUTPUT FILES
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def output_pdf(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    last = project.last_compile
    if not last or not last.pdf:
        return Response({"error": "No compiled PDF available"}, status=404)
    return FileResponse(
        last.pdf.open("rb"),
        content_type="application/pdf",
        filename=f"{project.name}.pdf",
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def output_log(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    last = project.last_compile
    if not last:
        return Response({"error": "No compilation exists yet"}, status=404)
    return HttpResponse(last.log, content_type="text/plain; charset=utf-8")


# ═══════════════════════════════════════════════════════════════
# PREVIEW PDF (for React – inline viewing)
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def preview_pdf(request, project_id):
    """Stream the latest compiled PDF inline (for iframe/embed)."""
    project = _get_accessible_project(project_id, request.user)
    last = project.last_compile
    if not last or not last.pdf:
        return Response({"error": "No compiled PDF available"}, status=404)
    response = FileResponse(last.pdf.open("rb"), content_type="application/pdf")
    response["Content-Disposition"] = 'inline; filename="preview.pdf"'
    return response


# ═══════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def template_list(request):
    if request.method == "GET":
        templates = PaperTemplate.objects.filter(
            Q(is_public=True) | Q(author=request.user)
        )
        serializer = PaperTemplateSerializer(templates, many=True)
        return Response(serializer.data)
    elif request.method == "POST":
        serializer = PaperTemplateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(author=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def use_template(request, template_id):
    project_name = request.data.get("name", "New Project")
    project = PaperProject.objects.create(owner=request.user, name=project_name)
    _apply_template(project, template_id)
    return Response(PaperProjectDetailSerializer(project).data, status=201)


def _apply_template(project, template_id):
    try:
        template = PaperTemplate.objects.get(id=template_id)
    except PaperTemplate.DoesNotExist:
        return
    for filename, content in template.files.items():
        PaperFile.objects.create(project=project, filename=filename, content=content)
    template.usage_count += 1
    template.save(update_fields=["usage_count"])


# ═══════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def export_project(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in project.files.all():
            zf.writestr(f.filename, f.content)
    buffer.seek(0)
    return FileResponse(
        buffer,
        content_type="application/zip",
        as_attachment=True,
        filename=f"{project.name}.zip",
    )


# ═══════════════════════════════════════════════════════════════
# WORD COUNT
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def word_count(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    total = 0
    per_file = {}
    for f in project.files.all():
        if f.is_tex:
            words = len(f.content.split())
            per_file[f.filename] = words
            total += words
    return Response({"total_words": total, "per_file": per_file})


# ═══════════════════════════════════════════════════════════════
# COLLABORATION
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def add_collaborator(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can add collaborators"}, status=403)
    username = request.data.get("username")
    if not username:
        return Response({"error": "username required"}, status=400)
    from django.contrib.auth import get_user_model

    User = get_user_model()
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({"error": "User not found"}, status=404)
    project.collaborators.add(user)
    return Response(
        {
            "status": "added",
            "collaborators": [u.username for u in project.collaborators.all()],
        }
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def remove_collaborator(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response(
            {"error": "Only the owner can remove collaborators"}, status=403
        )
    username = request.data.get("username")
    if not username:
        return Response({"error": "username required"}, status=400)
    from django.contrib.auth import get_user_model

    User = get_user_model()
    project.collaborators.remove(User.objects.get(username=username))
    return Response(
        {
            "status": "removed",
            "collaborators": [u.username for u in project.collaborators.all()],
        }
    )


# ═══════════════════════════════════════════════════════════════
# SHARE LINKS (Public)
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_share_link(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can create share links"}, status=403)
    expires_in_days = request.data.get("expires_in_days")
    expires_at = (
        timezone.now() + timedelta(days=expires_in_days) if expires_in_days else None
    )
    link = PaperShareLink.objects.create(
        project=project,
        token=secrets.token_urlsafe(32),
        created_by=request.user,
        password=request.data.get("password", ""),
        expires_at=expires_at,
        max_views=request.data.get("max_views"),
    )
    return Response(
        {
            "id": str(link.id),
            "token": link.token,
            "url": link.public_url,
            "expires_at": link.expires_at,
            "is_active": link.is_active,
        },
        status=201,
    )


@api_view(["GET"])
def view_shared_project(request, token):
    """Public – view project details and download PDF via share link."""
    link = get_object_or_404(PaperShareLink, token=token, is_active=True)
    if link.is_expired:
        return Response({"error": "This share link has expired"}, status=410)
    if link.password and request.query_params.get("password", "") != link.password:
        return Response(
            {"error": "Password required", "password_required": True}, status=401
        )
    link.view_count += 1
    link.save(update_fields=["view_count"])
    project = link.project
    data = {
        "project_name": project.name,
        "description": project.description,
        "files": [
            {"filename": f.filename, "content": f.content} for f in project.files.all()
        ],
        "has_pdf": bool(project.last_compile and project.last_compile.pdf),
    }
    return Response(data)


@api_view(["GET"])
def shared_project_pdf(request, token):
    """Public – download compiled PDF from share link."""
    link = get_object_or_404(PaperShareLink, token=token, is_active=True)
    if link.is_expired:
        return Response({"error": "Expired"}, status=410)
    if link.password and request.query_params.get("password", "") != link.password:
        return Response({"error": "Password required"}, status=401)
    last = link.project.last_compile
    if not last or not last.pdf:
        return Response({"error": "No PDF available"}, status=404)
    return FileResponse(
        last.pdf.open("rb"),
        content_type="application/pdf",
        filename=f"{link.project.name}.pdf",
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_share_links(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Not allowed"}, status=403)
    links = project.share_links.all()
    data = [
        {
            "id": str(l.id),
            "token": l.token,
            "url": l.public_url,
            "is_active": l.is_active,
            "view_count": l.view_count,
            "expires_at": l.expires_at,
            "created_at": l.created_at,
        }
        for l in links
    ]
    return Response(data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def revoke_share_link(request, project_id, link_id):
    link = get_object_or_404(PaperShareLink, id=link_id, project_id=project_id)
    if link.created_by != request.user:
        return Response({"error": "Not allowed"}, status=403)
    link.is_active = False
    link.save(update_fields=["is_active"])
    return Response({"status": "revoked"})


# ═══════════════════════════════════════════════════════════════
# GIT SYNC
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def git_push(request, project_id):
    """Push project files to a configured git remote."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can push"}, status=403)
    if not project.git_remote:
        return Response({"error": "No git remote configured"}, status=400)
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in project.files.all():
            with open(os.path.join(tmpdir, f.filename), "w") as fh:
                fh.write(f.content)
        try:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True, check=True)
            subprocess.run(
                ["git", "add", "-A"], cwd=tmpdir, capture_output=True, check=True
            )
            subprocess.run(
                ["git", "commit", "-m", f"Auto-commit {timezone.now().isoformat()}"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "remote", "add", "origin", project.git_remote],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "push", "-u", "origin", "main", "--force"],
                cwd=tmpdir,
                capture_output=True,
                check=True,
            )
            return Response({"status": "pushed"})
        except subprocess.CalledProcessError as e:
            return Response({"error": str(e.stderr)}, status=500)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def git_pull(request, project_id):
    """Pull project files from a configured git remote."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can pull"}, status=403)
    if not project.git_remote:
        return Response({"error": "No git remote configured"}, status=400)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", project.git_remote, tmpdir],
                capture_output=True,
                check=True,
            )
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    filepath = os.path.join(root, file)
                    relpath = os.path.relpath(filepath, tmpdir)
                    if os.path.getsize(filepath) < 5_000_000:
                        with open(filepath, "r") as fh:
                            content = fh.read()
                        PaperFile.objects.update_or_create(
                            project=project,
                            filename=relpath,
                            defaults={"content": content},
                        )
            project.version += 1
            project.save(update_fields=["version", "updated_at"])
            return Response({"status": "pulled"})
        except subprocess.CalledProcessError as e:
            return Response({"error": str(e.stderr)}, status=500)


# ═══════════════════════════════════════════════════════════════
# COMMENTS (Track Changes / Review)
# ═══════════════════════════════════════════════════════════════


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def comment_list(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if request.method == "GET":
        comments = project.comments.all()
        file_id = request.query_params.get("file_id")
        if file_id:
            comments = comments.filter(file_id=file_id)
        unresolved = request.query_params.get("unresolved")
        if unresolved == "true":
            comments = comments.filter(is_resolved=False)
        paginator = StandardPagination()
        page = paginator.paginate_queryset(comments, request)
        serializer = PaperCommentSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)
    elif request.method == "POST":
        serializer = PaperCommentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(project=project, author=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def comment_detail(request, project_id, comment_id):
    comment = get_object_or_404(PaperComment, id=comment_id, project_id=project_id)
    if request.method == "PATCH":
        if comment.author != request.user:
            return Response({"error": "Not allowed"}, status=403)
        comment.content = request.data.get("content", comment.content)
        comment.is_resolved = request.data.get("is_resolved", comment.is_resolved)
        comment.save()
        return Response(PaperCommentSerializer(comment).data)
    elif request.method == "DELETE":
        if comment.author != request.user:
            return Response({"error": "Not allowed"}, status=403)
        comment.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# SNAPSHOTS (Version History)
# ═══════════════════════════════════════════════════════════════


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def snapshot_list(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    if request.method == "GET":
        snapshots = project.snapshots.all()
        paginator = StandardPagination()
        page = paginator.paginate_queryset(snapshots, request)
        serializer = PaperVersionSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)
    elif request.method == "POST":
        files_dict = {}
        for f in project.files.all():
            files_dict[f.filename] = f.content
        snap = PaperVersion.objects.create(
            project=project,
            version_number=project.version,
            files_snapshot=files_dict,
            created_by=request.user,
            message=request.data.get("message", ""),
        )
        return Response(PaperVersionSerializer(snap).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def snapshot_detail(request, project_id, version_number):
    project = _get_accessible_project(project_id, request.user)
    snap = get_object_or_404(
        PaperVersion, project=project, version_number=version_number
    )
    return Response(PaperVersionSerializer(snap).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def restore_snapshot(request, project_id, version_number):
    """Restore project files to a previous snapshot."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can restore"}, status=403)
    snap = get_object_or_404(
        PaperVersion, project=project, version_number=version_number
    )
    for filename, content in snap.files_snapshot.items():
        PaperFile.objects.update_or_create(
            project=project,
            filename=filename,
            defaults={"content": content, "version": models.F("version") + 1},
        )
    project.version += 1
    project.save(update_fields=["version", "updated_at"])
    return Response({"status": "restored", "from_version": version_number})


# ═══════════════════════════════════════════════════════════════
# COMPILE HISTORY
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def compile_history(request, project_id):
    project = _get_accessible_project(project_id, request.user)
    compiles = project.compiles.order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(compiles, request)
    serializer = PaperCompileSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# HUGGINGFACE UPLOAD
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upload_to_huggingface(request, project_id):
    """Upload compiled PDF to HuggingFace as a Dataset or Space."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can publish"}, status=403)
    last = project.last_compile
    if not last or not last.pdf:
        return Response({"error": "Compile the project first"}, status=400)
    try:
        import json

        from huggingface_hub import create_repo, upload_folder

        token = getattr(settings, "HUGGINGFACE_TOKEN", None)
        username = getattr(settings, "HUGGINGFACE_USERNAME", "unknown")
        if not token:
            return Response({"error": "HUGGINGFACE_TOKEN not set"}, status=500)
        repo_id = f"{username}/paper-{project.id.hex[:8]}"
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(last.pdf.path, os.path.join(tmpdir, f"{project.name}.pdf"))
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write(f"# {project.name}\n\n{project.description}\n\nCompiled paper.")
            with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
                json.dump(
                    {
                        "project_name": project.name,
                        "compiled_at": last.created_at.isoformat(),
                    },
                    f,
                )
            upload_folder(
                folder_path=tmpdir, repo_id=repo_id, token=token, repo_type="dataset"
            )
        return Response(
            {"status": "uploaded", "url": f"https://huggingface.co/datasets/{repo_id}"}
        )
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════


def _get_accessible_project(project_id, user):
    project = get_object_or_404(PaperProject, id=project_id)
    if project.owner != user and not project.collaborators.filter(id=user.id).exists():
        from django.core.exceptions import PermissionDenied

        raise PermissionDenied("You do not have access to this project.")
    return project
