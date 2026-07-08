from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Authorship, Citation, Publication
from .serializers import (CitationSerializer, PublicationSerializer,
                          ReviewSerializer)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def publication_list(request):
    if request.method == "GET":
        qs = Publication.objects.annotate(cc=Count("citations_in")).order_by(
            "-publication_date"
        )
        pub_type = request.query_params.get("type")
        if pub_type:
            qs = qs.filter(publication_type=pub_type)
        search = request.query_params.get("q")
        if search:
            qs = qs.filter(Q(title__icontains=search) | Q(abstract__icontains=search))
        tag = request.query_params.get("tag")
        if tag:
            qs = qs.filter(tags__contains=[tag])
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            PublicationSerializer(page, many=True).data
        )
    elif request.method == "POST":
        serializer = PublicationSerializer(data=request.data)
        if serializer.is_valid():
            pub = serializer.save()
            # Add authors
            author_ids = request.data.get("author_ids", [request.user.id])
            for i, author_id in enumerate(author_ids):
                user = get_object_or_404(get_user_model(), id=author_id)
                Authorship.objects.create(
                    publication=pub,
                    author=user,
                    author_order=i,
                    corresponding_author=(i == 0),
                )
            return Response(PublicationSerializer(pub).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def publication_detail(request, pub_id):
    pub = get_object_or_404(Publication, id=pub_id)
    if request.method == "GET":
        pub.view_count += 1
        pub.save(update_fields=["view_count"])
        return Response(PublicationSerializer(pub).data)
    elif request.method == "PATCH":
        serializer = PublicationSerializer(pub, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        pub.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def add_citation(request):
    source_id = request.data.get("source_id")
    target_id = request.data.get("target_id")
    if not source_id or not target_id:
        return Response({"error": "source_id and target_id required"}, status=400)
    source = get_object_or_404(Publication, id=source_id)
    target = get_object_or_404(Publication, id=target_id)
    cite, created = Citation.objects.get_or_create(source=source, target=target)
    if created:
        source.citation_count = source.citations_out.count()
        source.save()
        target.citation_count = target.citations_in.count()
        target.save()
    return Response(CitationSerializer(cite).data, status=201 if created else 200)


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def review_list(request, pub_id):
    pub = get_object_or_404(Publication, id=pub_id)
    if request.method == "GET":
        reviews = pub.reviews.all().order_by("-created_at")
        return Response(ReviewSerializer(reviews, many=True).data)
    elif request.method == "POST":
        serializer = ReviewSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(publication=pub, reviewer=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
