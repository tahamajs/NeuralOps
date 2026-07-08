import django_filters
from django.db.models import Q

from .models import ResearcherNote


class NoteFilter(django_filters.FilterSet):
    title = django_filters.CharFilter(lookup_expr="icontains")
    content = django_filters.CharFilter(lookup_expr="icontains")
    # REMOVE the tags line that uses 'contains'
    # tags = django_filters.CharFilter(field_name='tags', lookup_expr='contains')   # ❌
    tags = django_filters.CharFilter(method="filter_tags")  # ✅ custom filter
    created_at_after = django_filters.DateTimeFilter(
        field_name="created_at", lookup_expr="gte"
    )
    created_at_before = django_filters.DateTimeFilter(
        field_name="created_at", lookup_expr="lte"
    )
    author_username = django_filters.CharFilter(
        field_name="author__username", lookup_expr="iexact"
    )
    run_id = django_filters.UUIDFilter(field_name="run__id")
    is_public = django_filters.BooleanFilter()
    pinned = django_filters.BooleanFilter()
    search = django_filters.CharFilter(method="full_text_search")

    class Meta:
        model = ResearcherNote
        fields = [
            "title",
            "content",
            "tags",
            "author_username",
            "run_id",
            "is_public",
            "pinned",
            "created_at_after",
            "created_at_before",
        ]

    def filter_tags(self, queryset, name, value):
        # Filter notes whose tags list contains the given value (Python side)
        # Since JSONField list contains is not supported on SQLite, we do it manually
        # We can use Q expression? Better: iterate and keep only those where value is in the list.
        # But for performance, we can use __icontains on the JSON field as a workaround.
        # However, text representation contains the value.
        return queryset.filter(tags__icontains=f'"{value}"')  # safe for SQLite

    def full_text_search(self, queryset, name, value):
        return queryset.filter(Q(title__icontains=value) | Q(content__icontains=value))
