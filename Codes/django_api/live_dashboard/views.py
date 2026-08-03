import logging

from django.db import connection
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import Dashboard, DashboardWidget, SavedSQLQuery
from .serializers import (
    DashboardSerializer,
    DashboardWidgetSerializer,
    SavedSQLQuerySerializer,
)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Dashboards
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def dashboard_list(request):
    """List my dashboards or create a new one."""
    if request.method == "GET":
        qs = Dashboard.objects.filter(user=request.user).order_by("-updated_at")
        return Response(DashboardSerializer(qs, many=True).data)

    elif request.method == "POST":
        serializer = DashboardSerializer(data=request.data)
        if serializer.is_valid():
            dashboard = serializer.save(user=request.user)
            return Response(DashboardSerializer(dashboard).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def dashboard_detail(request, dashboard_id):
    """Get, update, or delete a dashboard."""
    dashboard = get_object_or_404(Dashboard, id=dashboard_id, user=request.user)

    if request.method == "GET":
        return Response(DashboardSerializer(dashboard).data)

    elif request.method == "PATCH":
        serializer = DashboardSerializer(dashboard, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        dashboard.delete()
        return Response(status=204)


@api_view(["GET"])
@permission_classes([AllowAny])
def public_dashboard(request, dashboard_id):
    """View a public dashboard."""
    dashboard = get_object_or_404(Dashboard, id=dashboard_id, is_public=True)
    return Response(DashboardSerializer(dashboard).data)


# ═══════════════════════════════════════════════════════════════
# Widgets
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def add_widget(request, dashboard_id):
    """Add a widget to a dashboard."""
    dashboard = get_object_or_404(Dashboard, id=dashboard_id, user=request.user)
    serializer = DashboardWidgetSerializer(data=request.data)
    if serializer.is_valid():
        widget = serializer.save(dashboard=dashboard)
        return Response(DashboardWidgetSerializer(widget).data, status=201)
    return Response(serializer.errors, status=400)


@api_view(["PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def widget_detail(request, widget_id):
    """Update or delete a widget."""
    widget = get_object_or_404(
        DashboardWidget, id=widget_id, dashboard__user=request.user
    )

    if request.method == "PATCH":
        serializer = DashboardWidgetSerializer(widget, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        widget.delete()
        return Response(status=204)


@api_view(["PUT"])
@permission_classes([IsAuthenticated])
def save_layout(request, dashboard_id):
    """Save the entire layout (positions, sizes) of widgets in a dashboard."""
    dashboard = get_object_or_404(Dashboard, id=dashboard_id, user=request.user)
    widgets_data = request.data.get("widgets", [])
    for wdata in widgets_data:
        widget = dashboard.widgets.filter(id=wdata.get("id")).first()
        if widget:
            for field in ["position_x", "position_y", "width", "height"]:
                if field in wdata:
                    setattr(widget, field, wdata[field])
            widget.save()
    return Response({"status": "saved"})


# ═══════════════════════════════════════════════════════════════
# SQL Queries
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def sql_query_list(request):
    """List my saved SQL queries or save a new one."""
    if request.method == "GET":
        qs = SavedSQLQuery.objects.filter(user=request.user).order_by("-created_at")
        return Response(SavedSQLQuerySerializer(qs, many=True).data)

    elif request.method == "POST":
        serializer = SavedSQLQuerySerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.save(user=request.user)
            return Response(SavedSQLQuerySerializer(query).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def execute_sql(request):
    """
    Execute a read‑only SQL query and return results.
    Only SELECT queries are allowed.
    """
    sql = request.data.get("sql", "").strip()
    if not sql:
        return Response({"error": "sql required"}, status=400)

    # Security: only allow SELECT queries
    if not sql.lower().startswith("select"):
        return Response({"error": "Only SELECT queries are allowed"}, status=400)

    # Block dangerous keywords
    dangerous = ["drop", "delete", "insert", "update", "alter", "create", "truncate"]
    if any(word in sql.lower() for word in dangerous):
        return Response({"error": "Query contains forbidden keywords"}, status=400)

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
            return Response({"columns": columns, "rows": data, "count": len(data)})
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# ═══════════════════════════════════════════════════════════════
# Live Experiment Feed (WebSocket-based)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def live_experiment_data(request):
    """
    Return live data for the currently running experiments.
    Used by dashboard widgets that poll for updates.
    """
    from experiment.models import TrainingRun

    runs = TrainingRun.objects.filter(user=request.user, status="running").order_by(
        "-created_at"
    )[:10]

    return Response(
        [
            {
                "id": str(r.id),
                "name": r.name,
                "current_step": r.current_step,
                "total_steps": r.total_steps,
                "progress": round(r.current_step / max(r.total_steps, 1) * 100, 1),
                "status": r.status,
            }
            for r in runs
        ]
    )
