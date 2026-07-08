from django.urls import path

from . import views

urlpatterns = [
    # ── Clusters ────────────────────────────────────────────────
    path("clusters/", views.cluster_list, name="cluster_list"),
    path("clusters/<uuid:cluster_id>/", views.cluster_detail, name="cluster_detail"),
    # ── Nodes ───────────────────────────────────────────────────
    path("clusters/<uuid:cluster_id>/nodes/", views.add_node, name="add_node"),
    path(
        "clusters/<uuid:cluster_id>/nodes/<uuid:node_id>/",
        views.remove_node,
        name="remove_node",
    ),
    path(
        "clusters/<uuid:cluster_id>/nodes/<uuid:node_id>/heartbeat/",
        views.node_heartbeat,
        name="node_heartbeat",
    ),
    # ── Tasks ───────────────────────────────────────────────────
    path("clusters/<uuid:cluster_id>/tasks/", views.task_list, name="task_list"),
    path("tasks/<uuid:task_id>/", views.task_detail, name="task_detail"),
    path("tasks/<uuid:task_id>/launch/", views.launch_task, name="launch_task"),
    path("tasks/<uuid:task_id>/cancel/", views.task_cancel, name="task_cancel"),
    path("tasks/<uuid:task_id>/result/", views.task_result, name="task_result"),
    # ── Provider Dashboard ──────────────────────────────────────
    path(
        "dashboard/provider/",
        views.provider_federation_dashboard,
        name="provider_federation_dashboard",
    ),
]
