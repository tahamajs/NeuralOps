from django.urls import path

from . import views

urlpatterns = [
    # Feature Stores
    path("stores/", views.feature_store_list, name="feature_store_list"),
    # Feature Groups
    path("groups/", views.feature_group_list, name="feature_group_list"),
    # Drift Monitors
    path("monitors/", views.drift_monitor_list, name="drift_monitor_list"),
    path(
        "monitors/<uuid:monitor_id>/check/",
        views.run_drift_check,
        name="drift_check_run",
    ),
    path(
        "monitors/<uuid:monitor_id>/history/",
        views.drift_check_history,
        name="drift_check_history",
    ),
    # Alerts
    path("alerts/", views.alert_list, name="alert_list"),
    path(
        "alerts/<uuid:alert_id>/acknowledge/",
        views.acknowledge_alert,
        name="alert_acknowledge",
    ),
    # Alert Channels
    path("channels/", views.alert_channel_list, name="alert_channel_list"),
    path(
        "channels/<uuid:channel_id>/",
        views.alert_channel_delete,
        name="alert_channel_delete",
    ),
    # Performance Snapshots
    path(
        "performance/",
        views.performance_snapshot_list,
        name="performance_snapshot_list",
    ),
]
