urlpatterns = [
    # Git & Cloud
    path("cloud/connect/", views.connect_cloud, name="cloud_connect"),
    path("cloud/connections/", views.list_connections, name="cloud_list"),
    path(
        "cloud/connections/<uuid:conn_id>/",
        views.disconnect_cloud,
        name="cloud_disconnect",
    ),
    path(
        "cloud/connections/<uuid:conn_id>/files/", views.list_files, name="cloud_files"
    ),
    path("cloud/connections/<uuid:conn_id>/sync/", views.start_sync, name="cloud_sync"),
    # Webhooks
    path("webhooks/", views.webhook_list, name="webhook_list"),
    path("webhooks/<uuid:sub_id>/", views.webhook_delete, name="webhook_delete"),
    path("webhooks/<uuid:sub_id>/test/", views.webhook_test, name="webhook_test"),
    # Sandbox
    path("sandbox/start/", views.start_sandbox, name="sandbox_start"),
    path("sandbox/<uuid:session_id>/", views.sandbox_status, name="sandbox_status"),
    path(
        "sandbox/<uuid:session_id>/execute/",
        views.execute_in_sandbox,
        name="sandbox_execute",
    ),
    path("sandbox/<uuid:session_id>/stop/", views.stop_sandbox, name="sandbox_stop"),
    # Plugins
    path("plugins/", views.plugin_list, name="plugin_list"),
    path(
        "plugins/<slug:plugin_slug>/enable/", views.enable_plugin, name="plugin_enable"
    ),
    path(
        "plugins/<slug:plugin_slug>/disable/",
        views.disable_plugin,
        name="plugin_disable",
    ),
    path(
        "plugins/<slug:plugin_slug>/manifest/",
        views.plugin_manifest,
        name="plugin_manifest",
    ),
]
