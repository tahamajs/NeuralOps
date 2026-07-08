from django.urls import path

from . import views

urlpatterns = [
    # ── Global version listing ─────────────────────────────────
    path("versions/", views.all_model_versions, name="all_model_versions"),
    # ── Per‑run versions ───────────────────────────────────────
    path(
        "runs/<uuid:run_id>/versions/",
        views.list_model_versions,
        name="list_model_versions",
    ),
    path(
        "runs/<uuid:run_id>/versions/cleanup/",
        views.cleanup_versions,
        name="cleanup_versions",
    ),
    path(
        "runs/<uuid:run_id>/versions/<int:version_number>/set_production/",
        views.set_production_version,
        name="set_production_version",
    ),
    path(
        "runs/<uuid:run_id>/versions/compare/<int:version_a>/<int:version_b>/",
        views.compare_versions,
        name="compare_versions",
    ),
    # ── Single version actions ─────────────────────────────────
    path(
        "versions/<uuid:version_id>/",
        views.model_version_detail,
        name="model_version_detail",
    ),
    path(
        "versions/<uuid:version_id>/publish/",
        views.publish_to_hub,
        name="publish_to_hub",
    ),
    path(
        "versions/<uuid:version_id>/download/",
        views.download_model_version,
        name="download_model_version",
    ),
    path(
        "versions/<uuid:version_id>/archive/",
        views.archive_to_cold_storage,
        name="archive_to_cold_storage",
    ),
    path(
        "versions/<uuid:version_id>/restore/",
        views.restore_from_cold_storage,
        name="restore_from_cold_storage",
    ),
    # ── Diff creation & reconstruction ─────────────────────────
    path(
        "versions/diff/<uuid:from_version_id>/<uuid:to_version_id>/",
        views.create_diff_view,
        name="create_diff",
    ),
    path(
        "versions/reconstruct/<uuid:base_version_id>/<uuid:diff_id>/",
        views.reconstruct_version,
        name="reconstruct_version",
    ),
    # ── Lineage graph ──────────────────────────────────────────
    path(
        "versions/<uuid:version_id>/lineage/", views.lineage_graph, name="lineage_graph"
    ),
    # ── Compression methods catalog ────────────────────────────
    path(
        "compression-methods/",
        views.list_compression_methods,
        name="list_compression_methods",
    ),
    # ── Lifecycle migration (staff only) ───────────────────────
    path(
        "versions/migrate-lifecycle/",
        views.run_lifecycle_migration,
        name="run_lifecycle_migration",
    ),
    # ── Governance ──────────────────────────────────────────────
    path(
        "versions/<uuid:version_id>/request-approval/",
        views.request_approval,
        name="request_approval",
    ),
    path(
        "approvals/<uuid:approval_id>/review/",
        views.review_approval,
        name="review_approval",
    ),
    path("approvals/pending/", views.pending_approvals, name="pending_approvals"),
    path(
        "versions/<uuid:version_id>/access/",
        views.version_access_list,
        name="version_access_list",
    ),
    path(
        "versions/<uuid:version_id>/access/grant/",
        views.grant_access,
        name="grant_access",
    ),
    path("access/<uuid:access_id>/revoke/", views.revoke_access, name="revoke_access"),
    path(
        "versions/<uuid:version_id>/scans/",
        views.security_scan_results,
        name="security_scan_results",
    ),
    path(
        "versions/<uuid:version_id>/scans/trigger/",
        views.trigger_security_scan,
        name="trigger_security_scan",
    ),
    path("versions/<uuid:version_id>/usage/", views.usage_log, name="usage_log"),
    path("versions/<uuid:version_id>/usage/log/", views.log_usage, name="log_usage"),
    path(
        "dashboard/governance/", views.governance_dashboard, name="governance_dashboard"
    ),
]
