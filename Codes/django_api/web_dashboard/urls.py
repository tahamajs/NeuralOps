from django.urls import path

from . import views

urlpatterns = [
    # ─── HTML Pages (SPA fallback / direct access) ─────────────
    path("", views.index, name="dashboard_index"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("dashboard/job/<uuid:job_id>/", views.job_detail, name="dashboard_job_detail"),
    # ─── Authentication API ────────────────────────────────────
    path("api/register/", views.api_register, name="api_register"),
    path("api/login/", views.api_login, name="api_login"),
    path("api/logout/", views.api_logout, name="api_logout"),
    # ─── Widget Management ─────────────────────────────────────
    path("api/widgets/", views.dashboard_widgets, name="dashboard_widgets"),
    path(
        "api/widgets/layout/", views.dashboard_widgets, name="dashboard_widgets_layout"
    ),
    path("api/widgets/<uuid:widget_id>/data/", views.widget_data, name="widget_data"),
    path(
        "api/widgets/<uuid:widget_id>/config/",
        views.widget_config,
        name="widget_config",
    ),
    # ─── Dashboard Statistics & Config ─────────────────────────
    path("api/dashboard/stats/", views.dashboard_stats, name="dashboard_stats"),
    path(
        "api/dashboard/settings/", views.dashboard_settings, name="dashboard_settings"
    ),
    path("api/config/all/", views.launch_config_options, name="launch_config_options"),
    # ─── Model Card Generation ─────────────────────────────────
    path(
        "api/runs/<uuid:run_id>/model-card/",
        views.generate_model_card,
        name="generate_model_card",
    ),
    # ─── Consumer GPU Marketplace ────────────────────────────
    path(
        "api/marketplace/listings/",
        views.marketplace_listings,
        name="marketplace_listings",
    ),
    path("api/marketplace/orders/", views.consumer_orders, name="consumer_orders"),
    path("api/marketplace/jobs/", views.consumer_jobs, name="consumer_jobs"),
    path(
        "api/marketplace/jobs/<uuid:job_id>/",
        views.consumer_job_detail,
        name="consumer_job_detail",
    ),
    path("api/marketplace/balance/", views.consumer_balance, name="consumer_balance"),
    path(
        "api/marketplace/invoices/", views.consumer_invoices, name="consumer_invoices"
    ),
    path("api/marketplace/usage/", views.consumer_usage, name="consumer_usage"),
    path(
        "api/marketplace/leaderboard/",
        views.consumer_leaderboard,
        name="consumer_leaderboard",
    ),
    path(
        "api/marketplace/reputation/",
        views.consumer_reputation,
        name="consumer_reputation",
    ),
]
