"""
Main URL configuration for the NeuralOps platform.

All app URLs are included under clear, non‑conflicting prefixes.
WebSocket routes are registered at the bottom.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import HttpResponse
from django.urls import include, path
# ----------------------------------------------------------------------
# Local imports – adjust paths if your apps live elsewhere
# ----------------------------------------------------------------------
from experiment import views as experiment_views
from paper.consumers import CollaborationConsumer  # WebSocket consumer
from sweeps import views as sweeps_views


# ----------------------------------------------------------------------
# Helper view (e.g. for well‑known paths)
# ----------------------------------------------------------------------
def well_known_ignore(request):
    return HttpResponse(status=204)


# ----------------------------------------------------------------------
# URL Patterns
# ----------------------------------------------------------------------
urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),
    # Main dashboard (SPA)
    path("", include("web_dashboard.urls")),
    # REST API root
    path("api/", include("api.urls")),
    # ------------------------------------------------------------------
    # Core experiment & project management
    # ------------------------------------------------------------------
    path("api/experiments/", include("experiment.urls")),
    path("api/sweeps/", include("sweeps.urls")),
    path("api/notes/", include("notes.urls")),
    path("api/model-versioning/", include("model_versioning.urls")),
    path("api/jobs/", include("jobscheduler.urls")),
    path("api/terminal/", include("terminal.urls")),
    path("api/paper/", include("paper.urls")),
    path("api/portfolios/", include("portfolios.urls")),
    
    
    # ------------------------------------------------------------------
    # Discovery, exploration & dashboards
    # ------------------------------------------------------------------
    path("api/explorer/", include("experiment_explorer.urls")),
    path("api/live-dash/", include("live_dashboard.urls")),
    path("api/model-hub/", include("model_hub.urls")),
    path("api/lab/", include("lab_assistant.urls")),
    path("api/health-monitor/", include("health_monitor.urls")),
    path("api/community-lab/", include("community_lab.urls")),
    path("api/playground/", include("playground.urls")),
    path("api/datasets/", include("dataset_app.urls")),
    path("api/llm/", include("llm_hub.urls")),
    # ------------------------------------------------------------------
    # Social, community & profiles
    # ------------------------------------------------------------------
    path("api/profiles/", include("profiles.urls")),
    path("api/messaging/", include("messaging.urls")),
    path("api/leaderboards/", include("leaderboards.urls")),
    path("api/orgs/", include("organizations.urls")),
    path("api/community/", include("community.urls")),
    path("api/social/", include("social.urls")),
    path(
        "api/social-network/", include("social_network.urls")
    ),  # separate from 'social'
    path("api/support/", include("support.urls")),
    # ------------------------------------------------------------------
    # GPU marketplace & finance
    # ------------------------------------------------------------------
    path("api/gpu/", include("gpu_marketplace.urls")),
    path("api/gpu/escrow/", include("gpu_escrow.urls")),
    path("api/gpu/benchmarks/", include("gpu_benchmark.urls")),
    path("api/gpu/federation/", include("gpu_federation.urls")),
    path("api/gpu/reputation/", include("gpu_reputation.urls")),
    path("api/gpu/validator/", include("gpu_validator.urls")),
    # GPU finance (separate prefix to avoid clash)
    path("api/gpu-finance/", include("gpu_finance.urls")),
    # ------------------------------------------------------------------
    # Gamification, streaks & reputation
    # ------------------------------------------------------------------
    path("api/gamification/", include("gamification.urls")),
    path("api/streaks/", include("streaks.urls")),
    path("api/reputation/", include("reputation.urls")),  # general reputation
    path("api/experts/", include("expert_wall.urls")),
    # ------------------------------------------------------------------
    # AI research agents & assistants
    # ------------------------------------------------------------------
    path("api/collaborators/", include("collaborator_matching.urls")),
    path("api/impact/", include("impact_predictor.urls")),
    path("api/literature/", include("literature_review.urls")),
    path("api/discovery/", include("literature_discovery.urls")),
    path("api/coscientist/", include("ai_coscientist.urls")),
    path("api/ideas/", include("idea_tournament.urls")),
    path("api/computational/", include("computational_discovery.urls")),
    path("api/accelerator/", include("experiment_accelerator.urls")),
    path("api/anomaly-hpo/", include("anomaly_hpo.urls")),
    path("api/autonomous-research/", include("autonomous_research.urls")),
    path("api/marketplace-ai/", include("marketplace_ai.urls")),
    path("api/code/", include("coding_assistant.urls")),
    path("api/recommender/", include("experiment_recommender.urls")),
    path("api/forensics/", include("failure_forensics.urls")),
    path("api/coach/", include("research_coach.urls")),
    path("api/agent/", include("research_agent.urls")),
    path("api/agents/", include("agents.urls")),
    # ------------------------------------------------------------------
    # Additional research & integration services
    # ------------------------------------------------------------------
    path("api/research/", include("research.urls")),  # core research app
    path("api/research-discovery/", include("research_discovery.urls")),
    path("api/governance/", include("governance.urls")),
    path("api/integrations/", include("integrations.urls")),
    path("api/mlops/", include("mlops.urls")),  # data quality, monitoring
    # ------------------------------------------------------------------
    # Stand‑alone views (parameter interactions, PDF export)
    # ------------------------------------------------------------------
    path(
        "parameter-interactions/",
        sweeps_views.parameter_interactions,
        name="parameter_interactions",
    ),
    path(
        "parameter-scatter/",
        sweeps_views.run_parameter_scatter,
        name="parameter_scatter",
    ),
    path(
        "runs/<uuid:run_id>/export-report-pdf/",
        experiment_views.export_experiment_report_pdf,
        name="export_report_pdf",
    ),
    # ------------------------------------------------------------------
    # WebSocket routes
    # ------------------------------------------------------------------
    path(
        "ws/collab/<uuid:project_id>/",
        CollaborationConsumer.as_asgi(),
        name="collab_ws",
    ),
]

# ----------------------------------------------------------------------
# Serve static & media files in development
# ----------------------------------------------------------------------
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
