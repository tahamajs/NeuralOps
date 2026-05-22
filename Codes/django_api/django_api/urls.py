from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse

# ----------------------------------------------------------------------
# Helper view for well‑known (if needed)
# ----------------------------------------------------------------------
def well_known_ignore(request):
    return HttpResponse(status=204)

# ----------------------------------------------------------------------
# URL Patterns
# ----------------------------------------------------------------------
urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # Dashboard (main UI)
    path('', include('web_dashboard.urls')),

    # API root
    path('api/', include('api.urls')),

    # ------------------------------
    # Core NeuralOps Apps
    # ------------------------------
    path('api/notes/', include('notes.urls')),
    path('api/model-versioning/', include('model_versioning.urls')),
    path('api/experiments/', include('experiment.urls')),
    path('api/jobs/', include('jobscheduler.urls')),
    path('api/terminal/', include('terminal.urls')),
    path('api/paper/', include('paper.urls')),
    path('api/profiles/', include('profiles.urls')),
    path('api/messaging/', include('messaging.urls')),
    path('api/leaderboards/', include('leaderboards.urls')),
    path('api/orgs/', include('organizations.urls')),
    path('api/portfolios/', include('portfolios.urls')),
    path('api/community/', include('community.urls')),
    path('api/social/', include('social.urls')),
    path('api/social/', include('social_network.urls')),          # if separate
    path('api/research/', include('research.urls')),
    path('api/support/', include('support.urls')),

    # ------------------------------
    # GPU Marketplace & Finance
    # ------------------------------
    path('api/gpu/', include('gpu_marketplace.urls')),
    path('api/escrow/', include('gpu_escrow.urls')),
    path('api/benchmarks/', include('gpu_benchmark.urls')),
    path('api/federation/', include('gpu_federation.urls')),
    path('api/reputation/', include('gpu_reputation.urls')),
    path('api/validator/', include('gpu_validator.urls')),

    # ------------------------------
    # Gamification & Engagement
    # ------------------------------
    path('api/gamification/', include('gamification.urls')),
    path('api/streaks/', include('streaks.urls')),
    path('api/experts/', include('expert_wall.urls')),
    path('api/reputation/', include('reputation.urls')),

    # ------------------------------
    # AI & Research Agents
    # ------------------------------
    path('api/collaborators/', include('collaborator_matching.urls')),
    path('api/impact/', include('impact_predictor.urls')),
    path('api/literature/', include('literature_review.urls')),
    path('api/discovery/', include('literature_discovery.urls')),
    path('api/coscientist/', include('ai_coscientist.urls')),
    path('api/ideas/', include('idea_tournament.urls')),
    path('api/discovery/comp/', include('computational_discovery.urls')),  # fixed typo
    path('api/accelerator/', include('experiment_accelerator.urls')),
    path('api/anomaly-hpo/', include('anomaly_hpo.urls')),
    path('api/research/', include('autonomous_research.urls')),
    path('api/marketplace-ai/', include('marketplace_ai.urls')),
    path('api/code/', include('coding_assistant.urls')),
    path('api/recommender/', include('experiment_recommender.urls')),
    path('api/forensics/', include('failure_forensics.urls')),
    path('api/coach/', include('research_coach.urls')),
    path('api/agent/', include('research_agent.urls')),

    # ------------------------------
    # Sweeps (Hyperparameter Optimization)
    # ------------------------------
    path('api/sweeps/', include('sweeps.urls')),   # was previously under api/ directly – fixed
    path('api/explorer/', include('experiment_explorer.urls')),
path('api/live-dash/', include('live_dashboard.urls')),
path('api/model-hub/', include('model_hub.urls')),
path('api/lab/', include('lab_assistant.urls')),
path('api/health-monitor/', include('health_monitor.urls')),
path('api/community-lab/', include('community_lab.urls')),
path('api/playground/', include('playground.urls')),
path('api/datasets/', include('datasets.urls')),
path('parameter-interactions/', views.parameter_interactions, name='parameter_interactions'),
path('parameter-scatter/', views.run_parameter_scatter, name='parameter_scatter'),
path('runs/<uuid:run_id>/export-report-pdf/', views.export_experiment_report_pdf, name='export_report_pdf'),
path('api/llm/', include('llm_hub.urls')),
path('api/projects/', include('projects.urls')),
path('api/protocols/', include('protocols.urls')),
path('ws/collab/<uuid:project_id>/', CollaborationConsumer.as_asgi()),
path('api/governance/', include('governance.urls')),

]

# ----------------------------------------------------------------------
# Static & media serving in development
# ----------------------------------------------------------------------
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)