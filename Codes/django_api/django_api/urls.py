from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from paper import views as paper_views

def well_known_ignore(request):
    return HttpResponse(status=204)

from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('web_dashboard.urls')),
    path('api/', include('api.urls')),
    path('api/notes/', include('notes.urls')),
    path('api/model-versioning/', include('model_versioning.urls')),
    path('api/experiments/', include('experiment.urls')),
    path('api/jobs/', include('jobscheduler.urls')),
    # path('api/agents/', include('agents.urls')),
    path('api/', include('sweeps.urls')),
    path('api/terminal/', include('terminal.urls')),
    path('api/paper/', include('paper.urls')),        
    path('api/profiles/', include('profiles.urls')),
    path('api/messaging/', include('messaging.urls')),
    path('api/leaderboards/', include('leaderboards.urls')),
    path('api/gpu/', include('gpu_marketplace.urls')),
    path('api/gpu/', include('gpu_marketplace.urls')),
    path('api/escrow/', include('gpu_escrow.urls')),           # escrow app
    path('api/benchmarks/', include('gpu_benchmark.urls')),   # benchmark app
    path('api/federation/', include('gpu_federation.urls')),  # federation app
    path('api/reputation/', include('gpu_reputation.urls')),  # reputation app
    path('api/validator/', include('gpu_validator.urls')),    # validator app
    path('api/orgs/', include('organizations.urls')),
    path('api/portfolios/', include('portfolios.urls')),
    path('api/community/', include('community.urls')),
    path('api/reputation/', include('reputation.urls')),
    path('api/collaborators/', include('collaborator_matching.urls')),
    path('api/gamification/', include('gamification.urls')),
    path('api/streaks/', include('streaks.urls')),
    path('api/experts/', include('expert_wall.urls')),
    path('api/impact/', include('impact_predictor.urls')),
    path('api/literature/', include('literature_review.urls')),
path('api/discovery/', include('literature_discovery.urls')),
path('api/coscientist/', include('ai_coscientist.urls')),
path('api/ideas/', include('idea_tournament.urls')),
path('api/discovery/', include('computational_discovry.urls')),
path('api/accelerator/', include('experiment_accelerator.urls')),
path('api/anomaly-hpo/', include('anomaly_hpo.urls')),
path('api/research/', include('autonomous_research.urls')),
path('api/marketplace-ai/', include('marketplace_ai.urls')),
path('api/code/', include('coding_assistant.urls')),
path('api/recommender/', include('experiment_recommender.urls')),

    path('api/social/', include('social_network.urls')),

]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)