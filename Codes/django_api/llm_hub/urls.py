from django.urls import path
from . import views

urlpatterns = [
    path('playground/chat/', views.playground_chat, name='playground_chat'),
    path('playground/compare/', views.playground_compare, name='playground_compare'),
    path('prompts/', views.prompt_list, name='prompt_list'),
    path('prompts/<uuid:prompt_id>/version/', views.prompt_new_version, name='prompt_new_version'),
    path('eval/start/', views.start_evaluation, name='start_evaluation'),
    path('cost/', views.cost_dashboard, name='cost_dashboard'),
    path('memory/', views.memory_list, name='memory_list'),
    path('memory/context/', views.memory_context, name='memory_context'),
    # ... add full CRUD for tools, schemas, suites, etc.
]