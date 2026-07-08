from django.urls import path

from . import views

urlpatterns = [
    # Prompts
    path("prompts/", views.prompt_list, name="prompt_list"),
    path("prompts/<uuid:template_id>/", views.prompt_detail, name="prompt_detail"),
    path(
        "prompts/<uuid:template_id>/new-version/",
        views.prompt_new_version,
        name="prompt_new_version",
    ),
    path(
        "prompts/<uuid:template_id>/test-render/",
        views.prompt_test_render,
        name="prompt_test_render",
    ),
    # Model Configs
    path("model-configs/", views.model_config_list, name="model_config_list"),
    path(
        "model-configs/<uuid:config_id>/",
        views.model_config_detail,
        name="model_config_detail",
    ),
    # Playground Sessions
    path("sessions/", views.session_list, name="session_list"),
    path(
        "sessions/<uuid:session_id>/send/",
        views.session_send_message,
        name="session_send_message",
    ),
    # Comparisons
    path("comparisons/", views.comparison_list, name="comparison_list"),
    # Tools
    path("tools/", views.tool_list, name="tool_list"),
]
