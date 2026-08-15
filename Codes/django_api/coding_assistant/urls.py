from django.urls import path

from . import views

urlpatterns = [
    path("", views.session_list, name="coding_assistant_root"),
    # Sessions
    path("sessions/", views.session_list, name="code_sessions"),
    path(
        "sessions/<uuid:session_id>/", views.session_detail, name="code_session_detail"
    ),
    path(
        "sessions/<uuid:session_id>/send/", views.send_message, name="code_send_message"
    ),
    path(
        "sessions/<uuid:session_id>/stream/",
        views.stream_message,
        name="code_stream_message",
    ),
    # Generated Code
    path(
        "sessions/<uuid:session_id>/files/",
        views.save_generated_code,
        name="code_save_file",
    ),
    path("files/<uuid:file_id>/", views.generated_code_detail, name="code_file_detail"),
    # Code Review
    path("files/<uuid:file_id>/review/", views.request_review, name="code_review"),
    path("files/<uuid:file_id>/reviews/", views.get_review, name="code_get_reviews"),
    # Test Generation
    path("files/<uuid:file_id>/tests/", views.generate_test, name="code_generate_test"),
    path("files/<uuid:file_id>/tests/get/", views.get_test, name="code_get_test"),
    # Documentation
    path(
        "files/<uuid:file_id>/document/",
        views.generate_documentation,
        name="code_generate_documentation",
    ),
    # Experiment Config
    path(
        "generate-config/",
        views.generate_experiment_config,
        name="code_generate_config",
    ),
]
