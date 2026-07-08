from django.urls import path

from . import views

urlpatterns = [
    path("sessions/", views.list_sessions, name="terminal_sessions"),
    path("sessions/<uuid:session_id>/kill/", views.kill_session, name="terminal_kill"),
]
