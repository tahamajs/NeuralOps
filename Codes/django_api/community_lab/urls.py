from django.urls import path

from . import views

urlpatterns = [
    path("", views.lab_list, name="community_lab_root"),
    # Labs
    path("labs/", views.lab_list, name="lab_list"),
    path("labs/<uuid:lab_id>/", views.lab_detail, name="lab_detail"),
    # Agent Roles
    path("labs/<uuid:lab_id>/agents/", views.agent_role_list, name="agent_role_list"),
    # Chatrooms
    path("labs/<uuid:lab_id>/chatrooms/", views.chatroom_list, name="chatroom_list"),
    path(
        "labs/<uuid:lab_id>/chatrooms/<uuid:room_id>/messages/",
        views.chatroom_messages,
        name="chatroom_messages",
    ),
    # Workflows
    path("labs/<uuid:lab_id>/workflows/", views.workflow_list, name="workflow_list"),
    # Discoveries
    path("discoveries/", views.discovery_list, name="discovery_list"),
    path(
        "labs/<uuid:lab_id>/discoveries/",
        views.discovery_list,
        name="lab_discovery_list",
    ),
    path(
        "discoveries/<uuid:submission_id>/review/",
        views.submit_review,
        name="submit_review",
    ),
    # Invitations
    path("labs/<uuid:lab_id>/invite/", views.invite_to_lab, name="invite_to_lab"),
    path("invitations/", views.my_invitations, name="my_invitations"),
]
