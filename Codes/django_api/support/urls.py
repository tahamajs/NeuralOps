from django.urls import path

from . import views, views_solutions

urlpatterns = [
    # Tickets
    path("tickets/", views.ticket_list, name="ticket_list"),
    path("tickets/<uuid:ticket_id>/", views.ticket_detail, name="ticket_detail"),
    path(
        "tickets/<uuid:ticket_id>/messages/",
        views.ticket_messages,
        name="ticket_messages",
    ),
    path(
        "tickets/<uuid:ticket_id>/feedback/",
        views.submit_feedback,
        name="ticket_feedback",
    ),
    # Solutions
    path("solutions/", views.solution_list, name="solution_list"),
    path(
        "solutions/<uuid:solution_id>/upvote/",
        views.upvote_solution,
        name="upvote_solution",
    ),
]
path("support-solutions/", views_solutions.get_support_solutions),
