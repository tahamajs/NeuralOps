from django.urls import path

from . import views

urlpatterns = [
    path("follow/<int:user_id>/", views.follow_user, name="follow_user"),
    path("unfollow/<int:user_id>/", views.unfollow_user, name="unfollow_user"),
    path("followers/<int:user_id>/", views.followers_list, name="followers_list"),
    path("following/<int:user_id>/", views.following_list, name="following_list"),
    path("feed/", views.activity_feed, name="activity_feed"),
    path("groups/", views.group_list, name="group_list"),
    path("groups/<uuid:group_id>/", views.group_detail, name="group_detail"),
    path("groups/<uuid:group_id>/join/", views.join_group, name="join_group"),
    path("groups/<uuid:group_id>/leave/", views.leave_group, name="leave_group"),
    path(
        "groups/<uuid:group_id>/discussions/",
        views.discussion_list,
        name="discussion_list",
    ),
    path(
        "discussions/<uuid:discussion_id>/",
        views.discussion_detail,
        name="discussion_detail",
    ),
    path(
        "discussions/<uuid:discussion_id>/reply/",
        views.reply_to_discussion,
        name="reply_to_discussion",
    ),
    path("alerts/", views.my_alerts, name="my_alerts"),
    path("alerts/<uuid:alert_id>/read/", views.mark_alert_read, name="mark_alert_read"),
    path(
        "alerts/mark-all-read/", views.mark_all_alerts_read, name="mark_all_alerts_read"
    ),
    path("suggested/", views.suggested_researchers, name="suggested_researchers"),
]
